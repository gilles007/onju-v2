"""
Offline test for the "pause flush" in pipeline/main.py process_utterances():
a held sentence is played as soon as the agent's text stream goes quiet for
conversation.pause_flush_s, instead of waiting for the next sentence.

No network, no audio hardware, no TTS/ASR: the modules that need them are
replaced with stubs before pipeline.main is imported, and the device, ASR,
TTS and audio sends are fakes that record what would have been sent.
(send_sentence still writes its debug WAVs to /tmp/tts-debug; the test
removes the ones it created.)

Run from the repo root:  python3 tests/test_pause_flush.py
"""
import asyncio
import glob
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _noop(*a, **k):
    return None


async def _anoop(*a, **k):
    return None


# Third-party packages: stub only if they are not installed.
for _pkg, _attrs in (
    ("numpy", {"int16": "int16"}),
    ("yaml", {"safe_load": _noop}),
    ("openai", {"AsyncOpenAI": object}),
):
    try:
        __import__(_pkg)
    except ImportError:
        _stub(_pkg, **_attrs)

# Pipeline modules that pull in opus, VAD, HTTP clients or sockets.
_stub("pipeline.audio", decode_ulaw=_noop, opus_encode=_noop,
      opus_frames_to_tcp_payload=_noop, pcm_to_wav=_noop, OpusStreamEncoder=object)
_stub("pipeline.device", Device=object, DeviceManager=object)
_stub("pipeline.protocol", send_audio=_anoop, send_state=_anoop, send_led_blink=_anoop,
      open_led_connection=_anoop, write_led_blink=_noop, close_led_connection=_anoop,
      open_audio_connection=_anoop, write_audio_frames=_noop, close_audio_connection=_anoop)
_services = _stub("pipeline.services")
_services.asr = _stub("pipeline.services.asr", transcribe=_anoop)
_services.tts = _stub("pipeline.services.tts", synthesize_stream=_noop)

import pipeline.main as m  # noqa: E402

TAG = "Zzpf"          # every test sentence starts with this, to find debug WAVs
MIC = 60              # default_mic_timeout in the fake config
PAUSE = 0.3           # pause_flush_s used by the tests (production default 0.8)


class FakeEncoder:
    def __init__(self, *a):
        pass

    def encode_chunk(self, pcm):
        return [b"f"] if pcm else []

    def flush(self):
        return []


class FakeWriter:
    async def drain(self):
        pass


class Recorder:
    """Records every audio connection the pipeline would open to the Nest."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.events = []        # (t, kind, sentence or None, mic_timeout)
        self._mic = None

    def now(self):
        return time.monotonic() - self.t0

    async def open_audio_connection(self, ip, port, mic_timeout, volume, fade):
        self._mic = mic_timeout
        return FakeWriter()

    async def synthesize_stream(self, sentence, voice, config):
        self.events.append((self.now(), "sentence", sentence, self._mic))
        await asyncio.sleep(0.02)
        yield b"\0\0" * 16000     # 1 s of silence; enough for one batch

    async def send_audio(self, ip, port, payload, mic_timeout, volume, fade):
        self.events.append((self.now(), "empty", None, mic_timeout))

    def sent(self):
        return [(s, mic) for _, k, s, mic in self.events if k == "sentence"]

    def empties(self):
        return [mic for _, k, _, mic in self.events if k == "empty"]

    def time_of(self, sentence):
        return next(t for t, k, s, _ in self.events if s == sentence)


class FakeConversation:
    """Streams text deltas; a float in the script means 'go quiet that long'."""

    def __init__(self, script, fail_at_end=False):
        self.script = script
        self.fail_at_end = fail_at_end

    async def stream(self, text, extra_context=None):
        for item in self.script:
            if isinstance(item, float):
                await asyncio.sleep(item)
            else:
                yield item
        if self.fail_at_end:
            raise RuntimeError("stream broke")

    def commit(self, text):
        self.committed = text


class FakeAudio:
    def astype(self, _):
        return self

    def tobytes(self):
        return b"\0\0"


def make_device(conv, ptt=False):
    return types.SimpleNamespace(
        hostname="onju-test", ip="127.0.0.1", ptt=ptt, voice="pepper",
        processing=False, interrupted=asyncio.Event(), vad_writer=None,
        conversation=conv, last_user_text=None, last_response=None,
    )


def make_config(pause=PAUSE, stall=False):
    conv = {"backend": "agentic" if stall else "conversational"}
    if pause is not None:
        conv["pause_flush_s"] = pause
    return {
        "network": {"tcp_port": 3001},
        "device": {"default_mic_timeout": MIC, "default_volume": 10, "led_fade": 6},
        "audio": {"sample_rate": 16000, "opus_frame_size": 320},
        "asr": {"url": "fake"},
        "conversation": conv,
    }


async def run_turn(script, *, pause=PAUSE, ptt=False, stall=None, fail_at_end=False,
                   interrupt_after=None):
    rec = Recorder()
    m.asr.transcribe = lambda *a, **k: _ret({"text": "hello", "no_speech_prob": 0.0})
    m.stall_mod.decide_stall = lambda *a, **k: _ret(stall)
    m.tts.synthesize_stream = rec.synthesize_stream
    m.open_audio_connection = rec.open_audio_connection
    m.write_audio_frames = lambda w, f: True
    m.close_audio_connection = _anoop
    m.send_audio = rec.send_audio
    m.OpusStreamEncoder = FakeEncoder

    device = make_device(FakeConversation(script, fail_at_end), ptt=ptt)
    if interrupt_after is not None:
        # Like udp_listener does when the user talks over the reply.
        asyncio.get_running_loop().call_later(interrupt_after, device.interrupted.set)
    q: asyncio.Queue = asyncio.Queue()
    await q.put((device, FakeAudio()))
    worker = asyncio.create_task(m.process_utterances(make_config(pause, stall is not None), None, q))
    # Abort paths `continue` before utterance_queue.task_done(), so q.join()
    # can't be used; the turn is over once the item is taken and
    # device.processing is back to False (set in process_utterances' finally).
    try:
        deadline = time.monotonic() + 20
        await asyncio.sleep(0)
        while not q.empty() or device.processing:
            if time.monotonic() > deadline:
                raise TimeoutError("turn did not finish")
            await asyncio.sleep(0.01)
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    return rec, device


async def _ret(value):
    return value


FAILS = []
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if not cond and detail else ""))
    if not cond:
        FAILS.append(name)


async def main():
    a, b, c = f"{TAG} let me look that up.", f"{TAG} the answer is forty two.", f"{TAG} anything else?"

    # 1. Progress line, long silence, then the answer.
    rec, dev = await run_turn([a + " ", 1.5, b + " ", c])
    check("gap: held sentence played during the gap, non-final",
          rec.sent()[0] == (a, 0) and rec.time_of(a) < 1.2,
          f"{rec.events}")
    check("gap: later sentences keep lookahead, last one is final",
          rec.sent()[1:] == [(b, 0), (c, MIC)], f"{rec.sent()}")
    check("gap: no extra mic reopen when the last sentence was final",
          rec.empties() == [], f"{rec.empties()}")
    check("gap: last_response has the whole reply", dev.last_response == f"{a} {b} {c}")

    # 2. The last sentence gets flushed early, then the stream just ends.
    rec, dev = await run_turn([a + " ", 1.0])
    check("early flush of last sentence: sent non-final during the gap",
          rec.sent() == [(a, 0)] and rec.time_of(a) < 0.9, f"{rec.events}")
    check("early flush of last sentence: turn ends with a mic reopen",
          rec.empties() == [MIC], f"{rec.empties()}")
    check("early flush of last sentence: last_response kept", dev.last_response == a)

    # 3. Fast stream: behaves exactly as before (lookahead, no pause flush).
    rec, _ = await run_turn([a + " ", b + " ", c])
    check("fast stream: unchanged (partial, partial, final)",
          rec.sent() == [(a, 0), (b, 0), (c, MIC)] and rec.empties() == [], f"{rec.events}")

    # 4. Tap-to-talk (PTT): same end as the old 'stall played, nothing else' path.
    rec, _ = await run_turn([a + " ", 1.0], ptt=True)
    check("ptt early flush: sentence sent, no mic reopen (as for PTT before)",
          rec.sent() == [(a, 0)] and rec.empties() == [], f"{rec.events}")

    # 5. pause_flush_s: 0 turns it off (old behaviour: held until the stream ends).
    rec, _ = await run_turn([a + " ", 1.0], pause=0)
    check("pause_flush_s=0: sentence held until the end, sent final",
          rec.sent() == [(a, MIC)] and rec.time_of(a) >= 1.0, f"{rec.events}")

    # 6. Old config without the key: default 0.8 s applies.
    rec, _ = await run_turn([a + " ", 1.5, b], pause=None)
    t = rec.time_of(a)
    check("no config key: default ~0.8 s pause flush",
          0.7 <= t < 1.4 and rec.sent() == [(a, 0), (b, MIC)], f"t={t:.2f} {rec.sent()}")

    # 7. Stall line plays first; stream error after an early flush reopens the mic.
    rec, _ = await run_turn([a + " ", 1.0], stall=f"{TAG} one sec.", fail_at_end=True)
    check("stall first, then held sentence; stream error -> LLM failed path reopens mic",
          rec.sent() == [(f"{TAG} one sec.", 0), (a, 0)] and rec.empties() == [MIC], f"{rec.events}")

    # 8. Interrupted while a sentence is held: aborts, and the producer task
    #    is cancelled instead of waiting out the agent's 5 s of silence.
    t0 = time.monotonic()
    rec, _ = await run_turn([a + " ", b + " ", 5.0, c], interrupt_after=0.1)
    check("interrupt: turn aborted quickly, nothing after the interrupt is sent",
          time.monotonic() - t0 < 2.0 and (c, MIC) not in rec.sent(), f"{rec.events}")

    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(CHECKS)
    print(f"{total - len(FAILS)}/{total} passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
