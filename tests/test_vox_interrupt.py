"""
Offline test for VOX barge-in (pipeline/bargein.py and its use in
pipeline/main.py):

- a single mic frame over vad.threshold no longer interrupts a turn;
- about 300 ms (vad.interrupt_min_ms) of continuous speech does, while the
  pod is playing, and the interrupt is logged with the pod's name, the
  speech duration, peak/average probability and the playback state;
- speech while nothing is playing (waiting for ASR or the agent) does not
  interrupt, and is logged once per turn;
- a longer dip below the threshold starts the count over (one 32 ms dip is
  tolerated), and every turn starts from zero;
- config overrides (interrupt_min_ms, interrupt_only_while_playing);
- the playback estimate built from the audio send_sentence streams;
- through main.py's real udp_listener over a local UDP socket: VOX needs
  the sustained speech, PTT still interrupts on its first frame.

Reuses the stubs and fakes of tests/test_pause_flush.py (no network other
than localhost, no audio hardware). Run from the repo root:
python3 tests/test_vox_interrupt.py
"""
import asyncio
import glob
import logging
import os
import socket
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs)
import test_led_state as ls    # noqa: E402

from pipeline import bargein   # noqa: E402

m = pf.m
TAG = pf.TAG
FRAME = 512 / 16000            # one mic frame: 512 samples at 16 kHz = 32 ms
MARGIN = bargein.PLAYBACK_MARGIN_S


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def take(self, prefix=""):
        out = [x for x in self.lines if x.startswith(prefix)]
        self.lines.clear()
        return out


LOG = LogCapture()


def check(name, cond, detail=""):
    pf.check(name, cond, detail)


def make(cfg=None, playing=True, host="onju-test"):
    """A BargeIn at t=100 s; if `playing`, a 10 s reply is being played."""
    vad = {"threshold": 0.5}
    vad.update(cfg or {})
    b = bargein.BargeIn(host, vad)
    if playing:
        start = b.start_segment(now=100.0)
        b.audio_sent(start, 10.0, now=100.0)
    return b


def feed(b, probs, frame_s=FRAME, t0=100.5):
    """Feed probabilities one frame at a time; return the 1-based index of
    the frame that interrupted, or None."""
    for i, p in enumerate(probs):
        if b.on_frame(p, frame_s, now=t0 + i * frame_s):
            return i + 1
    return None


def unit_tests():
    # 1. One frame over the threshold (the old trigger) does nothing.
    b = make()
    LOG.take()
    check("single frame over threshold during playback: no interrupt",
          feed(b, [0.9, 0.1, 0.1]) is None)

    # 2. Sustained speech during playback: interrupts on the frame that
    #    reaches 300 ms (10 frames x 32 ms = 320 ms), not before; logged.
    b = make()
    LOG.take()
    i = feed(b, [0.6, 0.8, 0.9, 0.7, 0.8, 0.9, 0.95, 0.8, 0.7, 0.6, 0.9])
    lines = LOG.take("VOX")
    check("~300 ms sustained speech during playback: interrupts on frame 10 (320 ms)",
          i == 10, f"i={i}")
    check("interrupt is logged with pod name, duration, peak/avg, playback state",
          lines == ["VOX  interrupt from onju-test: 320 ms of speech "
                    "(peak 0.95, avg 0.78), playback active"], f"{lines}")

    # 3. Sustained speech while nothing plays (waiting for the agent):
    #    no interrupt, one log line for the whole turn.
    b = make(playing=False)
    LOG.take()
    i = feed(b, [0.9] * 40 + [0.1] * 3 + [0.9] * 40)
    lines = LOG.take("VOX")
    check("2.5 s of speech while nothing is playing: no interrupt", i is None, f"i={i}")
    check("ignored speech (nothing playing) logged once per turn",
          len(lines) == 1 and "speech ignored from onju-test (nothing playing)" in lines[0]
          and "320 ms" in lines[0] and "playback not active" in lines[0], f"{lines}")

    # 3b. The 09:07 incident: a lone noisy frame during the wait.
    b = make(playing=False)
    LOG.take()
    i = feed(b, [0.1, 0.55, 0.1, 0.1, 0.1])
    lines = LOG.take("VOX")
    check("incident replay: one 32 ms frame while waiting -> no interrupt, one log line",
          i is None and len(lines) == 1 and "(nothing playing): 32 ms" in lines[0], f"{i} {lines}")

    # 4. Reset: two quiet frames (64 ms) start the count over; a single
    #    quiet frame (32 ms) is tolerated but not counted.
    b = make()
    i = feed(b, [0.9] * 9 + [0.2, 0.2] + [0.9] * 9)
    check("two quiet frames reset the count (9 + 9 frames: no interrupt)", i is None, f"i={i}")
    b = make()
    i = feed(b, [0.9] * 9 + [0.2] + [0.9] * 2)
    check("one quiet frame is tolerated, not counted (interrupt on the 10th speech frame)",
          i == 11, f"i={i}")
    b = make()
    LOG.take()
    feed(b, [0.9] * 3 + [0.1] * 2 + [0.9] * 2 + [0.1] * 2)
    lines = LOG.take("VOX")
    check("too-short speech during playback logged once per turn",
          len(lines) == 1 and "(too short): 96 ms" in lines[0] and "need 300 ms" in lines[0]
          and "playback active" in lines[0], f"{lines}")

    # 5. Per-turn reset: a run can't carry over into the next turn, and the
    #    'ignored' line may be logged again in the new turn.
    b = make()
    feed(b, [0.9] * 8)
    b.new_turn()
    i = feed(b, [0.9] * 2)
    check("new turn: speech count starts from zero", i is None, f"i={i}")
    b = make(playing=False)
    LOG.take()
    feed(b, [0.9, 0.1, 0.1])
    b.new_turn()
    feed(b, [0.9, 0.1, 0.1])
    check("new turn: ignored-speech line logged again (once per turn)",
          len(LOG.take("VOX  speech ignored")) == 2)

    # 6. Config overrides.
    b = make({"interrupt_min_ms": 100})
    check("interrupt_min_ms: 100 -> interrupts on frame 4 (128 ms)",
          feed(b, [0.9] * 10) == 4)
    b = make({"interrupt_only_while_playing": False}, playing=False)
    LOG.take()
    i = feed(b, [0.9] * 12)
    lines = LOG.take("VOX")
    check("interrupt_only_while_playing: false -> sustained speech interrupts with nothing playing",
          i == 10 and len(lines) == 1 and lines[0].endswith("playback not active"), f"{i} {lines}")
    b = make({"interrupt_min_ms": 0, "interrupt_only_while_playing": False}, playing=False)
    check("interrupt_min_ms: 0 + only_while_playing: false -> old behaviour (first frame)",
          feed(b, [0.1, 0.9]) == 2)
    b = make({"threshold": 0.8})
    check("vad.threshold is used (0.7 is below a 0.8 threshold)",
          feed(b, [0.7] * 20) is None and feed(b, [0.9] * 10) == 10)
    b = make()
    check("duration comes from the frame length (16 ms frames: 19 frames = 304 ms)",
          feed(b, [0.9] * 30, frame_s=256 / 16000) == 19)
    b = bargein.BargeIn("x", {"threshold": 0.5})
    check("defaults without the new keys: 300 ms, only while playing",
          abs(b.min_s - 0.3) < 1e-9 and b.only_while_playing is True)

    # 7. Playback estimate.
    b = bargein.BargeIn("x", {"threshold": 0.5})
    check("idle pod: not playing", not b.is_playing(now=50.0))
    s1 = b.start_segment(now=100.0)
    check("pod counts as playing as soon as the audio connection is open",
          s1 == 100.0 and b.is_playing(now=100.1))
    b.audio_sent(s1, 2.0, now=100.2)              # 2 s of audio sent quickly
    s2 = b.start_segment(now=100.5)               # next sentence while it plays
    b.audio_sent(s2, 1.0, now=100.7)
    exp = 100.0 + 2.0 + MARGIN + 1.0 + MARGIN
    check("second sentence queues behind the first: end = both durations + a margin each",
          abs(s2 - (102.0 + MARGIN)) < 1e-9 and abs(b.playing_until - exp) < 1e-9,
          f"s2={s2} until={b.playing_until} exp={exp}")
    check("playing just before the estimated end, not after",
          b.is_playing(now=exp - 0.01) and not b.is_playing(now=exp + 0.01))
    b = bargein.BargeIn("x", {"threshold": 0.5})
    s = b.start_segment(now=10.0)
    b.audio_sent(s, 0.5, now=13.0)                # TTS slower than real time
    check("slow TTS: still playing until the last frame sent + margin",
          b.is_playing(now=13.0 + MARGIN - 0.01) and not b.is_playing(now=13.0 + MARGIN + 0.01))


class Encoder50:
    """Fake Opus encoder that makes one frame per 320 samples (20 ms), like
    the real one, so the frame count gives the true audio length."""
    def __init__(self, *a):
        self.carry = 0

    def encode_chunk(self, pcm):
        n = (self.carry + len(pcm)) // 640
        self.carry = (self.carry + len(pcm)) % 640
        return [b"f"] * n

    def flush(self):
        if self.carry:
            self.carry = 0
            return [b"f"]
        return []


class PlayRecorder(ls.StateRecorder):
    """Also notes the pod's estimated playback state when each sentence's
    audio connection is open (TTS starts) and after its audio was sent."""

    def __init__(self):
        super().__init__()
        self.play_log = []     # (sentence, "open"/"sent", is_playing, playing_until - t0)

    def _note(self, sentence, what):
        b = bargein._by_host["onju-test"]
        self.play_log.append((sentence, what, b.is_playing(), b.playing_until - self.t0))

    async def synthesize_stream(self, sentence, voice, config):
        self._note(sentence, "open")
        async for pcm in super().synthesize_stream(sentence, voice, config):
            yield pcm
        self._note(sentence, "sent")   # runs after main.py wrote the frames


async def turn_playback_test():
    """A real process_utterances turn: stall line, 3 s agent wait, answer.
    The pod counts as playing while the stall and the answer are sent and
    played, and not during the wait in between."""
    stall, answer = f"{TAG} one sec.", f"{TAG} the answer is forty two."
    samples = []
    stop = asyncio.Event()

    async def sample():
        while not stop.is_set():
            b = bargein._by_host.get("onju-test")
            samples.append((time.monotonic(), bool(b and b.is_playing())))
            await asyncio.sleep(0.02)

    # run_turn installs pf.FakeEncoder (one frame per batch); swap in one
    # that makes 50 frames per second of audio, like the real encoder.
    pf_enc = pf.FakeEncoder
    pf.FakeEncoder = Encoder50
    bargein._by_host.pop("onju-test", None)
    # Leftover from before the turn: 8 frames of speech and a logged
    # 'ignored' line. The turn must start from zero.
    left = bargein.for_device(types.SimpleNamespace(hostname="onju-test"), {"vad": {}})
    feed(left, [0.9] * 8 + [0.1, 0.1] + [0.9] * 8, t0=0.0)
    sampler = asyncio.create_task(sample())
    try:
        rec, device, _ = await ls.run_turn([3.0, answer], rec=PlayRecorder(), stall=stall)
    finally:
        stop.set()
        await sampler
        pf.FakeEncoder = pf_enc
    check("turn: process_utterances resets the speech count and log-once flags",
          left.speech_s == 0 and left.n_frames == 0 and not left._logged,
          f"speech_s={left.speech_s} logged={left._logged}")
    log_ = {(s, w): (p, u) for s, w, p, u in rec.play_log}
    t_stall, t_ans = rec.time_of(stall), rec.time_of(answer)
    # Each fake sentence is 1 s of audio (50 frames of 20 ms).
    check("turn: pod counts as playing once the stall's audio connection is open",
          log_[(stall, "open")][0] is True, f"{rec.play_log}")
    until = log_[(stall, "sent")][1]
    check("turn: stall estimated to play until its start + 1 s + margin",
          log_[(stall, "sent")][0] and abs(until - (t_stall + 1.0 + MARGIN)) < 0.1,
          f"until={until:.2f} t_stall={t_stall:.2f}")
    gap = [p for t, p in samples if until + 0.1 < t - rec.t0 < t_ans - 0.1]
    check("turn: not playing while waiting for the agent after the stall",
          len(gap) > 10 and not any(gap), f"{len(gap)} samples, {sum(gap)} playing")
    until = log_[(answer, "sent")][1]
    check("turn: playing again for the answer, until its start + 1 s + margin",
          log_[(answer, "open")][0] and log_[(answer, "sent")][0]
          and abs(until - (t_ans + 1.0 + MARGIN)) < 0.1,
          f"until={until:.2f} t_ans={t_ans:.2f} {rec.play_log}")


class FakeVAD:
    def __init__(self):
        self.speech_prob = 0.0
        self.recording = False
        self.buffer = []
        self.next = 0.0

    def process_frame(self, pcm):
        self.speech_prob = self.next
        return None


async def udp_test():
    """main.udp_listener on a real localhost UDP socket, with a fake VAD
    whose probability the test sets before each packet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    config = {
        "network": {"udp_port": port, "tcp_port": 3001},
        "audio": {"chunk_size": 512, "sample_rate": 16000},
        "device": {"led_power": 50, "led_update_period": 0.25, "led_fade": 6},
        "vad": {"threshold": 0.5},
    }
    m.decode_ulaw = lambda data: [0] * len(data)   # 1 byte of u-law = 1 sample

    def make_dev(host, ptt):
        return types.SimpleNamespace(
            hostname=host, ip="127.0.0.1", ptt=ptt, processing=True,
            interrupted=asyncio.Event(), vad=None if ptt else FakeVAD(),
            ptt_buffer=[], vad_writer=None, led_power=0, led_update_time=0.0)

    manager = types.SimpleNamespace(devices={}, get_by_ip=None, get_most_recent=lambda: None)
    q: asyncio.Queue = asyncio.Queue()
    listener = asyncio.create_task(m.udp_listener(config, manager, q))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    await asyncio.sleep(0.05)

    async def send(dev, prob):
        if dev.vad is not None:
            dev.vad.next = prob
        tx.sendto(b"\xff" * 512, ("127.0.0.1", port))
        await asyncio.sleep(0.01)

    try:
        # VOX, reply playing: 9 speech frames don't interrupt, the 10th does.
        vox = make_dev("onju-vox", ptt=False)
        manager.devices = {vox.hostname: vox}
        manager.get_by_ip = lambda ip: vox
        bargein._by_host.pop(vox.hostname, None)
        b = bargein.for_device(vox, config)
        b.new_turn()
        b.audio_sent(b.start_segment(), 10.0)
        LOG.take()
        await send(vox, 0.9)
        await send(vox, 0.1)
        await send(vox, 0.1)
        check("udp_listener: one loud frame during playback no longer interrupts",
              not vox.interrupted.is_set())
        for _ in range(9):
            await send(vox, 0.9)
        after9 = vox.interrupted.is_set()
        await send(vox, 0.9)
        lines = LOG.take("VOX  interrupt")
        check("udp_listener: interrupt after 10 frames (320 ms) of speech, logged once",
              not after9 and vox.interrupted.is_set() and len(lines) == 1
              and "onju-vox" in lines[0], f"after9={after9} {lines}")
        for _ in range(12):
            await send(vox, 0.9)
        check("udp_listener: no repeat interrupt log once the turn is interrupted",
              LOG.take("VOX  interrupt") == [])

        # VOX, waiting for the agent (nothing playing): 1 s of speech ignored.
        vox.interrupted.clear()
        b.playing_until = 0.0
        b.new_turn()
        for _ in range(31):
            await send(vox, 0.9)
        lines = LOG.take("VOX")
        check("udp_listener: 1 s of speech while nothing plays -> no interrupt, logged once",
              not vox.interrupted.is_set() and len(lines) == 1 and "nothing playing" in lines[0],
              f"{lines}")

        # PTT: first frame during a turn interrupts, playback or not, as before.
        ptt = make_dev("onju-ptt", ptt=True)
        manager.devices = {ptt.hostname: ptt}
        manager.get_by_ip = lambda ip: ptt
        LOG.take()
        await send(ptt, None)
        lines = LOG.take()
        check("udp_listener: PTT still interrupts on its first frame, nothing playing, logged",
              ptt.interrupted.is_set() and lines == ["PTT  interrupt from onju-ptt"]
              and len(ptt.ptt_buffer) == 1, f"{lines}")
        await send(ptt, None)
        check("udp_listener: PTT keeps buffering the next utterance",
              len(ptt.ptt_buffer) == 2 and LOG.take() == [])
    finally:
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass
        tx.close()


async def main():
    logging.getLogger().addHandler(LOG)
    logging.getLogger().setLevel(logging.INFO)
    # Keep the console quiet: only our capture handler gets INFO lines.
    for h in logging.getLogger().handlers:
        if h is not LOG:
            h.setLevel(logging.WARNING)
    unit_tests()
    await turn_playback_test()
    await udp_test()

    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(pf.CHECKS)
    print(f"{total - len(pf.FAILS)}/{total} passed")
    return 1 if pf.FAILS else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
