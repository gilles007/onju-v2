"""
Offline test for the LED status (0xEE) messages sent by
pipeline/main.py process_utterances():

- THINKING is sent when the utterance is taken for ASR (user stopped
  talking), before any audio of the reply;
- THINKING is sent again after the stall line and after each sentence
  played early by the pause flush, which includes the grok bridge's
  keepalive line (the pod goes SPEAKING -> IDLE after every playback),
  but not between fast back-to-back sentences;
- IDLE is always the last state of a turn, whatever path the turn took
  (answer, no speech, LLM failure, PTT with empty reply, interrupt,
  ASR error);
- state sends never hold up the turn, even when they are slow or hang.

Reuses the stubs and fakes of tests/test_pause_flush.py (no network,
no audio hardware). Run from the repo root:  python3 tests/test_led_state.py
"""
import asyncio
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs)

m = pf.m
TAG = pf.TAG
MIC = pf.MIC
THINKING, IDLE = 2, 0
NAMES = {0: "IDLE", 2: "THINKING"}


class StateRecorder(pf.Recorder):
    """Records audio sends (via pf.Recorder) and 0xEE state sends in one
    timeline. `state_delay` makes each state send slow; `state_hang`
    makes it never finish (the pipeline must time it out)."""

    def __init__(self, state_delay=0.0, state_hang=False, asr=None, asr_delay=0.1):
        super().__init__()
        self.state_delay = state_delay
        self.state_hang = state_hang
        self.asr_result = asr if asr is not None else {"text": "hello", "no_speech_prob": 0.0}
        self.asr_delay = asr_delay

    async def send_state(self, ip, port, state, timeout=0.2):
        self.events.append((self.now(), "state", NAMES.get(state, state), None))
        if self.state_hang:
            await asyncio.Event().wait()
        if self.state_delay:
            await asyncio.sleep(self.state_delay)

    async def transcribe(self, pcm, config):
        await asyncio.sleep(self.asr_delay)
        self.events.append((self.now(), "asr_done", None, None))
        if isinstance(self.asr_result, Exception):
            raise self.asr_result
        return self.asr_result

    def timeline(self):
        """Compact list like ['THINKING', 'asr', 'S:<sentence>', 'E', 'IDLE']."""
        out = []
        for _, kind, s, _ in self.events:
            if kind == "state":
                out.append(s)
            elif kind == "asr_done":
                out.append("asr")
            elif kind == "sentence":
                out.append("S:" + s.replace(TAG + " ", ""))
            elif kind == "empty":
                out.append("E")
        return out

    def states(self):
        return [s for _, k, s, _ in self.events if k == "state"]


async def run_turn(script, *, rec=None, pause=pf.PAUSE, ptt=False, stall=None,
                   fail_at_end=False, interrupt_after=None):
    rec = rec or StateRecorder()
    m.asr.transcribe = rec.transcribe
    m.stall_mod.decide_stall = lambda *a, **k: pf._ret(stall)
    m.tts.synthesize_stream = rec.synthesize_stream
    m.open_audio_connection = rec.open_audio_connection
    m.write_audio_frames = lambda w, f: True
    m.close_audio_connection = pf._anoop
    m.send_audio = rec.send_audio
    m.send_state = rec.send_state
    m.OpusStreamEncoder = pf.FakeEncoder

    device = pf.make_device(pf.FakeConversation(script, fail_at_end), ptt=ptt)
    if interrupt_after is not None:
        asyncio.get_running_loop().call_later(interrupt_after, device.interrupted.set)
    q: asyncio.Queue = asyncio.Queue()
    await q.put((device, pf.FakeAudio()))
    t0 = time.monotonic()
    worker = asyncio.create_task(m.process_utterances(pf.make_config(pause, stall is not None), None, q))
    try:
        deadline = time.monotonic() + 20
        await asyncio.sleep(0)
        # Turn over = item taken, processing back to False, and every
        # background state send finished (the final IDLE is sent in the
        # finally right after processing = False).
        while not q.empty() or device.processing or m._state_tasks:
            if time.monotonic() > deadline:
                raise TimeoutError("turn did not finish")
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.02)
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    return rec, device, time.monotonic() - t0


def check(name, cond, detail=""):
    pf.check(name, cond, detail)


def idle_last(rec):
    """IDLE is the last thing sent in the turn, after all audio."""
    tl = rec.timeline()
    return bool(tl) and tl[-1] == "IDLE"


async def main():
    a, b, c = f"{TAG} let me look that up.", f"{TAG} the answer is forty two.", f"{TAG} anything else?"
    stall = f"{TAG} one sec."

    # 1. Plain answer, fast stream: THINKING at utterance end (before ASR is
    #    done and before any audio), no re-assert between sentences, IDLE last.
    rec, _, _ = await run_turn([a + " ", b + " ", c])
    tl = rec.timeline()
    check("THINKING sent at utterance end, before ASR result and any audio",
          tl[:2] == ["THINKING", "asr"], f"{tl}")
    check("fast stream: no THINKING between back-to-back sentences, IDLE at end",
          rec.states() == ["THINKING", "IDLE"] and idle_last(rec), f"{tl}")

    # 2. Stall line, then the answer after a while: THINKING re-sent right
    #    after the stall line, before the answer.
    rec, _, _ = await run_turn([0.5, a + " ", b], stall=stall)
    tl = rec.timeline()
    i_stall = tl.index("S:one sec.")
    i_answer = tl.index("S:let me look that up.")
    check("stall: THINKING re-sent after the stall line, before the answer",
          "THINKING" in tl[i_stall + 1:i_answer], f"{tl}")
    check("stall: IDLE last", idle_last(rec), f"{tl}")

    # 3. Stall interrupted: no THINKING after the stall line.
    rec, _, _ = await run_turn([2.0, a], stall=stall, interrupt_after=0.05)
    tl = rec.timeline()
    check("stall + interrupt: no THINKING re-assert, IDLE last",
          rec.states() == ["THINKING", "IDLE"] and idle_last(rec), f"{tl}")

    # 4. Pause flush: held sentence played during the gap, then THINKING
    #    again while the agent keeps working, then the rest, then IDLE.
    rec, _, _ = await run_turn([a + " ", 1.2, b + " ", c])
    tl = rec.timeline()
    i_a = tl.index("S:let me look that up.")
    i_b = tl.index("S:the answer is forty two.")
    check("pause flush: THINKING re-sent after the early sentence, before the next",
          tl[i_a + 1:i_b] == ["THINKING"], f"{tl}")
    check("pause flush: IDLE last", idle_last(rec), f"{tl}")

    # 5. Two pauses -> two re-asserts; stall + pause flush together.
    rec, _, _ = await run_turn([a + " ", 1.0, b + " ", 1.0, c], stall=stall)
    check("stall + two pause flushes: THINKING after each non-final early send",
          rec.timeline().count("THINKING") == 4 and idle_last(rec), f"{rec.timeline()}")

    # 5b. The grok bridge's keepalive ("I'm still working on it, one
    #     moment.", every GROK_RELAY_KEEPALIVE s) reaches Onju as a sentence
    #     followed by silence, so it goes out through the pause flush and
    #     THINKING comes back after it.
    keep = f"{TAG} still working on it, one moment."
    rec, _, _ = await run_turn([keep + " ", 1.0, keep + " ", 1.0, b], stall=stall)
    tl = rec.timeline()
    k = "S:still working on it, one moment."
    check("keepalive lines: THINKING re-sent after each, answer last, then IDLE",
          [x for x in tl if x in (k, "THINKING")][1:] == ["THINKING", k, "THINKING", k, "THINKING"]
          and tl[-2:] == ["S:the answer is forty two.", "IDLE"], f"{tl}")

    # 6. IDLE in finally on every early-exit path.
    rec, _, _ = await run_turn([a], rec=StateRecorder(asr={"text": "", "no_speech_prob": 0.9}))
    check("no speech (VOX): THINKING then IDLE, after the mic-reopen send",
          rec.timeline() == ["THINKING", "asr", "E", "IDLE"], f"{rec.timeline()}")

    rec, _, _ = await run_turn([a], rec=StateRecorder(asr=RuntimeError("asr down")))
    check("ASR error: IDLE sent", rec.states() == ["THINKING", "IDLE"], f"{rec.timeline()}")

    rec, _, _ = await run_turn([], fail_at_end=True)
    check("LLM failure, VOX, no stall: IDLE sent",
          rec.states() == ["THINKING", "IDLE"] and idle_last(rec), f"{rec.timeline()}")

    rec, _, _ = await run_turn([], ptt=True)
    check("PTT with empty agent reply (nothing played): IDLE sent",
          rec.timeline() == ["THINKING", "asr", "IDLE"], f"{rec.timeline()}")

    rec, _, _ = await run_turn([a + " ", b + " ", 5.0, c], interrupt_after=0.2)
    check("interrupted mid-answer: IDLE last", idle_last(rec), f"{rec.timeline()}")

    # 7. Non-blocking: slow (0.4 s each) and hanging state sends must not
    #    delay the reply audio.
    base, _, base_total = await run_turn([a + " ", b])
    t_base = base.time_of(a)
    slow, _, _ = await run_turn([a + " ", b], rec=StateRecorder(state_delay=0.4))
    check("slow state sends (0.4 s) don't delay the first sentence",
          slow.time_of(a) - t_base < 0.1, f"base={t_base:.2f} slow={slow.time_of(a):.2f}")
    check("slow state sends still arrive in order, IDLE last",
          slow.states() == ["THINKING", "IDLE"], f"{slow.timeline()}")
    hang, _, hang_total = await run_turn([a + " ", b], rec=StateRecorder(state_hang=True))
    check("hanging state sends don't delay the first sentence",
          hang.time_of(a) - t_base < 0.1, f"base={t_base:.2f} hang={hang.time_of(a):.2f}")
    check("hanging state sends are timed out: turn still ends (bounded)",
          hang_total - base_total < 1.6 and hang.states() == ["THINKING", "IDLE"],
          f"base={base_total:.2f}s hang={hang_total:.2f}s {hang.timeline()}")

    # 8. Quick turn with a slow THINKING: IDLE must not overtake it.
    rec, _, _ = await run_turn([a], rec=StateRecorder(state_delay=0.3, asr_delay=0.0,
                                                    asr={"text": "", "no_speech_prob": 0.9}))
    order = rec.states()
    t_think = next(t for t, k, s, _ in rec.events if s == "THINKING")
    t_idle = next(t for t, k, s, _ in rec.events if s == "IDLE")
    check("quick turn: IDLE sent only after the slow THINKING finished",
          order == ["THINKING", "IDLE"] and t_idle - t_think >= 0.29,
          f"{order} dt={t_idle - t_think:.2f}")

    # 9. The real protocol.send_state (the module is stubbed above, so load
    #    the file directly): exact bytes on a local socket, and a pod that
    #    isn't there costs at most the short timeout and raises nothing.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "real_protocol", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "pipeline", "protocol.py"))
    proto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)
    got = []

    async def handle(reader, writer):
        got.append(await reader.read())
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    await proto.send_state("127.0.0.1", port, proto.STATE_THINKING)
    await asyncio.sleep(0.05)
    server.close()
    await server.wait_closed()
    check("real send_state: 6-byte header EE 02 00 00 00 00 on its own connection",
          got == [bytes([0xEE, 2, 0, 0, 0, 0])], f"{got}")
    t = time.monotonic()
    await proto.send_state("127.0.0.1", port, proto.STATE_IDLE)        # nobody listening now
    await proto.send_state("10.255.255.1", 3001, proto.STATE_IDLE)     # unroutable: connect timeout
    check("real send_state: unreachable pod returns within the 0.2 s timeout, no exception",
          time.monotonic() - t < 0.5, f"{time.monotonic() - t:.2f}s")

    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(pf.CHECKS)
    print(f"{total - len(pf.FAILS)}/{total} passed")
    return 1 if pf.FAILS else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
