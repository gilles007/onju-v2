"""
Offline test for when send_sentence() in pipeline/main.py opens the audio
connection to the pod (2026-10-01).

The pod waits at most 2 s for the first Opus frame after the 6-byte header,
then closes the connection and plays nothing. Whole-sentence TTS (backend
local with Qwen) takes 2-3 s, so the connection is now opened only when the
first audio is ready:

- slow TTS: the connection is opened after the first chunk, and frames
  follow the header at once; still one connection per sentence (also with
  several streamed chunks), closed once at the end;
- no audio at all, or less than one batch: the connection is opened at the
  end, so the pod still gets the header (mic timeout) and end-of-segment;
  the SEND line does not crash on a missing tts_first;
- TTS error before any audio: header + end-of-segment, as before;
- interrupt before the first audio: no connection (a header would stop
  the pod's mic while the user talks); IDLE still ends the turn;
- open fails: no retry, nothing written or closed;
- the pod closed its end (reader at EOF): one warning per sentence.

Reuses the stubs and fakes of tests/test_pause_flush.py and
tests/test_led_state.py. Run from the repo root:
python3 tests/test_pod_connection.py
"""
import asyncio
import glob
import logging
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs)
import test_led_state as ls     # noqa: E402

m = pf.m
TAG = pf.TAG
MIC = pf.MIC
SEC = b"\0\0" * 16000          # 1 s of 16 kHz s16 audio


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append((record.levelname, record.getMessage()))


class Reader:
    def __init__(self, eof):
        self.eof = eof

    def at_eof(self):
        return self.eof


class Writer:
    def __init__(self, rec, pod_closed=False):
        self.rec = rec
        if pod_closed:
            self.pod_reader = Reader(True)

    async def drain(self):
        pass


class ConnRecorder(ls.StateRecorder):
    """Timeline of TTS and pod-connection events for one sentence.
    `chunks` is a list of (delay_s, pcm) the fake TTS yields; `fail`
    raises after the delays; `open_ok` False makes the open fail."""

    def __init__(self, chunks, fail=False, open_ok=True, pod_closed=False):
        super().__init__()
        self.chunks = chunks
        self.fail = fail
        self.open_ok = open_ok
        self.pod_closed = pod_closed
        self.log = []          # (t, kind, detail)

    def mark(self, kind, detail=None):
        self.log.append((self.now(), kind, detail))

    def kinds(self):
        return [k for _, k, _ in self.log]

    def t(self, kind):
        return next(t for t, k, _ in self.log if k == kind)

    async def synthesize_stream(self, sentence, voice, config):
        self.events.append([self.now(), "sentence", sentence, None])
        self.mark("tts_start")
        for delay, pcm in self.chunks:
            await asyncio.sleep(delay)
            self.mark("chunk", len(pcm))
            yield pcm
        if self.fail:
            raise RuntimeError("tts broke")

    async def open_audio_connection(self, ip, port, mic_timeout, volume, fade):
        self.mark("open", mic_timeout)
        if not self.open_ok:
            return None
        await super().open_audio_connection(ip, port, mic_timeout, volume, fade)
        return Writer(self, self.pod_closed)

    def write(self, writer, frames):
        self.mark("write", len(frames))
        return True

    async def close(self, writer):
        self.mark("close")


async def run(rec, script, interrupt_after=None):
    # ls.run_turn installs the fakes; write/close are swapped in after it
    # did, through a wrapper that re-patches them as soon as the turn starts.
    orig_ss = rec.synthesize_stream

    async def ss(sentence, voice, config):
        m.write_audio_frames = rec.write
        m.close_audio_connection = rec.close
        async for pcm in orig_ss(sentence, voice, config):
            yield pcm
    rec.synthesize_stream = ss
    cap = LogCapture()
    level, propagate = m.log.level, m.log.propagate
    m.log.addHandler(cap)
    m.log.setLevel(logging.INFO)       # SEND / interrupt lines are INFO
    m.log.propagate = False            # captured, not printed
    try:
        await ls.run_turn(script, rec=rec, interrupt_after=interrupt_after)
    finally:
        m.log.removeHandler(cap)
        m.log.setLevel(level)
        m.log.propagate = propagate
    return cap.lines


def check(name, cond, detail=""):
    pf.check(name, cond, detail)


async def main():
    a = f"{TAG} the answer is forty two."

    # 1. Slow whole-sentence TTS (2.3 s in production; 0.3 s here).
    rec = ConnRecorder([(0.3, SEC)])
    logs = await run(rec, [a])
    k = rec.kinds()
    check("slow TTS: connection opened only after the first chunk",
          k.index("tts_start") < k.index("chunk") < k.index("open") < k.index("write"), f"{k}")
    check("slow TTS: frames follow the header at once (no wait after open)",
          rec.t("write") - rec.t("open") < 0.05, f"{rec.log}")
    check("slow TTS: one open, one close, final header carries the mic timeout",
          k.count("open") == 1 and k.count("close") == 1 and rec.log[k.index("open")][2] == MIC,
          f"{rec.log}")
    check("slow TTS: IDLE still the last state", ls.idle_last(rec), f"{rec.timeline()}")

    # 2. Streamed chunks: still one connection per sentence.
    rec = ConnRecorder([(0.05, SEC[:19200]), (0.05, SEC[:19200]), (0.05, SEC[:6400])])
    await run(rec, [a])
    k = rec.kinds()
    check("stream: one connection for the whole sentence, opened after chunk 1",
          k.count("open") == 1 and k.count("close") == 1
          and k.index("chunk") < k.index("open") < k.index("write") and k[-1] == "close"
          and k.count("write") >= 2, f"{k}")

    # 3. No audio at all: header + end-of-segment at the end, no crash.
    rec = ConnRecorder([])
    logs = await run(rec, [a])
    k = rec.kinds()
    send = [msg for _, msg in logs if msg.startswith("SEND")]
    check("no audio: connection still opened (mic timeout) and closed, nothing written",
          k == ["tts_start", "open", "close"] and rec.log[1][2] == MIC, f"{k}")
    check("no audio: SEND logged with 'no audio', 0 frames",
          len(send) == 1 and "no audio" in send[0] and " 0 frames" in send[0], f"{send}")
    check("no audio: IDLE still the last state", ls.idle_last(rec), f"{rec.timeline()}")

    # 4. Less than one batch (0.2 s): opened at the end, tail written.
    rec = ConnRecorder([(0.05, SEC[:6400])])
    await run(rec, [a])
    check("short audio: opened at the end, tail written, closed",
          rec.kinds() == ["tts_start", "chunk", "open", "write", "close"], f"{rec.kinds()}")

    # 5. TTS error before any audio: header + end-of-segment, as before.
    rec = ConnRecorder([], fail=True)
    logs = await run(rec, [a])
    check("TTS error: connection opened and closed (pod gets the header), error logged",
          rec.kinds() == ["tts_start", "open", "close"]
          and any(msg.startswith("TTS  failed") for _, msg in logs), f"{rec.kinds()} {logs}")

    # 6. Interrupt while the TTS is still working: no connection at all.
    rec = ConnRecorder([(0.6, SEC)])
    logs = await run(rec, [a], interrupt_after=0.3)
    k = rec.kinds()
    check("interrupt before first audio: no connection opened, nothing written",
          "tts_start" in k and "open" not in k and "write" not in k, f"{k}")
    check("interrupt before first audio: logged, IDLE still last",
          any("before the first audio" in msg for _, msg in logs) and ls.idle_last(rec),
          f"{logs} {rec.timeline()}")

    # 7. Open fails: one attempt, nothing written or closed.
    rec = ConnRecorder([(0.05, SEC)], open_ok=False)
    logs = await run(rec, [a])
    k = rec.kinds()
    check("open fails: one attempt, no write, no close, error logged",
          k.count("open") == 1 and "write" not in k and "close" not in k
          and any("Failed to open audio connection" in msg for _, msg in logs), f"{k}")

    # 8. Pod already closed its end: one warning for the sentence.
    rec = ConnRecorder([(0.05, SEC), (0.05, SEC)], pod_closed=True)
    logs = await run(rec, [a])
    warns = [msg for lvl, msg in logs if lvl == "WARNING" and "closed the audio connection" in msg]
    check("pod closed: warned once per sentence", len(warns) == 1, f"{warns}")

    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(pf.CHECKS)
    print(f"{total - len(pf.FAILS)}/{total} passed")
    return 1 if pf.FAILS else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
