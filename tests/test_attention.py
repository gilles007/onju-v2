"""
Offline test for attention gating, the voice LEDs, mic-off and the
end-of-speech / pod-close confirmation (2026-10-19; pipeline/attention.py,
main.py udp_listener / stream_sentence / process_utterances, protocol.py).

Gilles's rule, checked through main.py's real udp_listener (localhost UDP
in, a fake pod on a real localhost TCP listener that drops a connection
after 500 ms without a command, like onjuino.ino):

1. While the server listens, every utterance flashes the LEDs: idle
   speech gets a persistent LED connection with blinks, kept alive through
   pauses by level-0 keepalives; barge-in speech during a reply gets
   one-shot blinks.
2. While it doesn't (a turn waiting for ASR / the agent, a say, nothing
   playing): no blink reaches the pod, nothing interrupts, nothing is
   queued, and every ignored utterance is logged.

Also: the old stuck-flag bug (an utterance cut by a turn start left the
next one without any LED connection) stays fixed; audio heard while deaf
never carries over into a turn; LedLink reconnects after the pod hangs up,
warns on connect failures and never blocks; protocol end-of-speech marker,
half-close and await_pod_close (closed / reset / timeout); the turn closes
the mic at its start and after every partial sentence and always reopens
it at the end.

Reuses the stubs of tests/test_pause_flush.py; pipeline/protocol.py itself
is loaded for real. Run from the repo root:  python3 tests/test_attention.py
"""
import asyncio
import glob
import importlib.util
import logging
import os
import socket
import struct
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs)
import test_led_state as ls     # noqa: E402

import numpy as np              # noqa: E402

m = pf.m
from pipeline import attention, bargein  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "real_protocol_for_attention",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "pipeline", "protocol.py"))
proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto)

TAG = pf.TAG
MIC = pf.MIC


def check(name, cond, detail=""):
    pf.check(name, cond, detail)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))

    def take(self, prefix=None, level=None):
        """Pop and return the matching messages (all of them if no filter)."""
        def hit(lv, msg):
            return (prefix is None or msg.startswith(prefix)) and (level is None or lv == level)
        out = [msg for lv, msg in self.records if hit(lv, msg)]
        self.records = [(lv, msg) for lv, msg in self.records if not hit(lv, msg)]
        return out


LOG = LogCapture()


# ---- fake pod ---------------------------------------------------------------

class FakePod:
    """Pod side of the TCP protocol, one record per connection:
    {"blinks": [levels], "cmds": [header bytes], "audio": n_frames,
     "marker": bool}. Like onjuino.ino, a connection that sends no command
    for IDLE_S is dropped (the 500 ms header wait)."""

    IDLE_S = 0.5

    def __init__(self, behaviour="normal"):
        self.conns = []
        self.behaviour = behaviour     # normal | reset | hold
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def _handle(self, reader, writer):
        c = {"blinks": [], "cmds": [], "audio": None, "marker": False, "t": time.monotonic()}
        self.conns.append(c)
        try:
            if self.behaviour == "reset":
                sock = writer.get_extra_info("socket")
                await asyncio.sleep(0.05)   # let the client send its data, unread
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                writer.transport.abort()
                return
            while True:
                try:
                    header = await asyncio.wait_for(reader.readexactly(6), self.IDLE_S)
                except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                    break
                c["cmds"].append(header[0])
                if header[0] == 0xCC:
                    c["blinks"].append(header[1])
                elif header[0] == 0xAA:
                    frames = 0
                    while True:
                        try:
                            n = int.from_bytes(await reader.readexactly(2), "big")
                        except asyncio.IncompleteReadError:
                            break
                        if n == 0:
                            c["marker"] = True
                            break
                        await reader.readexactly(n)
                        frames += 1
                    c["audio"] = frames
                    if self.behaviour == "hold":
                        await asyncio.sleep(3600)
                    break
        finally:
            if self.behaviour != "reset":
                writer.close()

    def blink_conns(self):
        return [c for c in self.conns if 0xCC in c["cmds"]]

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---- 1. protocol -------------------------------------------------------------

async def protocol_tests():
    pod = await FakePod().start()
    try:
        w = await proto.open_audio_connection("127.0.0.1", pod.port, mic_timeout=0)
        proto.write_audio_frames(w, [b"x" * 10, b"y" * 12])
        await proto.close_audio_connection(w, half_close=True)
        res = await proto.await_pod_close(w, 2.0)
        c = pod.conns[-1]
        check("protocol: frames, then end-of-speech marker the pod sees",
              c["audio"] == 2 and c["marker"], f"{c}")
        check("protocol: half-close, pod closes after reading -> 'closed'", res == "closed", res)

        w = await proto.open_audio_connection("127.0.0.1", pod.port)
        await proto.close_audio_connection(w)
        await asyncio.sleep(0.05)
        check("protocol: plain close also ends with the marker",
              pod.conns[-1]["marker"] and pod.conns[-1]["audio"] == 0, f"{pod.conns[-1]}")

        await proto.send_audio("127.0.0.1", pod.port, b"", mic_timeout=5)
        await asyncio.sleep(0.05)
        check("protocol: send_audio (empty mic reopen) ends with the marker too",
              pod.conns[-1]["marker"], f"{pod.conns[-1]}")

        lw = await proto.open_led_connection("127.0.0.1", pod.port)
        check("protocol: LED connection keeps the pod reader",
              lw is not None and hasattr(lw, "pod_reader"))
        ok1 = proto.write_led_blink(lw, 40)
        await asyncio.sleep(FakePod.IDLE_S + 0.3)       # pod hangs up after 500 ms idle
        dead = lw.pod_reader.at_eof()
        lw.close()
        ok2 = proto.write_led_blink(lw, 40)
        check("protocol: pod hang-up visible as EOF; write on a closed transport -> False",
              ok1 and dead and ok2 is False, f"ok1={ok1} dead={dead} ok2={ok2}")
    finally:
        await pod.stop()

    pod = await FakePod("reset").start()
    try:
        w = await proto.open_audio_connection("127.0.0.1", pod.port)
        proto.write_audio_frames(w, [b"z" * 2000] * 20)
        await proto.close_audio_connection(w, half_close=True)
        res = await proto.await_pod_close(w, 2.0)
        check("protocol: pod resets with audio unread -> 'reset'", res == "reset", res)
    finally:
        await pod.stop()

    pod = await FakePod("hold").start()
    try:
        w = await proto.open_audio_connection("127.0.0.1", pod.port)
        await proto.close_audio_connection(w, half_close=True)
        t0 = time.monotonic()
        res = await proto.await_pod_close(w, 0.3)
        check("protocol: pod never closes -> 'timeout' after the bound",
              res == "timeout" and time.monotonic() - t0 < 1.0, res)
    finally:
        pod.server.close()


# ---- 2. listening() ----------------------------------------------------------

def listening_tests():
    cfg = {"vad": {"threshold": 0.5}}

    def dev(**kw):
        d = types.SimpleNamespace(hostname="onju-att-unit", processing=False,
                                  interrupted=asyncio.Event(), say_active=False)
        d.__dict__.update(kw)
        return d

    bargein._by_host.pop("onju-att-unit", None)
    d = dev()
    check("listening: no turn -> idle", attention.listening(d, cfg) == (True, "idle"))
    d.processing = True
    check("listening: turn, nothing playing -> deaf (turn in progress)",
          attention.listening(d, cfg) == (False, "turn in progress"))
    d.say_active = True
    check("listening: say, nothing playing -> deaf (say in progress)",
          attention.listening(d, cfg) == (False, "say in progress"))
    d.say_active = False
    b = bargein.for_device(d, cfg)
    b.audio_sent(b.start_segment(), 5.0)
    check("listening: reply playing -> barge-in", attention.listening(d, cfg) == (True, "barge-in"))
    b.playing_until = 0.0
    d.interrupted.set()
    check("listening: already interrupted -> keeps listening (next turn)",
          attention.listening(d, cfg) == (True, "interrupted"))
    bargein._by_host.pop("onju-att-unit", None)
    d2 = dev(processing=True)
    cfg2 = {"vad": {"threshold": 0.5, "interrupt_only_while_playing": False}}
    check("listening: interrupt_only_while_playing false -> barge-in during the whole turn",
          attention.listening(d2, cfg2) == (True, "barge-in"))
    bargein._by_host.pop("onju-att-unit", None)


# ---- 3. LedLink --------------------------------------------------------------

async def ledlink_tests():
    attention.protocol = proto
    pod = await FakePod().start()
    try:
        link = attention.LedLink("onju-link")
        t0 = time.monotonic()
        link.ensure("127.0.0.1", pod.port)
        took = time.monotonic() - t0
        check("LedLink: ensure() returns at once, connects in the background",
              took < 0.01 and link.opening and not link.alive(), f"took={took:.3f}")
        await asyncio.sleep(0.05)
        check("LedLink: connected", link.alive())
        link.blink(60, 6)
        link.blink(0, 6)
        await asyncio.sleep(0.05)
        check("LedLink: voice blink and level-0 keepalive reach the pod",
              pod.conns[-1]["blinks"] == [60, 0], f"{pod.conns}")
        LOG.take()
        await asyncio.sleep(FakePod.IDLE_S + 0.3)       # nothing sent: pod hangs up
        check("LedLink: pod hang-up detected", not link.alive())
        link.ensure("127.0.0.1", pod.port)
        await asyncio.sleep(0.05)
        link.blink(30, 6)
        await asyncio.sleep(0.05)
        check("LedLink: reopened after the hang-up, logged",
              link.alive() and len(pod.conns) == 2 and pod.conns[-1]["blinks"] == [30]
              and any("closed by the pod" in s for s in LOG.take("LED")), f"{pod.conns}")
        link.close()
        await asyncio.sleep(0.05)
        check("LedLink: close() closes it", not link.alive() and link.writer is None)

        # close() while connecting: the late connection is discarded.
        link.ensure("127.0.0.1", pod.port)
        link.close()
        await asyncio.sleep(0.1)
        check("LedLink: close() during connect leaves nothing open",
              link.writer is None and not link.opening)
    finally:
        await pod.stop()

    # Connect failures: WARNING, retry only after RETRY_S, never blocks.
    port = free_port()
    link = attention.LedLink("onju-nolink")
    LOG.take()
    link.ensure("127.0.0.1", port)
    await asyncio.sleep(0.1)
    warns = LOG.take("LED", level="WARNING")
    link.ensure("127.0.0.1", port)          # too soon: no new attempt
    again_soon = link.opening
    await asyncio.sleep(attention.LedLink.RETRY_S + 0.05)
    link.ensure("127.0.0.1", port)
    retried = link.opening
    await asyncio.sleep(0.1)
    LOG.take()
    link.close()
    gave_up = LOG.take("LED  gave up")
    check("LedLink: connect failure -> WARNING once, retry after RETRY_S, give-up logged",
          len(warns) == 1 and "onju-nolink" in warns[0] and not again_soon and retried
          and len(gave_up) == 1, f"{warns} soon={again_soon} retried={retried} {gave_up}")


# ---- 4. udp_listener: the rule end to end -------------------------------------

class ScriptVAD:
    """VAD stand-in: the test sets the probability of the next frame;
    >= 0.5 starts recording; end_next makes that frame end the utterance."""

    def __init__(self):
        self.speech_prob = 0.0
        self.recording = False
        self.buffer = []
        self.next = 0.0
        self.end_next = False
        self.resets = 0

    def reset(self):
        self.buffer = []
        self.recording = False
        self.speech_prob = 0.0
        self.resets += 1

    def process_frame(self, pcm):
        self.speech_prob = self.next
        if not self.recording:
            if self.next >= 0.5:
                self.recording = True
                self.buffer.append(np.zeros(512, dtype=np.int16))
            return None
        self.buffer.append(np.zeros(512, dtype=np.int16))
        if self.end_next:
            self.end_next = False
            audio = np.concatenate(self.buffer)
            self.buffer = []
            self.recording = False
            self.speech_prob = 0.0
            return audio
        return None


async def udp_tests():
    attention.protocol = proto
    m.send_led_blink = proto.send_led_blink
    pod = await FakePod().start()
    udp_port = free_port()
    config = {
        "network": {"udp_port": udp_port, "tcp_port": pod.port},
        "audio": {"chunk_size": 512, "sample_rate": 16000},
        "device": {"led_power": 50, "led_update_period": 0.25, "led_fade": 6},
        "vad": {"threshold": 0.5},
    }
    m.decode_ulaw = lambda data: np.zeros(len(data), dtype=np.int16)
    dev = types.SimpleNamespace(
        hostname="onju-att", ip="127.0.0.1", ptt=False, processing=False, say_active=False,
        interrupted=asyncio.Event(), vad=ScriptVAD(), ptt_buffer=[],
        led_power=0, led_update_time=0.0, last_mic_at=0.0)
    manager = types.SimpleNamespace(devices={dev.hostname: dev}, get_by_ip=lambda ip: dev,
                                    get_most_recent=lambda: dev)
    bargein._by_host.pop(dev.hostname, None)
    attention._by_host.pop(dev.hostname, None)
    q: asyncio.Queue = asyncio.Queue()
    listener = asyncio.create_task(m.udp_listener(config, manager, q))
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    await asyncio.sleep(0.05)

    async def frames(prob, n=1, end=False):
        for i in range(n):
            dev.vad.next = prob
            dev.vad.end_next = end and i == n - 1
            tx.sendto(b"\xff" * 512, ("127.0.0.1", udp_port))
            await asyncio.sleep(0.032)              # real time, like the pod

    def blinks(conns):
        return [lv for c in conns for lv in c["blinks"]]

    try:
        # Rule 1, idle: speech -> one persistent LED connection with voice
        # blinks; a 0.8 s pause inside the utterance is bridged by level-0
        # keepalives (no 500 ms hang-up); the utterance is queued.
        LOG.take()
        await frames(0.9, 15)
        await frames(0.0, 25)                       # 0.8 s pause, still recording
        await frames(0.9, 10, end=True)
        await asyncio.sleep(0.1)
        bc = pod.blink_conns()
        lv = blinks(bc)
        check("idle: utterance flashes the LEDs over one persistent connection",
              len(bc) == 1 and any(x > 0 for x in lv), f"{pod.conns}")
        check("idle: pause bridged by level-0 keepalives, pod never hung up mid-utterance",
              0 in lv and len(bc) == 1 and lv[-1] > 0, f"{lv}")
        check("idle: utterance queued", q.qsize() == 1 and
              any(s.startswith("VAD  utterance from onju-att") for s in LOG.take("VAD")))
        q.get_nowait()

        # Stuck-flag regression: speech starts idle, a turn starts while it is
        # recording -> cut off (VAD reset, logged), no more blinks; after the
        # turn the next utterance gets its LED connection again.
        n0 = len(pod.blink_conns())
        await frames(0.9, 10)
        dev.processing = True                       # e.g. the worker took the previous utterance
        await asyncio.sleep(0.05)
        before = sum(len(c["blinks"]) for c in pod.conns)
        await frames(0.9, 10)
        cut = LOG.take("VAD  ignored")
        mid = len(pod.blink_conns())
        after = sum(len(c["blinks"]) for c in pod.conns)
        await frames(0.9, 5, end=True)
        not_queued = q.empty()
        dev.processing = False
        await frames(0.0, 2)
        await frames(0.9, 12, end=True)
        await asyncio.sleep(0.1)
        bc = pod.blink_conns()
        check("cut by a turn start: logged as ignored, VAD reset, not queued",
              len(cut) == 1 and "cut off" in cut[0] and not_queued, f"{cut}")
        check("cut by a turn start: no blinks once the server stopped listening",
              mid == n0 + 1 and after == before and bc[n0]["blinks"],
              f"before={before} after={after} {pod.conns[n0:]}")
        check("after the turn: next utterance has its LED connection again (stuck-flag bug)",
              len(bc) == n0 + 2 and any(x > 0 for x in bc[-1]["blinks"]) and q.qsize() == 1,
              f"{pod.conns[n0:]}")
        q.get_nowait()
        LOG.take()

        # Rule 2, deaf: a turn waiting for the agent, nothing playing. Two
        # utterances: no blink reaches the pod, no interrupt, nothing queued,
        # each one logged.
        dev.processing = True
        n0 = len(pod.conns)
        for _ in range(2):
            await frames(0.9, 15)
            await frames(0.9, 1, end=True)
            await frames(0.0, 3)
        await asyncio.sleep(0.3)
        lines = LOG.take("VAD  ignored")
        check("deaf: nothing reaches the pod (no LED connection, no blink)",
              len(pod.conns) == n0, f"{pod.conns[n0:]}")
        check("deaf: no interrupt, nothing queued",
              not dev.interrupted.is_set() and q.empty())
        check("deaf: every ignored utterance logged (not once per turn)",
              len(lines) == 2 and all("not listening (turn in progress)" in s for s in lines),
              f"{lines}")
        check("deaf: barge-in not even fed (no VOX lines)", LOG.take("VOX") == [])

        # Deaf -> listening: audio heard while deaf doesn't carry over.
        await frames(0.9, 10)                       # recording while deaf
        resets = dev.vad.resets
        dev.processing = False
        await frames(0.9, 1)
        carried = LOG.take("VAD  ignored")
        check("deaf -> idle: speech started while deaf is dropped (VAD reset, logged)",
              dev.vad.resets == resets + 1 and len(carried) == 1
              and "started before the server listened" in carried[0], f"{carried}")
        await frames(0.9, 4, end=True)
        await asyncio.sleep(0.05)
        got = q.get_nowait() if not q.empty() else None
        check("deaf -> idle: what is queued is only the part heard while listening",
              got is not None and len(got[1]) <= 5 * 512, f"{None if got is None else len(got[1])}")
        LOG.take()

        # Rule 1 during a turn: reply playing -> barge-in armed. One-shot
        # blinks (each on its own short connection), the interrupt after
        # 300 ms, and the rest of the speech becomes the next turn.
        dev.processing = True
        dev.interrupted.clear()
        b = bargein.for_device(dev, config)
        b.new_turn()
        b.audio_sent(b.start_segment(), 10.0)
        await frames(0.0, 1)
        n0 = len(pod.conns)
        await frames(0.9, 20)
        await frames(0.9, 1, end=True)
        await asyncio.sleep(0.2)
        new = pod.conns[n0:]
        check("barge-in: voice blinks reach the pod as one-shot connections",
              len(new) >= 2 and all(c["cmds"] == [0xCC] and c["blinks"][0] > 0 for c in new),
              f"{new}")
        check("barge-in: interrupts the turn (logged)",
              dev.interrupted.is_set() and len(LOG.take("VOX  interrupt")) == 1)
        check("barge-in: the interrupting speech is queued as the next turn",
              q.qsize() == 1 and any("barge-in" in s for s in LOG.take("VAD")))
        q.get_nowait()

        # Barge-in armed but too short to interrupt: blinks, then ignored, logged.
        dev.interrupted.clear()
        b.new_turn()
        b.audio_sent(b.start_segment(), 10.0)
        await frames(0.9, 4)
        await frames(0.2, 1, end=True)
        lines = LOG.take("VAD  ignored")
        check("barge-in too short: no interrupt, not queued, logged as ignored",
              not dev.interrupted.is_set() and q.empty() and len(lines) == 1
              and "did not interrupt" in lines[0], f"{lines}")

        # Packet timeout while deaf: logged, not queued.
        b.playing_until = 0.0
        await frames(0.9, 10)
        await asyncio.sleep(0.9)
        lines = LOG.take("VAD  ignored")
        check("mic stops while deaf: utterance ignored (logged), not queued",
              q.empty() and len(lines) == 1 and "mic stopped" in lines[0], f"{lines}")
        dev.processing = False
    finally:
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass
        tx.close()
        attention.led_for(dev).close()
        await asyncio.sleep(0.05)
        await pod.stop()


# ---- 5. mic off / reopen in a turn --------------------------------------------

class MicRecorder(ls.StateRecorder):
    async def send_stop_listening(self, ip, port, hold_s=0):
        self.events.append((self.now(), "mic_off", None, hold_s))

    def mic_timeline(self):
        out = []
        for _, kind, s, mic in self.events:
            if kind == "mic_off":
                out.append("OFF")
            elif kind == "sentence":
                out.append(f"S{mic}")
            elif kind == "empty":
                out.append(f"E{mic}")
            elif kind == "asr_done":
                out.append("asr")
        return out


async def mic_tests():
    a, b = f"{TAG} let me look.", f"{TAG} the answer."

    async def turn(script, **kw):
        rec = kw.pop("rec", None) or MicRecorder()
        m.send_stop_listening = rec.send_stop_listening
        try:
            r, dev, _ = await ls.run_turn(script, rec=rec, **kw)
        finally:
            m.send_stop_listening = proto.send_stop_listening
        return r

    rec = await turn([a + " ", b], stall=f"{TAG} one sec.")
    check("turn: mic off at start, after the stall and each partial; final opens it; no reopen",
          rec.mic_timeline() == ["OFF", "asr", "S0", "OFF", "S0", "OFF", f"S{MIC}"],
          f"{rec.mic_timeline()}")

    rec = await turn([a + " ", b])
    check("turn without stall: OFF, partial + OFF, final",
          rec.mic_timeline() == ["OFF", "asr", "S0", "OFF", f"S{MIC}"], f"{rec.mic_timeline()}")

    rec = await turn([a], rec=MicRecorder(asr=RuntimeError("asr down")))
    check("ASR failure: the mic closed at turn start is reopened",
          rec.mic_timeline() == ["OFF", "asr", f"E{MIC}"], f"{rec.mic_timeline()}")

    rec = await turn([a], rec=MicRecorder(asr={"text": "", "no_speech_prob": 0.9}))
    check("no speech: reopened once",
          rec.mic_timeline() == ["OFF", "asr", f"E{MIC}"], f"{rec.mic_timeline()}")

    rec = await turn([a + " ", b + " ", 5.0, f"{TAG} never."], interrupt_after=0.15)
    tl = rec.mic_timeline()
    check("interrupt: nothing sent after the interrupt, mic reopened once at the end",
          tl == ["OFF", "asr", "S0", "OFF", f"E{MIC}"], f"{tl}")

    rec = await turn([a + " ", b], ptt=True)
    check("PTT pod: no mic off, no reopen (button-gated mic)",
          "OFF" not in rec.mic_timeline() and not any(x.startswith("E") for x in rec.mic_timeline()),
          f"{rec.mic_timeline()}")


# ---- 6. pod close confirmation in stream_sentence ------------------------------

async def confirm_tests():
    attention.protocol = proto
    saved = (m.open_audio_connection, m.write_audio_frames, m.close_audio_connection,
             m.await_pod_close, m.send_stop_listening, m.OpusStreamEncoder, m.tts.synthesize_stream)
    config = pf.make_config()

    async def one(behaviour, slack=None):
        pod = await FakePod(behaviour).start()
        config["network"]["tcp_port"] = pod.port
        if slack is not None:
            m.POD_CLOSE_SLACK_S = slack
        dev = pf.make_device(None)
        dev.hostname = f"onju-confirm-{behaviour}"

        async def tts(sentence, voice, cfg):
            yield b"\1\0" * 16000

        m.open_audio_connection = proto.open_audio_connection
        m.write_audio_frames = proto.write_audio_frames
        m.close_audio_connection = proto.close_audio_connection
        m.await_pod_close = proto.await_pod_close
        m.send_stop_listening = proto.send_stop_listening
        m.OpusStreamEncoder = pf.FakeEncoder
        m.tts.synthesize_stream = tts
        LOG.take()
        ok, n = await m.stream_sentence(config, dev, f"{TAG} confirm {behaviour}.",
                                        is_final=True, turn_t0=time.monotonic())
        bargein._by_host.pop(dev.hostname, None)
        for _ in range(100):
            if not m._confirm_tasks:
                break
            await asyncio.sleep(0.05)
        warns = LOG.take(level="WARNING")
        pod.server.close()
        return ok, pod, warns

    try:
        ok, pod, warns = await one("normal")
        check("stream_sentence: pod got the marker; normal close -> no warning",
              ok and pod.conns[0]["marker"] and warns == [], f"{pod.conns} {warns}")
        ok, pod, warns = await one("reset")
        check("stream_sentence: pod reset -> WARNING naming the sentence",
              len(warns) == 1 and "reset the audio connection" in warns[0]
              and "confirm reset" in warns[0], f"{warns}")
        ok, pod, warns = await one("hold", slack=0.3)
        check("stream_sentence: pod never closes -> WARNING after the bound",
              len(warns) == 1 and "did not close" in warns[0], f"{warns}")
    finally:
        m.POD_CLOSE_SLACK_S = 15.0
        (m.open_audio_connection, m.write_audio_frames, m.close_audio_connection,
         m.await_pod_close, m.send_stop_listening, m.OpusStreamEncoder,
         m.tts.synthesize_stream) = saved


async def main():
    root = logging.getLogger()
    root.addHandler(LOG)
    root.setLevel(logging.DEBUG)
    for h in root.handlers:
        if h is not LOG:
            h.setLevel(logging.CRITICAL)
    listening_tests()
    await protocol_tests()
    await ledlink_tests()
    await udp_tests()
    await mic_tests()
    await confirm_tests()
    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(pf.CHECKS)
    print(f"{total - len(pf.FAILS)}/{total} passed")
    return 1 if pf.FAILS else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
