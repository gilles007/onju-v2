"""
Offline test for the "say" back door (POST /say, pipeline/say.py and
main.py say_server / say_request / play_say):

- off unless say.enabled: main() doesn't start the server at all (nothing
  listens on the port); on, unknown paths give 404;
- busy pod -> 409 with a JSON reason and nothing touched (no connection to
  the pod, no state changed): voice turn in progress, turn queued, playing
  (barge-in estimate), recording (VAD), listening (recent mic packets), a
  say already running;
- unknown pod / no pod -> 404; missing text, bad JSON, unknown bot -> 400;
  wrong token -> 401;
- happy path through a fake pod (a real TCP listener, real
  pipeline/protocol.py): one audio connection per sentence with mic
  timeout 0, Opus frames, each followed by 0xDD 0 to keep the mic closed
  (2026-10-19: firmware before 20261019 opens it for 20 s after every
  playback, whatever the header says); the agent's voice goes to the TTS;
  200 with the duration; one JSON line in the say log; the say reaches
  that bot's next voice turn as a note in extra_context (once);
  listen=true keeps the mic open (last header carries the default mic
  timeout, no 0xDD after it); wait=false answers "queued" at once; an
  interrupt stops it (no 0xDD after the interrupted sentence: the user is
  talking) and only the played sentences are remembered;
- voice/bot resolution with named agents;
- staged announce: optional filler + think_s + delay_s (THINKING held, mic
  closed between parts, both lines in say.log / tell_agent; interrupt aborts
  the rest; a second /say during the sequence gets 409).
- say.render: whole (tts.backend local only): every sentence of a part is
  rendered before the first is sent; the answer renders during the
  filler/think phase, THINKING stays on until it is ready; a tap during
  the render sends nothing; sentence (default) and local_stream unchanged.
- "wav" (pre-rendered audio instead of text): played through the same path
  with no TTS call (filler still uses the TTS), 24 kHz stereo downmixed and
  resampled to 16 kHz mono, label in say.log, listen/mic-off as for text;
  bad base64, non-WAV, 8-bit, too long, and neither text nor wav -> 400.

Reuses the stubs of tests/test_pause_flush.py. Run from the repo root:
python3 tests/test_say.py
"""
import asyncio
import glob
import importlib.util
import json
import os
import socket
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs)

m = pf.m
say = m.say_mod
TAG = pf.TAG
MIC = pf.MIC
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The real wire protocol (test_pause_flush stubs pipeline.protocol).
_spec = importlib.util.spec_from_file_location("real_protocol_under_test",
                                               os.path.join(ROOT, "pipeline", "protocol.py"))
proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto)

# Real agents module for the router (pure Python).
from pipeline import agents  # noqa: E402

AGENTS_CFG = {"default": "pepper", "list": [
    {"name": "pepper", "aliases": ["peppa"], "voice": "pepper"},
    {"name": "robin", "aliases": ["tim"], "voice": "tim-allen"},
    {"name": "data", "voice": "jack-nicholson"},
]}


def check(name, cond, detail=""):
    pf.check(name, cond, detail)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class FakePod:
    """TCP listener speaking the pod side of the protocol: 0xAA header +
    length-prefixed Opus frames until EOF, or a 6-byte command."""

    def __init__(self):
        self.conns = []        # dicts: kind, mic, frames
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def _handle(self, reader, writer):
        try:
            header = await reader.readexactly(6)
            if header[0] == 0xAA:
                frames = 0
                while True:
                    try:
                        n = int.from_bytes(await reader.readexactly(2), "big")
                    except asyncio.IncompleteReadError:
                        break
                    if n == 0:
                        break
                    await reader.readexactly(n)
                    frames += 1
                self.conns.append({"kind": "audio", "mic": header[1] << 8 | header[2],
                                   "frames": frames})
            elif header[0] == 0xDD:
                self.conns.append({"kind": "mic_cmd", "mic": header[1] << 8 | header[2]})
            else:
                self.conns.append({"kind": hex(header[0])})
        finally:
            writer.close()

    def audio(self):
        return [c for c in self.conns if c["kind"] == "audio"]

    def mic_cmds(self):
        return [c for c in self.conns if c["kind"] == "mic_cmd"]

    async def stop(self):
        self.server.close()
        await self.server.wait_closed()


class FakeTTS:
    def __init__(self, delay=0.02):
        self.calls = []        # (sentence, voice_override)
        self.delay = delay

    async def synthesize_stream(self, sentence, voice, config, voice_override=None):
        self.calls.append((sentence, voice_override))
        await asyncio.sleep(self.delay)
        yield b"\0\0" * 16000          # 1 s -> one batch -> encoder gives 1 frame


def make_device(host="onju-test"):
    return types.SimpleNamespace(
        hostname=host, ip="127.0.0.1", ptt=False, voice="pepper",
        processing=False, interrupted=asyncio.Event(), vad_writer=None,
        vad=types.SimpleNamespace(recording=False), ptt_buffer=[],
        last_mic_at=0.0, say_active=False,
        conversation=None, last_user_text=None, last_response=None,
    )


class FakeManager:
    def __init__(self, *devices):
        self.devices = {d.hostname: d for d in devices}

    def get_most_recent(self):
        return next(reversed(self.devices.values())) if self.devices else None


def make_config(pod_port, say_port, log_path, agents_cfg=AGENTS_CFG, **say_extra):
    cfg = pf.make_config()
    cfg["network"]["tcp_port"] = pod_port
    cfg["say"] = {"enabled": True, "host": "127.0.0.1", "port": say_port, "log": log_path,
                  **say_extra}
    if agents_cfg:
        cfg["agents"] = agents_cfg
    return cfg


async def http(port, method, path, obj=None, headers=None, raw=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    body = raw if raw is not None else (json.dumps(obj).encode() if obj is not None else b"")
    head = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n"
    for k, v in (headers or {}).items():
        head += f"{k}: {v}\r\n"
    writer.write(head.encode() + b"\r\n" + body)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(), 20)
    writer.close()
    status_line, _, rest = data.partition(b"\r\n")
    status = int(status_line.split()[1])
    payload = rest.split(b"\r\n\r\n", 1)[1]
    return status, json.loads(payload)


class Env:
    """Fake pod + say server + fakes installed in pipeline.main."""

    def __init__(self, devices=None, agents_cfg=AGENTS_CFG, queue=None, tts_backend=None,
                 **say_extra):
        self.devices = devices if devices is not None else [make_device()]
        self.agents_cfg = agents_cfg
        self.tts_backend = tts_backend
        self.queue = queue if queue is not None else asyncio.Queue()
        self.say_extra = say_extra

    async def __aenter__(self):
        m.bargein._by_host.clear()
        say.MEMORY._said.clear()
        self.pod = FakePod()
        await self.pod.start()
        self.tts = FakeTTS()
        self.tmp = tempfile.mkdtemp(prefix="say-test-")
        self.log_path = os.path.join(self.tmp, "say.log")
        self.port = free_port()
        self.config = make_config(self.pod.port, self.port, self.log_path, self.agents_cfg,
                                  **self.say_extra)
        if self.tts_backend:
            self.config["tts"] = {"backend": self.tts_backend}
        m.tts.synthesize_stream = self.tts.synthesize_stream
        m.open_audio_connection = proto.open_audio_connection
        m.write_audio_frames = proto.write_audio_frames
        m.close_audio_connection = proto.close_audio_connection
        m.send_stop_listening = proto.send_stop_listening
        m.OpusStreamEncoder = pf.FakeEncoder
        self.manager = FakeManager(*self.devices)
        ready = asyncio.Event()
        self.server = asyncio.create_task(m.say_server(self.config, self.manager, self.queue, ready))
        await asyncio.wait_for(ready.wait(), 5)
        return self

    async def __aexit__(self, *exc):
        self.server.cancel()
        try:
            await self.server
        except asyncio.CancelledError:
            pass
        await self.pod.stop()

    def log_lines(self):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]


async def test_disabled():
    started = []

    async def fake_say_server(*a, **k):
        started.append(True)

    async def noop(*a, **k):
        return None

    saved = {k: getattr(m, k) for k in ("load_config", "DeviceManager", "udp_listener",
                                         "multicast_listener", "process_utterances",
                                         "control_server", "say_server")}
    try:
        m.udp_listener = m.multicast_listener = m.process_utterances = m.control_server = noop
        m.say_server = fake_say_server
        m.DeviceManager = lambda config: FakeManager()
        for enabled in (False, True):
            started.clear()
            cfg = pf.make_config()
            cfg["tts"] = {"backend": "local"}
            cfg["logging"] = {"level": "WARNING"}
            if enabled:
                cfg["say"] = {"enabled": True}
            m.load_config = lambda path=None, cfg=cfg: cfg
            await m.main()
            check(f"say {'enabled' if enabled else 'absent/disabled'}: server "
                  f"{'started' if enabled else 'not started'}", bool(started) == enabled)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
    check("SayConfig: off by default, sensible defaults",
          not say.SayConfig({}).enabled and say.SayConfig({}).port == 3103
          and say.SayConfig({}).host == "127.0.0.1" and say.SayConfig({}).tell_agent)


async def test_refusals():
    async with Env() as env:
        dev = env.devices[0]
        st, body = await http(env.port, "GET", "/nope")
        check("unknown path -> 404", st == 404, f"{st} {body}")
        st, body = await http(env.port, "POST", "/say", raw=b"not json")
        check("bad JSON -> 400", st == 400, f"{st} {body}")
        st, body = await http(env.port, "POST", "/say", {"voice": "robin"})
        check("missing text -> 400", st == 400 and "text" in body["error"], f"{st} {body}")
        st, body = await http(env.port, "POST", "/say", {"text": "hi", "bot": "nobody"})
        check("unknown bot -> 400 with the bot list", st == 400 and "robin" in body.get("bots", []),
              f"{st} {body}")
        st, body = await http(env.port, "POST", "/say", {"text": "hi", "device": "onju-other"})
        check("unknown pod -> 404", st == 404 and body["error"] == "unknown pod", f"{st} {body}")

        cases = [
            ("voice turn in progress", lambda: setattr(dev, "processing", True), "voice turn"),
            ("say already running", lambda: setattr(dev, "say_active", True), "say"),
            ("recording", lambda: setattr(dev.vad, "recording", True), "recording"),
            ("listening (mic packets)", lambda: setattr(dev, "last_mic_at", time.time()), "listening"),
            ("playing (estimate)", lambda: m.bargein.for_device(dev, env.config).audio_sent(
                time.monotonic(), 3.0), "playing"),
            ("turn queued", lambda: env.queue.put_nowait(("x", "y")), "queued"),
        ]
        for name, make_busy, word in cases:
            make_busy()
            before = (dev.processing, dev.say_active, dev.interrupted.is_set())
            st, body = await http(env.port, "POST", "/say", {"text": f"{TAG} hello.", "voice": "robin"})
            check(f"busy ({name}) -> 409 with a reason",
                  st == 409 and body.get("error") == "busy" and word in body.get("reason", ""),
                  f"{st} {body}")
            check(f"busy ({name}): nothing touched",
                  not env.pod.conns and not env.tts.calls
                  and (dev.processing, dev.say_active, dev.interrupted.is_set()) == before,
                  f"{env.pod.conns} {env.tts.calls}")
            # reset
            dev.processing = dev.say_active = False
            dev.vad.recording = False
            dev.last_mic_at = 0.0
            m.bargein._by_host.clear()
            while not env.queue.empty():
                env.queue.get_nowait()
        check("refusals: nothing logged to say.log", env.log_lines() == [], f"{env.log_lines()}")

    async with Env(devices=[]) as env:
        st, body = await http(env.port, "POST", "/say", {"text": "hi"})
        check("no pod known -> 404", st == 404 and "no pod" in body["error"], f"{st} {body}")

    async with Env(token="s3cret") as env:
        st, body = await http(env.port, "POST", "/say", {"text": f"{TAG} hi."})
        check("token set, none given -> 401", st == 401 and not env.pod.conns, f"{st} {body}")
        st, body = await http(env.port, "POST", "/say", {"text": f"{TAG} hi."},
                              headers={"Authorization": "Bearer s3cret"})
        check("token set, right token -> 200", st == 200, f"{st} {body}")


async def test_happy():
    async with Env() as env:
        dev = env.devices[0]
        text = f"{TAG} hello from the pod. {TAG} this is Robin."
        st, body = await http(env.port, "POST", "/say", {"text": text, "voice": "robin"})
        await asyncio.sleep(0.2)          # let the pod handler record the 0xDD
        audio = env.pod.audio()
        check("happy: 200 played, 2 sentences, duration from the frames",
              st == 200 and body["status"] == "played" and body["sentences"] == 2
              and body["played"] == 2 and abs(body["duration_s"] - 0.04) < 1e-6,
              f"{st} {body}")
        check("happy: robin's voice to the TTS, bot robin",
              [v for _, v in env.tts.calls] == ["tim-allen", "tim-allen"]
              and body["bot"] == "robin" and body["voice"] == "tim-allen", f"{env.tts.calls} {body}")
        check("happy: one audio connection per sentence, mic timeout 0, frames delivered",
              len(audio) == 2 and all(c["mic"] == 0 and c["frames"] == 1 for c in audio),
              f"{env.pod.conns}")
        check("happy: mic kept closed (0xDD 0 after each sentence, last one after the audio)",
              [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd", "audio", "mic_cmd"]
              and all(c["mic"] == 0 for c in env.pod.mic_cmds()), f"{env.pod.conns}")
        check("happy: pod released (processing/say_active False)",
              not dev.processing and not dev.say_active)
        lines = env.log_lines()
        check("happy: one JSON line in say.log with time, bot, text",
              len(lines) == 1 and lines[0]["bot"] == "robin" and lines[0]["text"] == text
              and lines[0]["status"] == "played" and lines[0].get("time"), f"{lines}")
        st2, body2 = await http(env.port, "POST", "/say", {"text": f"{TAG} again."})
        check("happy: a second say right after is refused while the pod plays (409)",
              st2 == 409 and "playing" in body2["reason"], f"{st2} {body2}")

        # The say reaches robin's next turn on this pod, once; not pepper's.
        check("memory: nothing for pepper", say.MEMORY.take(dev.hostname, "pepper") is None)
        note = say.MEMORY.take(dev.hostname, "robin")
        check("memory: robin's next turn gets a note with the text",
              note is not None and "hello from the pod." in note and "this is Robin." in note, note)
        check("memory: only once", say.MEMORY.take(dev.hostname, "robin") is None)

        st, body = await http(env.port, "GET", "/say")
        check("GET /say: pods with busy state, agents with voices",
              st == 200 and dev.hostname in body["devices"]
              and body["agents"]["robin"] == "tim-allen", f"{st} {body}")

    # listen=true: last header carries the default mic timeout, no 0xDD.
    async with Env() as env:
        st, body = await http(env.port, "POST", "/say",
                              {"text": f"{TAG} one. {TAG} two.", "voice": "pepper", "listen": True})
        await asyncio.sleep(0.2)
        check("listen: last sentence opens the mic, no 0xDD after it",
              st == 200 and [c["mic"] for c in env.pod.audio()] == [0, MIC]
              and [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd", "audio"],
              f"{st} {env.pod.conns}")

    # wait=false: answered at once, played in the background.
    async with Env() as env:
        env.tts.delay = 0.3
        t0 = time.monotonic()
        st, body = await http(env.port, "POST", "/say", {"text": f"{TAG} later.", "voice": "data"})
        quick = time.monotonic() - t0
        check("wait=true (default) answers after the audio was sent", quick >= 0.3, f"{quick:.2f}")
        await asyncio.sleep(0.5)
        env.pod.conns.clear()
        m.bargein._by_host.clear()        # forget the first say's playback estimate
        t0 = time.monotonic()
        st, body = await http(env.port, "POST", "/say",
                              {"text": f"{TAG} later.", "voice": "data", "wait": False})
        quick = time.monotonic() - t0
        check("wait=false: 200 queued at once", st == 200 and body["status"] == "queued"
              and quick < 0.25, f"{st} {body} {quick:.2f}")
        await asyncio.sleep(0.8)
        check("wait=false: still played", len(env.pod.audio()) == 1, f"{env.pod.conns}")

    # Interrupt (barge-in) after the first sentence: stops, no 0xDD, only
    # the played sentence remembered.
    async with Env(agents_cfg=None) as env:
        dev = env.devices[0]
        orig = env.tts.synthesize_stream

        async def ss(sentence, voice, config, voice_override=None):
            if len(env.tts.calls) == 1:
                dev.interrupted.set()
            async for pcm in orig(sentence, voice, config, voice_override):
                yield pcm
        m.tts.synthesize_stream = ss
        st, body = await http(env.port, "POST", "/say",
                              {"text": f"{TAG} first. {TAG} second.", "voice": "pepper"})
        await asyncio.sleep(0.2)
        note = say.MEMORY.take(dev.hostname, None)
        check("interrupt: status interrupted, 1 played, no 0xDD after the interrupt, pod released",
              st == 200 and body["status"] == "interrupted" and body["played"] == 1
              and [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd"]
              and not dev.processing and not dev.say_active,
              f"{st} {body} {env.pod.conns}")
        check("interrupt: only the played sentence remembered",
              note is not None and "first." in note and "second." not in note, note)
        check("no agents: voice passed to the TTS as is",
              env.tts.calls and env.tts.calls[0][1] == "pepper", f"{env.tts.calls}")


async def test_turn_note():
    """A say's note goes into the next voice turn's extra_context (once)."""
    say.MEMORY._said.clear()
    say.MEMORY.add("onju-test", None, f"{TAG} said by the pod.")
    seen = []

    class Conv(pf.FakeConversation):
        async def stream(self, text, extra_context=None):
            seen.append(extra_context)
            async for d in super().stream(text, extra_context):
                yield d

    async def run(conv):
        rec = pf.Recorder()
        m.asr.transcribe = lambda *a, **k: pf._ret({"text": "hello", "no_speech_prob": 0.0})
        m.stall_mod.decide_stall = lambda *a, **k: pf._ret(None)
        m.tts.synthesize_stream = rec.synthesize_stream
        m.open_audio_connection = rec.open_audio_connection
        m.write_audio_frames = lambda w, f: True
        m.close_audio_connection = pf._anoop
        m.send_audio = rec.send_audio
        m.send_state = pf._anoop
        m.OpusStreamEncoder = pf.FakeEncoder
        device = pf.make_device(conv)
        q = asyncio.Queue()
        await q.put((device, pf.FakeAudio()))
        worker = asyncio.create_task(m.process_utterances(pf.make_config(), None, q))
        deadline = time.monotonic() + 10
        await asyncio.sleep(0)
        while not q.empty() or device.processing or m._state_tasks:
            if time.monotonic() > deadline:
                break
            await asyncio.sleep(0.01)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    await run(Conv([f"{TAG} ok."]))
    await run(Conv([f"{TAG} ok."]))
    check("turn: the say reaches the next turn as a note in extra_context",
          len(seen) == 2 and seen[0] and "said by the pod." in seen[0], f"{seen}")
    check("turn: and only that one", seen[1] is None, f"{seen}")


def test_resolve():
    r = agents.router_from_config({"agents": AGENTS_CFG})
    cases = [
        (("robin", ""), ("tim-allen", "robin")),
        (("Tim", ""), ("tim-allen", "robin")),          # alias
        (("tim-allen", ""), ("tim-allen", "robin")),    # TTS voice -> its agent
        (("", ""), ("pepper", "pepper")),                # default agent
        (("", "data"), ("jack-nicholson", "data")),      # bot gives the voice
        (("pepper", "robin"), ("pepper", "robin")),      # voice and bot apart
        (("some-voice", ""), ("some-voice", None)),      # unknown voice: as is
    ]
    for (voice, bot), want in cases:
        got = say.resolve_voice(r, voice, bot)
        check(f"resolve voice={voice!r} bot={bot!r} -> {want}", got == want, f"{got}")
    check("resolve without agents: as is", say.resolve_voice(None, "x", "Y") == ("x", "y"))
    # raw_voice: the name is a TTS voice even when an agent has that name
    raw_cases = [
        (("data", ""), ("data", None)),                  # voice "data", not agent data
        (("jack-nicholson", ""), ("jack-nicholson", "data")),  # agent with that voice
        (("tim-allen", ""), ("tim-allen", "robin")),
        (("data", "data"), ("data", "data")),            # bot given: it owns the line
        (("Tim", ""), ("Tim", None)),                    # alias not resolved when raw
    ]
    for (voice, bot), want in raw_cases:
        got = say.resolve_voice(r, voice, bot, raw=True)
        check(f"resolve raw voice={voice!r} bot={bot!r} -> {want}", got == want, f"{got}")
    p = say.parse_request(say.SayConfig({"enabled": True}), b'{"text": "hi", "voice": "data", "raw_voice": true}')
    check("parse: raw_voice kept", p["raw_voice"] is True, p)
    p = say.parse_request(say.SayConfig({"enabled": True}), b'{"text": "hi"}')
    check("parse: raw_voice defaults to false", p["raw_voice"] is False, p)


def test_parse_staged():
    scfg = say.SayConfig({"enabled": True})
    r = say.parse_request(scfg, b'{"text": "answer.", "filler": "  Let me look.  ", "think_s": 3, "delay_s": 5}')
    check("parse: filler cleaned, think_s/delay_s kept",
          r["text"] == "answer." and r["filler"] == "Let me look."
          and r["think_s"] == 3.0 and r["delay_s"] == 5.0, r)
    r = say.parse_request(scfg, b'{"text": "hi."}')
    check("parse: plain request has no filler and zero think/delay",
          r["filler"] is None and r["think_s"] == 0.0 and r["delay_s"] == 0.0, r)
    r = say.parse_request(scfg, b'{"text": "hi.", "think_s": 100, "delay_s": 999}')
    check("parse: think_s capped at 30, delay_s at 120",
          r["think_s"] == say.MAX_THINK_S and r["delay_s"] == say.MAX_DELAY_S, r)
    for bad in (b'{"text": "hi.", "think_s": -1}',
                b'{"text": "hi.", "think_s": "x"}',
                b'{"text": "hi.", "filler": 1}'):
        try:
            say.parse_request(scfg, bad)
            check(f"parse: bad {bad} raises", False)
        except say.SayError as e:
            check(f"parse: bad {bad} -> 400", e.status == 400, e.body)
    r = say.parse_request(scfg, b'{"text": "hi.", "filler": "   "}')
    check("parse: blank filler is None", r["filler"] is None, r)


async def test_staged():
    """filler -> wait for play -> THINKING -> answer; delay; interrupt; busy."""
    states = []

    async with Env() as env:
        # Record fire_state (THINKING) without touching the real pod wire.
        real_fire = m.fire_state

        def record_fire(ip, port, state):
            states.append(state)
            return real_fire(ip, port, state)

        m.fire_state = record_fire
        try:
            text = f"{TAG} I do not know."
            filler = f"{TAG} Let me look that up."
            t0 = time.monotonic()
            st, body = await http(env.port, "POST", "/say", {
                "text": text, "voice": "pepper", "filler": filler, "think_s": 0.35,
            })
            elapsed = time.monotonic() - t0
            await asyncio.sleep(0.2)
            audio = env.pod.audio()
            check("staged: 200 played, filler+answer both played",
                  st == 200 and body["status"] == "played"
                  and body["played"] == 1 and body["filler_played"] == 1
                  and body.get("filler") == filler and body.get("think_s") == 0.35,
                  f"{st} {body}")
            check("staged: two audio connections (filler then answer), mic 0 on both",
                  len(audio) == 2 and all(c["mic"] == 0 for c in audio), f"{env.pod.conns}")
            check("staged: mic kept closed after the filler and after the answer",
                  [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd", "audio", "mic_cmd"],
                  f"{env.pod.conns}")
            check("staged: TTS got filler then answer in pepper's voice",
                  [s for s, _ in env.tts.calls] == [filler, text]
                  and all(v == "pepper" for _, v in env.tts.calls), f"{env.tts.calls}")
            check("staged: THINKING fired at least once during the hold",
                  states.count(2) >= 1, f"states={states}")
            # Filler play wait uses bargein margin (~0.5s) + think_s.
            check("staged: elapsed covers filler-play wait + think_s",
                  elapsed >= 0.35, f"{elapsed:.2f}")
            lines = env.log_lines()
            check("staged: say.log has filler line then answer line",
                  len(lines) == 2 and lines[0].get("filler") is True
                  and lines[0]["text"] == filler and lines[0]["status"] == "played"
                  and lines[1].get("filler") is None and lines[1]["text"] == text
                  and lines[1]["status"] == "played", f"{lines}")
            note = say.MEMORY.take(env.devices[0].hostname, "pepper")
            check("staged: tell_agent note marks the filler and includes the answer",
                  note is not None and "short filler" in note and "Let me look" in note
                  and "I do not know" in note, note)
        finally:
            m.fire_state = real_fire

    # delay_s: claims immediately (second /say -> 409), then speaks.
    async with Env() as env:
        env.tts.delay = 0.05
        started = asyncio.Event()
        done = asyncio.Event()

        async def delayed():
            started.set()
            return await http(env.port, "POST", "/say", {
                "text": f"{TAG} after delay.", "voice": "data", "delay_s": 0.4,
            })

        task = asyncio.create_task(delayed())
        await started.wait()
        await asyncio.sleep(0.05)
        st2, body2 = await http(env.port, "POST", "/say", {"text": f"{TAG} no."})
        check("delay: second /say during delay is 409 busy",
              st2 == 409 and "say" in body2["reason"], f"{st2} {body2}")
        st, body = await task
        check("delay: first say plays after delay_s",
              st == 200 and body["status"] == "played" and body.get("delay_s") == 0.4
              and body.get("delayed") is True, f"{st} {body}")

    # Interrupt during THINKING: filler was played, answer never starts.
    async with Env() as env:
        states.clear()
        real_fire = m.fire_state

        def record_fire(ip, port, state):
            states.append(state)
            # Trip the interrupt the first time THINKING is requested.
            if state == 2:
                env.devices[0].interrupted.set()
            return real_fire(ip, port, state)

        m.fire_state = record_fire
        try:
            st, body = await http(env.port, "POST", "/say", {
                "text": f"{TAG} answer never.", "voice": "robin",
                "filler": f"{TAG} looking.", "think_s": 2.0,
            })
            await asyncio.sleep(0.2)
            check("interrupt-in-think: status interrupted, answer not played",
                  st == 200 and body["status"] == "interrupted"
                  and body["played"] == 0 and body["filler_played"] == 1,
                  f"{st} {body}")
            check("interrupt-in-think: only filler audio and its 0xDD, nothing after the abort",
                  [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd"],
                  f"{env.pod.conns}")
            check("interrupt-in-think: pod released",
                  not env.devices[0].processing and not env.devices[0].say_active)
            lines = env.log_lines()
            check("interrupt-in-think: filler logged played, answer interrupted",
                  len(lines) == 2 and lines[0].get("filler") and lines[0]["status"] == "played"
                  and lines[1]["status"] == "interrupted" and not lines[1].get("filler"),
                  f"{lines}")
            note = say.MEMORY.take(env.devices[0].hostname, "robin")
            check("interrupt-in-think: only the filler is remembered",
                  note is not None and "looking" in note and "answer never" not in note, note)
        finally:
            m.fire_state = real_fire

    # listen=true on a staged say: only the answer opens the mic.
    async with Env() as env:
        st, body = await http(env.port, "POST", "/say", {
            "text": f"{TAG} final.", "voice": "pepper",
            "filler": f"{TAG} hold on.", "think_s": 0.2, "listen": True,
        })
        await asyncio.sleep(0.2)
        mics = [c["mic"] for c in env.pod.audio()]
        check("staged+listen: filler mic 0 (+0xDD), answer opens mic, no 0xDD after it",
              st == 200 and mics == [0, MIC]
              and [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd", "audio"],
              f"{st} {mics} {env.pod.conns}")



def test_render_mode():
    """say.render: sentence by default; whole only with tts.backend local."""
    c = say.SayConfig({})
    check("render: default is sentence", c.render == "sentence", c.render)
    check("render: unknown value falls back to sentence",
          say.SayConfig({"render": "bogus"}).render == "sentence")
    w = say.SayConfig({"render": " Whole "})
    check("render: whole parsed", w.render == "whole", w.render)
    check("render: whole + tts local -> whole",
          say.render_whole(w, {"tts": {"backend": "local"}}) is True)
    check("render: whole + local_stream / elevenlabs / no tts -> per sentence",
          not say.render_whole(w, {"tts": {"backend": "local_stream"}})
          and not say.render_whole(w, {"tts": {"backend": "elevenlabs"}})
          and not say.render_whole(w, {}))
    check("render: sentence + tts local -> per sentence",
          not say.render_whole(c, {"tts": {"backend": "local"}}))


async def test_render_whole():
    """render: whole renders every sentence before the first is sent."""
    text = f"{TAG} One. Two is here. Three!"

    async def run(backend, events, **say_extra):
        async with Env(tts_backend=backend, **say_extra) as env:
            fake_stream = env.tts.synthesize_stream
            real_open = m.open_audio_connection

            async def rec_stream(sentence, voice, config, **kw):
                events.append(("tts", sentence))
                async for pcm in fake_stream(sentence, voice, config, **kw):
                    yield pcm

            async def rec_open(ip, port, **kw):
                events.append(("open", kw.get("mic_timeout")))
                return await real_open(ip, port, **kw)

            m.tts.synthesize_stream = rec_stream
            m.open_audio_connection = rec_open
            st, body = await http(env.port, "POST", "/say", {"text": text, "voice": "pepper"})
            await asyncio.sleep(0.2)
            return env, st, body

    def kinds(events):
        return [k for k, _ in events]

    ev = []
    env, st, body = await run("local", ev, render="whole")
    check("whole: 200 played, 3/3",
          st == 200 and body["status"] == "played" and body["played"] == 3, f"{st} {body}")
    check("whole: all 3 sentences rendered before the first audio connection",
          kinds(ev) == ["tts"] * 3 + ["open"] * 3, f"{ev}")
    check("whole: 3 audio connections, mic 0, each followed by 0xDD 0",
          [c["mic"] for c in env.pod.audio()] == [0, 0, 0]
          and [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd"] * 3, f"{env.pod.conns}")
    check("whole: TTS got each sentence once, in order, in pepper's voice",
          [s for s, _ in env.tts.calls] == [f"{TAG} One.", "Two is here.", "Three!"]
          and all(v == "pepper" for _, v in env.tts.calls), f"{env.tts.calls}")
    lines = env.log_lines()
    check("whole: one say.log line, played",
          len(lines) == 1 and lines[0]["status"] == "played", f"{lines}")

    for backend, extra, label in (("local", {}, "sentence (default)"),
                                  ("local_stream", {"render": "whole"}, "whole + local_stream")):
        ev = []
        env, st, body = await run(backend, ev, **extra)
        check(f"{label}: unchanged, render and send one sentence at a time",
              st == 200 and body["played"] == 3
              and kinds(ev) == ["tts", "open"] * 3, f"{st} {body} {ev}")

    # Staged: the answer renders during the filler/think phase, before THINKING ends.
    states = []
    real_fire = m.fire_state
    m.fire_state = lambda ip, port, state: (states.append(("state", state)),
                                            real_fire(ip, port, state))[1]
    try:
        filler = f"{TAG} Hmm. Let me see."
        async with Env(tts_backend="local", render="whole") as env:
            fake_stream = env.tts.synthesize_stream
            real_open = m.open_audio_connection

            async def rec_stream(sentence, voice, config, **kw):
                states.append(("tts", sentence))
                async for pcm in fake_stream(sentence, voice, config, **kw):
                    yield pcm

            async def rec_open(ip, port, **kw):
                states.append(("open", kw.get("mic_timeout")))
                return await real_open(ip, port, **kw)

            m.tts.synthesize_stream = rec_stream
            m.open_audio_connection = rec_open
            states.clear()
            st, body = await http(env.port, "POST", "/say", {
                "text": text, "voice": "pepper", "filler": filler, "think_s": 0.3})
            await asyncio.sleep(0.2)
            k = kinds(states)
            check("whole staged: 200 played, filler 2/2, answer 3/3",
                  st == 200 and body["status"] == "played" and body["played"] == 3
                  and body["filler_played"] == 2, f"{st} {body}")
            check("whole staged: filler rendered, then answer renders before THINKING ends; "
                  "filler plays back to back, answer after THINKING",
                  k[:2] == ["tts", "tts"] and k.count("tts") == 5
                  and k.index("state") > max(i for i, x in enumerate(k) if x == "tts")
                  and k[-3:] == ["open"] * 3 and k.count("open") == 5,
                  f"{states}")
            lines = env.log_lines()
            check("whole staged: filler + answer logged played",
                  len(lines) == 2 and lines[0].get("filler") and lines[1]["status"] == "played",
                  f"{lines}")

        # Slow answer render: THINKING stays on past think_s until it is ready.
        async with Env(tts_backend="local", render="whole") as env:
            env.tts.delay = 0.25
            states.clear()
            t0 = time.monotonic()
            st, body = await http(env.port, "POST", "/say", {
                "text": text, "voice": "pepper", "think_s": 0.1})
            elapsed = time.monotonic() - t0
            check("whole slow render: played after the full render (>= 3 x 0.25 s)",
                  st == 200 and body["status"] == "played" and body["played"] == 3
                  and elapsed >= 0.75, f"{st} {body} {elapsed:.2f}")
            check("whole slow render: THINKING fired",
                  ("state", 2) in states, f"{states}")

        # Tap while waiting for the render: nothing sent, no 0xDD, pod released.
        async with Env(tts_backend="local", render="whole") as env:
            env.tts.delay = 0.5
            dev = env.devices[0]

            async def tap():
                await asyncio.sleep(0.2)
                dev.interrupted.set()

            tap_task = asyncio.create_task(tap())
            st, body = await http(env.port, "POST", "/say", {
                "text": text, "voice": "pepper", "think_s": 0.1})
            await tap_task
            await asyncio.sleep(0.1)
            check("whole interrupted during render: interrupted, nothing played",
                  st == 200 and body["status"] == "interrupted" and body["played"] == 0
                  and not env.pod.audio() and not env.pod.mic_cmds(), f"{st} {body} {env.pod.conns}")
            check("whole interrupted during render: pod released",
                  not dev.processing and not dev.say_active)
    finally:
        m.fire_state = real_fire


def make_wav(seconds, rate=16000, channels=1, width=2, value=1000):
    import base64, io, wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        n = int(seconds * rate) * channels
        sample = value.to_bytes(width, "little", signed=width > 1) if width > 1 else bytes([200])
        w.writeframes(sample * n)
    return base64.b64encode(buf.getvalue()).decode()


async def test_wav():
    pcm = say.decode_wav(make_wav(1.0, rate=24000, channels=2))
    check("wav decode: 24 kHz stereo -> 16 kHz mono 16-bit (1 s = 32000 bytes), level kept",
          len(pcm) == 32000 and int.from_bytes(pcm[100:102], "little", signed=True) == 1000,
          f"{len(pcm)}")
    check("wav decode: 16 kHz mono passes through unchanged",
          say.decode_wav(make_wav(0.5)) == (1000).to_bytes(2, "little") * 8000)

    async with Env() as env:
        dev = env.devices[0]
        st, body = await http(env.port, "POST", "/say",
                              {"wav": make_wav(1.0), "voice": "pepper", "text": f"{TAG} from a file."})
        await asyncio.sleep(0.2)
        audio = env.pod.audio()
        check("wav: 200 played, one part, no TTS call",
              st == 200 and body["status"] == "played" and body["sentences"] == 1
              and body["played"] == 1 and env.tts.calls == [], f"{st} {body} {env.tts.calls}")
        check("wav: one audio connection, 2 batches of 0.5 s, mic 0 then 0xDD 0",
              len(audio) == 1 and audio[0]["frames"] == 2 and audio[0]["mic"] == 0
              and [c["kind"] for c in env.pod.conns] == ["audio", "mic_cmd"], f"{env.pod.conns}")
        lines = env.log_lines()
        check("wav: say.log has the text label, pod released",
              len(lines) == 1 and lines[0]["text"] == f"{TAG} from a file."
              and not dev.processing and not dev.say_active, f"{lines}")

    async with Env() as env:
        st, body = await http(env.port, "POST", "/say",
                              {"wav": make_wav(1.0, rate=24000, channels=2), "listen": True,
                               "filler": f"{TAG} one sec.", "think_s": 0.1})
        await asyncio.sleep(0.2)
        audio = env.pod.audio()
        lines = env.log_lines()
        check("wav + filler + listen: TTS only for the filler, wav last with the mic timeout",
              st == 200 and [c for c, _ in env.tts.calls] == [f"{TAG} one sec."]
              and [c["mic"] for c in audio] == [0, MIC] and audio[1]["frames"] == 2,
              f"{st} {body} {env.tts.calls} {env.pod.conns}")
        check("wav without text: labelled [wav 1.0s] in say.log",
              any(x["text"] == "[wav 1.0s]" for x in lines), f"{lines}")

    async with Env() as env:
        bad = [({"wav": "not base64!"}, "base64"),
               ({"wav": "aGVsbG8="}, "not a PCM WAV"),
               ({"wav": make_wav(0.1, width=1)}, "16-bit"),
               ({"wav": make_wav(say.MAX_WAV_S + 1)}, "max"),
               ({"voice": "pepper"}, "wav")]
        results = []
        for obj, needle in bad:
            st, body = await http(env.port, "POST", "/say", obj)
            results.append((st, body.get("error", "")))
        check("wav refusals: bad base64, not a WAV, 8-bit, too long, nothing to say -> 400 "
              "with a clear error, nothing sent",
              all(st == 400 and needle in err for (st, err), (_, needle) in zip(results, bad))
              and env.pod.conns == [], f"{results}")

    # grok-voice tell_agent: a wav's text is remembered for the agent's next
    # turn like a text say; a wav without text leaves nothing.
    async with Env() as env:
        hostname = env.devices[0].hostname
        say.MEMORY._said.clear()
        st, body = await http(env.port, "POST", "/say",
                              {"wav": make_wav(0.5), "text": f"{TAG} filed line."})
        bot = body.get("bot")
        note = say.MEMORY.take(hostname, bot)
        m.bargein._by_host.clear()
        await http(env.port, "POST", "/say", {"wav": make_wav(0.5)})
        check("wav tell_agent: text remembered for the next turn, no text -> nothing told",
              note is not None and f"{TAG} filed line." in note
              and say.MEMORY.take(hostname, bot) is None, f"{st} {body} {note}")


async def main():
    await test_disabled()
    await test_refusals()
    await test_happy()
    await test_staged()
    await test_turn_note()
    test_resolve()
    test_parse_staged()
    test_render_mode()
    await test_render_whole()
    await test_wav()
    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(pf.CHECKS)
    print(f"{total - len(pf.FAILS)}/{total} passed")
    return 1 if pf.FAILS else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
