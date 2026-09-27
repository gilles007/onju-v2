"""
Offline test for multi-agent routing (pipeline/agents.py and its use in
pipeline/main.py, pipeline/services/tts.py, pipeline/conversation/agentic.py):

- address detection at the start of an utterance: greetings and fillers,
  punctuation rules, "Okay." not counting as a greeting, names later in the
  sentence ignored;
- fuzzy matching of speech-to-text misspellings, Ruby vs Robin kept apart
  (Rubi/Rubie/Rooby/Rudy -> Ruby, Robbin/Robyn/Robbie/Rubin -> Robin,
  Rob/Ruben -> nobody), a tie between two agents matches nobody;
- the address is stripped from the text sent upstream (or kept, when
  strip_address is off); a bare "Hey Ruby" becomes "Hey Ruby.";
- stickiness per device, the sticky timeout, touch() at the end of a turn;
- voice choice: the agent's voice, else the default agent's voice;
- TTS: voice_override wins in the local_stream and local payloads, and the
  payload is unchanged without it;
- agentic backend: X-Onju-Bot header only when a bot is given;
- through main.py's real process_utterances: without `agents:` the turn is
  exactly as before (no extra arguments anywhere); with it, the stall and
  the reply use the agent's voice, the bot goes upstream, the stall
  classifier and the agent get the stripped text, follow-ups stick.

Reuses the stubs and fakes of tests/test_pause_flush.py (no network, no
audio hardware). Run from the repo root: python3 tests/test_agents.py
"""
import asyncio
import glob
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs for pipeline.main)

from pipeline import agents  # noqa: E402

m = pf.m
TAG = "Zzag"
FAILS = []
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if not cond and detail else ""))
    if not cond:
        FAILS.append(name)


AGENTS_CFG = {
    "default": "pepper",
    "sticky_timeout_s": 300,
    "list": [
        {"name": "pepper", "aliases": ["peppa"], "voice": "pepper"},
        {"name": "data", "voice": "data"},
        {"name": "ruby"},
        {"name": "robin", "aliases": ["robyn"]},
    ],
}


def router(**over):
    cfg = dict(AGENTS_CFG, **over)
    return agents.router_from_config({"agents": cfg})


def who(r, text):
    a = r.detect(text)
    return a.agent if a else None


# ---------------------------------------------------------------------------
# 1. Detection and fuzzy matching
# ---------------------------------------------------------------------------

def test_detection():
    r = router()
    cases = [
        ("Hey Ruby, what's the weather?", "ruby"),
        ("Hey Robin what's on my calendar", "robin"),
        ("Ruby, set a timer.", "ruby"),
        ("Ruby?", "ruby"),
        ("Okay Data what time is it", "data"),
        ("Um, hey Pepper, go on.", "pepper"),
        ("Hi Data.", "data"),
        ("Hello, Robin!", "robin"),
        ("Data, read me the news.", "data"),
        # not an address
        ("Data shows the market is up.", None),          # bare start, no punctuation
        ("Okay. Data shows the market is up.", None),    # sentence ended after "Okay"
        ("What time is it, Ruby?", None),                 # name at the end
        ("Can Ruby hear me", None),
        ("Hey, what's the weather?", None),
        ("So data from yesterday looks fine", None),      # filler, not a greeting
        ("", None),
        ("Hey", None),
    ]
    for text, want in cases:
        got = who(r, text)
        check(f"detect {text!r} -> {want}", got == want, f"got {got}")


def test_fuzzy():
    r = router()
    cases = [
        # Ruby spellings
        ("Hey Rubi, hi.", "ruby"), ("Hey Rubie, hi.", "ruby"), ("Hey Rooby, hi.", "ruby"),
        ("Hey Ruby's there", "ruby"), ("Hey Rudy, hi.", "ruby"),
        # Robin spellings
        ("Hey Robbin, hi.", "robin"), ("Hey Robyn, hi.", "robin"), ("Hey Robbie, hi.", "robin"),
        ("Hey Rubin, hi.", "robin"),
        # too far from both: nobody
        ("Hey Rob, hi.", None), ("Hey Ruben, hi.", None), ("Hey Robert, hi.", None),
        # others
        ("Hey Peppa!", "pepper"), ("Hey Pepa, hi.", "pepper"), ("Hey Dana, hi.", "data"),
        ("Hey Dad, hi.", None), ("Hey Siri, hi.", None),
    ]
    for text, want in cases:
        got = who(r, text)
        check(f"fuzzy {text!r} -> {want}", got == want, f"got {got}")

    # Ruby and Robin can never match each other, whatever the aliases.
    check("fuzzy: 'Ruby' is not within reach of Robin",
          r.match_word("ruby") == ("ruby", 0) and r.match_word("robin") == ("robin", 0))
    check("fuzzy: normalised keys 2 edits apart (> 1 allowed)",
          agents._distance(agents._key("ruby"), agents._key("robin")) == 2
          and agents.allowed_distance(agents._key("robin")) == 1)
    # Rubin: 1 edit from both keys, settled by spelling; an explicit alias makes it exact.
    check("fuzzy: 'Rubin' -> robin at distance 1 without an alias",
          r.match_word("Rubin") == ("robin", 1), f"{r.match_word('Rubin')}")
    r2 = router(list=[{"name": "ruby"}, {"name": "robin", "aliases": ["rubin"]}], default="ruby")
    check("fuzzy: 'Rubin' as a robin alias is exact", r2.match_word("Rubin") == ("robin", 0))
    # A true tie between two agents: no guess.
    r3 = router(list=[{"name": "bella"}, {"name": "della"}], default="bella")
    check("fuzzy: tie between two agents matches nobody", r3.match_word("sella") is None,
          f"{r3.match_word('sella')}")
    # Short names are exact only.
    r4 = router(list=[{"name": "max"}, {"name": "pepper"}], default="pepper")
    check("fuzzy: 3-letter name is exact only",
          r4.match_word("max") == ("max", 0) and r4.match_word("mac") is None)


# ---------------------------------------------------------------------------
# 2. Stripping, stickiness, voices
# ---------------------------------------------------------------------------

def test_route():
    r = router()
    rt = r.route("nest", "Hey Rubin, what's on my calendar?", now=0)
    check("route: address stripped, capitalised, bot robin",
          (rt.agent, rt.text, rt.reason) == ("robin", "What's on my calendar?", "addressed"), f"{rt}")
    rt = r.route("nest", "Hey Ruby.", now=1)
    check("route: bare address keeps a correctly spelled greeting",
          (rt.agent, rt.text) == ("ruby", "Hey Ruby."), f"{rt}")
    rk = router(strip_address=False)
    rt = rk.route("nest", "Hey Ruby, what's up?", now=0)
    check("route: strip_address off keeps the text as heard",
          (rt.agent, rt.text) == ("ruby", "Hey Ruby, what's up?"), f"{rt}")
    rt = r.route("nest2", "What's the weather?", now=0)
    check("route: nobody named, first turn -> default pepper, text unchanged",
          (rt.agent, rt.reason, rt.text) == ("pepper", "default", "What's the weather?"), f"{rt}")


def test_sticky():
    r = router(sticky_timeout_s=60)
    r.route("a", "Hey Data, what time is it?", now=0)
    rt = r.route("a", "And in Tokyo?", now=30)
    check("sticky: follow-up stays with data", (rt.agent, rt.reason) == ("data", "sticky"), f"{rt}")
    rt = r.route("b", "What time is it?", now=31)
    check("sticky: per device (other pod gets the default)", rt.agent == "pepper", f"{rt}")
    rt = r.route("a", "Hey Robin, hi.", now=40)
    check("sticky: a new name switches", rt.agent == "robin", f"{rt}")
    rt = r.route("a", "What was I saying?", now=99)
    check("sticky: within timeout of the last turn", rt.agent == "robin", f"{rt}")
    rt = r.route("a", "Anything new?", now=99 + 61)
    check("sticky: timeout -> back to default", (rt.agent, rt.reason) == ("pepper", "timeout"), f"{rt}")
    r.route("c", "Hey Data, tell me a long story.", now=0)
    r.touch("c", now=100)       # the reply played until t=100
    rt = r.route("c", "Go on.", now=150)
    check("sticky: touch() at turn end restarts the clock", rt.agent == "data", f"{rt}")


def test_voices():
    r = router()
    check("voice: agent's own voice", r.voice_for("data") == "data")
    check("voice: agent without a voice uses the default agent's (ruby -> pepper)",
          r.voice_for("ruby") == "pepper" and r.voice_for("robin") == "pepper")
    r2 = router(list=[{"name": "pepper"}, {"name": "data", "voice": "data"}])
    check("voice: default agent without a voice -> None (TTS config voice)",
          r2.voice_for("pepper") is None and r2.voice_for("data") == "data")


def test_unconfigured():
    check("unconfigured: no agents section -> no router",
          agents.router_from_config({"tts": {}}) is None
          and agents.router_from_config({"agents": None}) is None)
    try:
        agents.router_from_config({"agents": {"default": "zed", "list": [{"name": "pepper"}]}})
        ok = False
    except ValueError:
        ok = True
    check("config: unknown default agent is an error at startup", ok)


# ---------------------------------------------------------------------------
# 3. TTS payload and the X-Onju-Bot header
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    async def aiter_bytes(self):
        yield self.body

    @property
    def content(self):
        return self.body


class _FakeStreamCtx:
    def __init__(self, resp):
        self.resp = resp

    async def __aenter__(self):
        return self.resp

    async def __aexit__(self, *a):
        return False


def _wav_header(sr=16000):
    import struct
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


async def test_tts_payload():
    # Load the real tts module from its file (test_pause_flush replaced
    # pipeline.services.tts with a stub); stub its HTTP/audio imports if absent.
    for name, attrs in (("httpx", {}), ("pydub", {"AudioSegment": object}),
                        ("audioop", {"ratecv": None})):
        try:
            __import__(name)
        except ImportError:
            pf._stub(name, **attrs)
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "pipeline", "services", "tts.py")
    spec = importlib.util.spec_from_file_location("real_tts_under_test", path)
    real_tts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_tts)
    sent = []

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, method, url, json):
            sent.append(json)
            return _FakeStreamCtx(_FakeResp(_wav_header() + b"\0\0" * 8))

    real_tts.httpx = types.SimpleNamespace(AsyncClient=FakeClient)
    cfg = {"audio": {"sample_rate": 16000},
           "tts": {"backend": "local_stream",
                   "local_stream": {"url": "http://x", "voice": "pepper"}}}
    out = [c async for c in real_tts.synthesize_stream("Hi.", "Emma", cfg)]
    check("tts: no override -> config voice (unchanged)", sent[-1]["voice"] == "pepper" and out, f"{sent[-1]}")
    base = dict(sent[-1])
    out = [c async for c in real_tts.synthesize_stream("Hi.", "Emma", cfg, voice_override="data")]
    check("tts: voice_override wins over the config voice", sent[-1]["voice"] == "data", f"{sent[-1]}")
    check("tts: only the voice differs", {k: v for k, v in sent[-1].items() if k != "voice"}
          == {k: v for k, v in base.items() if k != "voice"})
    out = [c async for c in real_tts.synthesize_stream("Hi.", "Emma", cfg, voice_override=None)]
    check("tts: voice_override=None -> same payload as before", sent[-1] == base)

    posted = []

    async def fake_local_post(url, json):
        posted.append(json)
        return _FakeResp(b"")

    class FakeClient2(FakeClient):
        async def post(self, url, json):
            return await fake_local_post(url, json)

    class FakeSeg:
        raw_data = b"\0\0"

        @staticmethod
        def from_wav(_):
            return FakeSeg()

        def set_channels(self, *_):
            return self

        set_frame_rate = set_sample_width = set_channels

        def __len__(self):
            return 1

    real_tts.httpx = types.SimpleNamespace(AsyncClient=FakeClient2)
    real_tts.AudioSegment = FakeSeg
    kcfg = {"audio": {"sample_rate": 16000},
            "tts": {"backend": "local", "local": {"url": "http://k", "model": "kokoro", "voice": "af_heart"}}}
    await real_tts.synthesize("Hi.", "Emma", kcfg)
    await real_tts.synthesize("Hi.", "Emma", kcfg, voice_override="af_bella")
    check("tts local: config voice without override, override wins with it",
          [p["voice"] for p in posted] == ["af_heart", "af_bella"], f"{posted}")


async def test_bot_header():
    import pipeline.conversation.agentic as ag
    calls = []

    class FakeOpenAI:
        def __init__(self, **k):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

        async def create(self, **kw):
            calls.append(kw)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok"))])

    ag.AsyncOpenAI = FakeOpenAI
    b = ag.AgenticBackend({"base_url": "http://b", "model": "pepper"}, "nest")
    k = b._build_kwargs("hello")
    check("agentic: no bot -> no extra headers (unchanged)", "extra_headers" not in k, f"{k}")
    k = b._build_kwargs("hello", None, "ruby")
    check("agentic: bot -> X-Onju-Bot header", k.get("extra_headers") == {"X-Onju-Bot": "ruby"}, f"{k}")
    b2 = ag.AgenticBackend({"base_url": "http://b", "provider_model": "grok"}, "nest")
    k = b2._build_kwargs("hello", None, "data")
    check("agentic: bot header merged with x-openclaw-model",
          k.get("extra_headers") == {"x-openclaw-model": "grok", "X-Onju-Bot": "data"}, f"{k}")
    await b.send("hi", bot="robin")
    check("agentic: send() passes the bot too", calls[-1]["extra_headers"] == {"X-Onju-Bot": "robin"})


# ---------------------------------------------------------------------------
# 4. Through main.process_utterances
# ---------------------------------------------------------------------------

class TurnRecorder:
    def __init__(self):
        self.tts = []      # (sentence, voice, kwargs)
        self.stall_text = []

    async def synthesize_stream(self, sentence, voice, config, **kwargs):
        self.tts.append((sentence, voice, kwargs))
        await asyncio.sleep(0.01)
        yield b"\0\0" * 16000


class Conv:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []    # (text, kwargs)

    async def stream(self, text, **kwargs):
        self.calls.append((text, kwargs))
        yield self.reply

    def commit(self, text):
        pass


def turn_config(with_agents, stall=True):
    cfg = pf.make_config(pause=0.3, stall=stall)
    if with_agents:
        cfg["agents"] = AGENTS_CFG
    return cfg


async def run_turns(utterances, with_agents, stall_line=f"{TAG} one sec."):
    rec = TurnRecorder()
    heard = []

    async def fake_asr(*a, **k):
        return {"text": heard_q.pop(0), "no_speech_prob": 0.0}

    async def fake_stall(text, config, **k):
        rec.stall_text.append(text)
        return stall_line

    heard_q = list(utterances)
    m.asr.transcribe = fake_asr
    m.stall_mod.decide_stall = fake_stall
    m.tts.synthesize_stream = rec.synthesize_stream
    m.open_audio_connection = pf.Recorder().open_audio_connection
    m.write_audio_frames = lambda w, f: True
    m.close_audio_connection = pf._anoop
    m.send_audio = pf._anoop
    m.OpusStreamEncoder = pf.FakeEncoder

    conv = Conv(f"{TAG} here you go.")
    device = pf.make_device(conv)
    q: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(m.process_utterances(turn_config(with_agents), None, q))
    try:
        for _ in utterances:
            await q.put((device, pf.FakeAudio()))
            await asyncio.sleep(0)
            for _ in range(2000):
                if q.empty() and not device.processing:
                    break
                await asyncio.sleep(0.01)
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
    return rec, conv


async def test_pipeline():
    rec, conv = await run_turns(["Hey Ruby, what's the weather?"], with_agents=False)
    check("pipeline unconfigured: TTS called with no extra arguments",
          rec.tts and all(kw == {} for _, _, kw in rec.tts), f"{rec.tts}")
    check("pipeline unconfigured: upstream text untouched, no bot",
          conv.calls == [("Hey Ruby, what's the weather?", {"extra_context": conv.calls[0][1].get("extra_context")})]
          and "bot" not in conv.calls[0][1], f"{conv.calls}")
    check("pipeline unconfigured: stall classifier saw the full text",
          rec.stall_text == ["Hey Ruby, what's the weather?"], f"{rec.stall_text}")

    rec, conv = await run_turns(
        ["Hey Data, what time is it in Paris?", "And in Tokyo?", "Hey Robbin, what's on my list?"],
        with_agents=True)
    voices = [kw.get("voice_override") for _, _, kw in rec.tts]
    check("pipeline agents: stall and reply both in data's voice, then robin -> pepper voice",
          voices == ["data", "data", "data", "data", "pepper", "pepper"], f"{rec.tts}")
    check("pipeline agents: stall line is the first sentence of each turn",
          [s for s, _, _ in rec.tts][0::2] == [f"{TAG} one sec."] * 3, f"{rec.tts}")
    check("pipeline agents: bot sent upstream (data, data sticky, robin)",
          [kw.get("bot") for _, kw in conv.calls] == ["data", "data", "robin"], f"{conv.calls}")
    check("pipeline agents: address stripped for agent and stall classifier",
          [t for t, _ in conv.calls] == ["What time is it in Paris?", "And in Tokyo?", "What's on my list?"]
          and rec.stall_text == [t for t, _ in conv.calls], f"{conv.calls} {rec.stall_text}")


async def main():
    test_detection()
    test_fuzzy()
    test_route()
    test_sticky()
    test_voices()
    test_unconfigured()
    await test_tts_payload()
    await test_bot_header()
    await test_pipeline()
    for f in glob.glob(f"/tmp/tts-debug/*_{TAG}*.wav"):
        os.remove(f)
    total = len(CHECKS)
    print(f"{total - len(FAILS)}/{total} passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
