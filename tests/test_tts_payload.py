"""
Offline test for the TTS request payloads in pipeline/services/tts.py:

- local backend (_local): without sampling keys in tts.local the payload is
  exactly as before (model, input, voice, response_format; Kokoro/Hermes
  unaffected); temperature, top_k, top_p, do_sample, repetition_penalty and
  seed are sent when set, null counts as unset, unknown keys are not sent,
  ref_audio/ref_text still work;
- local_stream: payload unchanged without top_p/seed; they are added when set
  and do not override its own temperature/top_k/do_sample/repetition_penalty.

No network: httpx and pydub are replaced with fakes. Run from the repo root:
python3 tests/test_tts_payload.py
"""
import asyncio
import importlib.util
import os
import struct
import sys
import types

CHECKS, FAILS = [], []


def check(name, ok, detail=""):
    CHECKS.append(name)
    if not ok:
        FAILS.append(name)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if not ok and detail else ""))


for _name, _attrs in (("httpx", {}), ("pydub", {"AudioSegment": object}),
                      ("audioop", {"ratecv": None})):
    try:
        __import__(_name)
    except ImportError:
        _mod = types.ModuleType(_name)
        _mod.__dict__.update(_attrs)
        sys.modules[_name] = _mod

_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "pipeline", "services", "tts.py")
_spec = importlib.util.spec_from_file_location("tts_under_test", _path)
tts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tts)

POSTED, STREAMED = [], []


def _wav_header(sr=16000):
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + b"fmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


class _Resp:
    def __init__(self, body):
        self.content = body

    def raise_for_status(self):
        pass

    async def aiter_bytes(self):
        yield self.content


class _StreamCtx:
    def __init__(self, resp):
        self.resp = resp

    async def __aenter__(self):
        return self.resp

    async def __aexit__(self, *a):
        return False


class FakeClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json):
        POSTED.append(json)
        return _Resp(b"")

    def stream(self, method, url, json):
        STREAMED.append(json)
        return _StreamCtx(_Resp(_wav_header() + b"\0\0" * 8))


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


tts.httpx = types.SimpleNamespace(AsyncClient=FakeClient)
tts.AudioSegment = FakeSeg


def local_cfg(**extra):
    sec = {"url": "http://q", "model": "qwen3-tts-fast", "voice": "pepper"}
    sec.update(extra)
    return {"audio": {"sample_rate": 16000}, "tts": {"backend": "local", "local": sec}}


async def test_local():
    await tts.synthesize("Hi.", "Emma", local_cfg())
    check("local: no sampling keys -> the payload of before",
          POSTED[-1] == {"model": "qwen3-tts-fast", "input": "Hi.", "voice": "pepper",
                         "response_format": "wav"}, f"{POSTED[-1]}")
    kokoro = {"audio": {"sample_rate": 16000},
              "tts": {"backend": "local", "local": {"url": "http://k", "model": "kokoro",
                                                    "voice": "af_heart", "ref_audio": "",
                                                    "ref_text": ""}}}
    await tts.synthesize("Hi.", "Emma", kokoro)
    check("local: Kokoro config (empty ref_audio/ref_text) unchanged",
          POSTED[-1] == {"model": "kokoro", "input": "Hi.", "voice": "af_heart",
                         "response_format": "wav"}, f"{POSTED[-1]}")
    greedy = dict(temperature=0.1, top_k=1, top_p=0.95, do_sample=False,
                  repetition_penalty=1.2, seed=7)
    await tts.synthesize("Hi.", "Emma", local_cfg(**greedy))
    p = POSTED[-1]
    check("local: all six sampling keys sent with their values",
          all(p.get(k) == v for k, v in greedy.items()), f"{p}")
    check("local: do_sample false is sent (not dropped as falsy)",
          p["do_sample"] is False, f"{p}")
    await tts.synthesize("Hi.", "Emma", local_cfg(seed=0, top_k=0))
    check("local: seed 0 and top_k 0 are sent", POSTED[-1].get("seed") == 0
          and POSTED[-1].get("top_k") == 0, f"{POSTED[-1]}")
    await tts.synthesize("Hi.", "Emma", local_cfg(temperature=None, seed=None, top_k=3))
    check("local: null counts as unset", "temperature" not in POSTED[-1]
          and "seed" not in POSTED[-1] and POSTED[-1]["top_k"] == 3, f"{POSTED[-1]}")
    await tts.synthesize("Hi.", "Emma", local_cfg(timeout=5, chunk_size=8, stream=True))
    check("local: other config keys are not sent",
          set(POSTED[-1]) == {"model", "input", "voice", "response_format"}, f"{POSTED[-1]}")
    await tts.synthesize("Hi.", "Emma", local_cfg(ref_audio="/r/a.wav", ref_text="hello", seed=1))
    check("local: ref_audio/ref_text still sent next to the sampling keys",
          POSTED[-1]["ref_audio"] == "/r/a.wav" and POSTED[-1]["ref_text"] == "hello"
          and POSTED[-1]["seed"] == 1, f"{POSTED[-1]}")


async def test_local_stream():
    cfg = {"audio": {"sample_rate": 16000},
           "tts": {"backend": "local_stream",
                   "local_stream": {"url": "http://q", "voice": "pepper"}}}
    [c async for c in tts.synthesize_stream("Hi.", "Emma", cfg)]
    base = dict(STREAMED[-1])
    check("local_stream: no top_p/seed in the default payload",
          "top_p" not in base and "seed" not in base and base["temperature"] == 0.1
          and base["top_k"] == 1 and base["do_sample"] is False
          and base["repetition_penalty"] == 1.2, f"{base}")
    cfg["tts"]["local_stream"].update(seed=7, top_p=0.9)
    [c async for c in tts.synthesize_stream("Hi.", "Emma", cfg)]
    p = STREAMED[-1]
    check("local_stream: seed and top_p added when set, nothing else changes",
          {k: v for k, v in p.items() if k not in ("seed", "top_p")} == base
          and p["seed"] == 7 and p["top_p"] == 0.9, f"{p}")
    cfg["tts"]["local_stream"].update(temperature=0.5)
    [c async for c in tts.synthesize_stream("Hi.", "Emma", cfg)]
    check("local_stream: its own keys still read from config", STREAMED[-1]["temperature"] == 0.5)


async def main():
    await test_local()
    await test_local_stream()
    total = len(CHECKS)
    print(f"{total - len(FAILS)}/{total} passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
