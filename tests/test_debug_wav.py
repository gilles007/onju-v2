"""
Unit test for pipeline/debug_wav.py: the names of the debug WAVs in
/tmp/tts-debug (time prefix, '__' for stall lines, safe preview, no
overwrite in the same second, voice and time zone choice).

Run from the repo root:  python3 tests/test_debug_wav.py
"""
import datetime
import fnmatch
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import debug_wav as dw

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS  {name}")
    else:
        failed += 1
        print(f"FAIL  {name}  {detail}")


T = datetime.datetime(2026, 10, 8, 14, 9, 12)


def main():
    n = dw.wav_name("You are most welcome.", T, voice="pepper")
    check("normal sentence", n == "20261008-140912_pepper_You_are_most_welcome.wav", n)
    n = dw.wav_name('"Let me check."', T, voice="pepper", stall=True)
    check("stall: quotes dropped, two underscores after the voice",
          n == "20261008-140912_pepper__Let_me_check.wav", n)
    n = dw.wav_name("Bonjour, I am Gilles, Today I will spend eleven hours.", T, voice="gilles")
    check("30-char preview", n == "20261008-140912_gilles_Bonjour,_I_am_Gilles,_Today_I.wav", n)
    n = dw.wav_name("\u201cOne sec, I\u2019ll look.\u201d", T, voice="robin", stall=True)
    check("curly quotes and apostrophe dropped", n == "20261008-140912_robin__One_sec,_Ill_look.wav", n)
    n = dw.wav_name("a/b\\c: *?<>|$x  y", T, voice="../we ird/")
    check("no slashes or shell characters (preview and voice)",
          n == "20261008-140912_we_ird_abc_x_y.wav", n)
    n = dw.wav_name("hi", T, voice="")
    check("empty voice: default", n == "20261008-140912_default_hi.wav", n)
    n = dw.wav_name("  hi", T, voice="pepper")
    check("leading space does not look like a stall", n == "20261008-140912_pepper_hi.wav", n)
    n = dw.wav_name('"?!" ...', T, voice="pepper")
    check("nothing left: placeholder", n == "20261008-140912_pepper_empty.wav", n)
    for stall in (False, True):
        n = dw.wav_name("Zzpf one sec.", T, voice="pepper", stall=stall)
        check(f"tests' cleanup glob *_Zzpf*.wav matches (stall={stall})",
              fnmatch.fnmatch(n, "*_Zzpf*.wav"), n)

    tts = {"backend": "local", "local": {"voice": "pepper"},
           "local_stream": {"voice": "stream-v"},
           "elevenlabs": {"default_voice": "Emma"}}
    check("voice: override wins", dw.tts_voice({"tts": tts}, "Emma", "data") == "data")
    check("voice: local -> tts.local.voice", dw.tts_voice({"tts": tts}, "Emma") == "pepper")
    check("voice: local without voice -> af_heart (as services/tts.py)",
          dw.tts_voice({"tts": {"backend": "local", "local": {}}}, "Emma") == "af_heart")
    check("voice: local_stream -> tts.local_stream.voice",
          dw.tts_voice({"tts": dict(tts, backend="local_stream")}, "Emma") == "stream-v")
    check("voice: local_stream without voice -> device voice",
          dw.tts_voice({"tts": {"backend": "local_stream", "local_stream": {}}}, "Emma") == "Emma")
    check("voice: elevenlabs -> device voice",
          dw.tts_voice({"tts": dict(tts, backend="elevenlabs")}, "Emma") == "Emma")
    check("voice: no tts config, no voice -> af_heart, no exception",
          dw.tts_voice({}, None) == "af_heart")

    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        name = dw.wav_name("Same second.", T, voice="pepper")
        p1 = dw.unique_path(d, name)
        p1.touch()
        p2 = dw.unique_path(d, name)
        p2.touch()
        p3 = dw.unique_path(d, name)
        check("first name unchanged", p1.name == "20261008-140912_pepper_Same_second.wav", p1.name)
        check("same second: -2 then -3", (p2.name, p3.name) ==
              ("20261008-140912_pepper_Same_second-2.wav", "20261008-140912_pepper_Same_second-3.wav"),
              (p2.name, p3.name))

    old_tz = os.environ.pop("TZ", None)
    try:
        now = dw.local_now({})
        check("default zone is America/Los_Angeles", str(now.tzinfo) == "America/Los_Angeles",
              now.tzinfo)
        now = dw.local_now({"debug": {"tz": "Europe/Paris"}})
        check("debug.tz from the config", str(now.tzinfo) == "Europe/Paris", now.tzinfo)
        dw._warned = False
        now = dw.local_now({"debug": {"tz": "Not/AZone"}})
        check("unknown zone: UTC, no exception", now.utcoffset() == datetime.timedelta(0), now)
        check("unknown zone: warned once", dw._warned)
        os.environ["TZ"] = "UTC"
        now = dw.local_now({"debug": {"tz": "Europe/Paris"}})
        check("TZ set: process local time (naive), ignores debug.tz", now.tzinfo is None, now)
    finally:
        os.environ.pop("TZ", None)
        if old_tz is not None:
            os.environ["TZ"] = old_tz

    print(f"\n{passed}/{passed + failed} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
