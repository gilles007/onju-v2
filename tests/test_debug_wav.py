"""
Unit test for pipeline/debug_wav.py: the names of the debug WAVs in
/tmp/tts-debug (time prefix, '__' for stall lines, safe preview, no
overwrite in the same second, time zone choice).

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
    n = dw.wav_name("You are most welcome to it.", T)
    check("normal sentence", n == "20261008-140912_You_are_most_welcome.wav", n)
    n = dw.wav_name('"Let me check."', T, stall=True)
    check("stall: quotes dropped, two underscores", n == "20261008-140912__Let_me_check.wav", n)
    n = dw.wav_name("\u201cOne sec, I\u2019ll look.\u201d", T, stall=True)
    check("curly quotes and apostrophe dropped", n == "20261008-140912__One_sec,_Ill_look.wav", n)
    n = dw.wav_name("a/b\\c: *?<>|$x  y", T)
    check("no slashes or shell characters", n == "20261008-140912_abc_x_y.wav", n)
    n = dw.wav_name("  hi", T)
    check("leading space does not look like a stall", n == "20261008-140912_hi.wav", n)
    n = dw.wav_name('"?!" ...', T)
    check("nothing left: placeholder", n == "20261008-140912_empty.wav", n)
    n = dw.wav_name("Bonjour, c'est Gilles. Ça va très bien.", T)
    check("accented letters kept, 20-char preview", n == "20261008-140912_Bonjour,_cest_Gilles.wav", n)
    for stall in (False, True):
        n = dw.wav_name("Zzpf one sec.", T, stall=stall)
        check(f"tests' cleanup glob *_Zzpf*.wav matches (stall={stall})",
              fnmatch.fnmatch(n, "*_Zzpf*.wav"), n)

    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        name = dw.wav_name("Same second.", T)
        p1 = dw.unique_path(d, name)
        p1.touch()
        p2 = dw.unique_path(d, name)
        p2.touch()
        p3 = dw.unique_path(d, name)
        check("first name unchanged", p1.name == "20261008-140912_Same_second.wav", p1.name)
        check("same second: -2 then -3", (p2.name, p3.name) ==
              ("20261008-140912_Same_second-2.wav", "20261008-140912_Same_second-3.wav"),
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
