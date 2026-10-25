"""
Host-side test of the rolling playback buffer in onjuino/play_ring.h (the FIFO,
the TCP/Opus frame parser, the playback-start gate, and a simulation of the
pass loop used by opusDecodeTask). Compiles tests/test_play_ring.cpp with g++.

Run from the repo root:  python3 tests/test_play_ring.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        print("SKIP  no C++ compiler found")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "test_play_ring")
        build = subprocess.run(
            [cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-o", exe,
             os.path.join(HERE, "test_play_ring.cpp")],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stdout + build.stderr)
            print("FAIL  test_play_ring.cpp does not compile")
            return 1
        run = subprocess.run([exe], capture_output=True, text=True)
        print(run.stdout, end="")
        return run.returncode


if __name__ == "__main__":
    sys.exit(main())
