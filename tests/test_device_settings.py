"""
Offline test for the 0xEF device setting command (20261011): the
post-playback mic holdoff is sent to a pod when it announces itself.

- protocol.setting_header / send_mic_holdoff: exact 6-byte header
  EF <id> <value hi> <value lo> 00 00 (big-endian, like the 0xAA mic
  timeout), on its own connection; out-of-range values are refused;
  an unreachable pod costs at most the short timeout and raises nothing;
- main.greet_device: nothing is sent when device.mic_holdoff_ms is unset
  (or invalid); when set it is sent once, after the LED pulse / greeting,
  values above 2000 are clamped; a slow or hanging send never holds up
  greet_device; it is chained behind a pending 0xEE state send to that pod.

Reuses the module stubs of tests/test_pause_flush.py (no network except a
local socket, no audio hardware). Run from the repo root:
python3 tests/test_device_settings.py
"""
import asyncio
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pause_flush as pf  # noqa: E402  (installs the module stubs)

m = pf.m
check = pf.check

_spec = importlib.util.spec_from_file_location(
    "real_protocol", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "pipeline", "protocol.py"))
proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto)


def make_config(holdoff="unset"):
    cfg = pf.make_config()
    cfg["device"]["greeting"] = False
    if holdoff != "unset":
        cfg["device"]["mic_holdoff_ms"] = holdoff
    return cfg


class Log:
    def __init__(self, holdoff_delay=0.0, holdoff_hang=False):
        self.events = []
        self.holdoff_delay = holdoff_delay
        self.holdoff_hang = holdoff_hang

    async def send_led_blink(self, ip, port, **k):
        self.events.append(("blink", ip, port))

    async def send_state(self, ip, port, state, timeout=0.2):
        await asyncio.sleep(0.1)
        self.events.append(("state", state))

    async def send_mic_holdoff(self, ip, port, ms, timeout=0.2):
        if self.holdoff_hang:
            await asyncio.Event().wait()
        if self.holdoff_delay:
            await asyncio.sleep(self.holdoff_delay)
        self.events.append(("holdoff", ip, port, ms, timeout))


def install(log):
    m.send_led_blink = log.send_led_blink
    m.send_state = log.send_state
    m.send_mic_holdoff = log.send_mic_holdoff


async def settle():
    for _ in range(100):
        if not m._state_tasks:
            return
        await asyncio.sleep(0.02)


async def greet(holdoff="unset", log=None):
    log = log or Log()
    install(log)
    dev = pf.make_device(None)
    await m.greet_device(dev, make_config(holdoff))
    await settle()
    return log


async def main():
    # 1. Header bytes.
    check("header: 256 ms -> EF 01 01 00 00 00",
          proto.setting_header(proto.SETTING_MIC_HOLDOFF, 256) == bytes([0xEF, 1, 0x01, 0x00, 0, 0]))
    check("header: 512 ms -> EF 01 02 00 00 00",
          proto.setting_header(1, 512) == bytes([0xEF, 1, 0x02, 0x00, 0, 0]))
    check("header: 64 ms -> EF 01 00 40 00 00",
          proto.setting_header(1, 64) == bytes([0xEF, 1, 0x00, 0x40, 0, 0]))
    check("header: 2000 ms -> EF 01 07 D0 00 00 (big-endian, like 0xAA mic timeout)",
          proto.setting_header(1, 2000) == bytes([0xEF, 1, 0x07, 0xD0, 0, 0]))
    check("header: 0 ms -> EF 01 00 00 00 00",
          proto.setting_header(1, 0) == bytes([0xEF, 1, 0, 0, 0, 0]))
    check("header: always 6 bytes (header-only command)", len(proto.setting_header(7, 65535)) == 6)
    for bad in ((1, -1), (1, 65536), (256, 1), (-1, 1)):
        try:
            proto.setting_header(*bad)
            ok = False
        except ValueError:
            ok = True
        check(f"header: out of range {bad} refused", ok)
    check("setting id 1 = mic holdoff, max 2000 ms",
          proto.SETTING_MIC_HOLDOFF == 1 and proto.MIC_HOLDOFF_MAX_MS == 2000)

    # 2. Real send over a local socket.
    got = []

    async def handle(reader, writer):
        got.append(await reader.read())
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    await proto.send_mic_holdoff("127.0.0.1", port, 512)
    await asyncio.sleep(0.05)
    server.close()
    await server.wait_closed()
    check("real send_mic_holdoff: EF 01 02 00 00 00 alone on its own connection",
          got == [bytes([0xEF, 1, 2, 0, 0, 0])], f"{got}")
    t = time.monotonic()
    await proto.send_mic_holdoff("127.0.0.1", port, 256)       # nobody listening now
    await proto.send_mic_holdoff("10.255.255.1", 3001, 256)    # unroutable: connect timeout
    check("real send_mic_holdoff: unreachable pod returns within the 0.2 s timeout, no exception",
          time.monotonic() - t < 0.5, f"{time.monotonic() - t:.2f}s")

    # 3. Not sent when unset / invalid.
    log = await greet()
    check("unset: LED pulse sent, no 0xEF", [e[0] for e in log.events] == ["blink"], f"{log.events}")
    check("unset: mic_holdoff_ms() is None", m.mic_holdoff_ms(make_config()) is None)
    check("mic_holdoff_ms: null -> None", m.mic_holdoff_ms(make_config(None)) is None)
    for bad in ("256", -5, 2.5, True):
        log = await greet(bad)
        check(f"invalid {bad!r}: not sent", [e[0] for e in log.events] == ["blink"], f"{log.events}")
    check("no device section: None", m.mic_holdoff_ms({}) is None)

    # 4. Sent when set, after the greeting.
    log = await greet(512)
    check("512: sent once after the LED pulse, own short timeout",
          log.events == [("blink", "127.0.0.1", 3001), ("holdoff", "127.0.0.1", 3001, 512, m.STATE_TIMEOUT_S)],
          f"{log.events}")
    log = await greet(256)
    check("256: sent", log.events[-1][:4] == ("holdoff", "127.0.0.1", 3001, 256), f"{log.events}")
    log = await greet(0)
    check("0: sent (holdoff off)", log.events[-1][:4] == ("holdoff", "127.0.0.1", 3001, 0), f"{log.events}")
    log = await greet(5000)
    check("5000: clamped to 2000", log.events[-1][:4] == ("holdoff", "127.0.0.1", 3001, 2000), f"{log.events}")

    # 5. Never holds up greet_device.
    log = Log(holdoff_hang=True)
    install(log)
    t = time.monotonic()
    await m.greet_device(pf.make_device(None), make_config(512))
    dt = time.monotonic() - t
    check("hanging send: greet_device returns at once", dt < 0.05, f"{dt:.3f}s")
    t = time.monotonic()
    await settle()
    check("hanging send: background task timed out and cleaned up (< 1 s)",
          not m._state_tasks and time.monotonic() - t < 1.0, f"{time.monotonic() - t:.2f}s")

    async def boom(*a, **k):
        raise OSError("boom")
    m.send_mic_holdoff = boom
    m.send_led_blink = Log().send_led_blink
    await m.greet_device(pf.make_device(None), make_config(512))
    await settle()
    check("failing send: swallowed, no task left", not m._state_tasks)

    # 6. Chained behind a pending state send to the same pod.
    log = Log()
    install(log)
    m.fire_state("127.0.0.1", 3001, 2)
    await m.greet_device(pf.make_device(None), make_config(256))
    await settle()
    check("ordered after a pending 0xEE state",
          [e[0] for e in log.events] == ["blink", "state", "holdoff"], f"{log.events}")

    total = len(pf.CHECKS)
    print(f"{total - len(pf.FAILS)}/{total} passed")
    return 1 if pf.FAILS else 0


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
