"""Offline mute telemetry / away queue tests. Run: python tests/test_away.py."""
import asyncio
import base64
import io
import json
import logging
import sys
import time
import wave
from unittest.mock import patch

import test_say as ts
from pipeline.away import mute_state, receive_mute

m, say = ts.m, ts.say
check = ts.check


def test_status():
    manager = ts._real_device_manager()({"tts": {}})
    device = manager.create_device("pod", "192.0.2.1", ptt=True)
    device.voice = "kept-voice"
    conv = device.conversation
    check("status initially unknown", mute_state(device) is None)
    with patch("pipeline.away.log.info") as logged:
        check("MUTE packet consumed", receive_mute(manager, ["pod", "MUTE", "1"], device.ip))
        since = device.away_since
        receive_mute(manager, ["pod", "MUTE", "1"], device.ip)
        check("only change logged", logged.call_count == 1)
        check("away timestamp preserved", since == device.away_since and since > 0)
        receive_mute(manager, ["pod", "MUTE", "0"], "192.0.2.2")
        check("unmute logs once", logged.call_count == 2)
    check("existing identity preserved", manager.devices["pod"] is device and device.ptt
          and device.voice == "kept-voice" and device.conversation is conv)
    check("status updates IP and last heard", device.ip == "192.0.2.2" and device.last_heard_at > 0)
    check("unmute resets since", mute_state(device) is False and device.away_since is None)
    receive_mute(manager, ["pod", "MUTE", "1"], device.ip)
    check("devices exposes away", device.to_dict()["away"] is True
          and device.to_dict()["away_since"] is not None)
    check("35-second boundary", mute_state(device, device.mute_received_at + 35) is True)
    check("expired becomes unknown", mute_state(device, device.mute_received_at + 35.01) is None)
    device.mute_received_at -= 36
    check("devices exposes stale as unknown", device.to_dict()["away"] is None)
    for parts in (["bad", "MUTE"], ["bad", "MUTE", "2"], ["bad", "MUTE", "0", "PTT"]):
        check("invalid status consumed", receive_mute(manager, parts, "192.0.2.4"))
    check("invalid packet never registers", "bad" not in manager.devices)
    check("presence not consumed", not receive_mute(manager, ["other", "abc123", "PTT"], "192.0.2.3"))
    receive_mute(manager, ["new", "MUTE", "0"], "192.0.2.3")
    check("unknown pod registered from status", manager.devices["new"].ip == "192.0.2.3"
          and manager.get_most_recent() is manager.devices["new"])


async def test_listener():
    manager = ts._real_device_manager()({"tts": {}})
    packets = iter([(b"pod MUTE 1", ("192.0.2.1", 1)),
                    (b"bad MUTE 2", ("192.0.2.1", 1)),
                    (b"pod abc123 PTT", ("192.0.2.1", 1))])
    class Done(Exception):
        pass
    class Loop:
        async def sock_recvfrom(self, *args):
            try:
                return next(packets)
            except StopIteration:
                raise Done
    class Sock:
        def setsockopt(self, *args): pass
        def bind(self, *args): pass
        def setblocking(self, *args): pass
        def close(self): pass
    greeted = []
    async def greet(device, config):
        greeted.append(device.hostname)
    cfg = {"network": {"multicast_group": "239.0.0.1", "multicast_port": 12345}}
    with patch.object(m.socket, "socket", return_value=Sock()), \
         patch.object(m.asyncio, "get_event_loop", return_value=Loop()), \
         patch.object(m, "greet_device", greet):
        try:
            await m.multicast_listener(cfg, manager)
        except Done:
            pass
    check("listener greets presence only", greeted == ["pod"] and "bad" not in manager.devices)
    check("presence keeps mute state", mute_state(manager.devices["pod"]) is True)
    check("status before presence preserves PTT discovery", manager.devices["pod"].ptt
          and manager.devices["pod"].vad is None)


def set_away(device, value):
    device.away = value
    device.away_since = time.time() if value else None
    device.mute_received_at = time.time()


def tiny_wav():
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 1600)
    return base64.b64encode(out.getvalue()).decode()


async def test_queue():
    # Worker is tested deterministically via flush_away_queue; Env lifecycle
    # also runs it, but these operations finish before its next 1 s tick.
    d = ts.make_device()
    set_away(d, True)
    async with ts.Env(devices=[d], token="secret") as e:
        scfg = say.SayConfig.from_config(e.config)
        headers = {"Authorization": "Bearer secret"}
        status, r = await ts.http(e.port, "POST", "/say",
                                  {"text": "First.", "bot": "Data", "voice": "data"}, headers)
        check("away HTTP 202 with chat context", status == 202 and r["status"] == "queued_away"
              and r["text"] == "First." and r["bot"] == "data" and r["device"] == d.hostname)
        check("away no audio or TTS", not e.pod.conns and not e.tts.calls)
        check("away say.log", e.log_lines()[0]["status"] == "queued_away"
              and e.log_lines()[0]["text"] == "First.")
        await ts.http(e.port, "POST", "/say", {"text": "Second."}, headers)
        await ts.http(e.port, "POST", "/say", {"wav": tiny_wav(), "text": "WAV."}, headers)
        check("text and WAV FIFO retained", len(d.say_queue) == 3)
        status, r = await ts.http(e.port, "GET", "/say", headers=headers)
        check("status exposes away queue", r["devices"][d.hostname]["away"] is True
              and r["devices"][d.hostname]["queue_depth"] == 3)
        await m.flush_away_queue(e.config, scfg, None, e.manager, e.queue)
        check("muted queue held", len(d.say_queue) == 3 and not e.tts.calls)
        d.mute_received_at -= 36
        await m.flush_away_queue(e.config, scfg, None, e.manager, e.queue)
        check("unknown queue held", len(d.say_queue) == 3)
        set_away(d, False)
        d.processing = True
        await m.flush_away_queue(e.config, scfg, None, e.manager, e.queue)
        check("unmuted busy retains head", len(d.say_queue) == 3)
        d.processing = False
        # Fake playback keeps the pod busy for its estimated duration. Clear
        # only the fake estimate between flushes, never production busy checks.
        for _ in range(3):
            m.bargein._by_host.clear()
            await m.flush_away_queue(e.config, scfg, None, e.manager, e.queue)
        check("unmute FIFO text order", [s for s, _ in e.tts.calls] == ["First.", "Second."])
        check("WAV flush skips TTS", len(d.say_queue) == 0
              and [r["text"] for r in e.log_lines() if r["status"] == "played"]
              == ["First.", "Second.", "WAV."])
        check("claims released", not d.processing and not d.say_active)
        # Force bypasses away but not busy.
        set_away(d, True)
        m.bargein._by_host.clear()
        status, r = await ts.http(e.port, "POST", "/say", {"text": "Forced.", "force": True}, headers)
        check("force plays while away", status == 200 and r["status"] == "played")
        d.processing = True
        status, _ = await ts.http(e.port, "POST", "/say", {"text": "Busy.", "force": True}, headers)
        check("force respects busy", status == 409)
        d.processing = False
        status, _ = await ts.http(e.port, "POST", "/say", {"text": "No.", "force": "false"}, headers)
        check("force strictly boolean", status == 400)
        status, _ = await ts.http(e.port, "POST", "/say", {"text": "No."})
        check("away still authenticates", status == 401)
        status, _ = await ts.http(e.port, "POST", "/say", {"wav": "bad"}, headers)
        check("away validates WAV before queue", status == 400)
        # Stale mute does not block a fresh announce, even with an old backlog.
        await ts.http(e.port, "POST", "/say", {"text": "Held."}, headers)
        d.mute_received_at -= 36
        m.bargein._by_host.clear()
        status, r = await ts.http(e.port, "POST", "/say", {"text": "Unknown."}, headers)
        check("stale allows fresh announce", status == 200 and r["status"] == "played"
              and len(d.say_queue) == 1)
        set_away(d, True)
        for _ in range(31):
            await ts.http(e.port, "POST", "/say", {"text": "Bounded."}, headers)
        status, _ = await ts.http(e.port, "POST", "/say", {"text": "Full."}, headers)
        check("queue bounded", status == 429 and len(d.say_queue) == 32)


async def test_worker():
    d = ts.make_device()
    set_away(d, True)
    async with ts.Env(devices=[d]) as e:
        await ts.http(e.port, "POST", "/say", {"text": "Worker."})
        set_away(d, False)
        deadline = asyncio.get_running_loop().time() + 3
        while d.say_queue and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)
        check("background worker drains on unmute", not d.say_queue
              and any(x["status"] == "played" for x in e.log_lines()))
    check("worker cancelled with server", not any(
        t.get_coro().__name__ == "away_queue_worker" for t in asyncio.all_tasks()))


async def test_unicast_socket():
    """The real listener socket takes a unicast MUTE packet (pod -> server IP)."""
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("", 0))
    port = probe.getsockname()[1]
    probe.close()
    manager = ts._real_device_manager()({"tts": {}})
    cfg = {"network": {"multicast_group": "239.0.0.1", "multicast_port": port}}
    task = asyncio.create_task(m.multicast_listener(cfg, manager))
    await asyncio.sleep(0.1)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
        tx.sendto(b"pod MUTE 1", ("127.0.0.1", port))
    for _ in range(50):
        if "pod" in manager.devices:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    check("unicast MUTE reaches listener", "pod" in manager.devices
          and mute_state(manager.devices["pod"]) is True)


async def main():
    test_status()
    await test_listener()
    await test_unicast_socket()
    await test_queue()
    await test_worker()
    total = len(ts.pf.CHECKS)
    print(f"{total - len(ts.pf.FAILS)}/{total} passed")
    return bool(ts.pf.FAILS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
