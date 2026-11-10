"""Offline mute telemetry / away queue tests. Run: python tests/test_away.py."""
import asyncio
import base64
import io
import json
import logging
import sys
import time
import types
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
        check("queue bounded, oldest dropped", status == 202 and len(d.say_queue) == 32
              and json.loads(d.say_queue[-1][1])["text"] == "Full."
              and json.loads(d.say_queue[0][1])["text"] == "Bounded.")


async def test_backlog():
    """TTL expiry, max_items, and the unmute collapse (intro + newest, own voices)."""
    d = ts.make_device()
    set_away(d, True)
    away = {"ttl_s": 3600, "max_items": 5, "collapse_over": 3, "keep": 2}
    async with ts.Env(devices=[d], away=away) as e:
        scfg = say.SayConfig.from_config(e.config)
        for i in range(1, 7):
            status, r = await ts.http(e.port, "POST", "/say",
                                      {"text": f"Update {i}.", "voice": f"v{i}"})
        check("full queue keeps newest", status == 202 and r["queue_depth"] == 5
              and json.loads(d.say_queue[0][1])["text"] == "Update 2.")
        d.say_queue[0] = (d.say_queue[0][0] - 3601,) + d.say_queue[0][1:]
        with patch("pipeline.say.log.info") as logged:
            await m.flush_away_queue(e.config, scfg, None, e.manager, e.queue)
        check("expired dropped while muted", len(d.say_queue) == 4 and not e.tts.calls
              and any("expired" in c.args[0] and c.args[1] == 1 for c in logged.call_args_list))
        set_away(d, False)
        for _ in range(4):
            m.bargein._by_host.clear()
            await m.flush_away_queue(e.config, scfg, None, e.manager, e.queue)
        spoken = [(t, v) for t, v in e.tts.calls]
        intro = " ".join(t for t, v in spoken if v == "v5" and not t.startswith("Update"))
        check("collapse intro first", intro == "You have 4 updates. Here are the latest 2; "
              "the rest are in the chat." and not spoken[0][0].startswith("Update"), str(spoken))
        check("only newest played, own voices",
              [x for x in spoken if x[0].startswith("Update")] == [("Update 5.", "v5"), ("Update 6.", "v6")])
        check("backlog drained", not d.say_queue)
        if hasattr(say, "MEMORY"):
            note = say.MEMORY.take(d.hostname, None) or ""
            check("intro not told to the agent", "Update 5." in note and "updates" not in note)
    # Pure helpers: ttl 0 keeps, collapse_over 0 plays all, one-left wording.
    q = types.SimpleNamespace(hostname="p", say_queue=None)
    keep_all = say.SayConfig({"away": {"ttl_s": 0, "collapse_over": 0}})
    for i in range(5):
        say.queue_away(q, json.dumps({"text": str(i)}).encode(), {"text": str(i)}, "", "",
                       keep_all, now=0)
    check("ttl 0 never expires", say.prune_away(q, keep_all, now=10 ** 9) == 0)
    check("collapse off", say.collapse_away(q, keep_all) == 0 and len(q.say_queue) == 5)
    one = say.SayConfig({"away": {"collapse_over": 1, "keep": 1}})
    while len(q.say_queue) > 2:
        q.say_queue.popleft()
    check("collapse one", say.collapse_away(q, one) == 1 and q.say_queue[0][2]
          and "Here is the latest; the other one is" in json.loads(q.say_queue[0][1])["text"])
    check("collapse once", say.collapse_away(q, one) == 0)
    check("defaults", (say.SayConfig({}).away_ttl_s, say.SayConfig({}).away_max,
                       say.SayConfig({}).away_collapse_over, say.SayConfig({}).away_keep)
          == (86400.0, 32, 3, 3))


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


def test_if_present_parse():
    scfg = say.SayConfig({"enabled": True})
    r = say.parse_request(scfg, b'{"text": "hi."}')
    check("parse: if_present defaults to false", r["if_present"] is False, r)
    r = say.parse_request(scfg, b'{"text": "hi.", "if_present": true}')
    check("parse: if_present true", r["if_present"] is True, r)
    try:
        say.parse_request(scfg, b'{"text": "hi.", "if_present": "true"}')
        check("parse: if_present string raises", False)
    except say.SayError as e:
        check("parse: if_present string -> 400", e.status == 400
              and "if_present" in e.body.get("error", ""), e.body)


async def test_if_present():
    """if_present: skip while away; default still queues; force still plays."""
    d = ts.make_device()
    set_away(d, True)
    async with ts.Env(devices=[d]) as e:
        status, r = await ts.http(e.port, "POST", "/say", {"text": "Hold this."})
        check("default still queues while away", status == 202 and r["status"] == "queued_away"
              and len(d.say_queue) == 1)
        status, r = await ts.http(e.port, "POST", "/say",
                                  {"text": "Chat echo.", "if_present": True})
        check("if_present skips while away", status == 200 and r["status"] == "skipped_away"
              and r["text"] == "Chat echo." and r["device"] == d.hostname
              and r["queue_depth"] == 1, r)
        check("if_present does not queue", len(d.say_queue) == 1
              and json.loads(d.say_queue[0][1])["text"] == "Hold this.")
        check("if_present no audio or TTS", not e.pod.conns and not e.tts.calls)
        check("if_present say.log", e.log_lines()[-1]["status"] == "skipped_away"
              and e.log_lines()[-1]["text"] == "Chat echo."
              and e.log_lines()[-1]["device"] == d.hostname)
        status, _ = await ts.http(e.port, "POST", "/say",
                                  {"text": "No.", "if_present": "true"})
        check("if_present strictly boolean", status == 400)
        check("bad if_present does not queue", len(d.say_queue) == 1)
        # Stale/unknown does not skip (same as a fresh announce).
        d.mute_received_at -= 36
        m.bargein._by_host.clear()
        status, r = await ts.http(e.port, "POST", "/say",
                                  {"text": "Stale.", "if_present": True})
        check("stale if_present plays, backlog kept", status == 200 and r["status"] == "played"
              and len(d.say_queue) == 1)
        # Present with a backlog: if_present still queues (it will be said).
        set_away(d, False)
        m.bargein._by_host.clear()
        status, r = await ts.http(e.port, "POST", "/say",
                                  {"text": "After backlog.", "if_present": True})
        check("if_present queues behind backlog when present",
              status == 202 and r["status"] == "queued_away" and len(d.say_queue) == 2
              and json.loads(d.say_queue[-1][1])["text"] == "After backlog.", r)
        # Drain the backlog so the next if_present can play.
        for _ in range(2):
            m.bargein._by_host.clear()
            await m.flush_away_queue(e.config, say.SayConfig.from_config(e.config),
                                     None, e.manager, e.queue)
        check("backlog drained", not d.say_queue)
        m.bargein._by_host.clear()
        status, r = await ts.http(e.port, "POST", "/say",
                                  {"text": "I am here.", "if_present": True})
        check("if_present plays when present", status == 200 and r["status"] == "played", r)
        # force wins over if_present while away.
        set_away(d, True)
        m.bargein._by_host.clear()
        depth = len(getattr(d, "say_queue", ()) or ())
        status, r = await ts.http(e.port, "POST", "/say",
                                  {"text": "Forced present.", "if_present": True, "force": True})
        check("force wins over if_present", status == 200 and r["status"] == "played"
              and len(getattr(d, "say_queue", ()) or ()) == depth)


async def main():
    test_status()
    test_if_present_parse()
    await test_listener()
    await test_unicast_socket()
    await test_queue()
    await test_backlog()
    await test_if_present()
    await test_worker()
    total = len(ts.pf.CHECKS)
    print(f"{total - len(ts.pf.FAILS)}/{total} passed")
    return bool(ts.pf.FAILS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
