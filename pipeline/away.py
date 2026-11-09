"""Pod mute/away status: refreshed by UDP status packets, unknown after 35 seconds."""
import logging
import time

log = logging.getLogger(__name__)
MUTE_STALE_S = 35.0


def mute_state(device, now=None):
    """True = away, False = present, None = unknown (old firmware/offline)."""
    now = time.time() if now is None else now
    seen = getattr(device, "mute_received_at", 0.0)
    if not seen or now - seen > MUTE_STALE_S:
        return None
    return device.away


def receive_mute(manager, parts, ip):
    """Consume MUTE packets, including malformed ones; never greet for status.

    A new pod can register before its presence packet. Existing voice/PTT and
    conversation state are preserved. Status is also proof of life for /say.
    """
    if len(parts) < 2 or parts[1] != "MUTE":
        return False
    if len(parts) == 3 and parts[2] in ("0", "1"):
        device = manager.create_device(parts[0], ip, heard=True, status_only=True)
        away = parts[2] == "1"
        log.debug("MUTE  %s %s from %s", parts[0], parts[2], ip)
        if not getattr(device, "mute_received_at", 0.0) or device.away != away:
            log.info("DEVICE %s AWAY %s", device.hostname, "on" if away else "off")
            device.away_since = time.time() if away else None
        device.away = away
        device.mute_received_at = time.time()
    return True
