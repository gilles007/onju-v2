"""Names for the debug WAVs stream_sentence() writes to /tmp/tts-debug.

    20261008-140912_You_are_most_welcome.wav    a sentence
    20261008-140912__Let_me_check.wav           a stall line (two underscores)
    20261008-140912_You_are_most_welcome-2.wav  same name in the same second

The time is local: the TZ environment variable when it is set, else
`debug.tz` from the config, else America/Los_Angeles (the container clock
is UTC). An unknown zone logs one warning and falls back to UTC.
"""
import datetime
import logging
import os
import pathlib
import re

log = logging.getLogger(__name__)

DEFAULT_TZ = "America/Los_Angeles"
PREVIEW_CHARS = 20
_warned = False


def local_now(config: dict) -> datetime.datetime:
    """Current time in the zone used for debug WAV names."""
    global _warned
    if os.environ.get("TZ"):
        return datetime.datetime.now()        # the process's local time (TZ)
    name = (config.get("debug") or {}).get("tz") or DEFAULT_TZ
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo(name))
    except Exception as e:
        if not _warned:
            _warned = True
            log.warning(f"DEBUG  time zone {name!r} unavailable ({e!r}); "
                        f"debug WAV names use UTC")
        return datetime.datetime.now(datetime.timezone.utc)


def wav_name(sentence: str, when: datetime.datetime, *, stall: bool = False) -> str:
    """YYYYMMDD-HHMMSS_<preview>.wav, or YYYYMMDD-HHMMSS__<preview>.wav for a
    stall line. The preview is the first 20 characters without quotes,
    spaces as underscores, and only letters, digits and . , - kept (none
    at either end)."""
    text = re.sub(r"[\"'\u2018\u2019\u201c\u201d`]", "", sentence).strip()
    preview = re.sub(r"\s+", "_", text[:PREVIEW_CHARS])
    preview = re.sub(r"[^\w.,-]", "", preview)
    preview = re.sub(r"_+", "_", preview).strip("_.,-") or "empty"
    return f"{when:%Y%m%d-%H%M%S}{'__' if stall else '_'}{preview}.wav"


def unique_path(directory: pathlib.Path, name: str) -> pathlib.Path:
    """directory/name, or name-2, name-3 ... if that file already exists."""
    path = directory / name
    n = 2
    while path.exists():
        path = directory / f"{pathlib.Path(name).stem}-{n}.wav"
        n += 1
    return path
