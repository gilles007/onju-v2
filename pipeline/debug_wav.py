"""Names for the debug WAVs stream_sentence() writes to /tmp/tts-debug.

    20261008-140912_pepper_You_are_most_welcome.wav    a sentence
    20261008-140912_pepper__Let_me_check.wav           a stall line (two
                                                       underscores after the voice)
    20261008-140912_pepper_You_are_most_welcome-2.wav  same name in the same second

The voice is the TTS voice asked for that sentence (see tts_voice()).

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
PREVIEW_CHARS = 30
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


def tts_voice(config: dict, voice: str | None = None,
              voice_override: str | None = None) -> str:
    """The voice name pipeline/services/tts.py sends for a sentence: the
    per-call voice_override, else the configured voice of the TTS backend
    (voice = the device's voice, used by elevenlabs and local_stream)."""
    tcfg = config.get("tts") or {}
    backend = tcfg.get("backend", "local")
    if voice_override:
        return voice_override
    if backend == "elevenlabs":
        return voice or ""
    if backend == "local_stream":
        return (tcfg.get("local_stream") or {}).get("voice") or voice or ""
    return (tcfg.get(backend) or tcfg.get("local") or {}).get("voice", "af_heart")


def _clean(text: str, chars: int) -> str:
    """First chars characters without quotes, spaces as underscores, only
    letters, digits and . , - kept (none at either end)."""
    text = re.sub(r"[\"'\u2018\u2019\u201c\u201d`]", "", text or "").strip()
    out = re.sub(r"\s+", "_", text[:chars])
    out = re.sub(r"[^\w.,-]", "", out)
    return re.sub(r"_+", "_", out).strip("_.,-")


def wav_name(sentence: str, when: datetime.datetime, *, voice: str = "",
             stall: bool = False) -> str:
    """YYYYMMDD-HHMMSS_<voice>_<preview>.wav, or ..._<voice>__<preview>.wav
    for a stall line. The preview is the first 30 characters of the
    sentence; voice and preview are cleaned by _clean()."""
    voice = _clean(voice, 40) or "default"
    preview = _clean(sentence, PREVIEW_CHARS) or "empty"
    return f"{when:%Y%m%d-%H%M%S}_{voice}{'__' if stall else '_'}{preview}.wav"


def unique_path(directory: pathlib.Path, name: str) -> pathlib.Path:
    """directory/name, or name-2, name-3 ... if that file already exists."""
    path = directory / name
    n = 2
    while path.exists():
        path = directory / f"{pathlib.Path(name).stem}-{n}.wav"
        n += 1
    return path
