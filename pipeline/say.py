"""
"Say" back door (2026-10-03, Ruby, for filming): make the pod speak a given
text in an agent's voice, without a voice turn. POST /say on its own small
HTTP server (main.py say_server), off unless the config enables it:

    say:
      enabled: true
      host: "127.0.0.1"     # 0.0.0.0 = reachable from the LAN (set a token)
      port: 3103
      token: ""             # optional; then send "Authorization: Bearer <token>"
      log: "/tmp/tts-debug/say.log"
      tell_agent: true      # pass the text to the agent's next turn

Request (JSON):
    {"text": "...",            required (the answer)
     "voice": "robin",         agent name/alias, or a TTS voice ("tim-allen")
     "raw_voice": false,       optional: "voice" is a TTS voice name, never an agent
     "bot": "robin",           optional: whose history gets the text
     "device": "onju-1415E8",  optional: hostname or IP (default: last pod seen)
     "listen": false,          optional: open the mic after the final answer
     "wait": true,             optional: answer after it was sent (false = at once)
     "filler": "...",          optional: short line before THINKING (same voice)
     "think_s": 3,             optional: seconds of THINKING LEDs after filler
     "delay_s": 5}             optional: wait before starting (re-check busy then)

Staged sequence when filler and/or think_s is set: optional delay → filler →
THINKING for think_s → answer. Plain announces (text only) are unchanged.
listen applies only after the final answer; the mic stays closed between parts.

Responses: 200 {"status": "played"|"queued"|"interrupted", "duration_s", ...}
("played" = all audio sent; the pod is still playing the last ~duration_s);
502 when the TTS or the pod connection failed;
409 {"error": "busy", "reason": ...} when the pod is in a turn, playing,
recording or listening (nothing is touched); 404 when no such pod is known
(or the endpoint is off: then nothing listens on the port at all);
400 bad request; 401 bad token.

This module keeps only the parts that don't need audio or the network
(config, voice/bot resolution, the busy check, the log, the agent memory),
so the offline tests can import it; main.py does the HTTP and the playback
through the same stream_sentence() a voice turn uses.
"""
import json
import logging
import os
import time
from datetime import datetime

log = logging.getLogger(__name__)

DEFAULT_PORT = 3103
DEFAULT_HOST = "127.0.0.1"
DEFAULT_LOG = "/tmp/tts-debug/say.log"
DEFAULT_MAX_CHARS = 1500
# The pod streams mic packets all the time its mic is open (the server runs
# the VAD), so packets within this window mean it is listening.
LISTEN_GRACE_S = 1.5
# Says waiting for the agent's next turn are dropped after this long
# (grok-bridge forgets a voice history after 1800 s idle, GROK_HISTORY_IDLE).
MEMORY_TTL_S = 1800.0
# Caps for the optional staged fields (filming). Firmware THINKING timeout
# is 200 s; 30 s is plenty for a staged pause and keeps a stuck request
# from holding the pod for minutes.
MAX_THINK_S = 30.0
MAX_DELAY_S = 120.0


class SayError(Exception):
    """A request the endpoint refuses: HTTP status + JSON body."""

    def __init__(self, status: int, body: dict):
        super().__init__(body.get("error", "error"))
        self.status = status
        self.body = body


class SayConfig:
    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.host = str(cfg.get("host") or DEFAULT_HOST)
        self.port = int(cfg.get("port") or DEFAULT_PORT)
        self.token = str(cfg.get("token") or "")
        self.log_path = str(cfg.get("log") or DEFAULT_LOG)
        self.tell_agent = bool(cfg.get("tell_agent", True))
        self.max_chars = int(cfg.get("max_chars") or DEFAULT_MAX_CHARS)

    @classmethod
    def from_config(cls, config: dict) -> "SayConfig":
        return cls(config.get("say"))


# ---- request -----------------------------------------------------------

def check_token(scfg: SayConfig, headers: dict) -> None:
    if not scfg.token:
        return
    auth = headers.get("authorization", "")
    given = auth[7:].strip() if auth.lower().startswith("bearer ") else headers.get("x-say-token", "")
    if given != scfg.token:
        raise SayError(401, {"error": "unauthorized"})


def _opt_seconds(req: dict, key: str, cap: float) -> float:
    """Optional non-negative seconds field, capped at `cap`. Missing/None/"" → 0."""
    if key not in req or req[key] is None or req[key] == "":
        return 0.0
    v = req[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SayError(400, {"error": f"\"{key}\" must be a number"})
    v = float(v)
    if v < 0:
        raise SayError(400, {"error": f"\"{key}\" must be >= 0"})
    return min(v, cap)


def _opt_text(scfg: SayConfig, req: dict, key: str) -> str | None:
    """Optional text field (same cleaning/length rules as \"text\"). Missing/blank → None."""
    if key not in req or req[key] is None:
        return None
    v = req[key]
    if not isinstance(v, str):
        raise SayError(400, {"error": f"\"{key}\" must be a string"})
    v = " ".join(v.split())
    if not v:
        return None
    if len(v) > scfg.max_chars:
        raise SayError(400, {"error": f"{key} longer than {scfg.max_chars} characters"})
    return v


def parse_request(scfg: SayConfig, body: bytes) -> dict:
    try:
        req = json.loads(body or b"{}")
    except ValueError:
        raise SayError(400, {"error": "body is not JSON"})
    if not isinstance(req, dict):
        raise SayError(400, {"error": "body must be a JSON object"})
    text = req.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SayError(400, {"error": "missing \"text\""})
    text = " ".join(text.split())
    if len(text) > scfg.max_chars:
        raise SayError(400, {"error": f"text longer than {scfg.max_chars} characters"})
    for key in ("voice", "bot", "device"):
        if req.get(key) is not None and not isinstance(req.get(key), str):
            raise SayError(400, {"error": f"\"{key}\" must be a string"})
    return {
        "text": text,
        "voice": (req.get("voice") or "").strip(),
        "bot": (req.get("bot") or "").strip(),
        "device": (req.get("device") or "").strip(),
        "listen": bool(req.get("listen", False)),
        "wait": bool(req.get("wait", True)),
        "filler": _opt_text(scfg, req, "filler"),
        "think_s": _opt_seconds(req, "think_s", MAX_THINK_S),
        "delay_s": _opt_seconds(req, "delay_s", MAX_DELAY_S),
        "raw_voice": bool(req.get("raw_voice", False)),
    }


def resolve_voice(router, voice: str, bot: str,
                  raw: bool = False) -> tuple[str | None, str | None]:
    """(TTS voice or None = configured default, bot or None) for a request.

    With named agents (router): `voice` may be an agent name or alias (its
    voice is used) or a TTS voice name (the agent with that voice, if any,
    is the bot). `bot`, if given, must be an agent; it wins for the history,
    and gives the voice when `voice` is empty. Without agents, `voice` is
    passed to the TTS as is and `bot` is only recorded.

    `raw` (request "raw_voice": true): `voice` is always a TTS voice name and
    is never looked up as an agent name/alias. Needed when a voice and an
    agent share a name: voice "data" (the old Data clip) vs agent "data"
    (jack-nicholson). The bot is still `bot`, else the agent whose configured
    voice is exactly that name, else none (played and logged, no agent told)."""
    if router is None:
        return (voice or None), (bot.lower() or None)
    agent_bot = None
    if bot:
        key = bot.lower()
        if key not in router.agents:
            hit = _agent_by_name(router, key)
            if hit is None:
                raise SayError(400, {"error": f"unknown bot {bot!r}",
                                     "bots": sorted(router.agents)})
            key = hit
        agent_bot = key
    tts_voice = None
    voice_bot = None
    if voice:
        hit = None if raw else _agent_by_name(router, voice.lower())
        if hit is not None:
            voice_bot = hit
            tts_voice = router.voice_for(hit)
        else:
            tts_voice = voice
            voice_bot = next((a.name for a in router.agents.values()
                              if a.voice and a.voice.lower() == voice.lower()), None)
    elif agent_bot:
        tts_voice = router.voice_for(agent_bot)
    else:
        voice_bot = router.default
        tts_voice = router.voice_for(router.default)
    return tts_voice, (agent_bot or voice_bot)


def _agent_by_name(router, word: str) -> str | None:
    """Agent whose name or alias is exactly `word` (case-insensitive).
    No fuzzy matching: a typo should fail loudly, not pick someone."""
    for a in router.agents.values():
        if word == a.name.lower() or word in (x.lower() for x in a.aliases):
            return a.name
    return None


def find_device(manager, name: str):
    if manager is None or not getattr(manager, "devices", None):
        return None
    if not name:
        return manager.get_most_recent()
    if name in manager.devices:
        return manager.devices[name]
    for d in manager.devices.values():
        if d.ip == name or d.hostname.lower() == name.lower():
            return d
    return None


# ---- busy check ----------------------------------------------------------

def busy_reason(device, barge, queued: int = 0, now: float | None = None,
                *, ignore_owned: bool = False) -> str | None:
    """Why the pod can't take a say right now, or None.

    - a say is already running on it;
    - a voice turn is in progress (ASR, agent, THINKING, reply) or queued;
    - the server estimates it is still playing audio (bargein.py);
    - it is recording (VAD has speech, or a PTT buffer is filling);
    - its mic is open (mic packets within LISTEN_GRACE_S): after a reply the
      pod listens for default_mic_timeout s, after a tap for 20 s.

    ignore_owned: skip the say_active / processing checks. Used after a
    staged delay_s, when this say already claimed the pod and is about to
    start the filler/answer — still refuse if something else (mic, queue,
    recording, leftover playback estimate) got in the way."""
    now = time.time() if now is None else now
    if not ignore_owned:
        if getattr(device, "say_active", False):
            return "a say is already playing"
        if getattr(device, "processing", False):
            return "a voice turn is in progress"
    if queued:
        return "a voice turn is queued"
    if barge is not None and barge.is_playing():
        return "the pod is playing audio"
    vad = getattr(device, "vad", None)
    if (vad is not None and getattr(vad, "recording", False)) or getattr(device, "ptt_buffer", None):
        return "the pod is recording speech"
    last_mic = getattr(device, "last_mic_at", 0.0) or 0.0
    if now - last_mic < LISTEN_GRACE_S:
        return f"the pod is listening (mic open, last packet {now - last_mic:.1f}s ago)"
    return None


# ---- record ----------------------------------------------------------------

def append_log(path: str, entry: dict) -> None:
    """One JSON object per line (time, device, bot, voice, status,
    duration_s, text; optional filler=true). Never raises."""
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        line = dict(entry)
        # ISO with the UTC offset: the container's clock is UTC, the agents
        # (and Gilles) read Pacific time; the offset keeps it unambiguous.
        line = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), **line}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning(f"SAY  could not write {path}: {e}")


class SayMemory:
    """Says waiting for the next voice turn of the bot that 'said' them.

    The agents' history lives in grok-bridge (per pod and bot, from the
    turns it relays); a say never goes through it. So the next turn routed
    to that bot on that pod carries the says as a note in extra_context,
    next to the stall note. grok-bridge keeps that note as part of the
    user's words, so it reaches the bot and stays in its history."""

    def __init__(self, ttl_s: float = MEMORY_TTL_S):
        self.ttl_s = ttl_s
        # (when, text, is_filler)
        self._said: dict[tuple[str, str], list[tuple[float, str, bool]]] = {}

    def add(self, hostname: str, bot: str | None, text: str,
            now: float | None = None, *, filler: bool = False) -> None:
        now = time.monotonic() if now is None else now
        self._said.setdefault((hostname, bot or ""), []).append((now, text, bool(filler)))

    def take(self, hostname: str, bot: str | None, now: float | None = None) -> str | None:
        """The note for this turn (and forget the says), or None."""
        now = time.monotonic() if now is None else now
        items = self._said.pop((hostname, bot or ""), [])
        kept = [(t, filler) for at, t, filler in items if now - at <= self.ttl_s]
        if not kept:
            return None
        parts = []
        for t, filler in kept:
            if filler:
                parts.append(f'(a short filler: "{t}")')
            else:
                parts.append(f'"{t}"')
        quoted = " ".join(parts)
        return (f"(Note: before this, the user had the speaker say this aloud in your voice, "
                f"as you: {quoted} Treat it as something you said.)")


MEMORY = SayMemory()
