"""
Multi-agent routing (Grok Bot, 2026-09-27): after ASR, pick which named agent
a turn is for, and with which TTS voice.

Optional. Without an `agents:` section in the config, router_from_config()
returns None and main.py does exactly what it did before.

    agents:
      default: pepper          # who answers when nobody is named
      sticky_timeout_s: 300    # follow-ups stay with the last agent this long
      strip_address: true      # drop "Hey Ruby," from the text sent upstream
      list:
        - name: pepper
          aliases: [peppa]
          voice: pepper
        - name: ruby           # no voice: uses the default agent's voice
          aliases: [rubi]

Addressing. A name counts when it is the first word of the utterance, after
at most three leading fillers/greetings ("um", "oh", "hey", "okay", ...):
"Hey Ruby, ...", "Ruby, ...", "Okay Data what's ...". Without a greeting the
name must be followed by punctuation ("Data, what time is it") or be the
whole utterance ("Ruby?"), so a sentence that merely starts with a word like
"Data shows ..." is not taken as an address. A greeting followed by a full
stop ("Okay. Data shows ...") does not count as a greeting.

Fuzzy matching, for speech-to-text misspellings. Each word is compared with
every name and alias twice: as spelled, and after a small phonetic
normalisation (_key: final -y/-ie/-ey/-ee -> i, oo -> u, y between
consonants -> i, ph -> f, ck -> k, doubled letters collapsed, trailing 's
dropped), so Rubie/Rooby -> "rubi", Robbin/Robyn -> "robin". The distance is
the smaller of the two edit distances (insert, delete, substitute, swap of
neighbours). Allowed distance by the normalised name's length: up to 3
letters exact only, 4-5 letters 1 edit, 6+ letters 2 edits. The closest
agent wins; ties on distance go to the smaller as-spelled distance; if two
different agents are still tied, nothing is matched (no guessing).
Ruby vs Robin: "rubi" and "robin" are 2 edits apart, more than either
allows, so neither name can match the other. "Rubin" is 1 edit from both
keys and is settled by spelling (1 edit from Robin, 2 from Ruby) -> Robin;
listing it as a Robin alias makes that explicit.

Stickiness is per device: a named agent stays chosen for follow-ups until
another name is spoken or sticky_timeout_s passes without a turn, then the
default agent answers again.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

GREETINGS = frozenset({
    "hey", "hi", "hello", "hiya", "heya", "hay", "hei", "yo", "oi",
    "okay", "ok", "morning", "evening",
})
FILLERS = frozenset({"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "oh", "well", "so"})
MAX_LEAD_WORDS = 3                      # fillers/greetings allowed before the name
DEFAULT_STICKY_S = 300.0

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_ADDRESS_PUNCT = re.compile(r"\s*[,.!?:;\u2014\u2013\-]")
_SEPARATOR = re.compile(r"[\s,.!?:;\u2014\u2013\-\"'\u201c\u201d]*")
_VOWELS = "aeiou"


def _norm(word: str) -> str:
    w = word.lower()
    w = re.sub(r"'s$", "", w)
    return w.replace("'", "")


def _key(word: str) -> str:
    """Rough phonetic key; see the module docstring."""
    w = _norm(word)
    w = w.replace("ph", "f").replace("ck", "k")
    w = re.sub(r"(ie|ey|ee|ea|y)$", "i", w)
    w = w.replace("oo", "u")
    w = re.sub(rf"(?<=[^{_VOWELS}])y(?=[^{_VOWELS}])", "i", w)
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


def _distance(a: str, b: str) -> int:
    """Edit distance with adjacent swaps (optimal string alignment)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (prev2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def allowed_distance(name_key: str) -> int:
    n = len(name_key)
    if n <= 3:
        return 0
    if n <= 5:
        return 1
    return 2


@dataclass
class Agent:
    name: str
    aliases: list[str] = field(default_factory=list)
    voice: str | None = None
    display: str = ""

    def spellings(self) -> list[str]:
        out = []
        for s in [self.name] + list(self.aliases):
            n = _norm(s)
            if n and n not in out:
                out.append(n)
        return out


@dataclass
class Address:
    agent: str                  # canonical agent name
    heard: str                  # the word as ASR wrote it
    distance: int               # 0 = exact name/alias
    rest: str                   # the utterance after the address phrase
    greeted: bool


@dataclass
class Route:
    agent: str
    voice: str | None           # TTS voice for this turn (None = TTS config default)
    text: str                   # text to send upstream (address stripped if configured)
    reason: str                 # "addressed", "sticky", "default", "timeout"
    address: Address | None = None


class AgentRouter:
    def __init__(self, agents: list[Agent], default: str,
                 sticky_timeout_s: float = DEFAULT_STICKY_S, strip_address: bool = True):
        if not agents:
            raise ValueError("agents: the list is empty")
        self.agents = {a.name: a for a in agents}
        if default not in self.agents:
            raise ValueError(f"agents.default {default!r} is not in agents.list")
        self.default = default
        self.sticky_timeout_s = float(sticky_timeout_s)
        self.strip_address = bool(strip_address)
        self._state: dict[str, tuple[str, float]] = {}   # device -> (agent, last turn time)
        self._spellings = [(a.name, s, _key(s)) for a in agents for s in a.spellings()]

    @classmethod
    def from_config(cls, config: dict) -> "AgentRouter | None":
        cfg = config.get("agents")
        if not cfg:
            return None
        entries = cfg.get("list") or []
        agents = []
        for e in entries:
            if isinstance(e, str):
                e = {"name": e}
            name = _norm(str(e.get("name", "")))
            if not name:
                raise ValueError(f"agents.list entry without a name: {e!r}")
            aliases = [str(a) for a in (e.get("aliases") or [])]
            voice = e.get("voice") or None
            display = str(e.get("display_name") or name.capitalize())
            agents.append(Agent(name, aliases, voice, display))
        default = _norm(str(cfg.get("default") or (agents[0].name if agents else "")))
        return cls(agents, default,
                   sticky_timeout_s=cfg.get("sticky_timeout_s", DEFAULT_STICKY_S),
                   strip_address=cfg.get("strip_address", True))

    # -- matching -----------------------------------------------------------

    def match_word(self, word: str) -> tuple[str, int] | None:
        """(agent, distance) for one spoken word, or None (no match or a tie)."""
        w = _norm(word)
        if not w:
            return None
        wk = _key(w)
        best: dict[str, tuple[int, int]] = {}
        for agent, spelled, key in self._spellings:
            raw = _distance(w, spelled)
            d = min(raw, _distance(wk, key))
            if d > allowed_distance(key):
                continue
            score = (d, raw)
            if agent not in best or score < best[agent]:
                best[agent] = score
        if not best:
            return None
        ranked = sorted(best.items(), key=lambda kv: kv[1])
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            log.info(f"AGENT  {word!r} is as close to {ranked[0][0]} as to "
                     f"{ranked[1][0]}; not routing on it")
            return None
        return ranked[0][0], ranked[0][1][0]

    def detect(self, text: str) -> Address | None:
        """The agent addressed at the start of `text`, if any."""
        words = list(_WORD.finditer(text or ""))
        i = 0
        greeted = False
        while i < len(words) and i < MAX_LEAD_WORDS:
            w = _norm(words[i].group())
            if w in GREETINGS:
                greeted = True
            elif w not in FILLERS:
                break
            i += 1
        if i >= len(words):
            return None
        if greeted and i > 0:
            # "Okay. Data shows ..." - a sentence ended after the greeting.
            gap = text[words[i - 1].end():words[i].start()]
            if re.search(r"[.!?]", gap):
                greeted = False
        cand = words[i]
        hit = self.match_word(cand.group())
        if hit is None:
            return None
        agent, dist = hit
        after = text[cand.end():]
        punct_after = bool(_ADDRESS_PUNCT.match(after))
        only_word = i == len(words) - 1
        if not (greeted or punct_after or only_word):
            return None
        rest = after[_SEPARATOR.match(after).end():].strip()
        return Address(agent, cand.group(), dist, rest, greeted)

    # -- routing ------------------------------------------------------------

    def voice_for(self, agent: str) -> str | None:
        a = self.agents.get(agent)
        if a and a.voice:
            return a.voice
        return self.agents[self.default].voice

    def display_name(self, agent: str) -> str:
        a = self.agents.get(agent)
        return a.display if a else agent.capitalize()

    def route(self, device_id: str, text: str, now: float | None = None) -> Route:
        now = time.monotonic() if now is None else now
        address = self.detect(text)
        prev = self._state.get(device_id)
        if address is not None:
            agent, reason = address.agent, "addressed"
        elif prev is not None and now - prev[1] <= self.sticky_timeout_s:
            agent, reason = prev[0], "sticky"
        else:
            agent = self.default
            reason = "timeout" if prev is not None else "default"
        self._state[device_id] = (agent, now)

        out = text
        if address is not None and self.strip_address:
            # Only the address goes: "Hey Rubin, what's up?" -> "what's up?".
            # A bare "Hey Ruby." keeps a short greeting, spelled correctly.
            out = address.rest or f"Hey {self.display_name(agent)}."
            if out[:1].islower():
                out = out[0].upper() + out[1:]
        return Route(agent, self.voice_for(agent), out, reason, address)

    def touch(self, device_id: str, now: float | None = None) -> None:
        """Restart the sticky clock at the end of a turn (long replies)."""
        prev = self._state.get(device_id)
        if prev is not None:
            self._state[device_id] = (prev[0], time.monotonic() if now is None else now)

    def current(self, device_id: str) -> str | None:
        prev = self._state.get(device_id)
        return prev[0] if prev else None

    def describe(self) -> str:
        parts = []
        for a in self.agents.values():
            v = a.voice or f"{self.voice_for(a.name)} (default)"
            parts.append(f"{a.name}{'*' if a.name == self.default else ''} voice={v}")
        return (f"{', '.join(parts)}; sticky {self.sticky_timeout_s:.0f}s; "
                f"strip_address={'on' if self.strip_address else 'off'}")


def router_from_config(config: dict) -> AgentRouter | None:
    return AgentRouter.from_config(config)
