"""
Attention gating and the voice LEDs (2026-10-19, Ruby).

One rule, per pod:

1. While the server is listening to the mic (anything it hears can start a
   turn or interrupt one), the pod's LEDs flash with the user's voice.
2. While it is not listening, what the mic picks up has no effect at all:
   no turn, no barge-in, no LED blinks. The server also closes the pod's
   mic for that time (0xDD 0, see main.py), so normally nothing streams.

listening() is the one place that decides which of the two applies.
LedLink keeps the LED connection that carries the voice blinks.

Only the standard library and pipeline.protocol/bargein are imported, so
the offline tests can use this module without audio packages.
"""
import asyncio
import logging
import time

from pipeline import bargein
from pipeline import protocol

log = logging.getLogger(__name__)


def listening(device, config: dict, now: float | None = None) -> tuple[bool, str]:
    """Is the server paying attention to this pod's mic right now?

    Returns (listening, why):
      (True, "idle")         no turn and no say: speech starts a turn.
      (True, "barge-in")     a turn or say is running and barge-in is armed
                             (by default: the reply is estimated to be
                             playing, see pipeline/bargein.py), so speech
                             can interrupt it.
      (True, "interrupted")  speech already interrupted this turn: keep
                             listening, it becomes the next turn.
      (False, reason)        deaf: a turn ("turn in progress") or a say
                             ("say in progress") is running and nothing
                             said now may change anything.
    """
    if not device.processing:
        return True, "idle"
    if device.interrupted.is_set():
        return True, "interrupted"
    if bargein.for_device(device, config).armed(now):
        return True, "barge-in"
    return False, ("say in progress" if getattr(device, "say_active", False)
                   else "turn in progress")


class LedLink:
    """The voice-blink connection to one pod.

    While idle-listening and the VAD is recording, udp_listener keeps one
    persistent TCP connection open and writes a 0xCC blink every
    led_update_period: the voice level, or level 0 as a keepalive when the
    user pauses (the pod drops a connection that sends nothing for 500 ms,
    onjuino.ino, and treats level 0 as a no-op). The connection is opened
    in the background, never awaited inline, so the mic loop is never held
    up; a failed connect is logged at WARNING and retried every RETRY_S
    while the utterance lasts. A connection the pod has closed (transport
    closing or EOF from the pod) is detected before each write and replaced.

    During a turn (barge-in listening) blinks go out as one-shot
    connections instead (pulse()): the pod serves one TCP client at a time,
    and a held connection would delay the reply's next audio connection.
    """

    RETRY_S = 0.5
    WARN_EVERY = 10   # after the first failure, warn again every N failures

    def __init__(self, hostname: str):
        self.hostname = hostname
        self.writer = None
        self._opening: asyncio.Task | None = None
        self._pulse: asyncio.Task | None = None
        self._gen = 0
        self._next_try = 0.0
        self._failures = 0
        self._bg: set[asyncio.Task] = set()

    # ---- state -------------------------------------------------------------

    def alive(self) -> bool:
        w = self.writer
        if w is None:
            return False
        transport = getattr(w, "transport", None)
        if transport is not None and transport.is_closing():
            return False
        reader = getattr(w, "pod_reader", None)
        if reader is not None and reader.at_eof():
            return False
        return True

    @property
    def opening(self) -> bool:
        return self._opening is not None

    # ---- persistent connection (idle listening) ----------------------------

    def ensure(self, ip: str, port: int, now: float | None = None) -> None:
        """Open the connection in the background unless it is open, being
        opened, or the last attempt failed less than RETRY_S ago."""
        if self.alive() or self._opening is not None:
            return
        if self.writer is not None:
            log.info(f"LED  connection to {self.hostname} was closed by the pod; reopening")
            self._drop()
        now = time.monotonic() if now is None else now
        if now < self._next_try:
            return
        self._opening = asyncio.create_task(self._open(ip, port, self._gen))

    async def _open(self, ip: str, port: int, gen: int) -> None:
        writer = None
        try:
            writer = await protocol.open_led_connection(ip, port)
        except asyncio.CancelledError:
            raise
        except Exception as e:   # open_led_connection returns None on errors; belt and braces
            log.debug(f"LED  open to {ip}:{port} raised {e!r}")
        finally:
            if gen == self._gen:
                self._opening = None
        if gen != self._gen:
            # close() ran while we were connecting: this connection is unwanted.
            if writer is not None:
                self._close_bg(writer)
            return
        if writer is None:
            self._failures += 1
            self._next_try = time.monotonic() + self.RETRY_S
            if self._failures == 1 or self._failures % self.WARN_EVERY == 0:
                log.warning(f"LED  cannot connect to {self.hostname} ({ip}:{port}) "
                            f"for voice blinks (attempt {self._failures}); retrying")
            return
        if self._failures:
            log.info(f"LED  connected to {self.hostname} after "
                     f"{self._failures} failed attempt(s)")
        self._failures = 0
        self.writer = writer

    def blink(self, level: int, fade: int) -> bool:
        """Write one blink (level 0 = keepalive). False if not connected."""
        if not self.alive():
            return False
        return protocol.write_led_blink(self.writer, level, fade=fade)

    def close(self) -> None:
        """End of the utterance (or attention lost): close and forget."""
        self._gen += 1
        if self._opening is not None:
            self._opening.cancel()
            self._opening = None
        self._drop()
        if self._failures:
            log.info(f"LED  gave up on {self.hostname} for this utterance after "
                     f"{self._failures} failed connect(s)")
        self._failures = 0
        self._next_try = 0.0

    def _drop(self) -> None:
        w, self.writer = self.writer, None
        if w is not None:
            self._close_bg(w)

    def _close_bg(self, writer) -> None:
        task = asyncio.create_task(protocol.close_led_connection(writer))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # ---- one-shot blinks (barge-in listening, during a turn) ---------------

    def pulse(self, ip: str, port: int, level: int, fade: int) -> None:
        """One blink on its own short connection, in the background. Skipped
        if the previous one hasn't finished (the pod may be busy playing)."""
        if level <= 0 or (self._pulse is not None and not self._pulse.done()):
            return
        self._pulse = asyncio.create_task(
            protocol.send_led_blink(ip, port, intensity=level, fade=fade))
        self._bg.add(self._pulse)
        self._pulse.add_done_callback(self._bg.discard)


_by_host: dict[str, LedLink] = {}


def led_for(device) -> LedLink:
    """The LedLink of this pod, created on first use."""
    link = _by_host.get(device.hostname)
    if link is None:
        link = _by_host[device.hostname] = LedLink(device.hostname)
    return link
