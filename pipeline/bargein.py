"""
VOX barge-in (2026-09-26, Ruby): decides when speech picked up by a VOX
pod's mic during a turn is allowed to interrupt that turn.

Before, a single 32 ms mic frame over vad.threshold at any moment of the
turn cancelled it silently, including while nothing was playing and the
server was just waiting for the agent's reply. Now:

1. Sustained speech: the speech probability must stay above vad.threshold
   for vad.interrupt_min_ms (default 300 ms) of audio. The duration is
   counted from the real length of each mic frame (samples / sample rate),
   so it doesn't depend on the pod's chunk size. One short dip below the
   threshold (up to GAP_TOLERANCE_S, i.e. a single 32 ms frame) is
   tolerated, because Silero's probability can drop for a frame between
   syllables; the dip doesn't count toward the duration. A longer dip
   starts the count over.
2. Only while the pod is playing: with vad.interrupt_only_while_playing
   (default true), speech counts only while the pod is estimated to be
   playing the reply. See start_segment() / audio_sent() for how that is
   estimated.
3. Logging: every interruption is logged with the pod's hostname, how long
   the speech lasted, its peak and average probability, and whether
   playback was active. Speech that is ignored (nothing playing, or too
   short) is logged once per turn and reason, so incidents can be traced
   without flooding the log.

PTT pods don't use this: the button press is the interrupt (main.py).

Note on the Nest (onjuino) firmware: it doesn't send mic audio while it is
playing (isPlaying), so during playback itself no frames arrive, and its
own speaker can't trigger an interrupt. Speech reaches the server while
the reply counts as playing when a tap has stopped the pod's playback
early (the server isn't told, so the rest of the reply still counts as
playing), or around the end of a playback, within the margin below. The
frames that dropped replies before came in while the server was waiting
for the agent: after the stall line has played, the pod's mic is open.

The state lives in one BargeIn object per pod, kept in a small registry
keyed by hostname (for_device()), like main.py keeps per-pod state for the
LED sends. Only the standard library is used, so the offline tests can
import it without audio packages.
"""
import logging
import time

log = logging.getLogger(__name__)

# How long after the estimated end of the audio the pod still counts as
# playing. It covers what the pod does around each playback: it waits for
# 64 to 256 ms of audio before starting the speaker (bufferThreshold), its
# I2S DMA buffers hold another 128 ms, and it writes 120 ms of silence at
# the end. It is added once per audio segment (sentence), because every
# segment goes through that again.
PLAYBACK_MARGIN_S = 0.5

# A dip below the threshold this long or shorter doesn't end a speech run.
GAP_TOLERANCE_S = 0.04

DEFAULT_MIN_MS = 300
DEFAULT_ONLY_WHILE_PLAYING = True


class BargeIn:
    def __init__(self, hostname: str, vad_cfg: dict):
        self.hostname = hostname
        # Same default as config.yaml.example; the VAD itself requires the key.
        self.threshold = vad_cfg.get("threshold", 0.5)
        self.min_s = float(vad_cfg.get("interrupt_min_ms", DEFAULT_MIN_MS)) / 1000
        self.only_while_playing = bool(vad_cfg.get("interrupt_only_while_playing",
                                                   DEFAULT_ONLY_WHILE_PLAYING))
        self.playing_until = 0.0   # time.monotonic() until which the pod is playing
        self.new_turn()

    # ---- per turn -------------------------------------------------------

    def new_turn(self):
        """Called when a turn starts: forget any speech run and allow the
        'ignored' log lines once more."""
        self._clear_run()
        self._logged: set[str] = set()

    def _clear_run(self):
        self.speech_s = 0.0    # speech counted in the current run
        self.gap_s = 0.0       # current dip below the threshold
        self.peak = 0.0
        self.prob_sum = 0.0
        self.n_frames = 0      # frames above the threshold in the current run

    # ---- playback estimate ----------------------------------------------
    #
    # The pod doesn't report when it is playing, so the server estimates
    # it from what it sends. Each sentence goes out on its own TCP
    # connection (send_sentence in main.py). The pod takes one connection
    # at a time: it is "playing" (and mutes its own mic) from the moment it
    # reads the audio header until that audio has played out, and it only
    # takes the next connection after that. The audio sent so far is known
    # exactly (Opus frames x frame length), so:
    #
    #   segment start = max(end of the previous segment, connection opened)
    #   playing_until = segment start + audio sent so far + margin
    #
    # updated after every batch of frames, and never earlier than "now +
    # margin" while frames are still going out (if the TTS is slower than
    # real time, the pod is still waiting for more audio).

    def start_segment(self, now: float | None = None) -> float:
        """An audio connection to the pod was just opened. Returns the
        estimated time its playback starts (pass it to audio_sent)."""
        now = time.monotonic() if now is None else now
        start = max(self.playing_until, now)
        self.playing_until = max(self.playing_until, now + PLAYBACK_MARGIN_S)
        return start

    def audio_sent(self, seg_start: float, audio_s: float, now: float | None = None):
        """`audio_s` seconds of audio in total have now been sent for the
        segment that started at `seg_start`."""
        now = time.monotonic() if now is None else now
        self.playing_until = max(self.playing_until,
                                 seg_start + audio_s + PLAYBACK_MARGIN_S,
                                 now + PLAYBACK_MARGIN_S)

    def is_playing(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now < self.playing_until

    # ---- decision ---------------------------------------------------------

    def on_frame(self, prob: float, frame_s: float, now: float | None = None) -> bool:
        """Feed the VAD speech probability of one mic frame received while
        a turn is running. Returns True when the turn should be interrupted
        (and logs it)."""
        now = time.monotonic() if now is None else now

        if prob <= self.threshold:
            if self.n_frames:
                self.gap_s += frame_s
                if self.gap_s > GAP_TOLERANCE_S + 1e-9:
                    # The speech run ended before it was long enough.
                    playing = self.is_playing(now)
                    if self.only_while_playing and not playing:
                        self._ignored("nothing playing", playing)
                    else:
                        self._ignored("too short", playing)
                    self._clear_run()
            return False

        self.gap_s = 0.0
        self.speech_s += frame_s
        self.n_frames += 1
        self.peak = max(self.peak, prob)
        self.prob_sum += prob

        if self.speech_s + 1e-9 < self.min_s:
            return False
        playing = self.is_playing(now)
        if self.only_while_playing and not playing:
            # Long enough, but nothing is playing (waiting for ASR or the
            # agent). Keep counting: if playback starts while the speech
            # goes on, it interrupts then.
            self._ignored("nothing playing", playing)
            return False

        log.info(f"VOX  interrupt from {self.hostname}: {self._run_desc()}, "
                 f"playback {'active' if playing else 'not active'}")
        self._clear_run()
        return True

    def _run_desc(self) -> str:
        avg = self.prob_sum / self.n_frames if self.n_frames else 0.0
        return (f"{self.speech_s * 1000:.0f} ms of speech "
                f"(peak {self.peak:.2f}, avg {avg:.2f})")

    def _ignored(self, reason: str, playing: bool):
        """Log ignored speech, at most once per turn for each reason."""
        if reason in self._logged:
            return
        self._logged.add(reason)
        extra = f", need {self.min_s * 1000:.0f} ms" if reason == "too short" else ""
        log.info(f"VOX  speech ignored from {self.hostname} ({reason}): "
                 f"{self._run_desc()}{extra}, "
                 f"playback {'active' if playing else 'not active'} "
                 f"(logged once per turn)")


_by_host: dict[str, BargeIn] = {}


def for_device(device, config: dict) -> BargeIn:
    """The BargeIn state of this pod, created on first use."""
    b = _by_host.get(device.hostname)
    if b is None:
        b = _by_host[device.hostname] = BargeIn(device.hostname, config.get("vad", {}))
    return b
