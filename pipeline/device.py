import asyncio
import logging
import time

from pipeline.conversation import ConversationBackend, create_backend
from pipeline.vad import VAD
from pipeline.away import mute_state

log = logging.getLogger(__name__)


class Device:
    def __init__(self, hostname: str, ip: str, config: dict, conversation: ConversationBackend,
                 voice: str | None = None, ptt: bool = False):
        self.hostname = hostname
        self.ip = ip
        self.config = config
        self.conversation = conversation
        el_cfg = config["tts"].get("elevenlabs", {})
        if voice:
            self.voice = voice
        elif ptt and el_cfg.get("default_voice_ptt"):
            self.voice = el_cfg["default_voice_ptt"]
        else:
            self.voice = el_cfg.get("default_voice", "Emma")
        self.ptt = ptt
        self.vad = None if ptt else VAD(config)
        self.last_user_text: str | None = None
        self.last_response: str | None = None
        self.led_power = 0
        self.led_update_time = 0.0
        self.last_mic_at = 0.0      # time.time() of the last mic packet (say busy check)
        self.say_active = False     # a POST /say is playing (pipeline/say.py)
        self.listen_until = 0.0     # time.monotonic() until which its mic window is open
        self.last_heard_at = 0.0    # time.time() of the last packet FROM the pod (UDP mic/VOX,
                                    # multicast announce); picks the default /say target

        self.away = False
        self.away_since = None
        self.mute_received_at = 0.0

        # PTT state
        self.ptt_buffer: list = []  # raw PCM frames during PTT
        self.processing = False     # True while ASR/LLM/TTS pipeline is running
        self.interrupted = asyncio.Event()

        # Add-on while thinking (2026-10-19, pipeline/attention.py and
        # main.py process_utterances). thinking: a VOX turn is waiting for
        # ASR / the agent and none of its audio has started playing, so the
        # server still listens. addon: speech in that window cancelled the
        # turn; its words are carried into the next turn (carry_text once
        # transcribed, else carry_audio). turn_seq: id of the current turn,
        # so a superseded turn's late answer is never played.
        self.thinking = False
        self.addon = False
        self.carry_text: str | None = None
        self.carry_audio = None
        self.turn_seq = 0

    def to_dict(self) -> dict:
        return {
            "hostname": self.hostname,
            "ip": self.ip,
            "voice": self.voice,
            "ptt": self.ptt,
            "away": mute_state(self),
            "away_since": self.away_since if mute_state(self) else None,
        }

    def __repr__(self):
        mode = "PTT" if self.ptt else "VOX"
        return f"<Device {self.hostname} {self.ip} {mode}>"


class DeviceManager:
    def __init__(self, config: dict):
        self.config = config
        self.devices: dict[str, Device] = {}

    def create_device(self, hostname: str, ip: str, ptt: bool = False,
                      heard: bool = False, status_only: bool = False) -> Device:
        device = self.devices.get(hostname)
        if device is None:
            conv = create_backend(self.config, hostname)
            device = Device(hostname, ip, self.config, conversation=conv, ptt=ptt)
            device.status_only = status_only
            self.devices[hostname] = device
            log.debug(f"New device: {device}")
        elif device.ip != ip:
            device.ip = ip
            log.debug(f"Updated {hostname} IP to {ip}")
        else:
            log.debug(f"Device {hostname} reconnected ({ip})")
        # A MUTE packet may beat boot presence (which carries PTT mode).
        # Only that provisional entry needs its mode completed; never reset
        # an established device's voice/conversation on periodic status.
        if not status_only and getattr(device, "status_only", False):
            device.ptt = ptt
            device.vad = None if ptt else device.vad
            if ptt:
                device.voice = self.config["tts"].get("elevenlabs", {}).get("default_voice_ptt") or device.voice
            device.status_only = False
        if heard:
            device.last_heard_at = time.time()
        return device

    def get_by_ip(self, ip: str) -> Device | None:
        for d in self.devices.values():
            if d.ip == ip:
                return d
        return None

    def get_most_recent(self) -> Device | None:
        """The pod we most recently HEARD from (any packet it sent), not the
        one added last: a stale manual/control entry (onju-1415C8) must not
        steal announces. Ties (e.g. nothing heard yet) go to the later-added."""
        best = None
        for d in self.devices.values():
            if best is None or getattr(d, "last_heard_at", 0.0) >= getattr(best, "last_heard_at", 0.0):
                best = d
        return best
