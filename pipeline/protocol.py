import asyncio
import logging
import struct

log = logging.getLogger(__name__)


async def send_tcp(ip: str, port: int, header: bytes, data: bytes | None = None, timeout: float = 5):
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        writer.write(header)
        if data:
            writer.write(data)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass  # non-critical, device may be busy


async def send_audio(ip: str, port: int, opus_payload: bytes, mic_timeout: int = 60, volume: int = 14, fade: int = 6):
    # header[0]   0xAA for audio
    # header[1:2] mic timeout in seconds (big-endian)
    # header[3]   volume
    # header[4]   fade rate
    # header[5]   compression type (2 = Opus)
    header = bytes([
        0xAA,
        (mic_timeout >> 8) & 0xFF,
        mic_timeout & 0xFF,
        volume,
        fade,
        2,  # Opus
    ])
    await send_tcp(ip, port, header, opus_payload)



# FIX 20261019
# Keep connection open until the entire sentence is sent (to prevent chopped audio)
#
async def open_audio_connection(ip: str, port: int, mic_timeout: int = 60,
                                 volume: int = 14, fade: int = 6,
                                 timeout: float = 5) -> asyncio.StreamWriter | None:
    """Open a persistent TCP connection and send the audio header.
    Returns the writer for incremental frame writes, or None on failure."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        # The pod never sends on this connection; its reader only sees EOF
        # when the pod closes its end (e.g. its 2 s wait for the next Opus
        # frame ran out). main.py checks pod_reader.at_eof() before writing.
        writer.pod_reader = reader
        header = bytes([
            0xAA,
            (mic_timeout >> 8) & 0xFF,
            mic_timeout & 0xFF,
            volume,
            fade,
            2,  # Opus
        ])
        writer.write(header)
        await writer.drain()
        return writer
    except (asyncio.TimeoutError, ConnectionError, OSError):
        return None


def write_audio_frames(writer: asyncio.StreamWriter, opus_frames: list[bytes]) -> bool:
    """Write length-prefixed Opus frames to an open audio connection.
    Returns False if the connection is dead."""
    try:
        for frame in opus_frames:
            writer.write(struct.pack('>H', len(frame)))
            writer.write(frame)
        return True
    except (ConnectionError, OSError):
        return False


async def close_audio_connection(writer: asyncio.StreamWriter):
    """Flush and close — EOF tells the pod this audio segment is done."""
    try:
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass
#
# End of Fix 20261019



# 20260819 LED Thinking Animation add-on
# Values for header[1] of the 0xEE state command. Must match
# DeviceVisualState in onjuino/onjuino.ino.
STATE_IDLE = 0
STATE_LISTENING = 1
STATE_THINKING = 2
STATE_SPEAKING = 3


async def send_state(ip: str, port: int, state: int, timeout: float = 0.2):
    """Send visual state command. 0=idle, 1=listening, 2=thinking, 3=speaking.

    Uses its own short TCP connection. The pod only accepts a new connection
    between playbacks, so a state sent while audio is playing is queued by
    the pod's TCP stack and applied right after that playback ends."""
    # header[0]   0xEE for visual state
    # header[1]   state (see STATE_* above)
    # header[2:5] not used
    header = bytes([0xEE, state, 0, 0, 0, 0])
    await send_tcp(ip, port, header, timeout=timeout)


# 20261011 0xEF device settings. Values for header[1] (setting id). Must
# match SETTING_* in onjuino/onjuino.ino. New settings get new ids: older
# firmware logs and ignores an unknown id.
SETTING_MIC_HOLDOFF = 1     # post-playback mic holdoff (VOX), in ms
MIC_HOLDOFF_MAX_MS = 2000   # the pod clamps to this too
SETTING_PREBUFFER = 2       # (20261012) playback pre-buffer, in ms
PREBUFFER_MIN_MS = 64       # the pod clamps to this range too
PREBUFFER_MAX_MS = 30000


def setting_header(setting_id: int, value: int) -> bytes:
    """Build the 6-byte 0xEF device setting header.

    header[0]   0xEF for device setting
    header[1]   setting id (SETTING_*)
    header[2:3] value, 16-bit big-endian (same byte order as the 0xAA mic timeout)
    header[4:5] reserved, 0
    """
    if not 0 <= setting_id <= 0xFF:
        raise ValueError(f"setting id out of range: {setting_id}")
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"setting value out of range: {value}")
    return bytes([0xEF, setting_id, (value >> 8) & 0xFF, value & 0xFF, 0, 0])


async def send_setting(ip: str, port: int, setting_id: int, value: int, timeout: float = 0.2):
    """Send a 0xEF device setting on its own short TCP connection (same style
    as send_state: a pod that is playing queues it and applies it right after
    that playback). Firmware older than 20261011 does not know 0xEF: it
    flashes red and drops that one connection, nothing else."""
    await send_tcp(ip, port, setting_header(setting_id, value), timeout=timeout)


async def send_mic_holdoff(ip: str, port: int, holdoff_ms: int, timeout: float = 0.2):
    """0xEF id 1: post-playback mic holdoff in ms (0..MIC_HOLDOFF_MAX_MS)."""
    await send_setting(ip, port, SETTING_MIC_HOLDOFF, holdoff_ms, timeout=timeout)


async def send_prebuffer(ip: str, port: int, prebuffer_ms: int, timeout: float = 0.2):
    """(20261012) 0xEF id 2: playback pre-buffer in ms (PREBUFFER_MIN_MS..PREBUFFER_MAX_MS)."""
    await send_setting(ip, port, SETTING_PREBUFFER, prebuffer_ms, timeout=timeout)


async def send_led_blink(ip: str, port: int, intensity: int, r: int = 255, g: int = 255, b: int = 255, fade: int = 6):
    # header[0]   0xCC for LED blink
    # header[1]   starting intensity
    # header[2:4] RGB
    # header[5]   fade rate
    header = bytes([0xCC, intensity, r, g, b, fade])
    await send_tcp(ip, port, header, timeout=0.1)


async def send_stop_listening(ip: str, port: int, hold_s: int = 0):
    """0xDD: set the pod's mic timeout to hold_s from now (0 = close the mic).
    Used by the say endpoint (main.py play_say) after its audio; also kept for
    sesame-esp32-bridge compatibility."""
    header = bytes([0xDD, (hold_s >> 8) & 0xFF, hold_s & 0xFF, 0, 0, 0])
    await send_tcp(ip, port, header, timeout=0.2)


async def open_led_connection(ip: str, port: int, timeout: float = 1) -> asyncio.StreamWriter | None:
    """Open a persistent TCP connection for streaming LED blink commands."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        return writer
    except (asyncio.TimeoutError, ConnectionError, OSError):
        return None


def write_led_blink(writer: asyncio.StreamWriter, intensity: int,
                    r: int = 255, g: int = 255, b: int = 255, fade: int = 6) -> bool:
    """Write a LED blink command to an open connection. Non-async (6-byte buffer write)."""
    header = bytes([0xCC, intensity, r, g, b, fade])
    try:
        writer.write(header)
        return True
    except (ConnectionError, OSError):
        return False


async def close_led_connection(writer: asyncio.StreamWriter):
    """Close a persistent LED connection."""
    try:
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass
