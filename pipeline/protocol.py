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
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
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




async def send_led_blink(ip: str, port: int, intensity: int, r: int = 255, g: int = 255, b: int = 255, fade: int = 6):
    # header[0]   0xCC for LED blink
    # header[1]   starting intensity
    # header[2:4] RGB
    # header[5]   fade rate
    header = bytes([0xCC, intensity, r, g, b, fade])
    await send_tcp(ip, port, header, timeout=0.1)


async def send_stop_listening(ip: str, port: int, hold_s: int = 0):
    """Unused by pipeline — kept for sesame-esp32-bridge compatibility."""
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
