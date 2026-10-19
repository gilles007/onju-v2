import argparse
import asyncio
import json
import logging
import os
import socket
import time
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

import numpy as np
import yaml

from pipeline import agents, attention, bargein
from pipeline import debug_wav
from pipeline import say as say_mod
from pipeline.audio import decode_ulaw, opus_encode, opus_frames_to_tcp_payload, pcm_to_wav, OpusStreamEncoder
from pipeline.conversation import create_backend, sentence_chunks
from pipeline.conversation import stall as stall_mod
from pipeline.device import Device, DeviceManager
from pipeline.protocol import STATE_IDLE, STATE_THINKING, send_audio, send_state, send_led_blink, open_led_connection, write_led_blink, close_led_connection, open_audio_connection, write_audio_frames, close_audio_connection, await_pod_close, send_stop_listening, send_mic_holdoff, MIC_HOLDOFF_MAX_MS, send_prebuffer, PREBUFFER_MIN_MS, PREBUFFER_MAX_MS
from pipeline.services import asr, tts

log = logging.getLogger(__name__)


# LED status (0xEE state command, 2026-09-25). The pod shows a THINKING
# chase from the moment the user stops talking until the answer plays.
# State sends never block a turn: each one is a background task with a
# short timeout whose errors are only logged at debug level. Sends to the
# same pod are chained so they reach it in the order they were issued
# (a quick turn's final IDLE can't overtake its own THINKING).
STATE_TIMEOUT_S = 0.2
_state_tasks: set[asyncio.Task] = set()        # strong refs so tasks aren't GC'd
_state_tail: dict[str, asyncio.Task] = {}      # last state task per pod IP


async def _send_state_after(prev: asyncio.Task | None, ip: str, port: int, state: int):
    if prev is not None and not prev.done():
        await asyncio.wait({prev})   # wait for it without cancelling it
    try:
        # send_tcp's timeout only covers connect; also bound write/close.
        await asyncio.wait_for(send_state(ip, port, state, timeout=STATE_TIMEOUT_S),
                               STATE_TIMEOUT_S + 0.3)
    except Exception as e:
        log.debug(f"STATE {state} to {ip} failed: {e!r}")


def fire_state(ip: str, port: int, state: int) -> asyncio.Task:
    """Send a 0xEE state to the pod in the background (fire-and-forget)."""
    task = asyncio.create_task(_send_state_after(_state_tail.get(ip), ip, port, state))
    _state_tasks.add(task)
    _state_tail[ip] = task

    def _done(t: asyncio.Task):
        _state_tasks.discard(t)
        if _state_tail.get(ip) is t:
            del _state_tail[ip]
    task.add_done_callback(_done)
    return task


# Mic off (0xDD 0, 2026-10-19). The server closes a VOX pod's mic whenever it
# isn't listening (pipeline/attention.py): at turn start, so the pod doesn't
# stream while ASR, the stall decision and the agent run, and after every
# sentence sent with mic timeout 0, because firmware before 20261019 opens
# the mic for at least MIC_LISTEN_MS (20 s) after any playback, whatever the
# header says. The mic opens again with the final sentence's header (or the
# empty send that ends a turn without one). Sent inline (short, bounded) so
# it reaches the pod in order: behind the sentence it follows and ahead of
# the next one (the pod takes connections one at a time, in order).
MIC_OFF_TIMEOUT_S = 0.5


async def mic_off(device, port: int, why: str) -> None:
    try:
        await asyncio.wait_for(send_stop_listening(device.ip, port, 0), MIC_OFF_TIMEOUT_S)
        log.debug(f"MIC  off for {device.hostname} ({why})")
    except Exception as e:
        log.debug(f"MIC  off for {device.hostname} ({why}) failed: {e!r}")


# Pod close confirmation (2026-10-19). After a sentence's end-of-speech marker
# the server only half-closes the audio connection and waits, in the
# background, for the pod to close its end, which it does once it has played
# the sentence. A reset means the pod closed with audio still unread (the
# tail was lost); no close at all within the expected playback time plus
# POD_CLOSE_SLACK_S means it never finished. Both are logged as warnings, so
# a clipped reply shows in the server log instead of looking sent.
POD_CLOSE_SLACK_S = 15.0
_confirm_tasks: set[asyncio.Task] = set()


def confirm_pod_close(writer, device, sentence: str, n_frames: int, play_left_s: float):
    timeout = play_left_s + POD_CLOSE_SLACK_S

    async def run():
        t0 = time.monotonic()
        result = await await_pod_close(writer, timeout)
        took = time.monotonic() - t0
        if result == "reset":
            log.warning(f"Pod {device.ip} reset the audio connection of {sentence!r} "
                        f"({n_frames} frames) after {took:.1f}s: its end may not have played")
        elif result == "timeout":
            log.warning(f"Pod {device.ip} did not close the audio connection of "
                        f"{sentence!r} ({n_frames} frames) within {timeout:.1f}s: "
                        f"it may not have played in full")
        else:
            log.debug(f"Pod {device.ip} closed the audio connection of {sentence!r} "
                      f"after {took:.1f}s")

    task = asyncio.create_task(run())
    _confirm_tasks.add(task)
    task.add_done_callback(_confirm_tasks.discard)
    return task


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


async def udp_listener(config: dict, manager: DeviceManager, utterance_queue: asyncio.Queue):
    """Receive u-law audio from ESP32 devices, run VAD or PTT buffering, queue complete utterances.

    VOX attention gating (2026-10-19, pipeline/attention.py): every frame
    is first checked against attention.listening(). While the server
    listens, the voice LEDs flash with the user's voice and speech can start
    a turn (idle) or interrupt one (barge-in). While it doesn't, the frame
    has no effect at all: no barge-in, no LED, nothing queued; an utterance
    heard then is only logged as ignored. Speech never carries across a
    change of attention: the VAD is reset, so a turn can't start from audio
    that was picked up while the server wasn't listening."""
    udp_port = config["network"]["udp_port"]
    chunk_bytes = config["audio"]["chunk_size"]
    sample_rate = config["audio"]["sample_rate"]

    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    sock.setblocking(False)
    log.info(f"UDP  listening on :{udp_port}")

    tcp_port = config["network"]["tcp_port"]
    dev_cfg = config["device"]

    ptt_timeout = 0.5
    min_utterance_samples = int(sample_rate * 0.3)
    last_packet_time: dict[str, float] = {}
    attending: dict[str, bool] = {}   # last attention.listening() result per VOX pod
    max_led_intensity = 150

    def vad_seconds(dev) -> float:
        return sum(len(b) for b in dev.vad.buffer) / sample_rate if dev.vad.buffer else 0.0

    def ignored(dev, seconds: float, why: str):
        log.info(f"VAD  ignored utterance from {dev.hostname} ({seconds:.1f}s): {why}")

    async def queue_utterance(dev, audio, how: str):
        log.info(f"VAD  {how} from {dev.hostname} ({len(audio)/sample_rate:.1f}s)")
        await utterance_queue.put((dev, audio))

    while True:
        # Short timeout to detect PTT release (packet stream stops)
        try:
            data, addr = await asyncio.wait_for(loop.sock_recvfrom(sock, chunk_bytes * 2), timeout=0.2)
        except asyncio.TimeoutError:
            now = time.time()
            for hostname, dev in list(manager.devices.items()):
                if hostname not in last_packet_time:
                    continue
                gap = now - last_packet_time[hostname]
                if gap < ptt_timeout:
                    continue

                # PTT device: flush buffer when packets stop
                if dev.ptt and dev.ptt_buffer:
                    audio = np.concatenate(dev.ptt_buffer)
                    dev.ptt_buffer.clear()
                    if len(audio) > min_utterance_samples:
                        log.info(f"PTT  end from {hostname} ({len(audio)/sample_rate:.1f}s)")
                        await utterance_queue.put((dev, audio))

                # VOX device: flush if VAD was recording but packets stopped
                elif not dev.ptt and dev.vad.recording:
                    attention.led_for(dev).close()
                    audio = np.concatenate(dev.vad.buffer) if dev.vad.buffer else None
                    dev.vad.reset()
                    if audio is None or len(audio) <= min_utterance_samples:
                        continue
                    hearing, why = attention.listening(dev, config)
                    if hearing and (not dev.processing or dev.interrupted.is_set()):
                        await queue_utterance(dev, audio, "timeout")
                    else:
                        ignored(dev, len(audio) / sample_rate,
                                f"mic stopped; not listening ({why})" if not hearing
                                else "mic stopped during the reply without interrupting it")
            continue

        device = manager.get_by_ip(addr[0])
        if device is None and addr[0] == "127.0.0.1":
            device = manager.get_most_recent()
        if device is None:
            continue

        now = time.time()
        last_packet_time[device.hostname] = now
        device.last_mic_at = now   # mic open: the say endpoint counts it as busy
        pcm = decode_ulaw(data)

        if device.ptt:
            # PTT frames arriving while we're still responding means the user
            # pressed the button to interrupt. The button press is the intent
            # signal; keep buffering so these frames become the next utterance
            # once the current turn bails out.
            if device.processing and not device.interrupted.is_set():
                log.info(f"PTT  interrupt from {device.hostname}")
                device.interrupted.set()
            device.ptt_buffer.append(pcm)
            continue

        # ---- VOX ---------------------------------------------------------
        led = attention.led_for(device)
        hearing, why = attention.listening(device, config)
        prev = attending.get(device.hostname)
        attending[device.hostname] = hearing
        if prev is not None and hearing != prev:
            # Attention changed. Speech in progress doesn't carry over: what
            # was heard while deaf must not start a turn, and speech that was
            # being listened to stops counting the moment the server stops
            # listening (the LEDs stop with it).
            if device.vad.recording:
                ignored(device, vad_seconds(device),
                        f"cut off, listening stopped ({why})" if not hearing
                        else "not listening (started before the server listened)")
                device.vad.reset()
            led.close()

        utterance = device.vad.process_frame(pcm)

        if not hearing:
            # Deaf: no barge-in, no LED, nothing queued. Logged per utterance.
            if utterance is not None:
                ignored(device, len(utterance) / sample_rate, f"not listening ({why})")
            continue

        # Listening: the LEDs flash with the voice (rule 1).
        if device.vad.speech_prob > 0.1:
            device.led_power = min(max_led_intensity, device.led_power + dev_cfg["led_power"])
        if device.processing:
            # Barge-in listening during a turn: one-shot blinks, so a held
            # connection can't delay the reply's next audio connection.
            led.close()
            if now - device.led_update_time > dev_cfg["led_update_period"]:
                device.led_update_time = now
                led.pulse(device.ip, tcp_port, device.led_power, dev_cfg["led_fade"])
                device.led_power = 0
            if not device.interrupted.is_set():
                if bargein.for_device(device, config).on_frame(
                        device.vad.speech_prob, len(pcm) / sample_rate):
                    device.interrupted.set()
            if utterance is not None:
                if device.interrupted.is_set():
                    # The speech that interrupted the turn is the next turn
                    # (like PTT); the worker takes it once this turn has ended.
                    await queue_utterance(device, utterance, "utterance (barge-in)")
                else:
                    ignored(device, len(utterance) / sample_rate,
                            "during the reply, did not interrupt it")
            continue

        # Idle listening: persistent LED connection while the VAD records,
        # a blink every led_update_period (level 0 = keepalive in pauses).
        if device.vad.recording:
            led.ensure(device.ip, tcp_port)
            # Until the connection is up the level keeps adding up (capped),
            # so the first blink goes out as soon as it is.
            if (now - device.led_update_time > dev_cfg["led_update_period"]
                    and led.alive()):
                device.led_update_time = now
                led.blink(device.led_power, dev_cfg["led_fade"])
                device.led_power = 0
        if utterance is not None:
            led.close()
            await queue_utterance(device, utterance, "utterance")
        elif not device.vad.recording:
            device.led_power = 0
            led.close()


def mic_holdoff_ms(config: dict) -> int | None:
    """device.mic_holdoff_ms from the config, or None when unset or invalid
    (then nothing is sent and the pod keeps its firmware default, 256 ms)."""
    value = (config.get("device") or {}).get("mic_holdoff_ms")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        log.warning(f"device.mic_holdoff_ms must be a whole number of ms >= 0, got {value!r}: not sent")
        return None
    if value > MIC_HOLDOFF_MAX_MS:
        log.warning(f"device.mic_holdoff_ms {value} above {MIC_HOLDOFF_MAX_MS}: clamped")
        value = MIC_HOLDOFF_MAX_MS
    return value


def prebuffer_ms(config: dict) -> int | None:
    """(20261012) device.prebuffer_ms from the config, or None when unset or
    invalid (then nothing is sent and the pod keeps its 1000 ms default).
    Clamped to PREBUFFER_MIN_MS..PREBUFFER_MAX_MS."""
    value = (config.get("device") or {}).get("prebuffer_ms")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        log.warning(f"device.prebuffer_ms must be a whole number of ms >= 0, got {value!r}: not sent")
        return None
    if value < PREBUFFER_MIN_MS:
        log.warning(f"device.prebuffer_ms {value} below {PREBUFFER_MIN_MS}: clamped")
        value = PREBUFFER_MIN_MS
    if value > PREBUFFER_MAX_MS:
        log.warning(f"device.prebuffer_ms {value} above {PREBUFFER_MAX_MS}: clamped")
        value = PREBUFFER_MAX_MS
    return value


async def _send_setting_after(prev: asyncio.Task | None, ip: str, port: int,
                              holdoff: int | None, prebuffer: int | None = None):
    if prev is not None and not prev.done():
        await asyncio.wait({prev})   # wait for it without cancelling it
    # Looked up at call time (module globals) so tests can stub them.
    sends = []
    if holdoff is not None:
        sends.append(("mic holdoff", lambda: send_mic_holdoff(ip, port, holdoff, timeout=STATE_TIMEOUT_S)))
    if prebuffer is not None:
        sends.append(("prebuffer", lambda: send_prebuffer(ip, port, prebuffer, timeout=STATE_TIMEOUT_S)))
    for name, make in sends:   # each on its own short connection; one failing doesn't stop the next
        try:
            await asyncio.wait_for(make(), STATE_TIMEOUT_S + 0.3)
        except Exception as e:
            log.debug(f"DEVICE {name} to {ip} failed: {e!r}")


def send_device_settings(device: Device, config: dict) -> asyncio.Task | None:
    """(20261011) Send the 0xEF device settings to a pod that just announced
    itself, in the background (fire-and-forget, like fire_state: own short
    connection, chained behind any pending state send to that pod, errors
    only logged at debug). Called after the greeting has been handed to the
    pod, so it never delays the greeting or the first turn; a pod that is
    still playing queues it and applies it after that playback.
    Nothing is sent for a setting that is unset (device.mic_holdoff_ms,
    device.prebuffer_ms (20261012)); nothing at all when both are unset."""
    holdoff = mic_holdoff_ms(config)
    prebuffer = prebuffer_ms(config)
    if holdoff is None and prebuffer is None:
        return None
    ip = device.ip
    port = config["network"]["tcp_port"]
    if holdoff is not None:
        log.info(f"DEVICE  {device.hostname} mic holdoff {holdoff} ms")
    if prebuffer is not None:
        log.info(f"DEVICE  {device.hostname} prebuffer {prebuffer} ms")
    task = asyncio.create_task(_send_setting_after(_state_tail.get(ip), ip, port, holdoff, prebuffer))
    _state_tasks.add(task)
    _state_tail[ip] = task

    def _done(t: asyncio.Task):
        _state_tasks.discard(t)
        if _state_tail.get(ip) is t:
            del _state_tail[ip]
    task.add_done_callback(_done)
    return task


async def greet_device(device: Device, config: dict):
    """Send greeting to a device, then its 0xEF settings (send_device_settings)."""
    try:
        await _send_greeting(device, config)
    finally:
        try:
            send_device_settings(device, config)
        except Exception as e:
            log.debug(f"DEVICE settings for {device.hostname} not sent: {e!r}")


async def _send_greeting(device: Device, config: dict):
    """Send greeting to a device. Always sends LED pulse for IP registration."""
    dev_cfg = config["device"]
    tcp_port = config["network"]["tcp_port"]

    # Always send LED pulse so the device learns our IP
    await send_led_blink(device.ip, tcp_port, intensity=100, r=0, g=255, b=50, fade=8)

    greeting_path = dev_cfg.get("greeting_wav")
    if not dev_cfg.get("greeting") or not greeting_path or not os.path.exists(greeting_path):
        return
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(greeting_path).set_channels(1).set_frame_rate(16000).set_sample_width(2)
        frames = opus_encode(audio.raw_data, config["audio"]["sample_rate"], config["audio"]["opus_frame_size"])
        payload = opus_frames_to_tcp_payload(frames)
        await send_audio(device.ip, tcp_port, payload,
                         mic_timeout=dev_cfg["default_mic_timeout"],
                         volume=dev_cfg["default_volume"],
                         fade=dev_cfg["led_fade"])
    except Exception as e:
        log.error(f"Failed to send greeting to {device.hostname}: {e}")


async def multicast_listener(config: dict, manager: DeviceManager):
    """Listen for ESP32 device announcements and send greeting audio."""
    mcast_group = config["network"]["multicast_group"]
    mcast_port = config["network"]["multicast_port"]

    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", mcast_port))
    group = socket.inet_aton(mcast_group)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group + socket.inet_aton("0.0.0.0"))
    sock.setblocking(False)
    log.info(f"MCAST  listening on {mcast_group}:{mcast_port}")

    while True:
        data, addr = await loop.sock_recvfrom(sock, 1024)
        try:
            parts = data.decode("utf-8").split()
        except UnicodeDecodeError:
            log.debug(f"MCAST  ignored non-UTF8 packet from {addr[0]}")
            continue
        if not parts:
            continue
        hostname = parts[0]
        ptt = "PTT" in (p.upper() for p in parts)
        log.info(f"DEVICE  {hostname} ({addr[0]}) {'PTT' if ptt else 'VOX'}")
        device = manager.create_device(hostname, addr[0], ptt=ptt)
        await greet_device(device, config)


async def _pcm_chunks(chunks: list[bytes]):
    for c in chunks:
        yield c


async def stream_sentence(config: dict, device, sentence: str, *, is_final: bool,
                          turn_t0: float, tts_kwargs: dict | None = None,
                          mic_timeout: int | None = None,
                          stall: bool = False,
                          pcm: list[bytes] | None = None) -> tuple[bool, int]:
    """Synthesize, encode, and stream one sentence over a single
    TCP connection. Opus frames are written as the TTS produces
    audio; the end-of-speech marker (frame length 0) and closing the
    connection signal end-of-segment to the pod.

    Returns (ok, frames sent). ok is False when the sentence was cut short
    (interrupt, TTS error, pod connection failed): the caller stops there.
    Used by a voice turn (process_utterances, through its send_sentence)
    and by the say endpoint (say_server). mic_timeout: the header's mic
    timeout; None = default_mic_timeout for a final sentence, else 0.
    When it is 0 on a VOX pod, an explicit mic-off (0xDD 0) follows the
    sentence (see mic_off()), so the pod's mic stays closed after it
    plays even on firmware that opens it for 20 s after every playback.
    stall: a stall or filler line; only changes its debug WAV name.
    pcm: audio already rendered for this sentence (say render: whole, see
    _say_render), sent instead of calling the TTS. None = call the TTS."""
    tcp_port = config["network"]["tcp_port"]
    dev_cfg = config["device"]
    sample_rate = config["audio"]["sample_rate"]
    opus_frame_size = config["audio"]["opus_frame_size"]
    tts_kwargs = tts_kwargs or {}
    if device.interrupted.is_set():
        log.info(f"Interrupted before TTS")
        return False, 0

    enc = OpusStreamEncoder(sample_rate, opus_frame_size)
    batch_pcm = b""
    batch_min = sample_rate * 2 // 2          # ~0.5s of s16 mono
    n_frames = 0
    t_first = None
    _dbg_pcm = b""
    max_audio_bytes = config["audio"]["sample_rate"] * 2 * 45  # 45s hard cap

    if mic_timeout is None:
        mic_timeout = dev_cfg["default_mic_timeout"] if is_final else 0
    barge = bargein.for_device(device, config)
    frame_s = opus_frame_size / sample_rate

    # The pod connection is opened only when the first audio is
    # ready to write (first batch, or the end of the sentence),
    # not before the TTS call (2026-10-01, Ruby). Once the pod
    # has read the 6-byte header it gives up on a stream that
    # sends nothing for TCP_STALL_TIMEOUT_MS ("Opus task: TCP
    # stalled waiting for frame length", onjuino.ino): 5000 ms
    # from the last byte received since 6dc6d0f, 2 s before.
    # Whole-sentence TTS (backend local with Qwen) takes 2-3 s,
    # so with the connection opened first, sentences were lost
    # on the 2 s firmware while the server still logged them as
    # sent. Still one connection per sentence (28134ec).
    writer = None
    seg_start = 0.0
    open_failed = False
    cancelled = False
    lost_warned = False

    async def open_pod() -> bool:
        nonlocal writer, seg_start, open_failed
        writer = await open_audio_connection(
            device.ip, tcp_port,
            mic_timeout=mic_timeout,
            volume=dev_cfg["default_volume"],
            fade=dev_cfg["led_fade"])
        if writer is None:
            open_failed = True
            log.error(f"Failed to open audio connection to {device.ip}")
            return False
        # Playback estimate for VOX barge-in: this sentence starts
        # playing when the previous one has finished (or now), and
        # lasts as long as the audio sent so far (+ a margin, see
        # pipeline/bargein.py). Updated after every batch of frames.
        seg_start = barge.start_segment()
        return True

    def pod_closed() -> bool:
        """The pod closed its end (e.g. its 5 s stall timeout, or a
        tap): frames written now are lost. Warned once per sentence."""
        nonlocal lost_warned
        reader = getattr(writer, "pod_reader", None)
        if lost_warned or reader is None or not reader.at_eof():
            return False
        lost_warned = True
        log.warning(f"Pod {device.ip} closed the audio connection after "
                    f"{n_frames} frames; the rest of {sentence!r} is lost "
                    f"(the pod gives up after 5 s without data, or was tapped)")
        return True

    audio = (_pcm_chunks(pcm) if pcm is not None else
             tts.synthesize_stream(sentence, device.voice, config, **tts_kwargs))
    try:
        async for pcm in audio:
            if device.interrupted.is_set():
                log.info(f"Interrupted mid-stream "
                         f"({n_frames} frames sent)")
                return False, n_frames
            if t_first is None:
                t_first = time.monotonic() - turn_t0
            _dbg_pcm += pcm
            # Detect possible TTS hallucination, and stop it before it becomes long babble
            if len(_dbg_pcm) > max_audio_bytes:
                log.warning(f"TTS runaway detected (audio len exceeds longest expected utterance)");
                break;
            batch_pcm += pcm
            ### 20261124, I'm still getting TTS Hallucination (due to short sentences?)
            ### Comment 7 lines below to send one sentence at a time? (I don't think it's networks stack related, but try)
            if len(batch_pcm) >= batch_min:
                frames = enc.encode_chunk(batch_pcm)
                if writer is None and not await open_pod():
                    return False, n_frames
                pod_closed()
                if frames and not write_audio_frames(writer, frames):
                    log.error(f"Connection lost mid-stream to {device.ip}")
                    return False, n_frames
                await writer.drain()
                n_frames += len(frames)
                barge.audio_sent(seg_start, n_frames * frame_s)
                batch_pcm = b""
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as e:
        log.error(f"TTS  failed: {e}")
        return False, n_frames
    finally:
        # Always close — even on error/interrupt, so the pod
        # gets EOF and doesn't hang waiting for more frames.
        tail = enc.encode_chunk(batch_pcm) + enc.flush()
        if writer is None and not (open_failed or cancelled):
            if device.interrupted.is_set():
                # Interrupted before any audio: nothing to play,
                # and a header now would stop the pod's mic
                # while the user is talking over us.
                log.info(f"Interrupted before the first audio; "
                         f"no connection opened for {sentence!r}")
            else:
                # Short sentence (under one batch), no audio at
                # all, runaway in a single chunk, or a TTS error:
                # the pod still gets the header (mic timeout)
                # and end-of-segment, as before.
                await open_pod()
        if writer is not None:
            if tail:
                pod_closed()
                write_audio_frames(writer, tail)
                n_frames += len(tail)
            await close_audio_connection(writer, half_close=True)
            barge.audio_sent(seg_start, n_frames * frame_s)
            # Lost tails show in the log: wait (in the background) for the
            # pod to close its end once it has played this sentence.
            confirm_pod_close(writer, device, sentence, n_frames,
                              max(0.0, barge.playing_until - time.monotonic()))
            if (mic_timeout == 0 and not device.ptt and not cancelled
                    and not device.interrupted.is_set()):
                # Keep the mic closed after this sentence (the next one, or
                # the end of the turn, decides when it opens). Not after an
                # interrupt: the user is talking, and the turn reopens it.
                await mic_off(device, tcp_port, "after sentence")

    log.info(f"SEND  [+{time.monotonic() - turn_t0:.2f}s] "
             f"tts_first[{f'+{t_first:.2f}s' if t_first is not None else 'no audio'}] "
             f"{n_frames} frames "
             f"to {device.ip} "
             f"({'final' if is_final else 'partial'}: {sentence!r})")

    # Debug dump
    import pathlib, wave as _wave
    _dbg_dir = pathlib.Path("/tmp/tts-debug")
    _dbg_dir.mkdir(exist_ok=True)
    _dbg_voice = debug_wav.tts_voice(config, device.voice, tts_kwargs.get("voice_override"))
    _dbg_path = debug_wav.unique_path(_dbg_dir, debug_wav.wav_name(
        sentence, debug_wav.local_now(config), voice=_dbg_voice, stall=stall))
    with _wave.open(str(_dbg_path), "wb") as _w:
        _w.setnchannels(1); _w.setsampwidth(2)
        _w.setframerate(config["audio"]["sample_rate"])
        _w.writeframes(_dbg_pcm)
    log.info(f"DEBUG  saved {len(_dbg_pcm)}B -> {_dbg_path}")

    return True, n_frames


async def process_utterances(config: dict, manager: DeviceManager, utterance_queue: asyncio.Queue):
    """Process complete utterances: ASR -> Conversation -> TTS -> Opus -> TCP."""
    tcp_port = config["network"]["tcp_port"]
    dev_cfg = config["device"]
    no_speech_threshold = 0.45
    # Multi-agent routing (2026-09-27): None unless the config has `agents:`.
    router = agents.router_from_config(config)

    while True:
        device, audio_int16 = await utterance_queue.get()
        route: agents.Route | None = None

        device.processing = True
        device.interrupted.clear()
        bargein.for_device(device, config).new_turn()
        # The pod's mic is closed for the turn (below) and must be open again
        # when it ends: by the final sentence's header, or by reopen_mic().
        mic_reopened = False

        async def reopen_mic():
            """Open the VOX pod's mic again (empty audio with the default mic
            timeout), once per turn. PTT pods: nothing to do."""
            nonlocal mic_reopened
            if mic_reopened or device.ptt:
                return
            mic_reopened = True
            try:
                await send_audio(device.ip, tcp_port, b"",
                                 mic_timeout=dev_cfg["default_mic_timeout"],
                                 volume=0, fade=0)
            except Exception as e:
                log.debug(f"MIC  reopen for {device.hostname} failed: {e!r}")

        try:
            # The server stops listening here (attention.listening() is False
            # while processing, except for barge-in): no voice LEDs, and the
            # pod's mic is closed until the reply's final sentence opens it.
            attention.led_for(device).close()
            if not device.ptt:
                await mic_off(device, tcp_port, "turn start")

            # LED THINKING starts here. udp_listener queued this utterance the
            # moment it decided the user stopped talking (VAD end of speech,
            # VAD packet timeout on VOX, or button release on PTT); this is
            # where it is taken off the queue and handed to ASR, before the
            # ASR, stall decision and LLM wait. Doing it here rather than in
            # udp_listener covers all three cases in one place, and it runs
            # after the previous turn's final IDLE (on a PTT interrupt the
            # next utterance is queued while the old turn is still ending).
            fire_state(device.ip, tcp_port, STATE_THINKING)

            # ASR
            pcm_bytes = audio_int16.astype(np.int16).tobytes()
            try:
                asr_result = await asr.transcribe(pcm_bytes, config)
            except Exception as e:
                log.error(f"ASR  failed ({config['asr']['url']}): {e}")
                continue
            text = asr_result.get("text", "").strip()
            nsp = asr_result.get("no_speech_prob")

            if not text or (nsp is not None and nsp > no_speech_threshold):
                log.info(f"ASR  (no speech)")
                await reopen_mic()
                continue

            # Check for interrupt before LLM
            if device.interrupted.is_set():
                log.info(f"Interrupted before LLM")
                continue

            # Multi-agent routing (optional `agents:` section, see
            # pipeline/agents.py). Decided here, before the stall, so the
            # stall line and the reply both use the chosen agent's voice.
            # The agent's name goes upstream as X-Onju-Bot; the address
            # ("Hey Ruby,") is dropped from the text if strip_address is on.
            # Without an `agents:` section none of this runs.
            tts_kwargs: dict = {}
            conv_kwargs: dict = {}
            if router is not None:
                route = router.route(device.hostname, text)
                conv_kwargs["bot"] = route.agent
                if route.voice:
                    tts_kwargs["voice_override"] = route.voice
                heard = ""
                if route.address is not None:
                    heard = f", heard {route.address.heard!r}"
                    if route.address.distance:
                        heard += f" ~{route.address.distance}"
                log.info(f"AGENT {route.agent} ({route.reason}{heard}) "
                         f"voice={route.voice or 'tts default'}"
                         f"{f' text={route.text!r}' if route.text != text else ''}")
                text = route.text

            # Says (POST /say, pipeline/say.py) played in this agent's voice
            # since its last turn on this pod: passed along as a note so the
            # agent knows what "it" said. Empty unless the say endpoint ran.
            say_note = None
            if say_mod.SayConfig.from_config(config).tell_agent:
                say_note = say_mod.MEMORY.take(device.hostname,
                                               route.agent if route is not None else None)
                if say_note:
                    log.info(f"SAY   passing earlier says to "
                             f"{route.agent if route is not None else 'the agent'}")

            # Streaming LLM → sentence-buffered TTS → Opus → TCP.
            # Intermediate sends use mic_timeout=0, each followed by a mic-off
            # (stream_sentence), so the mic only reopens after the final chunk
            # has played out.
            sample_rate = config["audio"]["sample_rate"]
            opus_frame_size = config["audio"]["opus_frame_size"]

            turn_t0 = time.monotonic()
            full_response: list[str] = []
            pending: str | None = None   # sentence waiting to be flushed
            sent_partial = False           # any non-final chunk already sent?
            answer_sent = False            # any answer (non-stall) sentence sent?
            first_sentence_at: float | None = None
            stream_start_at: float | None = None



            async def send_sentence(sentence: str, is_final: bool, stall: bool = False) -> bool:
                """One sentence of this turn (see stream_sentence)."""
                nonlocal sent_partial, mic_reopened
                ok, _ = await stream_sentence(config, device, sentence, is_final=is_final,
                                              turn_t0=turn_t0, tts_kwargs=tts_kwargs,
                                              stall=stall)
                if ok and not is_final:
                    sent_partial = True
                if ok and is_final:
                    mic_reopened = True   # its header carries default_mic_timeout
                return ok



            async def reopen_mic_if_needed():
                """The turn's mic-off closed the mic (and so did any partial
                audio): push an empty audio to reopen it on recovery."""
                await reopen_mic()

            # Stall decision (agentic mode only; blocking, capped by config timeout).
            # Passes the previous exchange so the classifier can recognize
            # continuations ("go on") and prefaces ("one more thing") as
            # conversational rather than tool-needing.
            stall_text: str | None = None
            if config["conversation"].get("backend") == "agentic":
                stall_text = await stall_mod.decide_stall(
                    text,
                    config,
                    prev_user=device.last_user_text,
                    prev_assistant=device.last_response,
                )
                stall_decided_at = time.monotonic() - turn_t0
                if stall_text:
                    log.info(f"STALL [+{stall_decided_at:.2f}s] decided: {stall_text!r}")
                else:
                    log.info(f"STALL [+{stall_decided_at:.2f}s] NONE")
            device.last_user_text = text

            # Fire stall TTS+send in parallel with OpenClaw warming up.
            stall_task: asyncio.Task | None = None
            extra_context: str | None = None
            if stall_text:
                stall_task = asyncio.create_task(send_sentence(stall_text, is_final=False,
                                                               stall=True))

                # The pod drops to IDLE when the stall line finishes playing.
                # If the answer hasn't started yet, ask for THINKING again; it
                # queues on the pod and applies right after the stall plays.
                def _after_stall(t: asyncio.Task):
                    if t.cancelled() or t.exception() is not None or not t.result():
                        return   # interrupted or failed: no THINKING
                    if not answer_sent and not device.interrupted.is_set():
                        fire_state(device.ip, tcp_port, STATE_THINKING)
                stall_task.add_done_callback(_after_stall)
                extra_context = (
                    f"(You already said aloud to the user: \"{stall_text}\" — "
                    f"don't repeat this phrase, continue naturally with the answer.)"
                )
            if say_note:
                # Its own paragraph: grok-bridge strips only the stall note,
                # so this one reaches the bot with the user's words.
                extra_context = f"{say_note}\n\n{extra_context}" if extra_context else say_note

            # Pause flush (2026-09-25, Ruby): normally each sentence is held as
            # `pending` until the NEXT one arrives (lookahead), so we know which
            # sentence is the last one and can send it as final. But when the
            # agent says a progress line ("Let me look that up.") and then goes
            # quiet for 30+ s while it works, that line would stay silent until
            # the answer arrives. Now, if no new sentence arrives within
            # conversation.pause_flush_s seconds (default 0.8; 0 disables),
            # the held sentence is played right away as non-final. When
            # sentences arrive quickly nothing changes.
            #
            # To wait with a timeout without killing the text stream, a
            # producer task iterates sentence_chunks() and feeds a queue; the
            # loop below waits on the queue. (asyncio.wait_for directly on the
            # generator's __anext__ would cancel and close the generator on
            # timeout.) Stream errors are handed over through the queue and
            # re-raised here, so the "LLM failed" path below still applies.
            pause_flush_s = float(config["conversation"].get("pause_flush_s", 0.8) or 0)
            sentence_q: asyncio.Queue = asyncio.Queue()
            _STREAM_END = object()   # sentinel: the text stream finished

            async def produce_sentences():
                try:
                    async for s in sentence_chunks(
                        device.conversation.stream(text, extra_context=extra_context,
                                                   **conv_kwargs)
                    ):
                        await sentence_q.put(s)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    await sentence_q.put(e)   # re-raised by the consumer
                    return
                await sentence_q.put(_STREAM_END)

            producer: asyncio.Task | None = None
            flushed_on_pause = 0   # sentences played early because the stream paused

            aborted = False
            try:
                stream_start_at = time.monotonic()
                producer = asyncio.create_task(produce_sentences())
                while True:
                    if pending is not None and pause_flush_s > 0:
                        try:
                            item = await asyncio.wait_for(sentence_q.get(), pause_flush_s)
                        except asyncio.TimeoutError:
                            # The text stream paused with a sentence held: play
                            # it now as non-final (mic stays closed, the turn
                            # goes on). The stall audio was already awaited
                            # before `pending` was set.
                            log.info(f"LLM  stream paused {pause_flush_s:.1f}s, "
                                     f"playing held sentence now")
                            answer_sent = True
                            if not await send_sentence(pending, is_final=False):
                                aborted = True
                                break
                            pending = None
                            flushed_on_pause += 1
                            # The agent is still working: back to THINKING
                            # once this sentence has played (queued on the pod).
                            fire_state(device.ip, tcp_port, STATE_THINKING)
                            continue
                    else:
                        item = await sentence_q.get()
                    if item is _STREAM_END:
                        break
                    if isinstance(item, BaseException):
                        raise item   # stream error from the producer
                    sentence = item

                    full_response.append(sentence)
                    if first_sentence_at is None:
                        first_sentence_at = time.monotonic()
                        ttfs_turn = first_sentence_at - turn_t0
                        ttfs_stream = first_sentence_at - stream_start_at
                        log.info(f"LLM  first sentence [+{ttfs_turn:.2f}s turn / "
                                 f"{ttfs_stream:.2f}s stream]: {sentence}")
                    else:
                        log.debug(f"LLM  sentence: {sentence}")

                    # Make sure the stall audio has finished sending before we
                    # start pushing OpenClaw content to the device.
                    if stall_task is not None and not stall_task.done():
                        await stall_task
                        stall_task = None

                    # Flush the *previous* sentence as non-final; whichever
                    # sentence is last when the stream ends becomes the final.
                    if pending is not None:
                        answer_sent = True
                        if not await send_sentence(pending, is_final=False):
                            aborted = True
                            break
                    pending = sentence
            except Exception as e:
                log.error(f"LLM  failed: {e}")
                if stall_task is not None:
                    try:
                        await stall_task
                    except Exception:
                        pass
                await reopen_mic_if_needed()
                continue
            finally:
                # Abort, interruption (send_sentence returned False) or error:
                # stop reading the text stream. No-op when it already ended.
                if producer is not None and not producer.done():
                    producer.cancel()
                    try:
                        await producer
                    except (asyncio.CancelledError, Exception):
                        pass

            # Drain the stall task if it's still pending (e.g. LLM stream
            # returned zero content).
            if stall_task is not None:
                try:
                    await stall_task
                except Exception:
                    pass
                stall_task = None

            if aborted:
                await reopen_mic_if_needed()
                continue

            if pending is not None:
                answer_sent = True
                if not await send_sentence(pending, is_final=True):
                    await reopen_mic_if_needed()
                    continue
            elif sent_partial:
                # The stall played but OpenClaw returned nothing — reopen mic.
                # Also the pause-flush case: the last sentence already played
                # as non-final (mic_timeout=0) during a pause, so close the
                # turn the same way: an empty send with the default timeout.
                await reopen_mic()
            else:
                # No content from the LLM and no stall — still reopen the mic.
                await reopen_mic()

            response_text = " ".join(full_response)
            if response_text:
                device.conversation.commit(response_text)
            device.last_response = response_text
            elapsed = time.monotonic() - turn_t0
            ttfs = f"{first_sentence_at - turn_t0:.2f}s" if first_sentence_at else "—"
            log.info(f"LLM  [{ttfs} first / {elapsed:.2f}s total / "
                     f"{len(full_response)} sentences / {len(response_text)} chars"
                     f"{f' / {flushed_on_pause} played on pause' if flushed_on_pause else ''}] "
                     f"{response_text}")

        except Exception as e:
            log.error(f"Pipeline error ({device.hostname}): {e}")
            await reopen_mic()
        finally:
            # Whatever path the turn took (ASR failure, interrupt, LLM error),
            # the mic closed at turn start must not stay closed.
            if not mic_reopened:
                await reopen_mic()
            device.processing = False
            if route is not None:
                router.touch(device.hostname)   # sticky clock runs from the turn's end
            # LED: every turn ends in IDLE, whatever path it took (answer,
            # no speech, interrupt, LLM failure with no stall, PTT with an
            # empty reply, pipeline error), so THINKING can't get stuck.
            # Short bounded wait, errors swallowed in fire_state. If the
            # final answer is still playing, the pod applies this only after
            # playback ends (it doesn't accept connections while playing),
            # and playback end already set IDLE, so it can't cut the answer.
            await asyncio.wait({fire_state(device.ip, tcp_port, STATE_IDLE)}, timeout=1.0)

        utterance_queue.task_done()


class ColorFormatter(logging.Formatter):
    GREY = "\033[90m"
    WHITE = "\033[37m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: WHITE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.WHITE)
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        return f"{self.GREY}{ts}{self.RESET}  {color}{msg}{self.RESET}"


async def control_server(config: dict, manager: DeviceManager):
    """Tiny HTTP endpoint for runtime device management."""
    port = config.get("network", {}).get("control_port", 3002)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            parts = request_line.decode().strip().split()
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1]

            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()

            body = b""
            if cl := int(headers.get("content-length", 0)):
                body = await reader.readexactly(cl)

            if path == "/devices" and method == "GET":
                data = {k: v.to_dict() for k, v in manager.devices.items()}
                resp = json.dumps(data, indent=2)
                _http_respond(writer, 200, resp)

            elif path == "/devices" and method == "POST":
                payload = json.loads(body)
                ip = payload["ip"]
                hostname = payload.get("hostname", f"manual-{ip.replace('.', '-')}")
                ptt = payload.get("ptt", False)
                device = manager.create_device(hostname, ip, ptt=ptt)
                log.info(f"DEVICE  {hostname} ({ip}) {'PTT' if ptt else 'VOX'} (control)")
                asyncio.create_task(greet_device(device, config))
                _http_respond(writer, 201, json.dumps({"hostname": hostname, "ip": ip, "ptt": ptt}))

            elif path == "/devices" and method == "DELETE":
                payload = json.loads(body)
                hostname = payload.get("hostname")
                if hostname and hostname in manager.devices:
                    del manager.devices[hostname]
                    log.info(f"DEVICE  {hostname} removed (control)")
                    _http_respond(writer, 200, json.dumps({"removed": hostname}))
                else:
                    _http_respond(writer, 404, json.dumps({"error": "not found"}))

            else:
                _http_respond(writer, 404, "not found")

            await writer.drain()
        except Exception as e:
            log.debug(f"Control request error: {e}")
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "0.0.0.0", port)
    log.info(f"CTRL listening on :{port}")
    async with server:
        await server.serve_forever()


def _http_respond(writer: asyncio.StreamWriter, status: int, body: str):
    reason = {200: "OK", 201: "Created", 404: "Not Found"}.get(status, "OK")
    writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}".encode())


# ---- say endpoint (pipeline/say.py) ----------------------------------------

async def _one_delta(text: str):
    yield text + " "


# How often to re-send STATE_THINKING while a staged say holds the LEDs.
# Firmware THINKING_TIMEOUT_MS is 200 s, so a single send covers think_s
# (capped at 30); refreshing matches how a normal turn re-arms THINKING
# after a stall line / pause-flush (SPEAKING would otherwise leave IDLE).
THINK_REFRESH_S = 10.0


async def _say_sleep(device, seconds: float) -> bool:
    """Sleep up to `seconds`. Returns False if the pod was interrupted."""
    if seconds <= 0:
        return not device.interrupted.is_set()
    end = time.monotonic() + seconds
    while True:
        if device.interrupted.is_set():
            return False
        left = end - time.monotonic()
        if left <= 0:
            return True
        await asyncio.sleep(min(0.05, left))


async def _say_wait_playback(device, barge) -> bool:
    """Wait until barge-in estimates the pod has finished playing.
    Returns False if interrupted. Accuracy is the playback estimate
    (audio sent + PLAYBACK_MARGIN_S); the pod does not report end-of-play."""
    while barge.is_playing():
        if device.interrupted.is_set():
            return False
        left = barge.playing_until - time.monotonic()
        await asyncio.sleep(min(0.05, max(0.0, left)))
    return not device.interrupted.is_set()


async def _say_hold_thinking(config: dict, device, think_s: float,
                             render: asyncio.Task | None = None) -> bool:
    """Hold STATE_THINKING for think_s (re-send every THINK_REFRESH_S).
    render (say render: whole): also wait until that render task is done,
    with THINKING still on if think_s > 0 (no LEDs if think_s is 0).
    Returns False if interrupted. Speaking LEDs take over when the answer
    audio connection opens (firmware sets VS_SPEAKING on 0xAA)."""
    if think_s <= 0 and render is None:
        return not device.interrupted.is_set()
    tcp_port = config["network"]["tcp_port"]
    if think_s > 0:
        fire_state(device.ip, tcp_port, STATE_THINKING)
    end = time.monotonic() + think_s
    next_refresh = time.monotonic() + THINK_REFRESH_S
    while True:
        if device.interrupted.is_set():
            return False
        now = time.monotonic()
        if now >= end and (render is None or render.done()):
            return True
        if think_s > 0 and now >= next_refresh:
            fire_state(device.ip, tcp_port, STATE_THINKING)
            next_refresh = now + THINK_REFRESH_S
        await asyncio.sleep(max(0.0, min(0.05, end - now, next_refresh - now)) if now < end else 0.05)


async def _say_render(config: dict, device, sentences: list[str], tts_kwargs: dict,
                      after: asyncio.Task | None = None) -> list[list[bytes]]:
    """say render: whole. Render every sentence's audio before any of it is
    sent (same TTS call as stream_sentence), so they play back to back.
    after: a render to finish first (the filler, before the answer).
    Stops early on interrupt or a TTS error: sentences it didn't render are
    synthesized by stream_sentence as usual (which then logs the error)."""
    if after is not None:
        await after
    t = time.monotonic()
    out = []
    for sentence in sentences:
        chunks = []
        try:
            async for pcm in tts.synthesize_stream(sentence, device.voice, config, **tts_kwargs):
                if device.interrupted.is_set():
                    return out
                chunks.append(pcm)
        except Exception as e:
            log.warning(f"SAY   render failed for {sentence!r}: {e}")
            return out
        out.append(chunks)
    log.info(f"SAY   rendered {len(out)} sentence(s) in {time.monotonic() - t:.2f}s")
    return out


async def _say_stream_part(config: dict, device, sentences: list[str], *,
                           t0: float, tts_kwargs: dict, listen: bool,
                           stall: bool = False,
                           rendered: list[list[bytes]] | None = None) -> tuple[bool, int, int]:
    """Stream sentences; listen only on the last if listen. Returns
    (ok, frames, played_count). stall: filler lines (debug WAV name only).
    rendered: audio from _say_render (say render: whole), one entry per
    sentence from the first; sentences without one call the TTS."""
    rendered = rendered or []
    frames = 0
    played = 0
    ok = True
    for i, sentence in enumerate(sentences):
        last = i == len(sentences) - 1
        mic = config["device"]["default_mic_timeout"] if (last and listen) else 0
        ok, n = await stream_sentence(config, device, sentence, is_final=last,
                                      turn_t0=t0, tts_kwargs=tts_kwargs, mic_timeout=mic,
                                      stall=stall,
                                      pcm=rendered[i] if i < len(rendered) else None)
        frames += n
        if not ok:
            break
        played += 1
    return ok, frames, played


async def play_say(config: dict, scfg: "say_mod.SayConfig", device,
                   sentences: list[str], text: str, tts_voice: str | None,
                   bot: str | None, listen: bool = False,
                   filler_sentences: list[str] | None = None,
                   filler_text: str | None = None,
                   think_s: float = 0.0, delay_s: float = 0.0,
                   barge=None, utterance_queue: asyncio.Queue | None = None) -> dict:
    """Play a say on a pod already claimed by say_request (say_active and
    processing set). Same path as a voice turn: stream_sentence() per
    sentence, so the speaking LEDs, barge-in estimate, SEND lines and debug
    WAVs all apply.

    Optional staging (all no-ops when unset, so plain announces are
    unchanged): delay_s → filler (mic closed) → wait for filler playback →
    STATE_THINKING for think_s → answer. listen applies only to the final
    answer sentence; between parts the mic stays closed (mic_timeout 0,
    and a 0xDD 0 after every sentence sent with it, see stream_sentence).

    say render: whole (tts.backend local only, say.render_whole): each part
    is rendered in full before it plays, so its sentences play back to back.
    The renders start right away (filler first, then the answer), so delay_s,
    the filler playback and think_s hide the answer's TTS time; if it isn't
    ready when think_s ends, THINKING stays on until it is."""
    tcp_port = config["network"]["tcp_port"]
    frame_s = config["audio"]["opus_frame_size"] / config["audio"]["sample_rate"]
    t0 = time.monotonic()
    tts_kwargs = {"voice_override": tts_voice} if tts_voice else {}
    if barge is None:
        barge = bargein.for_device(device, config)
    filler_sentences = filler_sentences or []
    frames = 0
    played = 0
    filler_frames = 0
    filler_played = 0
    ok = True
    stage = "start"
    delayed = False
    filler_render = answer_render = None
    if say_mod.render_whole(scfg, config):
        if filler_sentences:
            filler_render = asyncio.create_task(
                _say_render(config, device, filler_sentences, tts_kwargs))
        answer_render = asyncio.create_task(
            _say_render(config, device, sentences, tts_kwargs, after=filler_render))
    log.info(f"SAY   {device.hostname} voice={tts_voice or 'tts default'} bot={bot or '-'} "
             f"{len(sentences)} sentence(s)"
             f"{f', filler={filler_text!r}' if filler_text else ''}"
             f"{f', think_s={think_s:g}' if think_s else ''}"
             f"{f', delay_s={delay_s:g}' if delay_s else ''}"
             f": {text!r}")
    try:
        if delay_s > 0:
            stage = "delay"
            if not await _say_sleep(device, delay_s):
                ok = False
            else:
                delayed = True
                queued = utterance_queue.qsize() if utterance_queue is not None else 0
                reason = say_mod.busy_reason(device, barge, queued, ignore_owned=True)
                if reason:
                    log.info(f"SAY   aborted after delay on {device.hostname}: {reason}")
                    ok = False
                    stage = "delay_busy"
        if ok and filler_sentences:
            stage = "filler"
            ok = await _say_hold_thinking(config, device, 0, render=filler_render)
            if ok:
                ok, n, filler_played = await _say_stream_part(
                    config, device, filler_sentences, t0=t0, tts_kwargs=tts_kwargs,
                    listen=False, stall=True,
                    rendered=filler_render.result() if filler_render else None)
            else:
                n = 0
            filler_frames = n
            frames += n
            if ok and not await _say_wait_playback(device, barge):
                ok = False
        if ok and (think_s > 0 or answer_render is not None):
            stage = "think"
            if not await _say_hold_thinking(config, device, think_s, render=answer_render):
                ok = False
        if ok and sentences:
            stage = "answer"
            # Final part of a multi-stage say, or the only part of a plain one.
            ok, n, played = await _say_stream_part(
                config, device, sentences, t0=t0, tts_kwargs=tts_kwargs, listen=listen,
                rendered=answer_render.result() if answer_render else None)
            frames += n
        # Mic: every sentence sent with mic timeout 0 (all of them unless
        # listen) is followed by a 0xDD 0 in stream_sentence: firmware before
        # 20261019 opens the mic for at least 20 s after every playback
        # (MIC_LISTEN_MS in onjuino.ino), whatever the header says. The pod
        # is still playing, so each one waits in its TCP backlog and applies
        # right after the audio before it.
    finally:
        for task in (filler_render, answer_render):
            if task is not None and not task.done():
                task.cancel()
        device.processing = False
        device.say_active = False
    if ok:
        status = "played"
    elif stage == "delay_busy":
        status = "busy"
    elif device.interrupted.is_set():
        status = "interrupted"
    else:
        status = "failed"
    duration = round(frames * frame_s, 2)
    filler_duration = round(filler_frames * frame_s, 2)
    answer_duration = round((frames - filler_frames) * frame_s, 2)
    elapsed = round(time.monotonic() - t0, 2)

    def _part_status(part_stage: str, part_played: int, part_total: int) -> str:
        # A part that fully streamed is "played" even if a later stage
        # (think / answer) was interrupted or failed.
        if part_total and part_played == part_total:
            return "played"
        if status == "busy":
            return "busy"
        if part_played == 0 and stage != part_stage:
            # Never reached this part (e.g. aborted in think before answer).
            return status
        return "interrupted" if device.interrupted.is_set() else "failed"

    # Log each part that was reached. Filler lines carry filler=true so the
    # agent (and Gilles) can tell them apart from the answer.
    if filler_text and stage not in ("start", "delay", "delay_busy"):
        say_mod.append_log(scfg.log_path, {
            "device": device.hostname, "bot": bot, "voice": tts_voice,
            "status": _part_status("filler", filler_played, len(filler_sentences)),
            "duration_s": filler_duration, "text": filler_text, "filler": True,
        })
    if status == "busy":
        say_mod.append_log(scfg.log_path, {
            "device": device.hostname, "bot": bot, "voice": tts_voice,
            "status": "busy", "duration_s": 0, "text": text,
            "reason": "busy after delay",
        })
    elif stage not in ("start", "delay"):
        say_mod.append_log(scfg.log_path, {
            "device": device.hostname, "bot": bot, "voice": tts_voice,
            "status": _part_status("answer", played, len(sentences)),
            "duration_s": answer_duration, "text": text,
        })
    if scfg.tell_agent:
        if filler_text and filler_played:
            said_f = (filler_text if filler_played == len(filler_sentences)
                      else " ".join(filler_sentences[:filler_played]))
            say_mod.MEMORY.add(device.hostname, bot, said_f, filler=True)
        if played:
            said = (text if played == len(sentences)
                    else " ".join(sentences[:played]))
            say_mod.MEMORY.add(device.hostname, bot, said)
    log.info(f"SAY   {status}: answer {played}/{len(sentences)}"
             f"{f', filler {filler_played}/{len(filler_sentences)}' if filler_text else ''}"
             f", {duration:.2f}s of audio, {elapsed:.2f}s")
    result = {"status": status, "device": device.hostname, "bot": bot, "voice": tts_voice,
              "sentences": len(sentences), "played": played,
              "duration_s": duration, "elapsed_s": elapsed}
    if filler_text is not None:
        result["filler"] = filler_text
        result["filler_sentences"] = len(filler_sentences)
        result["filler_played"] = filler_played
    if think_s:
        result["think_s"] = think_s
    if delay_s:
        result["delay_s"] = delay_s
        result["delayed"] = delayed
    return result


async def say_request(config: dict, scfg: "say_mod.SayConfig", router, manager,
                      utterance_queue: asyncio.Queue | None,
                      headers: dict, body: bytes) -> tuple[int, dict]:
    """POST /say. Returns (HTTP status, JSON body); SayError for refusals."""
    say_mod.check_token(scfg, headers)
    req = say_mod.parse_request(scfg, body)
    tts_voice, bot = say_mod.resolve_voice(router, req["voice"], req["bot"],
                                             raw=req["raw_voice"])
    device = say_mod.find_device(manager, req["device"])
    if device is None:
        raise say_mod.SayError(404, {"error": "unknown pod" if req["device"] else "no pod known yet",
                                     "device": req["device"] or None})
    barge = bargein.for_device(device, config)
    queued = utterance_queue.qsize() if utterance_queue is not None else 0
    reason = say_mod.busy_reason(device, barge, queued)
    if reason:
        log.info(f"SAY   refused for {device.hostname}: {reason}")
        raise say_mod.SayError(409, {"error": "busy", "reason": reason, "device": device.hostname})
    # Claim the pod. No await since the busy check, so nothing can slip in
    # between; processing also keeps the VAD from starting a voice turn
    # (mic frames during the say only feed barge-in, like during a reply).
    # Claim covers delay_s too, so a second /say during the wait gets 409.
    device.say_active = True
    device.processing = True
    device.interrupted.clear()
    barge.new_turn()
    try:
        sentences = [s async for s in sentence_chunks(_one_delta(req["text"]))]
        filler_sentences = []
        if req["filler"]:
            filler_sentences = [s async for s in sentence_chunks(_one_delta(req["filler"]))]
    except BaseException:
        device.processing = False
        device.say_active = False
        raise
    task = asyncio.create_task(play_say(
        config, scfg, device, sentences, req["text"], tts_voice, bot,
        listen=req["listen"], filler_sentences=filler_sentences,
        filler_text=req["filler"], think_s=req["think_s"], delay_s=req["delay_s"],
        barge=barge, utterance_queue=utterance_queue))
    if not req["wait"]:
        _say_tasks.add(task)
        task.add_done_callback(_say_tasks.discard)
        body_out = {"status": "queued", "device": device.hostname, "bot": bot,
                    "voice": tts_voice, "sentences": len(sentences)}
        if req["filler"]:
            body_out["filler"] = req["filler"]
            body_out["filler_sentences"] = len(filler_sentences)
        if req["think_s"]:
            body_out["think_s"] = req["think_s"]
        if req["delay_s"]:
            body_out["delay_s"] = req["delay_s"]
        return 200, body_out
    result = await task
    if result["status"] == "busy":
        return 409, {"error": "busy", "reason": "busy after delay",
                     "device": device.hostname, **{k: result[k] for k in
                     ("bot", "voice", "delay_s") if k in result}}
    return (502 if result["status"] == "failed" else 200), result


_say_tasks: set[asyncio.Task] = set()   # strong refs for wait=false says


def say_status(config: dict, manager, utterance_queue, router) -> dict:
    """GET /say: known pods and whether each could take a say now."""
    queued = utterance_queue.qsize() if utterance_queue is not None else 0
    devices = {}
    for d in (manager.devices.values() if manager is not None else []):
        devices[d.hostname] = {
            "ip": d.ip,
            "busy": say_mod.busy_reason(d, bargein.for_device(d, config), queued),
        }
    voices = {}
    if router is not None:
        voices = {a.name: router.voice_for(a.name) for a in router.agents.values()}
    return {"devices": devices, "agents": voices}


async def _read_http_request(reader: asyncio.StreamReader, max_body: int = 65536):
    request_line = await asyncio.wait_for(reader.readline(), timeout=5)
    parts = request_line.decode("latin-1").strip().split()
    if len(parts) < 2:
        return None
    headers = {}
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=5)
        if line in (b"\r\n", b"\n", b""):
            break
        k, _, v = line.decode("latin-1").partition(":")
        headers[k.strip().lower()] = v.strip()
    try:
        length = int(headers.get("content-length", 0) or 0)
    except ValueError:
        length = -1
    if length < 0 or length > max_body:
        raise say_mod.SayError(400 if length < 0 else 413, {"error": "bad body size"})
    body = await asyncio.wait_for(reader.readexactly(length), timeout=10) if length else b""
    return parts[0].upper(), parts[1], headers, body


_HTTP_REASONS = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
                 405: "Method Not Allowed", 409: "Conflict", 413: "Payload Too Large",
                 500: "Internal Server Error", 502: "Bad Gateway"}


def _http_json(writer: asyncio.StreamWriter, status: int, obj) -> None:
    body = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    writer.write(f"HTTP/1.1 {status} {_HTTP_REASONS.get(status, 'OK')}\r\n"
                 f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                 f"Connection: close\r\n\r\n".encode() + body)


async def say_server(config: dict, manager, utterance_queue: asyncio.Queue | None,
                     ready: asyncio.Event | None = None):
    """POST /say and GET /say on say.host:say.port (only when say.enabled)."""
    scfg = say_mod.SayConfig.from_config(config)
    router = agents.router_from_config(config)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            try:
                req = await _read_http_request(reader)
                if req is None:
                    return
                method, path, headers, body = req
                path = path.split("?", 1)[0].rstrip("/") or "/"
                if path != "/say":
                    status, resp = 404, {"error": "not found"}
                elif method == "POST":
                    status, resp = await say_request(config, scfg, router, manager,
                                                     utterance_queue, headers, body)
                elif method == "GET":
                    say_mod.check_token(scfg, headers)
                    status, resp = 200, say_status(config, manager, utterance_queue, router)
                else:
                    status, resp = 405, {"error": "use POST (or GET for status)"}
            except say_mod.SayError as e:
                status, resp = e.status, e.body
            except Exception as e:
                log.error(f"SAY   request failed: {e!r}")
                status, resp = 500, {"error": str(e) or e.__class__.__name__}
            _http_json(writer, status, resp)
            await writer.drain()
        except Exception as e:
            log.debug(f"SAY   connection error: {e!r}")
        finally:
            writer.close()

    try:
        server = await asyncio.start_server(handle, scfg.host, scfg.port)
    except OSError as e:
        # e.g. the port is taken (another onju on this host): keep the voice
        # pipeline running without the say endpoint rather than exit.
        log.error(f"SAY   cannot listen on {scfg.host}:{scfg.port}: {e}; say endpoint off")
        return
    log.info(f"SAY   listening on {scfg.host}:{scfg.port} (POST /say)"
             f"{' with token' if scfg.token else ''}"
             f"{', render=whole' if say_mod.render_whole(scfg, config) else ''}")
    if scfg.render == "whole" and not say_mod.render_whole(scfg, config):
        log.warning(f"SAY   render: whole has no effect with tts.backend "
                    f"{config.get('tts', {}).get('backend')!r} (only local); rendering per sentence")
    if ready is not None:
        ready.set()
    async with server:
        await server.serve_forever()


def _log_startup_summary(config: dict) -> None:
    """Log the active endpoints and models so it's obvious at a glance how the
    pipeline is configured for this run."""
    conv_cfg = config["conversation"]
    backend_name = conv_cfg.get("backend", "conversational")
    backend_cfg = conv_cfg.get(backend_name, {})

    log.info("Pipeline server starting")
    log.info(f"  ASR   {config['asr']['url']}")

    if backend_name == "agentic":
        model = backend_cfg.get("provider_model") or backend_cfg.get("model", "?")
        log.info(f"  LLM   agentic: {model} @ {backend_cfg.get('base_url', '?')} "
                 f"(channel={backend_cfg.get('message_channel', '?')})")
        stall_cfg = conv_cfg.get("stall", {})
        if stall_cfg.get("enabled"):
            log.info(f"  STALL {stall_cfg.get('model', '?')} @ {stall_cfg.get('base_url', '?')} "
                     f"(timeout={stall_cfg.get('timeout', 1.5)}s)")
        else:
            log.info("  STALL disabled")
    else:
        log.info(f"  LLM   conversational: {backend_cfg.get('model', '?')} "
                 f"@ {backend_cfg.get('base_url', '?')}")

    router = agents.router_from_config(config)
    if router is not None:
        log.info(f"  AGENTS {router.describe()}")

    tts_cfg = config["tts"]
    tts_backend = tts_cfg.get("backend", "?")
    if tts_backend == "elevenlabs":
        el = tts_cfg.get("elevenlabs", {})
        vox = el.get("default_voice", "?")
        ptt = el.get("default_voice_ptt", vox)
        log.info(f"  TTS   elevenlabs: VOX={vox} PTT={ptt}")
    else:
        log.info(f"  TTS   {tts_backend}")


async def warmup(config: dict):
    """Validate conversation backend and TTS are reachable."""
    log.info("Warming up conversation backend and TTS...")

    async def _warmup_conversation():
        t0 = time.time()
        try:
            backend = create_backend(config, "_warmup")
            text = await backend.send("Hi")
            if not text.strip():
                log.error(f"LLM  warmup FAILED ({time.time() - t0:.1f}s): empty response")
            else:
                log.info(f"LLM  warmup OK  ({time.time() - t0:.1f}s) -> {text.strip()!r}")
        except Exception as e:
            log.error(f"LLM  warmup FAILED ({time.time() - t0:.1f}s): {e}")

    async def _warmup_tts():
        t0 = time.time()
        try:
            pcm = await tts.synthesize("Hello.", "default", config)
            duration_ms = len(pcm) / (16000 * 2) * 1000
            if duration_ms < 100:
                log.error(f"TTS  warmup FAILED ({time.time() - t0:.1f}s): audio too short ({duration_ms:.0f}ms)")
            else:
                log.info(f"TTS  warmup OK  ({time.time() - t0:.1f}s) -> {duration_ms:.0f}ms audio")
        except Exception as e:
            log.error(f"TTS  warmup FAILED ({time.time() - t0:.1f}s): {e}")

    await asyncio.gather(_warmup_conversation(), _warmup_tts())
    log.info("Warmup complete")


def _parse_device_arg(entry: str) -> tuple[str, str, bool]:
    """Parse device arg like 'name=ip:ptt' or 'ip:ptt' or 'ip'."""
    ptt = False
    if entry.endswith(":ptt"):
        ptt = True
        entry = entry[:-4]
    parts = entry.split("=", 1)
    if len(parts) == 2:
        hostname, ip = parts
    else:
        ip = parts[0]
        hostname = f"manual-{ip.replace('.', '-')}"
    return hostname, ip, ptt


async def main(config_path: str = None, do_warmup: bool = False, devices: list[str] = None):
    config = load_config(config_path)

    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logging.basicConfig(level=log_level, handlers=[handler])

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    manager = DeviceManager(config)

    startup_greets = []
    for entry in (devices or []):
        hostname, ip, ptt = _parse_device_arg(entry)
        device = manager.create_device(hostname, ip, ptt=ptt)
        log.info(f"DEVICE  {hostname} ({ip}) {'PTT' if ptt else 'VOX'} (manual)")
        startup_greets.append(greet_device(device, config))
    if startup_greets:
        await asyncio.gather(*startup_greets)

    if do_warmup:
        await warmup(config)

    utterance_queue = asyncio.Queue()

    _log_startup_summary(config)
    tasks = [
        udp_listener(config, manager, utterance_queue),
        multicast_listener(config, manager),
        process_utterances(config, manager, utterance_queue),
        control_server(config, manager),
    ]
    if say_mod.SayConfig.from_config(config).enabled:
        tasks.append(say_server(config, manager, utterance_queue))
    else:
        log.info("SAY   disabled (say.enabled is not set)")
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice pipeline server")
    parser.add_argument("config", nargs="?", default=None, help="Path to config YAML")
    parser.add_argument("--warmup", action="store_true", help="Warmup LLM and TTS on startup")
    parser.add_argument("--device", action="append", dest="devices", metavar="[NAME=]IP[:ptt]",
                        help="Register device (e.g. --device onju=192.168.1.50:ptt)")
    args = parser.parse_args()
    asyncio.run(main(args.config, do_warmup=args.warmup, devices=args.devices))
