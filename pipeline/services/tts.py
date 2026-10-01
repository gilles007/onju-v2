import io
import logging
import os
import audioop

import httpx
from pydub import AudioSegment

log = logging.getLogger(__name__)

# Optional generation fields of qwen3-tts-new's /v1/audio/speech (SpeechRequest
# in ruby-stack qwen3-tts-new/server.py). A key that is absent from the
# backend's config section, or set to null, is not sent: the server then uses
# its own default, so an unset key gives exactly the payload of before.
SAMPLING_KEYS = ("temperature", "top_k", "top_p", "do_sample",
                 "repetition_penalty", "seed")


def sampling_fields(cfg: dict) -> dict:
    """The SAMPLING_KEYS that are set (not None) in a tts.<backend> section."""
    return {k: cfg[k] for k in SAMPLING_KEYS if cfg.get(k) is not None}


async def synthesize(text: str, voice: str, config: dict) -> bytes:
    """Convert text to 16kHz mono PCM bytes using the configured TTS backend."""
    backend = config["tts"]["backend"]
    if backend == "elevenlabs":
        return await _elevenlabs(text, voice, config)
    if backend in ("local", "local_stream"):
        cfg = config["tts"].get(backend, config["tts"].get("local", {}))
        return await _local(text, config, cfg_override=cfg)
    raise ValueError(f"Unknown TTS backend: {backend}")



async def synthesize_stream(text: str, voice: str, config: dict):
    """Async generator yielding pipeline-rate (16k) mono s16le PCM chunks.

    backend != 'local_stream': one chunk — the whole sentence via synthesize(),
    byte-identical to today's behavior (Kokoro path untouched).
    backend 'local_stream': chunked WAV from a dual-mode server (qwen3-tts-fast);
    header parsed off the stream, stateful resample fallback if the server's rate
    differs from the pipeline rate.
    """
    backend = config["tts"]["backend"]
    if backend != "local_stream":
        yield await synthesize(text, voice, config)
        return

    cfg = config["tts"]["local_stream"]
    url = cfg["url"].rstrip("/") + "/v1/audio/speech"
    target_rate = config["audio"]["sample_rate"]
    payload = {
        "model": cfg.get("model", "qwen3-tts-fast"),
        "input": text,
        "voice": cfg.get("voice", "") or voice or "default",
        "response_format": "wav",
        "stream": True,
        "sampling_rate": target_rate,
        "temperature": cfg.get("temperature", 0.1),
        "top_k": cfg.get("top_k", 1),
        "do_sample": cfg.get("do_sample", False),
        # 1.2, not the server's default 1.05. With greedy decoding (the line
        # above) and the Pepper voice, the TTS sometimes starts with seconds
        # of silence (up to 9 s before "Hello there."), adds up to 5 s of
        # silence after the words, or says a long sentence twice. The penalty
        # stops it picking the same (silence) code over and over. Bench
        # (ruby-stack tools/tts_babble_bench.py, 40 Pepper takes each): more
        # than 1.5 s of dead air in 13 takes -> 1, the sentence said twice
        # once -> never, words unchanged. Set 1.05 here to get the old sound.
        "repetition_penalty": cfg.get("repetition_penalty", 1.2),
    }
    # top_p and seed: sent only when set in tts.local_stream (the keys above
    # keep their defaults). seed: same text + voice + settings -> same audio.
    for k, v in sampling_fields(cfg).items():
        payload.setdefault(k, v)

    header = b""
    src_rate = None
    ratecv_state = None
    stub = b""

    async with httpx.AsyncClient(timeout=cfg.get("timeout", 60)) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if src_rate is None:
                    header += chunk
                    if len(header) < 44:
                        continue
                    src_rate = int.from_bytes(header[24:28], "little")
                    if src_rate != target_rate:
                        log.warning(f"TTS stream at {src_rate}Hz, resampling to {target_rate}")
                    chunk = header[44:]
                    if not chunk:
                        continue
                pcm = stub + chunk
                if len(pcm) % 2:
                    pcm, stub = pcm[:-1], pcm[-1:]
                else:
                    stub = b""
                if not pcm:
                    continue
                if src_rate != target_rate:
                    pcm, ratecv_state = audioop.ratecv(
                        pcm, 2, 1, src_rate, target_rate, ratecv_state)
                yield pcm


async def _elevenlabs(text: str, voice_name: str, config: dict) -> bytes:
    el_cfg = config["tts"]["elevenlabs"]
    api_key = el_cfg["api_key"]
    voice_id = el_cfg["voices"].get(voice_name, el_cfg["voices"].get(el_cfg["default_voice"]))

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        mp3_bytes = resp.content

    audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    log.debug(f"TTS: {len(text)} chars -> {len(audio)}ms audio")
    return audio.raw_data


async def _local(text: str, config: dict, cfg_override=None) -> bytes:
    local_cfg = cfg_override or config["tts"]["local"]
    url = local_cfg["url"].rstrip("/") + "/v1/audio/speech"

    payload = {
        "model": local_cfg["model"],
        "input": text,
        "voice": local_cfg.get("voice", "af_heart"),
        "response_format": "wav",
    }

    # Voice cloning: pass ref_audio path (server reads from disk)
    ref_audio = local_cfg.get("ref_audio")
    if ref_audio:
        payload["ref_audio"] = os.path.abspath(ref_audio)
    ref_text = local_cfg.get("ref_text")
    if ref_text:
        payload["ref_text"] = ref_text

    # Sampling (qwen3-tts-new): only the keys set in tts.local are sent, so a
    # config without them (Kokoro, Hermes) posts exactly the payload of before.
    # E.g. temperature 0.1, top_k 1, do_sample false, repetition_penalty 1.2
    # (local_stream's greedy settings) and a seed for repeatable audio.
    payload.update(sampling_fields(local_cfg))

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        wav_bytes = resp.content

    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    log.debug(f"TTS local: {len(text)} chars -> {len(audio)}ms audio")
    return audio.raw_data
