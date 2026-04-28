"""Speech/media backend adapters for local and OpenAI-compatible runtimes."""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOCAL_STT_MODEL = os.environ.get("TURNSTONE_STT_LOCAL_MODEL", "base.en")
_DEFAULT_STT_MODEL = os.environ.get("TURNSTONE_STT_MODEL", "gpt-4o-mini-transcribe")
_DEFAULT_TTS_MODEL = os.environ.get("TURNSTONE_TTS_MODEL", "gpt-4o-mini-tts")
_LOCAL_KOKORO_MODEL_URL = os.environ.get(
    "TURNSTONE_KOKORO_MODEL_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
)
_LOCAL_KOKORO_VOICES_URL = os.environ.get(
    "TURNSTONE_KOKORO_VOICES_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
)


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    backend: str
    model_alias: str = ""


@dataclass(frozen=True)
class SpeechResult:
    audio_bytes: bytes
    media_type: str
    backend: str
    model_alias: str = ""


class MediaBackendError(RuntimeError):
    """A configured backend failed during execution."""


class MediaBackendUnavailable(RuntimeError):
    """No suitable backend is configured or installed."""


_kokoro_runtime: Any | None = None
_whisper_model_cache: dict[tuple[str, str, str], Any] = {}


def prefer_local_backend(selected_alias: str, env_flag: str) -> bool:
    alias = (selected_alias or "").strip().lower()
    if os.environ.get(env_flag, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return (
        alias.startswith("local")
        or alias.endswith("-local")
        or alias.startswith("kokoro")
        or alias.startswith("whisper")
    )


def _make_wav_bytes(samples: list[int], *, sample_rate: int = 24000) -> bytes:
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(s)))) for s in samples
        )
        wav.writeframes(frames)
    return buf.getvalue()


def fallback_tts_wav(text: str, *, sample_rate: int = 24000) -> bytes:
    text = (text or "").strip()[:240]
    if not text:
        text = "turnstone"
    samples: list[int] = []
    ramp = max(1, sample_rate // 400)
    per_char = max(sample_rate // 12, 1)
    silence = sample_rate // 80
    for idx, ch in enumerate(text):
        if ch.isspace():
            samples.extend([0] * (silence * 2))
            continue
        freq = 220 + ((ord(ch) + idx * 17) % 36) * 12
        amp = 7000
        for i in range(per_char):
            if i < ramp:
                env = i / float(ramp)
            elif i > per_char - ramp:
                env = max(0.0, (per_char - i) / float(ramp))
            else:
                env = 1.0
            value = int(amp * env * math.sin((2.0 * math.pi * freq * i) / sample_rate))
            samples.append(value)
        samples.extend([0] * silence)
    return _make_wav_bytes(samples, sample_rate=sample_rate)


def _openai_client_for_audio() -> Any | None:
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("TURNSTONE_OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return None
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _ensure_local_file(path: Path, url: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _get_kokoro_runtime() -> Any:
    global _kokoro_runtime
    if _kokoro_runtime is not None:
        return _kokoro_runtime
    from kokoro_onnx import Kokoro

    cache_dir = Path.home() / ".cache" / "turnstone" / "kokoro"
    model_path = _ensure_local_file(cache_dir / "kokoro-v1.0.onnx", _LOCAL_KOKORO_MODEL_URL)
    voices_path = _ensure_local_file(cache_dir / "voices-v1.0.bin", _LOCAL_KOKORO_VOICES_URL)
    _kokoro_runtime = Kokoro(str(model_path), str(voices_path))
    return _kokoro_runtime


def local_tts_wav(text: str, voice: str = "af_bella") -> bytes:
    runtime = _get_kokoro_runtime()
    audio, sample_rate = runtime.create(text, voice=voice or "af_bella", speed=1.0, lang="en-us")
    samples = [int(max(-1.0, min(1.0, float(x))) * 32767) for x in audio.tolist()]
    return _make_wav_bytes(samples, sample_rate=sample_rate)


def _get_whisper_model() -> Any:
    from faster_whisper import WhisperModel

    device = os.environ.get("TURNSTONE_STT_DEVICE", "cpu").strip() or "cpu"
    compute_type = os.environ.get("TURNSTONE_STT_COMPUTE_TYPE", "int8").strip() or "int8"
    key = (_LOCAL_STT_MODEL, device, compute_type)
    model = _whisper_model_cache.get(key)
    if model is None:
        model = WhisperModel(_LOCAL_STT_MODEL, device=device, compute_type=compute_type)
        _whisper_model_cache[key] = model
    return model


def local_transcribe_audio(data: bytes, filename: str = "speech.webm") -> str:
    suffix = Path(filename or "speech.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        model = _get_whisper_model()
        segments, _info = model.transcribe(tmp_path, temperature=0.0, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
    finally:
        with contextlib.suppress(Exception):
            os.unlink(tmp_path)


def transcribe_audio(data: bytes, filename: str, selected_alias: str = "") -> TranscriptionResult:
    if prefer_local_backend(selected_alias, "TURNSTONE_PREFER_LOCAL_STT"):
        try:
            return TranscriptionResult(
                transcript=local_transcribe_audio(data, filename or "speech.webm"),
                backend="local",
                model_alias=selected_alias,
            )
        except Exception as exc:
            raise MediaBackendError(f"Local transcription backend failed: {exc}") from exc

    client = _openai_client_for_audio()
    if client is not None:
        try:
            up = io.BytesIO(data)
            up.name = filename or "speech.webm"
            resp = client.audio.transcriptions.create(model=_DEFAULT_STT_MODEL, file=up)
            return TranscriptionResult(
                transcript=(getattr(resp, "text", "") or "").strip(),
                backend="openai-compatible",
                model_alias=selected_alias,
            )
        except Exception as exc:
            raise MediaBackendError(f"Transcription backend failed: {exc}") from exc

    try:
        return TranscriptionResult(
            transcript=local_transcribe_audio(data, filename or "speech.webm"),
            backend="local",
            model_alias=selected_alias,
        )
    except Exception as exc:
        raise MediaBackendUnavailable(
            "Speech transcription is not configured. Set OPENAI_API_KEY or install a local STT backend."
        ) from exc


def synthesize_speech(text: str, voice: str, selected_alias: str = "") -> SpeechResult:
    if prefer_local_backend(selected_alias, "TURNSTONE_PREFER_LOCAL_TTS"):
        try:
            return SpeechResult(
                audio_bytes=local_tts_wav(text, voice=voice or "af_bella"),
                media_type="audio/wav",
                backend="kokoro-local",
                model_alias=selected_alias,
            )
        except Exception as exc:
            raise MediaBackendError(f"Local TTS backend failed: {exc}") from exc

    client = _openai_client_for_audio()
    if client is not None:
        try:
            resp = client.audio.speech.create(
                model=_DEFAULT_TTS_MODEL,
                voice=voice,
                input=text,
                format="wav",
            )
            data = resp.read() if hasattr(resp, "read") else bytes(resp.content)
            return SpeechResult(
                audio_bytes=data,
                media_type="audio/wav",
                backend="openai-compatible",
                model_alias=selected_alias,
            )
        except Exception as exc:
            raise MediaBackendError(f"TTS backend failed: {exc}") from exc

    try:
        return SpeechResult(
            audio_bytes=local_tts_wav(text, voice=voice or "af_bella"),
            media_type="audio/wav",
            backend="kokoro-local",
            model_alias=selected_alias,
        )
    except Exception:
        return SpeechResult(
            audio_bytes=fallback_tts_wav(text),
            media_type="audio/wav",
            backend="fallback",
            model_alias=selected_alias,
        )
