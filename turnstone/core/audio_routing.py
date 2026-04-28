"""Helpers for speech / media-evaluator routing.

Keeps per-workstream media model override resolution in one place so UI,
server endpoints, and future provider adapters all follow the same rules.
"""

from __future__ import annotations

from typing import Any


def resolve_media_alias(
    *,
    session: Any | None = None,
    config_store: Any | None = None,
    role: str,
) -> str:
    """Return the effective alias for a media/evaluator *role*.

    Resolution order:
    1. Per-session override attribute on ``ChatSession``.
    2. Global ConfigStore default.
    3. Empty string when unresolved.
    """
    attr_map = {
        "stt": "_stt_model_alias",
        "tts": "_tts_model_alias",
        "vision_eval": "_vision_eval_model_alias",
        "av_eval": "_av_eval_model_alias",
        "intent_eval": "_intent_eval_model_alias",
    }
    setting_map = {
        "stt": "audio.stt_model_alias",
        "tts": "audio.tts_model_alias",
        "vision_eval": "audio.vision_eval_model_alias",
        "av_eval": "audio.av_eval_model_alias",
        "intent_eval": "audio.intent_eval_model_alias",
    }
    attr = attr_map.get(role, "")
    if attr and session is not None:
        value = str(getattr(session, attr, "") or "").strip()
        if value:
            return value
    key = setting_map.get(role, "")
    if key and config_store is not None:
        value = str(config_store.get(key) or "").strip()
        if value:
            return value
    return ""


def media_role_capability_flags(role: str) -> tuple[str, ...]:
    """Return capability keys that suggest a model is suitable for *role*."""
    return {
        "stt": ("supports_transcription", "supports_audio_eval"),
        "tts": ("supports_speech_synthesis",),
        "vision_eval": ("supports_vision",),
        "av_eval": ("supports_audio_eval", "supports_video_eval", "supports_vision"),
        "intent_eval": ("supports_audio_eval", "supports_text_eval"),
    }.get(role, ())
