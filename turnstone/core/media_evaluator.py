"""Multimodal evaluator helpers for image/audio/video attachments."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any

from turnstone.core.server_compat import merge_server_compat


@dataclass(frozen=True)
class EvaluationResult:
    role: str
    model_alias: str
    backend: str
    content: str
    parsed: dict[str, Any]


def _data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _default_prompt(kind: str, role: str) -> str:
    if role == "intent_eval":
        return (
            "Return a compact JSON object describing whether this media suggests the user is "
            "addressing the computer. Include keys: addressing_computer (boolean), "
            "wake_relevant (boolean), summary (string), confidence (number 0-1)."
        )
    if kind == "image":
        return (
            "Return a compact JSON object summarizing the scene. Include keys: modality, "
            "summary, people_present, gaze_direction, notable_objects."
        )
    if kind == "audio":
        return (
            "Return a compact JSON object that describes or transcribes this audio. "
            "Include keys: modality, transcript, contains_speech, summary."
        )
    if kind == "video":
        return (
            "Return a compact JSON object that summarizes this clip. Include keys: modality, "
            "summary, contains_audio, addressing_computer, wake_relevant."
        )
    return "Return a compact JSON object summarizing this attachment."


def _content_parts_for_attachment(row: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(row.get("kind") or "")
    mime = str(row.get("mime_type") or "application/octet-stream")
    raw = row.get("content") or b""
    url = _data_url(raw, mime)
    if kind == "image":
        return [{"type": "image_url", "image_url": {"url": url}}]
    if kind == "audio":
        return [{"type": "audio_url", "audio_url": {"url": url}}]
    if kind == "video":
        return [{"type": "video_url", "video_url": {"url": url}}]
    if kind == "text":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = "[unreadable text attachment]"
        return [{"type": "text", "text": text}]
    return [{"type": "text", "text": "[unsupported attachment]"}]


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _best_effort_parse_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    candidates = [raw]
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        candidates.insert(0, m.group(1))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {"value": obj}
        except Exception:
            continue
    return {}


def evaluate_attachment(
    *,
    registry: Any,
    model_alias: str,
    row: dict[str, Any],
    role: str,
    prompt: str = "",
    include_audio_in_video: bool = False,
) -> EvaluationResult:
    client, model_name, cfg = registry.resolve(model_alias)
    provider = registry.get_provider(model_alias)
    kind = str(row.get("kind") or "")
    parts = _content_parts_for_attachment(row)
    parts.append({"type": "text", "text": prompt.strip() or _default_prompt(kind, role)})
    messages = [{"role": "user", "content": parts}]

    temperature = 0.6
    reasoning_effort = "low"
    if kind in {"audio", "video"}:
        temperature = 0.0
        reasoning_effort = "none"

    extra_params = None
    if provider.provider_name == "openai-compatible":
        extra_params = merge_server_compat({"reasoning_effort": reasoning_effort}, cfg.server_compat)
        ctk = extra_params.setdefault("chat_template_kwargs", {})
        if kind in {"audio", "video"}:
            ctk.setdefault("enable_thinking", False)
        if kind == "video" and include_audio_in_video:
            mm = extra_params.setdefault("mm_processor_kwargs", {})
            mm.setdefault("use_audio_in_video", True)

    result = provider.create_completion(
        client=client,
        model=model_name,
        messages=messages,
        max_tokens=1024,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        extra_params=extra_params,
    )
    return EvaluationResult(
        role=role,
        model_alias=model_alias,
        backend=provider.provider_name,
        content=result.content or "",
        parsed=_best_effort_parse_json(result.content or ""),
    )
