"""Tests for multimodal attachment handling and Nemotron-friendly extra params."""

from __future__ import annotations

from turnstone.core.attachments import Attachment
from turnstone.core.providers._openai_common import sanitize_messages
from turnstone.core.session import _encode_media_data_uri


def test_sanitize_messages_preserves_audio_and_video_parts() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,BBBB"}},
                {"type": "text", "text": "describe this"},
            ],
        }
    ]
    out = sanitize_messages(msgs)
    assert out == msgs


def test_encode_media_data_uri_uses_base64_data_url() -> None:
    got = _encode_media_data_uri(b"abc", "audio/wav")
    assert got == "data:audio/wav;base64,YWJj"


def test_storage_utils_convert_audio_and_video_to_content_parts() -> None:
    from turnstone.core.storage._utils import _attachment_to_content_part

    audio = _attachment_to_content_part(
        {
            "kind": "audio",
            "mime_type": "audio/wav",
            "content": b"abc",
        }
    )
    video = _attachment_to_content_part(
        {
            "kind": "video",
            "mime_type": "video/mp4",
            "content": b"xyz",
        }
    )
    assert audio == {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,YWJj"}}
    assert video == {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,eHl6"}}


def test_attachment_kind_helpers_cover_audio_and_video() -> None:
    audio = Attachment("a1", "voice.wav", "audio/wav", "audio", b"x")
    video = Attachment("v1", "clip.mp4", "video/mp4", "video", b"y")
    assert audio.is_audio is True
    assert audio.is_video is False
    assert video.is_video is True
    assert video.is_audio is False
