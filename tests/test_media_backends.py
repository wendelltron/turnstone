"""Tests for media backend helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_synthesize_speech_uses_response_format_not_format() -> None:
    from turnstone.core.media_backends import synthesize_speech

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"RIFFdemo"
    mock_client.audio.speech.create.return_value = mock_resp

    with patch("turnstone.core.media_backends._openai_client_for_audio", return_value=mock_client), patch(
        "turnstone.core.media_backends.prefer_local_backend", return_value=False
    ):
        out = synthesize_speech("hello", "alloy", selected_alias="")

    kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert kwargs["response_format"] == "wav"
    assert "format" not in kwargs
    assert out.audio_bytes == b"RIFFdemo"
    assert out.backend == "openai-compatible"
