"""Tests for GET /v1/api/models."""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock

from starlette.testclient import TestClient

_TEST_JWT_SECRET = "test-jwt-secret-minimum-32-chars!"


def _server_jwt() -> str:
    from turnstone.core.auth import JWT_AUD_SERVER, create_jwt

    return create_jwt(
        user_id="test-models-endpoint",
        scopes=frozenset({"read", "write", "approve", "service"}),
        source="test",
        secret=_TEST_JWT_SECRET,
        audience=JWT_AUD_SERVER,
    )


_SERVER_AUTH_HEADERS = {"Authorization": f"Bearer {_server_jwt()}"}


class _FakeConfigStore:
    def __init__(self, **values: str) -> None:
        self._values = values

    def get(self, key: str, default: str = "") -> str:
        return self._values.get(key, default)


def _make_registry():
    from turnstone.core.model_registry import ModelConfig, ModelRegistry

    return ModelRegistry(
        models={
            "default": ModelConfig(
                "default",
                "http://localhost:8000/v1",
                "dummy",
                "text-model",
                capabilities={"supports_text_eval": True},
            ),
            "vision": ModelConfig(
                "vision",
                "http://localhost:8000/v1",
                "dummy",
                "vision-model",
                capabilities={"supports_vision": True},
            ),
            "speech": ModelConfig(
                "speech",
                "http://localhost:8000/v1",
                "dummy",
                "speech-model",
                capabilities={
                    "supports_transcription": True,
                    "supports_speech_synthesis": True,
                },
            ),
        },
        default="default",
    )


def test_models_endpoint_includes_capabilities_media_roles_and_defaults():
    from turnstone.server import create_app

    mock_mgr = MagicMock()
    mock_mgr.list_all.return_value = []
    mock_mgr.max_workstreams = 10

    app = create_app(
        workstreams=mock_mgr,
        global_queue=queue.Queue(),
        global_listeners=[],
        global_listeners_lock=threading.Lock(),
        skip_permissions=False,
        jwt_secret=_TEST_JWT_SECRET,
        registry=_make_registry(),
        config_store=_FakeConfigStore(
            **{
                "model.default_alias": "vision",
                "channels.default_model_alias": "speech",
            }
        ),
    )
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/v1/api/models", headers=_SERVER_AUTH_HEADERS)
    finally:
        client.close()

    assert resp.status_code == 200
    body = resp.json()
    assert body["default_alias"] == "vision"
    assert body["channel_default_alias"] == "speech"

    models = {item["alias"]: item for item in body["models"]}

    assert models["default"]["capabilities"] == {"supports_text_eval": True}
    assert models["default"]["media_roles"] == {
        "stt": False,
        "tts": False,
        "vision_eval": False,
        "av_eval": False,
        "intent_eval": True,
    }
    assert models["vision"]["media_roles"]["vision_eval"] is True
    assert models["speech"]["media_roles"]["stt"] is True
    assert models["speech"]["media_roles"]["tts"] is True
