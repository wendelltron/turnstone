"""Tests for POST /v1/api/workstreams/{ws_id}/attachments/{attachment_id}/evaluate."""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from tests.test_server_attachments_endpoints import PNG_1x1, _TEST_JWT_SECRET, _make_jwt


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_make_jwt(user_id)}"}


class _FakeConfigStore:
    def get(self, key: str, default: str = "") -> str:
        mapping = {
            "audio.vision_eval_model_alias": "vision-model",
            "audio.av_eval_model_alias": "av-model",
            "audio.intent_eval_model_alias": "intent-model",
        }
        return mapping.get(key, default)


def test_evaluate_attachment_endpoint_returns_structured_response(tmp_path):
    import sqlalchemy as sa

    import turnstone.server as srv_mod
    from turnstone.core.memory import register_workstream
    from turnstone.core.metrics import MetricsCollector
    from turnstone.core.model_registry import ModelConfig, ModelRegistry
    from turnstone.core.storage import init_storage, reset_storage
    from turnstone.core.storage._registry import get_storage
    from turnstone.core.storage._schema import workstreams as ws_tbl

    db_path = tmp_path / "test.db"
    reset_storage()
    init_storage("sqlite", path=str(db_path), run_migrations=False)
    srv_mod._metrics = MetricsCollector()
    register_workstream("ws-A", name="A")
    with get_storage()._conn() as conn:
        conn.execute(sa.update(ws_tbl).where(ws_tbl.c.ws_id == "ws-A").values(user_id="userA"))
        conn.commit()

    reg = ModelRegistry(
        models={
            "vision-model": ModelConfig("vision-model", "http://localhost:8000/v1", "dummy", "vision-model"),
            "av-model": ModelConfig("av-model", "http://localhost:8000/v1", "dummy", "av-model"),
            "intent-model": ModelConfig("intent-model", "http://localhost:8000/v1", "dummy", "intent-model"),
        },
        default="vision-model",
    )
    mock_mgr = MagicMock()
    mock_mgr.get.return_value = None
    mock_mgr.list_all.return_value = []
    mock_mgr.max_workstreams = 10
    app = srv_mod.create_app(
        workstreams=mock_mgr,
        global_queue=queue.Queue(),
        global_listeners=[],
        global_listeners_lock=threading.Lock(),
        skip_permissions=False,
        jwt_secret=_TEST_JWT_SECRET,
        registry=reg,
        config_store=_FakeConfigStore(),
    )
    client = TestClient(app, raise_server_exceptions=False)
    try:
        upload = client.post(
            "/v1/api/workstreams/ws-A/attachments",
            files={"file": ("tiny.png", PNG_1x1, "image/png")},
            headers=_auth("userA"),
        )
        aid = upload.json()["attachment_id"]
        fake = MagicMock()
        fake.role = "vision_eval"
        fake.model_alias = "vision-model"
        fake.backend = "openai-compatible"
        fake.content = '{"summary":"blue frame","addressing_computer":false}'
        fake.parsed = {"summary": "blue frame", "addressing_computer": False}
        with patch("turnstone.server._ws_session_from_request", return_value=None), patch(
            "turnstone.core.audio_routing.resolve_media_alias", return_value="vision-model"
        ), patch("turnstone.core.media_evaluator.evaluate_attachment", return_value=fake):
            resp = client.post(
                f"/v1/api/workstreams/ws-A/attachments/{aid}/evaluate",
                json={"role": "vision_eval"},
                headers=_auth("userA"),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "vision_eval"
        assert body["model_alias"] == "vision-model"
        assert body["parsed"]["summary"] == "blue frame"
    finally:
        client.close()
        reset_storage()
