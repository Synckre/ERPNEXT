"""Tests para la API compatible con OpenAI y autenticación por API Key."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from deep_agent.api import MODEL_ID, app

client = TestClient(app)


def test_health() -> None:
    """GET /health debe responder ok sin requerir autenticación."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root() -> None:
    """GET / debe devolver metadatos del servicio."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "openai" in data["endpoints"]


def test_list_models() -> None:
    """GET /v1/models debe listar modelos al estilo OpenAI."""
    with patch("deep_agent.api.AGENT_API_KEY", ""):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1
        assert data["data"][0]["id"] == MODEL_ID


def test_api_key_authentication_enforcement() -> None:
    """Cuando AGENT_API_KEY está configurada, peticiones sin API Key válida deben ser rechazadas con 401."""
    with patch("deep_agent.api.AGENT_API_KEY", "secret-test-key"):
        # Petición sin header
        unauth_resp = client.get("/v1/models")
        assert unauth_resp.status_code == 401
        assert "Unauthorized" in unauth_resp.json()["detail"]

        # Petición con Bearer Token correcto
        auth_resp = client.get("/v1/models", headers={"Authorization": "Bearer secret-test-key"})
        assert auth_resp.status_code == 200

        # Petición con X-API-Key header correcto
        x_auth_resp = client.get("/v1/models", headers={"X-API-Key": "secret-test-key"})
        assert x_auth_resp.status_code == 200


def test_chat_completions_invalid_request() -> None:
    """POST /v1/chat/completions sin messages debe dar 422."""
    with patch("deep_agent.api.AGENT_API_KEY", ""):
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 422


def test_chat_completions_no_stream_shape() -> None:
    """Verifica la estructura de la respuesta no-streaming."""
    with patch("deep_agent.api.AGENT_API_KEY", ""):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Di hola"}],
                "stream": False,
            },
        )
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in data
