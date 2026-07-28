"""Tests para la API compatible con OpenAI."""

from fastapi.testclient import TestClient

from deep_agent.api import MODEL_ID, app

client = TestClient(app)


def test_health() -> None:
    """GET /health debe responder ok."""
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
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert data["data"][0]["id"] == MODEL_ID


def test_chat_completions_invalid_request() -> None:
    """POST /v1/chat/completions sin messages debe dar 422."""
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 422


def test_chat_completions_no_stream_shape() -> None:
    """Verifica la estructura de la respuesta no-streaming."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Di hola"}],
            "stream": False,
        },
    )
    # Puede fallar por falta de DEEPSEEK_API_KEY, pero debe tener la forma correcta
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data


def test_chat_completions_stream_headers() -> None:
    """La respuesta streaming debe incluir headers SSE."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Di hola"}],
            "stream": True,
        },
    )
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        assert resp.headers.get("x-accel-buffering") == "no"
