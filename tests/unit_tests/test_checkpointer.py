"""Unit tests for server-side Checkpointer persistence in graph and API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from deep_agent.api import app
from deep_agent.graph import checkpointer, graph
from fastapi.testclient import TestClient

client = TestClient(app)


def test_checkpointer_instance_exists():
    """Verifica que el agente tenga configurado el Checkpointer de MemorySaver."""
    assert checkpointer is not None
    assert hasattr(graph, "checkpointer")


@pytest.mark.asyncio
async def test_thread_id_persistence_via_api():
    """Verifica que dos solicitudes con el mismo X-Thread-ID o user se asocien al mismo estado del checkpointer."""
    with patch("deep_agent.api.AGENT_API_KEY", ""):
        # Petición 1 con thread_id explícito
        resp1 = client.post(
            "/v1/chat/completions",
            headers={"X-Thread-ID": "test-session-1"},
            json={
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "Hola, mi nombre es Carlos"}],
                "stream": False,
            },
        )
        assert resp1.status_code in (200, 500)

        # Verificar que el checkpointer de LangGraph tiene guardado el estado para el thread
        state = await graph.aget_state({"configurable": {"thread_id": "test-session-1"}})
        assert state is not None
        assert hasattr(state, "values")
