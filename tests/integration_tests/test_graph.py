import os

import pytest

from deep_agent.graph import graph

pytestmark = pytest.mark.anyio

if not os.getenv("DEEPSEEK_API_KEY"):
    pytest.skip(
        "Set DEEPSEEK_API_KEY to run integration tests.", allow_module_level=True
    )


async def test_deep_agent_smoke() -> None:
    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello in one sentence.",
                }
            ]
        }
    )
    assert result is not None
    assert result.get("messages")
