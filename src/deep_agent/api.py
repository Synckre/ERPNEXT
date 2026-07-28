"""API compatible con OpenAI — para usar con Open WebUI, Cursor, etc."""

from __future__ import annotations

import os
import time
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from deep_agent.graph import graph  # noqa: E402

MODEL_ID = os.getenv("DEEP_AGENT_MODEL", "deepseek:deepseek-v4-flash").removeprefix(
    "deepseek:"
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(
    title="DeepSeek Agent API (OpenAI-compatible)",
    version="0.1.0",
    description="Endpoint compatible con OpenAI / Open WebUI",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Schemas (OpenAI format)
# ──────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "deepseek"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ──────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "service": "DeepSeek Agent API (OpenAI-compatible)",
        "endpoints": {
            "openai": {
                "models": "GET /v1/models",
                "chat": "POST /v1/chat/completions",
            },
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Lista los modelos disponibles (compatible con OpenAI)."""
    return ModelList(data=[ModelInfo(id=MODEL_ID)])


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    """Chat completions — compatible con OpenAI y Open WebUI."""
    langchain_messages = _to_langchain_messages(body.messages)

    if body.stream:
        return StreamingResponse(
            _stream_chat(body.model, langchain_messages),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-accel-buffering": "no",
            },
        )

    # ── No streaming ──────────────────────
    try:
        result = await graph.ainvoke({"messages": langchain_messages})
        assistant_msg = _last_assistant_message(result.get("messages", []), result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_msg,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": _estimate_tokens(langchain_messages),
            "completion_tokens": _estimate_tokens([assistant_msg])
            if assistant_msg
            else 0,
            "total_tokens": 0,
        },
    }


# ──────────────────────────────────────────────
#  Streaming
# ──────────────────────────────────────────────


async def _stream_chat(
    model: str,
    messages: list,
) -> AsyncGenerator[str, None]:
    """Genera SSE en formato OpenAI."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # Chunk inicial con el rol
    yield _sse_chunk(
        completion_id,
        created,
        model,
        choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    )

    full_content = ""

    try:
        async for event in graph.astream_events(
            {"messages": messages}, version="v2"
        ):
            if event.get("event") == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content or ""
                if content:
                    full_content += content
                    yield _sse_chunk(
                        completion_id,
                        created,
                        model,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    )
    except Exception:
        yield _sse_chunk(
            completion_id,
            created,
            model,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
        )
        yield "data: [DONE]\n\n"
        return

    # Chunk final
    yield _sse_chunk(
        completion_id,
        created,
        model,
        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
    )
    yield "data: [DONE]\n\n"


def _sse_chunk(completion_id: str, created: int, model: str, choices: list) -> str:
    """Helper para formatear un chunk SSE en formato OpenAI."""
    import json

    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": choices,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────


def _to_langchain_messages(
    msgs: list[ChatMessage],
) -> list:
    """Convierte mensajes OpenAI → LangChain."""
    result = []
    for m in msgs:
        match m.role:
            case "system":
                result.append(SystemMessage(content=m.content))
            case "assistant":
                result.append(AIMessage(content=m.content))
            case _:  # user, human, etc.
                result.append(HumanMessage(content=m.content))
    return result


def _last_assistant_message(messages: list, state_result: dict | None = None) -> str:
    """Extrae el último mensaje del asistente de la lista o describe las llamadas a herramientas."""
    import json

    if state_result and isinstance(state_result, dict) and "__interrupt__" in state_result:
        interrupts = state_result["__interrupt__"]
        return f"Acción pausada para aprobación humana (Human-in-the-loop): {interrupts}"

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) or (hasattr(msg, "type") and getattr(msg, "type", None) == "ai"):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                actions = []
                for tc in tool_calls:
                    tool_name = tc.get("name", "herramienta")
                    args = tc.get("args", {})
                    actions.append(
                        f"• Solicitando ejecutar '{tool_name}' con datos: {json.dumps(args, ensure_ascii=False)}"
                    )
                return "El agente ha preparado las siguientes operaciones en ERPNext:\n" + "\n".join(actions)

    return "Operación procesada por el agente."


def _estimate_tokens(messages: list | str) -> int:
    """Estimación simple de tokens (~4 chars por token)."""
    if isinstance(messages, list):
        text = " ".join(getattr(m, "content", str(m)) for m in messages)
    else:
        text = messages if isinstance(messages, str) else str(messages)
    return len(text) // 4
