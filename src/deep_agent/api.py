"""API compatible con OpenAI — para usar con Open WebUI, Cursor, etc."""

from __future__ import annotations

import os
import time
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

from deep_agent.graph import graph  # noqa: E402

MODEL_ID = os.getenv("DEEP_AGENT_MODEL", "deepseek:deepseek-v4-flash").removeprefix(
    "deepseek:"
)

AGENT_API_KEY = os.getenv("AGENT_API_KEY", "").strip()

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

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
security_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    header_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(security_bearer),
) -> str | None:
    """Valida la API Key vía 'Authorization: Bearer <key>' o 'X-API-Key: <key>'."""
    if not AGENT_API_KEY:
        return None

    provided_key = None
    if bearer and bearer.credentials:
        provided_key = bearer.credentials
    elif header_key:
        provided_key = header_key

    if not provided_key or provided_key != AGENT_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid API Key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return provided_key


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
    user: str | None = None


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


@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def list_models():
    """Lista los modelos disponibles (compatible con OpenAI)."""
    return ModelList(data=[ModelInfo(id=MODEL_ID)])


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(
    body: ChatCompletionRequest,
    x_thread_id: str | None = Header(default=None, alias="X-Thread-ID"),
):
    """Chat completions — compatible con OpenAI y Open WebUI con persistencia por hilo."""
    langchain_messages = _to_langchain_messages(body.messages)
    thread_id = x_thread_id or body.user or "default_thread"
    config = {"configurable": {"thread_id": thread_id}}

    if body.stream:
        return StreamingResponse(
            _stream_chat(body.model, langchain_messages, config=config),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-accel-buffering": "no",
            },
        )

    # ── No streaming ──────────────────────
    try:
        result = await graph.ainvoke({"messages": langchain_messages}, config=config)
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
    config: dict | None = None,
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
            {"messages": messages}, version="v2", config=config
        ):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk:
                    content = getattr(chunk, "content", "") or ""
                    if isinstance(content, str) and content:
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
    except Exception as exc:
        err_msg = f"\n\n⚠️ **Aviso del Agente**: La operación requiere interacción o fue interrumpida ({exc})."
        yield _sse_chunk(
            completion_id,
            created,
            model,
            choices=[
                {
                    "index": 0,
                    "delta": {"content": err_msg},
                    "finish_reason": "stop",
                }
            ],
        )
        yield "data: [DONE]\n\n"
        return

    # Si el stream finalizó sin contenido explícito de texto, enviar mensaje informativo
    if not full_content:
        fallback_msg = "Operación procesada correctamente por el agente."
        yield _sse_chunk(
            completion_id,
            created,
            model,
            choices=[
                {
                    "index": 0,
                    "delta": {"content": fallback_msg},
                    "finish_reason": None,
                }
            ],
        )

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
