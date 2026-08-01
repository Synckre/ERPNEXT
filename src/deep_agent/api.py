"""API compatible con OpenAI — para usar con Open WebUI, Cursor, etc."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
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

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Inicializa el checkpointer si es Postgres (crea las tablas necesarias).

    MemorySaver no necesita setup; AsyncPostgresSaver sí (`.setup()` crea las
    tablas de checkpoints). Se detecta por la presencia del método.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    setup = getattr(checkpointer, "setup", None)
    if setup is not None:
        result = setup()
        if inspect.isawaitable(result):
            await result
    yield


app = FastAPI(
    title="DeepSeek Agent API (OpenAI-compatible)",
    version="0.1.0",
    description="Endpoint compatible con OpenAI / Open WebUI",
    lifespan=lifespan,
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
    # Open WebUI no manda un session/chat id nativo en /v1/chat/completions.
    # `user` (campo estándar de OpenAI) y `metadata` son opcionales: si vienen
    # poblados se usan para derivar un thread_id más estable (ver
    # _derive_thread_id).
    user: str | None = None
    metadata: dict | None = None


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
    """Chat completions — compatible con OpenAI y Open WebUI.

    El `thread_id` se deriva del historial / `user` / `metadata` (ver
    _derive_thread_id) porque Open WebUI no manda un session id nativo en el
    protocolo OpenAI. Antes de invocar el grafo se comprueba si ese thread quedó
    pausado en un `__interrupt__` (Human-in-the-loop); si es así, el último
    mensaje del usuario se interpreta como la respuesta a la aprobación y se
    reanuda con `Command(resume=...)` en lugar de arrancar un grafo nuevo.
    """
    # ── Identificar la conversación y su estado ─────────────────────────
    thread_id = _derive_thread_id(body.messages, user=body.user, metadata=body.metadata)
    config = {"configurable": {"thread_id": thread_id}}

    # Comprobar el interrupt ANTES de reconstruir langchain_messages: si el
    # grafo quedó pausado esperando aprobación, el estado del thread ya contiene
    # el historial; el último mensaje del usuario es la decisión de la persona.
    pending, prior_messages = await _thread_state(graph, config)

    resume_value: dict | None = None
    if pending is not None:
        last_user_text = body.messages[-1].content if body.messages else ""
        resume_value = _build_resume_value(last_user_text, pending)

    langchain_messages = _to_langchain_messages(body.messages)

    # Sin interrupt: si el thread ya tiene historial guardado en el checkpointer,
    # sólo enviamos el último mensaje del body (el input nuevo). Reenviar todo el
    # historial duplicaría los mensajes en el estado del grafo en cada turno.
    if resume_value is None and prior_messages:
        graph_messages = [langchain_messages[-1]] if langchain_messages else []
    else:
        graph_messages = langchain_messages

    if body.stream:
        return StreamingResponse(
            _stream_chat(
                body.model,
                graph_messages,
                config=config,
                resume_value=resume_value,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "x-accel-buffering": "no",
            },
        )

    # ── No streaming ─────────────────────────────────────────────────────
    try:
        if resume_value is not None:
            graph_input: dict | Command = Command(resume=resume_value)
        else:
            graph_input = {"messages": graph_messages}
        # SIEMPRE pasamos config con thread_id para conservar el estado del grafo.
        result = await graph.ainvoke(graph_input, config=config)
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
    *,
    config: dict,
    resume_value: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Genera SSE en formato OpenAI (compatible con Open WebUI).

    - Siempre pasa `config` con `thread_id` para conservar el estado del grafo
      y poder reanudar interrupts (Human-in-the-loop).
    - Si `resume_value` no es None, se reanuda un grafo pausado en lugar de
      empezar una invocación nueva.
    - Además del texto del modelo se emiten eventos de progreso para que las
      tool calls sean visibles (on_tool_start / on_tool_end) y se avisa cuando
      la conversación queda pausada esperando aprobación.
    """
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
    notified_tool_call = False
    tool_names_seen: set[str] = set()

    graph_input: dict | Command = (
        Command(resume=resume_value)
        if resume_value is not None
        else {"messages": messages}
    )

    try:
        async for event in graph.astream_events(
            graph_input, config=config, version="v2"
        ):
            event_name = event.get("event")

            if event_name == "on_chat_model_stream":
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
                    continue

                # Cuando el modelo genera una tool call, `content` viene vacío y
                # los argumentos están en `tool_call_chunks`: sin esto el usuario
                # no vería señal de que el agente está trabajando.
                for tool_chunk in getattr(chunk, "tool_call_chunks", None) or []:
                    tool_name = tool_chunk.get("name")
                    if tool_name and tool_name not in tool_names_seen:
                        tool_names_seen.add(tool_name)
                        yield _sse_chunk(
                            completion_id,
                            created,
                            model,
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": f"\n🔧 Preparando `{tool_name}`...\n"
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )
                    elif (
                        not tool_name
                        and not tool_names_seen
                        and not notified_tool_call
                    ):
                        # Algunos modelos no incluyen el nombre hasta chunks
                        # posteriores: avisamos una sola vez de forma genérica.
                        notified_tool_call = True
                        yield _sse_chunk(
                            completion_id,
                            created,
                            model,
                            choices=[
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": "\n🔧 El agente está preparando una herramienta...\n"
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        )

            elif event_name == "on_tool_start":
                tool_name = event.get("name", "herramienta")
                yield _sse_chunk(
                    completion_id,
                    created,
                    model,
                    choices=[
                        {
                            "index": 0,
                            "delta": {
                                "content": f"\n🔧 Ejecutando `{tool_name}`...\n"
                            },
                            "finish_reason": None,
                        }
                    ],
                )

            elif event_name == "on_tool_end":
                tool_name = event.get("name", "herramienta")
                output = event["data"].get("output")
                brief = _format_tool_output(tool_name, output)
                if brief:
                    yield _sse_chunk(
                        completion_id,
                        created,
                        model,
                        choices=[
                            {
                                "index": 0,
                                "delta": {"content": brief},
                                "finish_reason": None,
                            }
                        ],
                    )
    except GraphInterrupt:
        # Defensivo: con checkpointer el stream suele cerrarse limpiamente al
        # llegar a un __interrupt__, pero si el runtime lanza GraphInterrupt lo
        # tratamos como pausa (no como error) y dejamos que el chequeo posterior
        # emita el aviso de aprobación.
        pass
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

    # Tras terminar el stream, comprobar si el grafo quedó pausado en un
    # `__interrupt__`. astream_events no lanza excepción en ese caso: el run
    # termina normal y el estado queda guardado en el checkpointer, así que
    # lo detectamos consultando el estado final del thread.
    pending = await _pending_interrupt(graph, config)
    if pending is not None:
        yield _sse_chunk(
            completion_id,
            created,
            model,
            choices=[
                {
                    "index": 0,
                    "delta": {"content": _format_interrupt_message([pending])},
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


# ──────────────────────────────────────────────
#  Thread_id / Human-in-the-loop
# ──────────────────────────────────────────────


def _stable_id(text: str) -> str:
    """Hash SHA-256 corto para usar como thread_id estable."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _history_fingerprint(messages: list[ChatMessage]) -> str:
    """Fingerprint estable del historial de la conversación.

    Por qué este método: Open WebUI no envía un session/chat id nativo en el
    body de /v1/chat/completions, así que no podemos fiarnos de un campo de
    sesión. En su lugar usamos un hash de los primeros N mensajes: dentro de
    una conversación el cliente manda el historial acumulado, por lo que esos
    primeros mensajes son idénticos turno a turno (thread_id estable para poder
    reanudar interrupts), mientras que dos conversaciones distintas casi seguro
    difieren (thread_id distinto). N=5 equilibra estabilidad y unicidad.
    """
    N = 5
    sample = "\n".join(f"{m.role}:{m.content}" for m in messages[:N])
    if not sample:
        # Sin historial no hay conversación que reanudar: id único para no
        # mezclar chats vacíos.
        return f"empty:{uuid.uuid4().hex}"
    return _stable_id(sample)


def _derive_thread_id(
    messages: list[ChatMessage],
    *,
    user: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Deriva un thread_id estable por conversación.

    Orden de preferencia:
    1. Un id explícito de sesión/hilo si el cliente lo envía en `metadata`
       (thread_id / chat_id / session_id).
    2. `user` (campo estándar de OpenAI) combinado con el fingerprint del
       historial: separa conversaciones del mismo usuario y sigue siendo
       estable dentro de una misma conversación.
    3. Fallback: hash de los primeros mensajes (ver _history_fingerprint).

    Es un workaround por la falta de session id nativo en el protocolo OpenAI.
    """
    for key in ("thread_id", "chat_id", "session_id"):
        value = (metadata or {}).get(key)
        if value:
            return _stable_id(f"meta:{key}:{value}")

    fingerprint = _history_fingerprint(messages)
    if user:
        return _stable_id(f"user:{user}:{fingerprint}")
    return fingerprint


def _interrupt_value(interrupt) -> dict | str:
    """Normaliza un elemento de `__interrupt__` a su valor (dict o str)."""
    return interrupt.value if hasattr(interrupt, "value") else interrupt


async def _thread_state(
    graph, config: dict
) -> tuple[dict | str | None, list | None]:
    """Devuelve (interrupt pendiente, mensajes previos del thread).

    Una sola lectura del checkpointer (aget_state) sirve para saber si el grafo
    quedó pausado en un `__interrupt__` esperando aprobación y si ya hay
    historial guardado en el thread (para no duplicar mensajes al reenviar el
    historial completo en cada request).

    Nota: la clave `__interrupt__` NO está en `snapshot.values` al consultar con
    `aget_state()` — sólo aparece en el dict que devuelve `graph.ainvoke()`
    cuando el grafo se pausa en ese mismo call. El interrupt pendiente vive en
    `snapshot.tasks`, en el atributo `.interrupts` de cada `PregelTask` (cada
    `Interrupt` tiene `.value` con el HITLRequest: `action_requests`/`review_configs`).
    """
    try:
        snapshot = await graph.aget_state(config)
    except Exception:
        return None, None
    if snapshot is None:
        return None, None

    # Leer los interrupts pendientes desde las tasks (ver nota del docstring).
    pending = None
    for task in snapshot.tasks or ():
        task_interrupts = getattr(task, "interrupts", None)
        if task_interrupts:
            pending = _interrupt_value(task_interrupts[0])
            break

    values = snapshot.values or {}
    return pending, values.get("messages")


async def _pending_interrupt(graph, config: dict) -> dict | str | None:
    """Devuelve el HITLRequest pendiente del thread, o None si no hay interrupt."""
    pending, _ = await _thread_state(graph, config)
    return pending


_REJECT_KEYWORDS = (
    "no", "nop", "rechaza", "rechazar", "rechazado", "cancela", "cancelar",
    "cancelado", "niega", "negar", "denegar", "detener", "para", "alto",
)


def _parse_hitl_decision(text: str) -> dict:
    """Interpreta el último mensaje del usuario como decisión HITL.

    - Palabras de rechazo → {"type": "reject", "message": <texto>}: el modelo
      recibe el motivo (vía ToolMessage) y no ejecuta la herramienta.
    - Cualquier otra cosa (sí, ok, instrucciones…) → {"type": "approve"}.
      Se asume consentimiento por defecto: el protocolo HITL de deepagents sólo
      permite approve/edit/reject, y parsear instrucciones libres a un edit
      estructurado no es fiable.

    Se usan límites de palabra (`\\b...\\b`) para evitar falsos positivos
    (p. ej. "nota" contiene "no" pero no es un rechazo).
    """
    norm = re.sub(r"[^\w\sáéíóúñüÁÉÍÓÚÑÜ]", " ", str(text or "").lower())
    for keyword in _REJECT_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", norm):
            return {"type": "reject", "message": str(text or "")}
    return {"type": "approve"}


def _build_resume_value(user_text: str, hitl_request) -> dict:
    """Construye el valor para Command(resume=...).

    HumanInTheLoopMiddleware espera {"decisions": [...]} con una decisión por
    tool call en espera; replicamos la decisión para cubrir varias llamadas.
    """
    decision = _parse_hitl_decision(user_text)
    if isinstance(hitl_request, dict):
        count = len(hitl_request.get("action_requests") or []) or 1
    else:
        count = 1
    return {"decisions": [decision] * count}


def _format_interrupt_message(interrupts: list) -> str:
    """Mensaje claro cuando la conversación queda pausada por un interrupt."""
    lines = [
        "⏸️ Conversación en pausa — esperando tu aprobación (Human-in-the-loop).",
        "",
        "Responde «sí» para aprobar y continuar, o «no» para rechazar.",
        "",
        "Acciones pendientes de aprobación:",
    ]
    for interrupt in interrupts:
        value = _interrupt_value(interrupt)
        if not isinstance(value, dict):
            lines.append(f"• {value}")
            continue
        for request in value.get("action_requests", []):
            name = request.get("name", "herramienta")
            args = request.get("args", {})
            description = request.get("description")
            lines.append(
                f"• 🔧 `{name}` con datos: {json.dumps(args, ensure_ascii=False)}"
            )
            if description:
                lines.append(f"  └ {description}")
    return "\n".join(lines)


def _format_tool_output(name: str, output) -> str | None:
    """Resumen corto del resultado de una herramienta para el stream."""
    if output is None:
        return None
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        text = " ".join(text.split())[:200]
        return f"  ✓ `{name}` → {text}\n"
    if isinstance(output, dict):
        text = json.dumps(output, ensure_ascii=False)[:200]
        return f"  ✓ `{name}` → {text}\n"
    return f"  ✓ `{name}` completado\n"


def _last_assistant_message(messages: list, state_result: dict | None = None) -> str:
    """Extrae el último mensaje del asistente o describe el estado.

    Si el grafo volvió a interrumpirse (p. ej. un segundo tool que también
    requiere aprobación), state_result contiene `__interrupt__`: en ese caso no
    hay una respuesta final normal, así que se comunica claramente que la
    conversación quedó en pausa esperando aprobación (ver
    _format_interrupt_message).
    """
    import json

    if state_result and isinstance(state_result, dict) and state_result.get("__interrupt__"):
        return _format_interrupt_message(state_result["__interrupt__"])

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
