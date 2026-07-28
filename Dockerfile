# Dockerfile para desplegar el Agente ERPNext en Coolify / Docker / Cloud
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Optimizaciones de compilación de bytecode y caché de uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copiar archivos de dependencias primero para optimizar la caché de Docker
COPY pyproject.toml uv.lock ./

# Instalar dependencias sin incluir el código fuente aún
RUN uv sync --frozen --no-dev --no-install-project

# Copiar el resto del código del proyecto
COPY src/ ./src/
COPY README.md Makefile langgraph.json ./

# Instalar el paquete principal
RUN uv sync --frozen --no-dev

# Etapa final de producción liviana
FROM python:3.12-slim

WORKDIR /app

# Copiar la aplicación e instalacion virtual de Python de la etapa anterior
COPY --from=builder /app /app

# Asegurar que los binarios del entorno virtual estén en el PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Comando de inicio del servidor API compatible con OpenAI
CMD ["uvicorn", "deep_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
