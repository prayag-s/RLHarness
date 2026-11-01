# RL-Native Agent Studio — container image
#
# The API executes model-generated Python, so bubblewrap is not optional here: the image sets
# RL_IDE_REQUIRE_ISOLATION=1 and refuses to start where user namespaces are unavailable.
#
# Bubblewrap needs unprivileged user namespaces inside the container. Run with either
#   docker run --security-opt seccomp=unconfined ...
# or the equivalent profile for your orchestrator, otherwise the API will (correctly) refuse to boot.
#
# Build:
#   docker build -t rl-studio:latest .
# Run:
#   docker run -p 8000:8000 -v rl-studio-data:/data \
#     -e DEEPSEEK_API_KEY=... rl-studio:latest

FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime

# `node` is a runtime dependency, not just a build tool: the arena generator validates the
# JavaScript it wrote before accepting it, and without an engine that check fails closed — the
# document is discarded after the provider has already been paid for it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RL_IDE_REQUIRE_ISOLATION=1 \
    RL_IDE_WORKSPACE_DIR=/data/workspaces \
    RL_IDE_FRONTEND_DIST=/app/frontend_dist

WORKDIR /app

# Pinned, resolved dependency set — the same one the test suite runs against.
COPY backend/requirements.lock ./requirements.lock
RUN pip install -r requirements.lock

COPY backend/ ./
COPY --from=frontend /app/frontend/dist /app/frontend_dist

RUN useradd --create-home --uid 10001 studio \
    && mkdir -p /data/workspaces /app/runs /app/workspaces \
    && chown -R studio:studio /data /app

USER studio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v4/settings/llm', timeout=4)"

CMD ["python", "-m", "uvicorn", "rl_ide.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
