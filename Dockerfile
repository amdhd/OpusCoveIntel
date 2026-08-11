# syntax=docker/dockerfile:1
#
# Multi-stage build. Dependencies resolve in `builder`; the runtime stage copies
# only the virtualenv, so build tooling never reaches production.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer, cached independently of source so a code edit does not
# trigger a full reinstall.
#
# --locked installs exactly what uv.lock pins and fails the build if the lock
# no longer matches pyproject.toml, so the image built from a commit today is
# the image built from it in six months. (--frozen would install from the lock
# but ignore pyproject.toml, quietly omitting a dependency someone added
# without re-locking.) --no-install-project keeps this layer to third-party
# dependencies; the application is copied into the runtime stage below, which
# is what lets a code edit reuse this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project


# The client app, built in its own stage so Node never reaches the runtime
# image. `npm ci` installs exactly what package-lock.json pins, for the same
# reason `uv sync --locked` does.
FROM node:22-slim AS client

WORKDIR /client

COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY frontend/ ./
# The build references the served stylesheet, so the one copy of it that exists
# has to be here too (see the `styles` array in angular.json).
COPY app/web/static/app.css /app/web/static/app.css
RUN npm run build


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# curl is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Never run as root.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app/ ./app/
COPY --chown=app:app pyproject.toml ./
# Static output only -- no Node, no node_modules, no build tooling.
COPY --from=client --chown=app:app /client/dist /app/frontend/dist

RUN mkdir -p /app/var/storage && chown -R app:app /app/var

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
