# Stage 1: Build virtual environment using uv
FROM ghcr.io/astral-sh/uv:python3.12-alpine AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install dependencies using cached mounts
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Stage 2: Clean, small runtime image
FROM python:3.12-alpine
WORKDIR /app

# Copy virtual environment and source folders
COPY --from=builder /app/.venv /app/.venv
COPY app/ /app/app/
COPY corpus/ /app/corpus/

# Ensure virtual env binaries are in PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
