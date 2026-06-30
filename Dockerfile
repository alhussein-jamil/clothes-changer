# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml uv.lock README.md ./
COPY outfit_studio/__init__.py outfit_studio/__init__.py
RUN uv venv --python 3.10 /app/.venv && \
    uv pip install -e . --python /app/.venv/bin/python && \
    uv pip uninstall onnxruntime -y --python /app/.venv/bin/python 2>/dev/null || true && \
    uv pip install --reinstall onnxruntime-gpu \
      nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
      nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 \
      --python /app/.venv/bin/python

COPY . .

RUN useradd --create-home --uid 1000 appuser && \
    mkdir -p /app/data /app/outputs /app/models && \
    chown -R appuser:appuser /app

USER appuser

ENV PATH="/app/.venv/bin:$PATH" \
    OUTFIT_STUDIO_HOST=0.0.0.0 \
    OUTFIT_STUDIO_PORT=7860 \
    OUTFIT_STUDIO_DB_PATH=/app/data/database.db \
    OUTFIT_STUDIO_OUTPUT_DIR=/app/outputs \
    OUTFIT_STUDIO_MODELS_DIR=/app/models

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://127.0.0.1:7860/health || exit 1

CMD ["python", "-m", "outfit_studio.main"]
