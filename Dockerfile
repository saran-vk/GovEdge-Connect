FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install runtime system packages (ffmpeg required for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project configuration and files
COPY pyproject.toml uv.lock* README.md ./
COPY gateway/ ./gateway/
COPY services/ ./services/
COPY middleware/ ./middleware/
COPY eval/ ./eval/
COPY models/ ./models/
COPY web-ui/ ./web-ui/
COPY data/ ./data/

# Install application and all dependencies
RUN uv pip install --system .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
