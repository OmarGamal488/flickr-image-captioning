# syntax=docker/dockerfile:1.7
# ----------------------------------------------------------------------------
# Flickr8k Image Captioning — inference-only image (CPU torch wheels).
#
# Serves BOTH the FastAPI microservice (port 8000) and the Gradio demo
# (port 7860). The two services share the same image — docker-compose.yml
# overrides the CMD to pick one or the other.
#
# Why CPU torch? Image is ~3 GB instead of ~8 GB for cu121, and the attention
# model does a single-image forward pass on CPU in ~150 ms — more than fast
# enough for an interactive demo.
# ----------------------------------------------------------------------------

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CHECKPOINT_PATH=/app/models/attention_lstm.pth \
    VOCAB_PATH=/app/data/processed/vocab.pkl \
    METRICS_PATH=/app/results/metrics_beam5.json

# Runtime shared libraries needed by Pillow (libjpeg/libpng) and torch (libgomp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Install CPU-only torch + torchvision from the official pytorch index first,
#    in its own layer so rebuilds don't re-download it every time code changes.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

# 2) Install the remaining runtime deps from PyPI. Small layer, rebuilds fast.
COPY requirements.docker.txt ./
RUN pip install -r requirements.docker.txt

# 3) Copy source + static assets. This layer is the one that changes on
#    every code change, so putting it last maximizes cache reuse above.
COPY src/ ./src/
COPY api/ ./api/
COPY app.py ./

# Model artifacts — the checkpoint is the biggest single file (~220 MB)
# and is deliberately baked in so the image is self-contained.
COPY models/attention_lstm.pth ./models/attention_lstm.pth
COPY data/processed/vocab.pkl ./data/processed/vocab.pkl
COPY results/metrics_beam5.json ./results/metrics_beam5.json

# Create a non-root user for the runtime.
RUN useradd --create-home --uid 10001 captioner \
    && chown -R captioner:captioner /app
USER captioner

EXPOSE 8000 7860

# Default command runs the FastAPI service. docker-compose overrides this
# for the Gradio container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
