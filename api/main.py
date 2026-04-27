"""FastAPI microservice for Flickr8k image captioning.

Endpoints:
    GET  /                   — API index (HTML blurb)
    GET  /health             — readiness + uptime
    GET  /model-info         — architecture, vocab size, test-set metrics
    POST /caption            — upload an image → best caption + beam alternatives
    POST /caption/attention  — upload an image → caption + per-token attention weights

Launch:
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import json
import math
import os
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

# Make `src` importable when running via `uvicorn api.main:app`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import generate_beam, load_attention_model  # noqa: E402
from src.utils import get_device  # noqa: E402
from src.visualize import attention_heatmap_for_image  # noqa: E402
from src.vocabulary import Vocabulary  # noqa: E402

from api.schemas import (  # noqa: E402
    AttentionResponse,
    BeamAlternative,
    CaptionResponse,
    HealthResponse,
    ModelInfoResponse,
)


CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", str(REPO_ROOT / "models" / "attention_lstm.pth")
)
VOCAB_PATH = os.environ.get(
    "VOCAB_PATH", str(REPO_ROOT / "data" / "processed" / "vocab.pkl")
)
METRICS_PATH = os.environ.get(
    "METRICS_PATH", str(REPO_ROOT / "results" / "metrics_beam5.json")
)
DEFAULT_BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "5"))
DEFAULT_MAX_LEN = int(os.environ.get("MAX_LEN", "20"))


# ---------------------------------------------------------------------------
# Lifespan — load model + vocab + metrics exactly once at startup
# ---------------------------------------------------------------------------

_state: dict = {
    "encoder": None,
    "decoder": None,
    "cfg": None,
    "vocab": None,
    "metrics": {},
    "device": None,
    "start_time": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    with open(VOCAB_PATH, "rb") as f:
        vocab: Vocabulary = pickle.load(f)
    encoder, decoder, cfg = load_attention_model(CHECKPOINT_PATH, len(vocab), device)

    metrics: dict = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    _state.update(
        encoder=encoder,
        decoder=decoder,
        cfg=cfg,
        vocab=vocab,
        metrics=metrics,
        device=device,
        start_time=time.perf_counter(),
    )
    print(f"[api] model loaded from {CHECKPOINT_PATH} on {device}")
    yield
    print("[api] shutting down")


app = FastAPI(
    title="Flickr8k Image Captioning API",
    description=(
        "Serves a ResNet50 + Bahdanau attention + LSTM decoder trained on "
        "Flickr8k. Inference uses beam search with length normalization and a "
        "repetition penalty."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_image_from_upload(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"could not decode image: {e}"
        ) from e
    finally:
        upload.file.close()


def _require_model():
    if _state["encoder"] is None or _state["decoder"] is None:
        raise HTTPException(status_code=503, detail="model is not loaded")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    return """
    <html><body style="font-family:system-ui;max-width:640px;margin:2rem auto;">
    <h1>Flickr8k Image Captioning API</h1>
    <p>ResNet50 + Bahdanau attention + LSTM decoder. Beam search (k=5) decoding.</p>
    <ul>
      <li><a href="/docs">/docs</a> — interactive Swagger UI</li>
      <li><a href="/health">/health</a></li>
      <li><a href="/model-info">/model-info</a></li>
      <li><code>POST /caption</code> — multipart image upload</li>
      <li><code>POST /caption/attention</code> — multipart image upload (returns attention weights)</li>
    </ul>
    </body></html>
    """


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    _require_model()
    return HealthResponse(
        status="ok",
        model_loaded=_state["encoder"] is not None,
        uptime_s=time.perf_counter() - _state["start_time"],
        device=str(_state["device"]),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    _require_model()
    cfg = _state["cfg"]
    vocab = _state["vocab"]
    test_metrics = {
        k: _state["metrics"].get(k)
        for k in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "CIDEr", "ROUGE-L")
        if _state["metrics"].get(k) is not None
    }
    return ModelInfoResponse(
        architecture="ResNet50 + Bahdanau attention + LSTM decoder",
        backbone="resnet50",
        decoder_type=cfg.rnn_type,
        vocab_size=len(vocab),
        embed_size=cfg.embed_size,
        hidden_size=cfg.hidden_size,
        attention_dim=cfg.attention_dim,
        checkpoint_path=CHECKPOINT_PATH,
        test_metrics=test_metrics,
    )


@app.post("/caption", response_model=CaptionResponse)
async def caption(
    file: UploadFile = File(..., description="Image file to caption"),
    beam_width: int = DEFAULT_BEAM_WIDTH,
    max_len: int = DEFAULT_MAX_LEN,
) -> CaptionResponse:
    _require_model()
    img = _load_image_from_upload(file)

    from src.inference import encode_image

    tensor = encode_image(img, _state["device"])
    beams = generate_beam(
        _state["encoder"],
        _state["decoder"],
        tensor,
        _state["vocab"],
        beam_width=beam_width,
        max_len=max_len,
    )
    best = beams[0]
    confidence = math.exp(best.score)  # length-normalized logp → pseudo-confidence in (0, 1)
    confidence = max(0.0, min(1.0, confidence))

    return CaptionResponse(
        caption=best.caption,
        method="beam",
        beam_width=beam_width,
        confidence=round(confidence, 4),
        alternatives=[
            BeamAlternative(caption=b.caption, score=round(b.score, 4))
            for b in beams
        ],
    )


@app.post("/caption/attention", response_model=AttentionResponse)
async def caption_attention(
    file: UploadFile = File(..., description="Image file to caption"),
    max_len: int = DEFAULT_MAX_LEN,
) -> AttentionResponse:
    _require_model()
    img = _load_image_from_upload(file)

    caption_str, tokens, alphas, _ = attention_heatmap_for_image(
        encoder=_state["encoder"],
        decoder=_state["decoder"],
        image=img,
        vocab=_state["vocab"],
        device=_state["device"],
        max_len=max_len,
    )

    # alphas is (n_tokens, H, W) — flatten H*W per row to match the response schema.
    attention_flat = [row.reshape(-1).tolist() for row in alphas]
    grid_shape = alphas.shape[1:] if alphas.ndim == 3 else (0, 0)

    return AttentionResponse(
        caption=caption_str,
        tokens=tokens,
        attention=attention_flat,
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
    )
