"""FastAPI endpoint smoke tests.

These tests need a real checkpoint + vocab on disk because the API loads them
during the lifespan event. If either is missing (e.g. CI runner without the
model artifacts), the whole module is skipped.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "models" / "attention_lstm.pth"
VOCAB = REPO_ROOT / "data" / "processed" / "vocab.pkl"

if not CHECKPOINT.exists() or not VOCAB.exists():
    pytest.skip(
        f"skipping API tests — missing checkpoint ({CHECKPOINT.name}) or vocab",
        allow_module_level=True,
    )

# Imported lazily so the skip above triggers before FastAPI tries to load anything.
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    # `with` triggers the FastAPI lifespan (model load) on enter / unload on exit.
    with TestClient(app) as c:
        yield c


def _png_bytes(size: tuple[int, int] = (224, 224)) -> bytes:
    img = Image.new("RGB", size, color=(120, 200, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["uptime_s"] >= 0
    assert "device" in body


def test_model_info_reports_architecture(client: TestClient):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["backbone"] == "resnet50"
    assert body["decoder_type"] in {"lstm", "gru"}
    assert body["vocab_size"] > 0
    assert body["embed_size"] > 0


def test_caption_endpoint_returns_caption_and_alternatives(client: TestClient):
    r = client.post(
        "/caption",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["caption"], str) and body["caption"]
    assert body["method"] == "beam"
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["alternatives"], list) and len(body["alternatives"]) >= 1
    # alternatives are sorted best-first
    scores = [a["score"] for a in body["alternatives"]]
    assert scores == sorted(scores, reverse=True)


def test_caption_endpoint_rejects_garbage_upload(client: TestClient):
    r = client.post(
        "/caption",
        files={"file": ("not_an_image.txt", b"this is not a PNG", "text/plain")},
    )
    assert r.status_code == 400
