# Flickr8k Image Captioning

> **ITI AI Track — Intake 46 — Deep Learning Project**
> Author: **Omar Gamal ElKady**

ResNet50 + Bahdanau attention + LSTM/GRU decoder, trained on **Flickr8k** (8,091 images × 5 captions). Trained with teacher forcing + doubly stochastic regularization, decoded with beam search (length normalisation + repetition penalty), evaluated with BLEU-1..4 / METEOR / CIDEr / ROUGE-L.

- **Live demo:** [huggingface.co/spaces/OmarGamal48812/flickr8k-captioning-demo](https://huggingface.co/spaces/OmarGamal48812/flickr8k-captioning-demo)
- **Model card:** [huggingface.co/OmarGamal48812/flickr8k-attention-lstm](https://huggingface.co/OmarGamal48812/flickr8k-attention-lstm)

## Test-set results (beam = 5, full 1,091-image test split)

| Model | BLEU-1 | BLEU-2 | BLEU-3 | **BLEU-4** | METEOR | CIDEr | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline (no attention) † | — | — | — | 0.1772 | — | — | — |
| **attention LSTM**     | 0.6488 | 0.4714 | 0.3378 | **0.2403** | 0.4270 | 0.6002 | 0.4788 |
| attention GRU          | 0.6455 | 0.4698 | 0.3373 | 0.2378 | 0.4380 | 0.6114 | 0.4839 |
| attention LSTM (HPO)   | 0.6251 | 0.4490 | 0.3166 | 0.2203 | 0.4257 | 0.5599 | 0.4706 |

† baseline is reported on validation only — the no-attention path uses
a different inference module. All other rows are full-test corpus
metrics.

Attention adds **+36 % relative BLEU-4** over the no-attention baseline
at the same encoder, decoder size, and training budget.

## Architecture

```
Image (3, 224, 224)
  └─ ResNet50 (pretrained, frozen 15 epochs, last 2 blocks unfrozen)
       (B, 2048, 7, 7) → reshape → (B, 49, 2048)
  └─ Bahdanau attention   V·tanh(W_enc + W_dec)
       context (B, 2048), α (B, 49)
  └─ LSTMCell (per-step) — embed=256, hidden=512, attn=256, dropout=0.5
  └─ Linear → vocab (V = 2,557)
```

Vocabulary is 2,557 tokens (frequency threshold 5, built from the train
split only). Specials: `<pad>=0, <start>=1, <end>=2, <unk>=3`. Captions
are lowercased and punctuation-stripped before tokenization at both
training and inference time so the learned and seen vocabularies are
byte-identical.

## Setup

```bash
# 1. Install uv if missing — https://docs.astral.sh/uv/
uv sync

# 2. Register the Jupyter kernel for the notebooks
uv run python -m ipykernel install --user \
    --name flickr8k-captioning \
    --display-name "Python (Flickr8k Captioning)"

# 3. Kaggle credentials (one-time) — https://www.kaggle.com/settings → API → Create token
chmod 600 ~/.kaggle/kaggle.json

# 4. Download Flickr8k (cached into ~/.cache/kagglehub/)
uv run python -c "import kagglehub; print(kagglehub.dataset_download('adityajn105/flickr8k'))"
ln -sfn <PRINTED_PATH> data/raw

# 5. Verify CUDA
uv run python -c "import torch; assert torch.cuda.is_available()"
```

## Training

```bash
# baseline (Phase 3 — no attention)
uv run python -m src.train --config configs/baseline.json

# attention LSTM (main result)
uv run python -m src.train_attention --config configs/attention_lstm.json

# attention GRU
uv run python -m src.train_attention --config configs/attention_gru.json
```

Checkpoints are saved to `models/<run_name>.pth` on best **val BLEU-4**
(not val loss). Training history per epoch is written alongside as
`models/<run_name>_history.json`.

## Inference

```bash
# greedy + beam=5 evaluation on the test split
uv run python -m src.evaluate \
    --checkpoint models/attention_lstm.pth \
    --method beam --beam-width 5 \
    --out results/metrics_beam5.json
```

Programmatic usage:

```python
import pickle
from src.inference import load_attention_model, caption_image
from src.utils import get_device

device = get_device()
with open("data/processed/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

encoder, decoder, cfg = load_attention_model(
    "models/attention_lstm.pth", len(vocab), device
)

caption, beams = caption_image(
    encoder, decoder, "your_image.jpg", vocab, device,
    method="beam", beam_width=5,
)
print(caption)            # → "a man in a red shirt is climbing a rocky cliff"
for b in beams[:3]:
    print(f"  {b.score:+.3f}  {b.caption}")
```

## Serving

| Component | File | Port | Launch |
|---|---|---:|---|
| Gradio demo | `app.py` | 7860 | `uv run python app.py` |
| FastAPI microservice | `api/main.py` | 8000 | `uv run uvicorn api.main:app --port 8000` |
| Both, containerised | `Dockerfile` + `docker-compose.yml` | 7860+8000 | `docker compose up` |

The FastAPI service exposes:

- `POST /caption` — multipart image upload → `{caption, alternatives, confidence}`
- `POST /caption/attention` — same upload → `{caption, tokens, attention_grid}`
- `GET /health` — readiness + uptime
- `GET /model-info` — architecture + test metrics

Model is loaded once via FastAPI's `lifespan` so per-request latency is
just inference (~25 ms on a recent GPU, ~250 ms on CPU).

## Docker

The multi-stage `Dockerfile` produces a single ~2.6 GB CPU-only image
that ships both services. `docker-compose.yml` runs them side-by-side.

```bash
# 1. Build (once, ~5 min — downloads CPU torch wheels + bakes in the checkpoint)
docker compose build

# 2. Run both services
docker compose up                  # foreground (Ctrl+C to stop)
docker compose up -d               # detached / background

# 3. Or run only one
docker compose up gradio           # http://localhost:7860
docker compose up api              # http://localhost:8000  (Swagger at /docs)

# 4. Stop
docker compose down
```

Smoke-test the API:

```bash
curl http://localhost:8000/health
curl -F "file=@your_image.jpg" http://localhost:8000/caption
```

The image bakes in `models/attention_lstm.pth`, `data/processed/vocab.pkl`,
and `results/metrics_beam5.json` — those files must exist locally before
`docker compose build` runs.

## Project structure

```
data/
  raw/               # symlink to Flickr8k (captions.txt + Images/)
  processed/         # vocab.pkl + {train,val,test}_data.pkl
notebooks/           # 01_eda, 02_modeling, 03_error_analysis
src/                 # dataset, vocabulary, encoder, attention, decoder,
                     # train, train_attention, inference, evaluate,
                     # visualize, utils
api/                 # FastAPI microservice (main.py, schemas.py)
app.py               # Gradio demo
configs/             # *.json training configs
models/              # *.pth checkpoints + *_history.json training logs
experiments/         # wandb logs + optuna_study.db
results/             # metrics_*.json, results_grid.png, attention_maps/, gradcam/
reports/             # project_report.pdf, analysis_report.md
hf_release/          # staged Hugging Face Hub artifacts (model card + checkpoint)
tests/               # pytest suite (placeholder)
```

## Hugging Face release

The trained `attention_lstm` checkpoint is published on the Hub with a
full model card (architecture, training details, test-set metrics,
limitations, citations):

**[huggingface.co/OmarGamal48812/flickr8k-attention-lstm](https://huggingface.co/OmarGamal48812/flickr8k-attention-lstm)**

The repo bundles `attention_lstm.pth`, `vocab.pkl`, the training
`config.json`, beam-5 + greedy `metrics_*.json`, and the model card
([`hf_release/README.md`](hf_release/README.md)).

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("OmarGamal48812/flickr8k-attention-lstm", "attention_lstm.pth")
vocab = hf_hub_download("OmarGamal48812/flickr8k-attention-lstm", "vocab.pkl")
```

## Documentation

- [`reports/project_report.pdf`](reports/project_report.pdf) — final
  report (4–6 pages: problem, architecture, experiments, results,
  error analysis, conclusions).
- [`reports/analysis_report.md`](reports/analysis_report.md) —
  complete error analysis (six failure modes, vocabulary coverage,
  caption diversity, per-image BLEU distribution, answers to project
  Q1–Q5).
- [`reports/presentation.pptx`](reports/presentation.pptx) — defense
  slide deck.

## Status

- [x] Phase 1 — Setup + data ingestion
- [x] Phase 2 — EDA + preprocessing (`notebooks/01_eda.ipynb`)
- [x] Phase 3 — Baseline model (`models/baseline.pth`)
- [x] Phase 4 — Attention + Optuna HPO (`models/attention_lstm.pth`,
      `attention_gru.pth`, `attention_lstm_hpo.pth`)
- [x] Phase 5 — Inference + evaluation (`src/evaluate.py`,
      `results/metrics_*.json`)
- [x] Phase 6 — Error analysis (`notebooks/03_error_analysis.ipynb`,
      `reports/analysis_report.md`)
- [x] Phase 7 — Deployment (Gradio + FastAPI + Docker + HF release + pytest + CI)
- [x] Phase 8 — Final report PDF (slides pending)

## License

MIT for code. The Flickr8k dataset and ImageNet weights have their own
licenses — please consult the upstream sources.
