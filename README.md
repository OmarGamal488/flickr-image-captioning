# Flickr Image Captioning

> **ITI AI Track — Intake 46 — Deep Learning Project**
> Author: **Omar Gamal ElKady**

ResNet50 + Bahdanau attention + LSTM/GRU decoder, trained on **Flickr8k + Flickr30k** (~39,874 images × 5 captions, vocab=10,111). Trained with teacher forcing, label smoothing, scheduled sampling, and doubly stochastic regularization. Decoded with beam search (length normalisation + repetition penalty), evaluated with BLEU-1..4 / METEOR / CIDEr / ROUGE-L.

- **Live demo:** [huggingface.co/spaces/OmarGamal48812/flickr8k-captioning-demo](https://huggingface.co/spaces/OmarGamal48812/flickr8k-captioning-demo)
- **Model card:** [huggingface.co/OmarGamal48812/flickr8k-attention-lstm](https://huggingface.co/OmarGamal48812/flickr8k-attention-lstm)

## Test-set results (beam = 5, full test split)

| Model | BLEU-1 | BLEU-2 | BLEU-3 | **BLEU-4** | METEOR | CIDEr | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline LSTM              | 0.6195 | 0.4425 | 0.3094 | 0.2159 | 0.4026 | 0.4897 | 0.4616 |
| Baseline GRU               | 0.6361 | 0.4610 | 0.3318 | 0.2390 | 0.4120 | 0.5277 | 0.4689 |
| Attention LSTM             | 0.6258 | 0.4770 | 0.3621 | 0.2750 | 0.4466 | 0.7316 | 0.5055 |
| Attention GRU              | 0.6182 | 0.4660 | 0.3474 | 0.2578 | 0.4352 | 0.6894 | 0.4947 |
| Attention LSTM + GloVe     | 0.6408 | 0.4935 | 0.3802 | 0.2932 | 0.4659 | 0.7842 | 0.5162 |
| **Attention GRU + GloVe**  | **0.6859** | **0.5289** | **0.4041** | **0.3093** | **0.4709** | **0.7961** | **0.5257** |

Attention adds **+27 % relative BLEU-4** over the no-attention baseline (0.2750 vs 0.2159). GloVe warm-start boosts the GRU decoder by **+20 %** to **BLEU-4 = 0.3093**.

## Architecture

```
Image (3, 224, 224)
  └─ ResNet50 (pretrained, frozen 9 epochs, last 2 blocks unfrozen from epoch 10)
       (B, 2048, 7, 7) → reshape → (B, 49, 2048)
  └─ Bahdanau attention   V·tanh(W_enc + W_dec)
       context (B, 2048), α (B, 49)
  └─ LSTMCell/GRUCell (per-step) — embed=256 (300 w/ GloVe), hidden=1024, attn=256, dropout=0.5
  └─ Linear → vocab
```

Vocabulary is built from the train split only (frequency threshold 3, vocab size = 10,111). Specials: `<pad>=0, <start>=1, <end>=2, <unk>=3`. Captions are lowercased and punctuation-stripped before tokenization.

## Setup

```bash
# 1. Install uv if missing — https://docs.astral.sh/uv/
uv sync

# 2. Register the Jupyter kernel for the notebooks
uv run python -m ipykernel install --user \
    --name flickr-captioning \
    --display-name "Python (Flickr Captioning)"

# 3. Kaggle credentials (one-time) — https://www.kaggle.com/settings → API → Create token
chmod 600 ~/.kaggle/kaggle.json

# 4. Download + merge Flickr8k and Flickr30k (auto-cached via kagglehub)
uv run python -c "from src.dataset import build_merged_raw; build_merged_raw()"
ln -sfn data/raw_merged data/raw

# 5. Verify CUDA
uv run python -c "import torch; assert torch.cuda.is_available()"
```

## Training

```bash
# Baseline models
uv run python -m src.train --config configs/baseline_lstm.json
uv run python -m src.train --config configs/baseline_gru.json

# Attention models
uv run python -m src.train_attention --config configs/attention_lstm.json
uv run python -m src.train_attention --config configs/attention_gru.json

# Attention models with GloVe initialisation (35 epochs)
uv run python -m src.train_attention --config configs/attention_lstm_glove.json
uv run python -m src.train_attention --config configs/attention_gru_glove.json
```

Checkpoints are saved to `models/<run_name>.pth` on best **val BLEU-4**.
Training history per epoch is written to `models/<run_name>_history.json`.

## Inference

```bash
# Beam-5 evaluation on the test split
uv run python -m src.evaluate \
    --checkpoint models/attention_gru_glove.pth \
    --method beam --beam-width 5 \
    --out results/metrics_attention_gru_glove_beam5.json
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
    "models/attention_gru_glove.pth", len(vocab), device
)

caption, beams = caption_image(
    encoder, decoder, "your_image.jpg", vocab, device,
    method="beam", beam_width=5,
)
print(caption)
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

## Docker

```bash
# Build (once)
docker compose build

# Run both services
docker compose up

# Or run only one
docker compose up gradio    # http://localhost:7860
docker compose up api       # http://localhost:8000  (Swagger at /docs)

docker compose down
```

Smoke-test the API:

```bash
curl http://localhost:8000/health
curl -F "file=@your_image.jpg" http://localhost:8000/caption
```

## Project structure

```
data/
  raw/               # symlink to merged dataset (captions.txt + Images/)
  processed/         # vocab.pkl + {train,val,test}_data.pkl
notebooks/           # 01_eda, 02_modeling, 03_error_analysis
src/                 # dataset, vocabulary, encoder, attention, decoder,
                     # train, train_attention, inference, evaluate,
                     # visualize, utils
api/                 # FastAPI microservice (main.py, schemas.py)
app.py               # Gradio demo
configs/             # *.json training configs
models/              # *.pth checkpoints + *_history.json training logs
results/             # metrics_*.json, results_grid.png, attention_maps/, gradcam/
reports/             # project_report.pdf, analysis_report.md
tests/               # pytest suite
```

## Hugging Face release

The trained `attention_gru_glove` checkpoint is published on the Hub:

**[huggingface.co/OmarGamal48812/flickr8k-attention-lstm](https://huggingface.co/OmarGamal48812/flickr8k-attention-lstm)**

```python
from huggingface_hub import hf_hub_download

ckpt  = hf_hub_download("OmarGamal48812/flickr8k-attention-lstm", "attention_gru_glove.pth")
vocab = hf_hub_download("OmarGamal48812/flickr8k-attention-lstm", "vocab.pkl")
```

## Documentation

- [`reports/project_report.pdf`](reports/project_report.pdf) — final report
- [`reports/analysis_report.md`](reports/analysis_report.md) — error analysis
- [`reports/presentation.pptx`](reports/presentation.pptx) — defense slides

## Status

- [x] Phase 1 — Setup + data ingestion
- [x] Phase 2 — EDA + preprocessing (`notebooks/01_eda.ipynb`)
- [x] Phase 3 — Baseline LSTM + GRU (`models/baseline_lstm.pth`, `baseline_gru.pth`)
- [x] Phase 4 — Attention LSTM + GRU + GloVe variants (`models/attention_*.pth`)
- [x] Phase 5 — Inference + evaluation (`src/evaluate.py`, `results/metrics_*.json`)
- [x] Phase 6 — Error analysis (`notebooks/03_error_analysis.ipynb`)
- [x] Phase 7 — Deployment (Gradio + FastAPI + Docker + HF release + pytest + CI)
- [ ] Phase 8 — Final report PDF + slides

## License

MIT for code. The Flickr8k/Flickr30k datasets and ImageNet weights have their own licenses — please consult the upstream sources.
