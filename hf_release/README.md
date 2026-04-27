---
language: en
license: mit
tags:
  - image-captioning
  - pytorch
  - resnet
  - attention
  - lstm
  - flickr8k
  - show-attend-and-tell
datasets:
  - nlphuji/flickr8k
metrics:
  - bleu
  - meteor
  - cider
  - rouge
library_name: pytorch
pipeline_tag: image-to-text
---

# Flickr8k Image Captioning — ResNet50 + Bahdanau Attention + LSTM Decoder

This model generates a natural-language description of an image. It uses a
**ResNet50** spatial-feature encoder, a **Bahdanau (additive)** attention
module, and an **LSTM decoder**, trained with teacher forcing and doubly
stochastic regularization on the **Flickr8k** dataset (8,091 images × 5
captions). It is the reference architecture from
[*Show, Attend and Tell* (Xu et al., 2015)](https://arxiv.org/abs/1502.03044).

## Test-set performance (beam search, k = 5)

| Metric | Value |
|---|---|
| BLEU-1 | 0.6488 |
| BLEU-2 | 0.4714 |
| BLEU-3 | 0.3378 |
| **BLEU-4** | **0.2403** |
| METEOR | 0.4270 |
| CIDEr | 0.6002 |
| ROUGE-L | 0.4788 |

Greedy decoding scores: BLEU-4 = 0.2073, METEOR = 0.4119, CIDEr = 0.5322.

Evaluated on the held-out 1,091-image test split (image-level split — no
captions cross train/val/test). Beam search uses length-normalized log-probs
(`alpha = 0.7`) and a repetition penalty of `1.2`.

## Architecture

```
Image (3, 224, 224)
  └─ ResNet50 (pretrained, frozen first 15 epochs, last 2 blocks fine-tuned)
       output: (B, 2048, 7, 7)  → reshape to (B, 49, 2048)
  └─ Bahdanau attention  V·tanh(W_enc(features) + W_dec(h_prev))
       output: context vector (B, 2048), attention weights (B, 49)
  └─ LSTMCell  (per timestep — re-queries attention each step)
       hidden state size: 512, embedding size: 256
  └─ Linear → vocab logits (V = 2,557)
```

Total parameters: **~36 M** (28 M frozen ResNet, 8 M trainable decoder/projection).

## Training details

- **Loss** — `CrossEntropyLoss(ignore_index=0)` plus doubly-stochastic
  regularization `α_c · ((1 − Σ_t α_t)²).mean()` with `α_c = 1.0`
- **Optimizer** — Adam, decoder LR `4e-4`, encoder LR `1e-5` (Phase B)
- **Schedule** — `ReduceLROnPlateau` on val BLEU-4, `factor=0.5`,
  `patience=3`
- **Two-phase training** — Phase A (15 epochs): freeze CNN, train decoder
  only. Phase B (10 epochs): unfreeze last 2 ResNet blocks.
- **Vocabulary** — 2,557 tokens (frequency threshold 5), built from train
  captions only. Special tokens: `<pad>=0, <start>=1, <end>=2, <unk>=3`.
- **Batch size** — 32, gradient clip 5.0
- **Seed** — 42

## Files in this repo

- `attention_lstm.pth` — PyTorch checkpoint (encoder + decoder state
  dicts, optimizer state, training config)
- `vocab.pkl` — pickled `Vocabulary` object built from the train split
- `config.json` — JSON copy of the training hyperparameters
- `metrics_beam5.json`, `metrics_greedy.json` — full test-set metrics

## Usage

The cleanest way to use this model is to clone the source repo so the
`Vocabulary`, encoder, and decoder classes are importable:

```bash
git clone https://github.com/OmarGamal488/flickr8k-image-captioning.git
cd flickr8k-image-captioning
uv sync
```

Then in Python:

```python
import pickle, torch
from huggingface_hub import hf_hub_download
from src.inference import load_attention_model, caption_image
from src.utils import get_device

repo_id = "OmarGamal48812/flickr8k-attention-lstm"
ckpt_path = hf_hub_download(repo_id=repo_id, filename="attention_lstm.pth")
vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.pkl")

device = get_device()
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

encoder, decoder, cfg = load_attention_model(ckpt_path, len(vocab), device)

caption, beams = caption_image(
    encoder, decoder, "your_image.jpg", vocab, device,
    method="beam", beam_width=5,
)
print(caption)
```

For interactive use, the same repo ships a Gradio demo (`app.py`) and a
FastAPI service (`api/main.py`).

## Limitations

- **Small training set.** Flickr8k has only 6,000 training images, so the
  model often falls back to "safe" generic captions (e.g. *a dog runs through
  the grass*) for unfamiliar scenes.
- **Vocabulary cap.** Words seen fewer than 5 times in the train split
  collapse to `<unk>`. Rare nouns and proper names are systematically lost.
- **Domain.** Trained exclusively on Flickr8k photos (mostly people, dogs,
  outdoor scenes). Performance degrades on cartoons, screenshots, abstract
  imagery, and any scene type not represented in Flickr8k.
- **Hallucinations.** Like all autoregressive captioners, the decoder can
  insert objects that aren't in the image when attention drifts.
- **English only.** Vocabulary and grammar are entirely English Flickr8k
  captions.

## Intended use

Educational demonstrations of the Show-Attend-Tell architecture and
research baselines. Not appropriate as the only data source for
accessibility tooling (alt-text generation should ideally use a model
trained on a much larger dataset).

## Citation

If you use this checkpoint, please credit the underlying paper:

```bibtex
@inproceedings{xu2015show,
  title     = {Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
  author    = {Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and
               Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
  booktitle = {ICML},
  year      = {2015}
}
```

and the dataset:

```bibtex
@article{hodosh2013framing,
  title   = {Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics},
  author  = {Hodosh, Micah and Young, Peter and Hockenmaier, Julia},
  journal = {Journal of Artificial Intelligence Research},
  year    = {2013}
}
```
