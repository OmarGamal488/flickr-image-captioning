---
title: Flickr8k Image Captioning
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
models:
  - OmarGamal48812/flickr8k-attention-lstm
---

# Flickr8k Image Captioning

Live demo of a **ResNet50 + Bahdanau attention + LSTM** image captioning
model trained on **Flickr8k** (8,091 images × 5 captions).

Upload an image and the Space will return:
- the top caption (beam search, k = 5)
- the top-k alternative captions
- a per-word attention heatmap showing what the decoder looked at for each token

## Test-set performance (beam = 5)

| BLEU-4 | METEOR | CIDEr | ROUGE-L |
|---:|---:|---:|---:|
| 0.2403 | 0.4270 | 0.6002 | 0.4788 |

## Model

The checkpoint, vocabulary, training config, and metrics are published in
the companion model repository:

[**OmarGamal48812/flickr8k-attention-lstm**](https://huggingface.co/OmarGamal48812/flickr8k-attention-lstm)

The Space downloads them at startup with `huggingface_hub.hf_hub_download`.

## Source code

[github.com/OmarGamal488/flickr8k-image-captioning](https://github.com/OmarGamal488/flickr8k-image-captioning)
