"""Gradio Space for Flickr8k image captioning.

On first launch, pulls the trained checkpoint + vocabulary from the
companion model repo on the Hub:
    https://huggingface.co/OmarGamal48812/flickr8k-attention-lstm

After that, every request runs locally on the Space's CPU.
"""

from __future__ import annotations

import io
import os
import pickle
from pathlib import Path

# --- Workaround for gradio_client schema parser crash on `additionalProperties: True` ---
# Some Pydantic / Gradio component schemas emit `additionalProperties: True` as a bool
# literal. The bundled gradio_client's `_json_schema_to_python_type` doesn't guard for
# non-dict schemas and crashes on `if "const" in schema`. Patch the function to fall
# back to "Any" for non-dict schemas. Must run BEFORE `import gradio`.
import gradio_client.utils as _gcu  # noqa: E402

_orig_jspt = _gcu._json_schema_to_python_type
_orig_get_type = _gcu.get_type


def _safe_jspt(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_jspt(schema, defs)


def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)


_gcu._json_schema_to_python_type = _safe_jspt
_gcu.get_type = _safe_get_type
# --- end workaround ---

import gradio as gr
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from src.inference import encode_image, generate_beam, load_attention_model
from src.utils import get_device
from src.visualize import attention_heatmap_for_image, plot_attention_heatmap
from src.vocabulary import Vocabulary


MODEL_REPO = os.environ.get("MODEL_REPO", "OmarGamal48812/flickr8k-attention-lstm")
DEFAULT_BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", "5"))
DEFAULT_MAX_LEN = int(os.environ.get("MAX_LEN", "20"))


# ---------------------------------------------------------------------------
# Pull artifacts + load model once
# ---------------------------------------------------------------------------

print(f"[gradio] downloading checkpoint + vocab from {MODEL_REPO} ...")
CHECKPOINT_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="attention_lstm.pth")
VOCAB_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="vocab.pkl")

print("[gradio] loading model ...")
_device = get_device()
with open(VOCAB_PATH, "rb") as f:
    _vocab: Vocabulary = pickle.load(f)
_encoder, _decoder, _cfg = load_attention_model(CHECKPOINT_PATH, len(_vocab), _device)
print(f"[gradio] model loaded on {_device}")


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def caption_image(image: Image.Image, beam_width: int = DEFAULT_BEAM_WIDTH):
    if image is None:
        return "Please upload an image.", "", None

    tensor = encode_image(image, _device)
    beams = generate_beam(
        _encoder, _decoder, tensor, _vocab,
        beam_width=beam_width, max_len=DEFAULT_MAX_LEN,
    )
    best = beams[0]
    best_md = (
        f"### {best.caption}\n\n"
        f"*(beam search k={beam_width}, normalized log-prob = {best.score:.3f})*"
    )

    alts_lines = ["**Alternative captions:**"]
    for i, b in enumerate(beams[1:], start=2):
        alts_lines.append(f"{i}. `{b.score:+.3f}`  {b.caption}")
    alts_md = "\n".join(alts_lines) if len(beams) > 1 else ""

    caption_str, tokens, alphas, img_tensor = attention_heatmap_for_image(
        encoder=_encoder,
        decoder=_decoder,
        image=image,
        vocab=_vocab,
        device=_device,
        max_len=DEFAULT_MAX_LEN,
    )
    fig = plot_attention_heatmap(
        img_tensor, tokens, alphas,
        title=f"Attention · {caption_str}",
    )
    heatmap_img = _fig_to_pil(fig)

    return best_md, alts_md, heatmap_img


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------


DESCRIPTION = f"""
### What this does
Upload an image and the model will generate a caption **plus** a per-word
attention heatmap showing which part of the image influenced each word.

### Model
ResNet50 + Bahdanau attention + `{_cfg.rnn_type.upper()}Cell` decoder
(embed={_cfg.embed_size}, hidden={_cfg.hidden_size}, attention={_cfg.attention_dim}).
Trained on Flickr8k (30K training captions). Test-set BLEU-4 = **0.2403** with beam search k=5.

Checkpoint: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
"""

with gr.Blocks(title="Flickr8k Image Captioning") as demo:
    gr.Markdown("# Flickr8k Image Captioning")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload image")
            beam_slider = gr.Slider(
                minimum=1, maximum=10, value=DEFAULT_BEAM_WIDTH, step=1,
                label="Beam width",
            )
            run_btn = gr.Button("Caption this image", variant="primary")
        with gr.Column(scale=1):
            caption_md = gr.Markdown(label="Generated caption")
            alts_md = gr.Markdown(label="Alternatives")

    heatmap_image = gr.Image(
        type="pil", label="Per-word attention heatmaps", interactive=False
    )

    run_btn.click(
        fn=caption_image,
        inputs=[image_input, beam_slider],
        outputs=[caption_md, alts_md, heatmap_image],
    )
    image_input.change(
        fn=lambda: ("", "", None),
        outputs=[caption_md, alts_md, heatmap_image],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_api=False,
    )
