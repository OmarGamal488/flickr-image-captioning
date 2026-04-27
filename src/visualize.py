"""Visualization helpers: per-word attention heatmaps and GradCAM overlays.

- ``attention_heatmap_for_image``  — runs the attention decoder on one image,
  returns ``(caption_tokens, alphas)``.
- ``plot_attention_heatmap``       — renders the Show-Attend-Tell style grid:
  input image + one heatmap panel per generated word.
- ``gradcam_for_image``            — runs GradCAM on the ResNet50 encoder's last
  conv block using ``pytorch-grad-cam``.
- ``plot_gradcam``                 — side-by-side image + CAM overlay.

All plots are saved to disk via ``save_path``; if omitted, the current figure is
left on the stack for a notebook cell to ``plt.show()``.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.decoder import DecoderAttention
from src.encoder import EncoderCNN_Attention
from src.inference import encode_image
from src.vocabulary import Vocabulary

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) ImageNet-normalized → (H, W, 3) in [0, 1]."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Attention heatmaps
# ---------------------------------------------------------------------------


@torch.no_grad()
def attention_heatmap_for_image(
    encoder: torch.nn.Module,
    decoder: DecoderAttention,
    image: str | Image.Image,
    vocab: Vocabulary,
    device: torch.device,
    max_len: int = 20,
) -> tuple[str, list[str], np.ndarray, torch.Tensor]:
    """Run greedy decoding and capture per-step attention weights.

    Returns:
        caption_str : human-readable caption (<start>/<end>/<pad> stripped)
        tokens      : list of token strings aligned with ``alphas``
        alphas      : (n_tokens, 7, 7) numpy array of attention weights
        image_tensor: the normalized input tensor for display later
    """
    encoder.eval()
    decoder.eval()

    image_tensor = encode_image(image, device)
    features = encoder(image_tensor)                              # (1, 49, 2048)
    P = features.size(1)
    side = int(round(P**0.5))                                     # 7 for ResNet

    gen_ids, per_step_alphas = decoder.generate_greedy(
        features, max_len=max_len, return_alphas=True
    )
    ids = gen_ids[0]
    alphas_list = per_step_alphas[0]

    # Trim past <end>
    tokens: list[str] = []
    alphas: list[np.ndarray] = []
    for tok, alpha in zip(ids, alphas_list):
        if tok == Vocabulary.END_IDX:
            break
        if tok == Vocabulary.PAD_IDX:
            continue
        tokens.append(vocab.itos.get(int(tok), Vocabulary.UNK_TOKEN))
        alphas.append(np.array(alpha).reshape(side, side))

    caption_str = vocab.denumericalize(ids)
    alphas_arr = np.stack(alphas) if alphas else np.zeros((0, side, side))
    return caption_str, tokens, alphas_arr, image_tensor.squeeze(0)


def plot_attention_heatmap(
    image_tensor: torch.Tensor,
    tokens: Sequence[str],
    alphas: np.ndarray,
    title: str | None = None,
    save_path: str | None = None,
    upscale_size: int = 224,
) -> plt.Figure:
    """Render a Show-Attend-Tell style figure for one caption."""
    img = _denormalize(image_tensor)
    n_words = len(tokens)
    n_cols = 5
    n_rows = (n_words + 1 + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    # First panel = raw image
    axes[0].imshow(img)
    axes[0].set_title("input", fontsize=9)
    axes[0].axis("off")

    for i, (word, alpha_map) in enumerate(zip(tokens, alphas), start=1):
        alpha_tensor = torch.tensor(alpha_map).unsqueeze(0).unsqueeze(0).float()
        upscaled = F.interpolate(
            alpha_tensor, size=(upscale_size, upscale_size), mode="bilinear", align_corners=False
        ).squeeze().numpy()
        upscaled = upscaled / (upscaled.max() + 1e-9)

        ax = axes[i]
        ax.imshow(img)
        ax.imshow(upscaled, alpha=0.55, cmap="jet")
        ax.set_title(word, fontsize=10)
        ax.axis("off")

    for j in range(n_words + 1, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------


def gradcam_for_image(
    encoder: EncoderCNN_Attention,
    image: str | Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GradCAM targeting the encoder's feature map magnitude.

    Returns:
        cam  : (H, W) normalized [0, 1] heat map
        img  : (H, W, 3) denormalized image suitable for imshow
    """
    from pytorch_grad_cam import GradCAM

    image_tensor = encode_image(image, device)
    # Encoder is frozen — enable input grads so activations carry a gradient fn.
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    img_rgb = _denormalize(image_tensor.squeeze(0).detach())

    # Target the last conv stage of ResNet50 — that's where spatial features
    # live before the attention decoder consumes them.
    target_layers = [encoder.features[-1]]

    class _FeatureMagnitudeTarget:
        """Sum of the encoder's spatial feature map — gives GradCAM a scalar."""

        def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
            return model_output.mean()

    # Temporarily enable grad on the target layer so pytorch-grad-cam can hook it.
    target_params = list(target_layers[0].parameters())
    previously_frozen = [p.requires_grad for p in target_params]
    for p in target_params:
        p.requires_grad_(True)
    try:
        cam = GradCAM(model=encoder, target_layers=target_layers)
        grayscale_cam = cam(
            input_tensor=image_tensor, targets=[_FeatureMagnitudeTarget()]
        )[0]  # (H, W)
    finally:
        for p, was in zip(target_params, previously_frozen):
            p.requires_grad_(was)
    # Normalize for display
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (
        grayscale_cam.max() - grayscale_cam.min() + 1e-9
    )
    return grayscale_cam.astype(np.float32), img_rgb


def plot_gradcam(
    image: np.ndarray,
    cam: np.ndarray,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    from pytorch_grad_cam.utils.image import show_cam_on_image

    overlay = show_cam_on_image(image.astype(np.float32), cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("input")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("GradCAM (ResNet50 last conv)")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig
