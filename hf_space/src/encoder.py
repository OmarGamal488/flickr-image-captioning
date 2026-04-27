"""CNN encoders for the captioning pipeline.

- ``EncoderCNN``            — baseline (global feature vector). Used in Phase 3.
- ``EncoderCNN_Attention``  — spatial feature map for Bahdanau attention. Phase 4.

Both wrap a pretrained ResNet50 by default, strip the classification head, and
optionally freeze the backbone so only the projection layer trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


def _build_resnet50(pretrained: bool = True) -> nn.Module:
    weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
    return tv_models.resnet50(weights=weights)


class EncoderCNN(nn.Module):
    """Baseline: ResNet50 + global avg pool + linear projection to ``embed_size``.

    Output shape: ``(B, embed_size)``.
    """

    def __init__(self, embed_size: int, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        backbone = _build_resnet50(pretrained=pretrained)
        # Drop the final FC layer; keep the adaptive avg-pool.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.project = nn.Linear(backbone.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.set_frozen(freeze)

    def set_frozen(self, frozen: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.features(images)            # (B, 2048, 1, 1)
        feats = feats.flatten(1)                 # (B, 2048)
        feats = self.project(feats)              # (B, embed_size)
        return self.bn(feats)


class EncoderCNN_Attention(nn.Module):
    """Attention-ready: ResNet50 w/o avg-pool → spatial feature map.

    Output shape: ``(B, num_pixels, encoder_dim)`` where ``num_pixels = H*W``.
    For 224x224 input and ResNet50, that's ``(B, 49, 2048)``.
    Used by Phase 4 Bahdanau attention decoder.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        backbone = _build_resnet50(pretrained=pretrained)
        # Drop avg-pool and fc — keep everything up to the last conv stage.
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.encoder_dim = backbone.fc.in_features  # 2048
        self.set_frozen(freeze)

    def set_frozen(self, frozen: bool) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def fine_tune_last_blocks(self, n_blocks: int = 2) -> None:
        """Unfreeze the last ``n_blocks`` residual blocks of ResNet50.

        Layer 3 is ``layer3``, layer 4 is ``layer4`` in torchvision's ResNet.
        With ``n_blocks=2`` both ``layer3`` and ``layer4`` become trainable.
        """
        # First freeze everything so the call is idempotent.
        for p in self.features.parameters():
            p.requires_grad = False
        # torchvision children order: conv1, bn1, relu, maxpool, layer1..4
        to_unfreeze = list(self.features.children())[-n_blocks:]
        for block in to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.features(images)            # (B, 2048, 7, 7)
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, 49, 2048)
        return feats
