"""Pydantic request/response models for the FastAPI captioning service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BeamAlternative(BaseModel):
    caption: str = Field(..., description="Candidate caption text")
    score: float = Field(..., description="Length-normalized log probability")


class CaptionResponse(BaseModel):
    caption: str = Field(..., description="Best caption selected by beam search")
    method: str = Field(..., description="Decoding strategy used", examples=["beam"])
    beam_width: int | None = Field(None, description="Beam width (if method=beam)")
    confidence: float = Field(
        ..., description="pseudo-confidence = exp(normalized best score) in [0, 1]"
    )
    alternatives: list[BeamAlternative] = Field(
        default_factory=list, description="Other beam candidates, best first"
    )


class AttentionResponse(BaseModel):
    caption: str
    tokens: list[str] = Field(
        ..., description="Caption tokens, aligned 1:1 with the attention rows"
    )
    attention: list[list[float]] = Field(
        ..., description="Per-token attention weights — shape (n_tokens, num_pixels)"
    )
    grid_shape: tuple[int, int] = Field(
        ..., description="(H, W) to reshape each row into a 2D heatmap"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_s: float
    device: str


class ModelInfoResponse(BaseModel):
    architecture: str
    backbone: str
    decoder_type: str
    vocab_size: int
    embed_size: int
    hidden_size: int
    attention_dim: int
    checkpoint_path: str
    test_metrics: dict = Field(
        default_factory=dict,
        description="Final test-set BLEU/METEOR/CIDEr/ROUGE-L from results/metrics_beam5.json",
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
