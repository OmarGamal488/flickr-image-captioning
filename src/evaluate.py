"""Full test-set evaluation for the attention captioning model.

Computes BLEU-1/2/3/4, METEOR, CIDEr, and ROUGE-L on the full Flickr8k test
split, comparing each generated caption against the 5 ground-truth references
for that image.

Usage:
    uv run python -m src.evaluate --checkpoint models/attention_lstm.pth --method beam
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

from src.inference import batch_generate, load_attention_model
from src.utils import get_device
from src.vocabulary import Vocabulary

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    PYCOCO_AVAILABLE = True
except Exception as e:  # pragma: no cover
    PYCOCO_AVAILABLE = False
    print(f"pycocoevalcap unavailable: {e} — CIDEr/ROUGE-L will be skipped")


def _load_split(processed_dir: str, name: str) -> dict:
    with open(os.path.join(processed_dir, f"{name}_data.pkl"), "rb") as f:
        return pickle.load(f)


def _compute_bleu(
    hypotheses: list[list[str]], references: list[list[list[str]]]
) -> dict[str, float]:
    smoothing = SmoothingFunction().method1
    return {
        "BLEU-1": corpus_bleu(
            references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing
        ),
        "BLEU-2": corpus_bleu(
            references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing
        ),
        "BLEU-3": corpus_bleu(
            references,
            hypotheses,
            weights=(1 / 3, 1 / 3, 1 / 3, 0),
            smoothing_function=smoothing,
        ),
        "BLEU-4": corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ),
    }


def _compute_meteor(
    hypotheses: list[list[str]], references: list[list[list[str]]]
) -> float:
    # nltk's meteor_score wants one hypothesis token list and a list of ref token lists
    scores = [
        meteor_score(refs, hyp) for refs, hyp in zip(references, hypotheses)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _compute_pycoco(
    hypotheses_str: dict[str, list[str]],
    references_str: dict[str, list[str]],
) -> dict[str, float]:
    if not PYCOCO_AVAILABLE:
        return {}
    # pycocoevalcap expects {id: [caption]} for res and {id: [refs]} for gts.
    cider_score, _ = Cider().compute_score(references_str, hypotheses_str)
    rouge_score, _ = Rouge().compute_score(references_str, hypotheses_str)
    return {"CIDEr": float(cider_score), "ROUGE-L": float(rouge_score)}


def _per_image_bleu4(
    hypotheses_tok: list[list[str]], references_tok: list[list[list[str]]]
) -> list[float]:
    """Per-image sentence-BLEU-4 so we can pick good/mediocre/bad for the qualitative grid."""
    from nltk.translate.bleu_score import sentence_bleu

    smoothing = SmoothingFunction().method1
    return [
        sentence_bleu(refs, hyp, weights=(0.25,) * 4, smoothing_function=smoothing)
        for refs, hyp in zip(references_tok, hypotheses_tok)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/attention_lstm.pth")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--images-dir", default="data/raw/Images")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--method", default="beam", choices=["greedy", "beam"])
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    with open(os.path.join(args.processed_dir, "vocab.pkl"), "rb") as f:
        vocab: Vocabulary = pickle.load(f)
    print(f"vocab size: {len(vocab)}")

    split = _load_split(args.processed_dir, args.split)
    image_ids = split["image_ids"]
    print(f"{args.split} split: {len(image_ids)} images")

    encoder, decoder, cfg = load_attention_model(args.checkpoint, len(vocab), device)
    print(f"loaded {args.checkpoint} (rnn={cfg.rnn_type})")

    # --- generate --------------------------------------------------------
    t0 = time.perf_counter()
    predictions = batch_generate(
        encoder=encoder,
        decoder=decoder,
        image_ids=image_ids,
        images_dir=args.images_dir,
        vocab=vocab,
        device=device,
        method=args.method,
        beam_width=args.beam_width,
        max_len=args.max_len,
        batch_size=32 if args.method == "greedy" else 1,
    )
    dt = time.perf_counter() - t0
    print(f"generated {len(predictions)} captions in {dt:.1f}s ({dt/len(predictions):.3f}s/img)")

    # --- shape data for every metric ------------------------------------
    hypotheses_tok: list[list[str]] = []
    references_tok: list[list[list[str]]] = []
    hypotheses_str: dict[str, list[str]] = {}
    references_str: dict[str, list[str]] = {}

    for img_id in image_ids:
        pred = predictions[img_id]
        refs_text = split["captions_text"][img_id]

        hypotheses_tok.append(pred.split())
        references_tok.append([Vocabulary.tokenize(c) for c in refs_text])

        hypotheses_str[img_id] = [pred]
        references_str[img_id] = [
            " ".join(Vocabulary.tokenize(c)) for c in refs_text
        ]

    # --- metrics ---------------------------------------------------------
    bleu = _compute_bleu(hypotheses_tok, references_tok)
    meteor = _compute_meteor(hypotheses_tok, references_tok)
    pycoco = _compute_pycoco(hypotheses_str, references_str)

    per_img_bleu4 = _per_image_bleu4(hypotheses_tok, references_tok)

    metrics = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_images": len(image_ids),
        "method": args.method,
        "beam_width": args.beam_width if args.method == "beam" else None,
        "max_len": args.max_len,
        "rnn_type": cfg.rnn_type,
        **{k: round(v, 4) for k, v in bleu.items()},
        "METEOR": round(meteor, 4),
        **{k: round(v, 4) for k, v in pycoco.items()},
        "wall_clock_s": round(dt, 1),
    }

    # --- persist ---------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n=== {args.split} metrics ({args.method}) ===")
    print(json.dumps(metrics, indent=2))
    print(f"saved → {args.out}")

    # Also persist predictions + per-image BLEU-4 so the qualitative grid can reuse them
    preds_path = args.out.replace(".json", "_predictions.json")
    with open(preds_path, "w") as f:
        json.dump(
            [
                {
                    "image_id": img_id,
                    "prediction": predictions[img_id],
                    "bleu4": round(score, 4),
                    "ground_truth": split["captions_text"][img_id],
                }
                for img_id, score in zip(image_ids, per_img_bleu4)
            ],
            f,
            indent=2,
        )
    print(f"per-image predictions saved → {preds_path}")


if __name__ == "__main__":
    main()
