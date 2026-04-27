"""Vocabulary for Flickr8k image captioning.

Indices 0-3 are fixed and load-bearing:
    <pad>=0, <start>=1, <end>=2, <unk>=3
The training loss uses ignore_index=0, so <pad> must stay at index 0.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable


class Vocabulary:
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    _PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
    _WS_RE = re.compile(r"\s+")

    def __init__(self, freq_threshold: int = 5) -> None:
        self.freq_threshold = freq_threshold
        self.itos: dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.START_IDX: self.START_TOKEN,
            self.END_IDX: self.END_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN,
        }
        self.stoi: dict[str, int] = {v: k for k, v in self.itos.items()}
        self.word_freqs: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        text = text.lower()
        text = cls._PUNCT_RE.sub(" ", text)
        text = cls._WS_RE.sub(" ", text).strip()
        return text.split(" ") if text else []

    def build_from_captions(self, captions: Iterable[str]) -> None:
        """Build the vocabulary from TRAIN captions only.

        Words appearing fewer than ``freq_threshold`` times are kept out of the
        vocabulary and map to ``<unk>`` at numericalization time.
        """
        self.word_freqs = Counter()
        for caption in captions:
            self.word_freqs.update(self.tokenize(caption))

        next_idx = len(self.itos)
        for word, freq in sorted(self.word_freqs.items()):
            if freq >= self.freq_threshold and word not in self.stoi:
                self.itos[next_idx] = word
                self.stoi[word] = next_idx
                next_idx += 1

    def numericalize(self, text: str, add_special: bool = True) -> list[int]:
        """Convert text to ``[<start>, ids..., <end>]`` (special tokens by default)."""
        ids = [self.stoi.get(tok, self.UNK_IDX) for tok in self.tokenize(text)]
        if add_special:
            return [self.START_IDX, *ids, self.END_IDX]
        return ids

    def denumericalize(self, ids: Iterable[int], strip_special: bool = True) -> str:
        """Inverse of ``numericalize``.

        When ``strip_special`` is True, <pad>/<start>/<end>/<unk> are removed.
        Generation stops at the first <end> encountered.
        """
        skip = {self.PAD_IDX, self.START_IDX, self.UNK_IDX} if strip_special else set()
        out: list[str] = []
        for i in ids:
            i = int(i)
            if i == self.END_IDX and strip_special:
                break
            if i in skip:
                continue
            out.append(self.itos.get(i, self.UNK_TOKEN))
        return " ".join(out)

    def coverage(self, captions: Iterable[str]) -> float:
        """Fraction of tokens in ``captions`` that are in-vocabulary (not <unk>)."""
        total = 0
        in_vocab = 0
        for caption in captions:
            for tok in self.tokenize(caption):
                total += 1
                if tok in self.stoi:
                    in_vocab += 1
        return in_vocab / total if total else 0.0
