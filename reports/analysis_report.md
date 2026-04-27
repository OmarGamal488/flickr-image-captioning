# Error Analysis — Flickr8k Image Captioning

**Model:** `models/attention_lstm.pth` — ResNet50 + Bahdanau attention + LSTM decoder
**Evaluation:** beam search k=5 on the Flickr8k test split (1091 images × 5 reference captions)
**Test-set metrics:** BLEU-4 = **0.2403**, METEOR = **0.4270**, CIDEr = **0.6002**, ROUGE-L = **0.4788**

Full Phase 6 notebook with rendered plots and image grid: [`notebooks/03_error_analysis.ipynb`](../notebooks/03_error_analysis.ipynb).

---

## 1. Per-image BLEU-4 distribution

Although **corpus** BLEU-4 is 0.24, the **per-image** median is only 0.13. The distribution is heavily right-skewed: 42.9% of predictions have sentence-BLEU-4 below 0.1 (essentially random), while 25.9% score above 0.3 (good) and 10.5% above 0.5 (excellent). Corpus BLEU is optimistic because short common phrases like *"a man is"* contribute to every prediction regardless of image content.

| Statistic | Value |
|---|---:|
| n images | 1091 |
| mean sentence BLEU-4 | 0.2106 |
| median sentence BLEU-4 | 0.1276 |
| fraction < 0.1 | 42.9% |
| fraction > 0.3 | 25.9% |
| fraction > 0.5 | 10.5% |

## 2. Six error categories

Phase 6 notebook §2 contains 12 hand-picked test examples (2 per category) with rendered images and per-example explanations. Summary:

| # | Type | Root cause |
|---|---|---|
| 1 | **Missing Object** | No count-aware representation — the context vector collapses object count. Model says "a dog" when GT says "two dogs" retrieving a stick. |
| 2 | **Wrong Attribute** | ImageNet features are discriminative for class but not for exact color. Model confidently emits the most common coat color ("brown dog") instead of describing the unique attribute ("blue collar"). |
| 3 | **Wrong Relationship** | No symbolic reasoning over subject-object interactions. Model identifies "boy + boogieboard + beach" correctly but the action *carrying* becomes *running*. |
| 4 | **Generic Caption** | Model stops after the first high-confidence phrase. "A man playing guitar sings into a microphone" → *"a man playing a guitar"*. |
| 5 | **Hallucination** | When visual evidence is weak, the language-model prior takes over. "A boy plays baseball" → *"a young girl is throwing a soccer ball"* (gender + sport both invented). |
| 6 | **Repetition** | LSTM hidden state enters an attractor where the previous word's top continuation is itself: *"a wooden wooden deck"*. Repetition penalty of 1.2 mitigates but doesn't fully fix this. |

## 3. Vocabulary coverage — the "safe word" bias

The training vocabulary has **2557 words** (frequency threshold 5 on the train split). Over the entire 1091-image test set the model predicts only **400 unique words** — **15.6% of the vocabulary**. In contrast, the ground-truth reference captions for the same images use 3281 distinct surface tokens across 5455 references (many OOV w.r.t. the training vocab).

The top-10 predicted tokens are dominated by function words:

| Rank | Predicted | Count | GT | GT count |
|---:|---|---:|---|---:|
| 1 | `a` | 2319 | `a` | 8443 |
| 2 | `in` | 620 | `the` | 2579 |
| 3 | `is` | 499 | `in` | 2518 |
| 4 | `the` | 366 | `on` | 1396 |
| 5 | `on` | 325 | `and` | 1264 |
| 6 | `man` | 289 | `is` | 1242 |
| 7 | `of` | 283 | `dog` | 1188 |
| 8 | `dog` | 274 | `with` | 1081 |
| 9 | `and` | 255 | `of` | 970 |
| 10 | `white` | 194 | `man` | 916 |

Only `man`, `dog`, and `white` carry real content. The model has essentially learned to emit a template "a [person/animal] is [verb] on/in a [surface]" and fills in the blanks from a small pool.

## 4. Caption diversity

| metric | predicted | ground truth | ratio |
|---|---:|---:|---:|
| distinct-1 | 0.0375 | 0.0555 | 0.68× |
| distinct-2 | 0.1099 | 0.3051 | **0.36×** |
| distinct-3 | 0.2214 | 0.6056 | **0.37×** |
| mean length | 9.77 | 10.83 | 0.91× |
| length std | 2.80 | 3.73 | 0.75× |

Predicted **bigrams are ~2.8× less diverse** than ground-truth bigrams. At the unigram level the model is only 30% less diverse, but at the phrase level the template pattern dominates — predictions settle into a small set of recurring bigrams like *"a man"*, *"is running"*, *"is standing"*, *"in the"*. The length distribution is ~10% shorter and ~25% narrower than GT, consistent with the model hitting `<end>` once the "safe" phrase is emitted.

## 5. Answers to required questions (PDF §16)

**Q1 — Why CNN for images?** Spatial locality, weight sharing, and hierarchical feature composition match the structure of visual data. Transfer learning from ImageNet is the other reason: our frozen ResNet50 features gave BLEU-4 = 0.2153 at epoch 12; unfreezing them on 8K images caused overfitting and dragged BLEU-4 down.

**Q2 — Why LSTM/GRU for text?** Captions have long-range dependencies that a feed-forward network cannot express. Gating mechanisms (LSTM: input/forget/output gates + cell state; GRU: reset + update gates) solve the vanishing-gradient problem of vanilla RNNs. Our own ablation: attention+GRU reached BLEU-4 = 0.2125 at **epoch 6** while attention+LSTM needed epoch 12 for 0.2153 — GRU is ~2× faster at the same quality.

**Q3 — Limitations?** Six concrete ones, all shown in §2: (1) no object counting, (2) no spatial reasoning, (3) safe-word bias (only 15% of vocab used), (4) dataset-size overfitting (confirmed by our HPO transfer failure), (5) fixed 224×224 encoder loses fine detail, (6) exposure bias from teacher forcing (which is why beam search gives a "free" +3.3 BLEU-4 points over greedy).

**Q4 — Why incorrect or repetitive captions?** Incorrect: language-model prior overrides weak visual evidence — hallucinations happen when the image is ambiguous and the decoder's internal trigram statistics dominate (*"girl throwing soccer ball"* is a higher-frequency sequence than *"boy plays baseball"*). Repetitive: the LSTM hidden state gets stuck in an attractor where its own output is its own most-likely next input. Repetition penalty helps but can't eliminate it without hurting legitimate reuse.

**Q5 — What to improve with more time/resources?** (1) Transformer decoder instead of LSTM; (2) MSCOCO 330K dataset (40× more data) to unlock fine-tuning; (3) bottom-up attention via Faster R-CNN object proposals instead of a 7×7 grid; (4) SCST — REINFORCE with CIDEr reward — to fix exposure bias; (5) ViT-L/16 or CLIP vision encoder; (6) scheduled sampling during training; (7) proper full-scale HPO (Hyperband or ASHA); (8) larger input resolution (384 or 448) for small-object recognition.

---

## Summary for the final report

The attention LSTM captioner reaches **BLEU-4 = 0.2403** on the Flickr8k test set — within the PDF's expected range for an attention model. It beats the no-attention baseline by **+21.5% relative**. The remaining failures are dominated by five systematic issues: the safe-word bias (only 15% of vocabulary used), template-like phrase diversity (2.8× less diverse bigrams than humans), the language-model prior overriding weak visual evidence on ambiguous images, the absence of object counting and spatial reasoning, and the dataset-size ceiling that prevents deeper fine-tuning. All five are **dataset-and-architecture limitations, not training bugs** — and they would be addressed in order by moving to MSCOCO, adding a Transformer decoder, using bottom-up attention, and applying SCST with CIDEr reward.
