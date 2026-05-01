# Error Analysis — Flickr Image Captioning

**Model:** `models/attention_gru_glove.pth` — ResNet50 + Bahdanau attention + GRU decoder + GloVe 300d
**Evaluation:** beam search k=5 on the merged Flickr8k+30k test split (1,873 images × 5 reference captions)
**Test-set metrics:** BLEU-4 = **0.3093**, METEOR = **0.4709**, CIDEr = **0.7961**, ROUGE-L = **0.5257**

Full Phase 6 notebook with rendered plots and image grid: [`notebooks/03_error_analysis.ipynb`](../notebooks/03_error_analysis.ipynb).

---

## 1. Per-image BLEU-4 distribution

Although **corpus** BLEU-4 is 0.3093, the **per-image** median is only 0.19. The distribution is heavily right-skewed: 31.1% of predictions have sentence-BLEU-4 below 0.1 (essentially random), while 36.6% score above 0.3 (good) and 19.4% above 0.5 (excellent). Corpus BLEU is optimistic because short common phrases like *"a man is"* contribute to every prediction regardless of image content.

| Statistic | Value |
|---|---:|
| n images | 1,873 |
| mean sentence BLEU-4 | 0.2817 |
| median sentence BLEU-4 | 0.1863 |
| fraction < 0.1 | 31.1% |
| fraction > 0.3 | 36.6% |
| fraction > 0.5 | 19.4% |

## 2. Six error categories

Phase 6 notebook §2 contains 12 hand-picked test examples (2 per category) with rendered images and per-example explanations. Summary:

| # | Type | Root cause |
|---|---|---|
| 1 | **Missing Object** | No count-aware representation — the context vector collapses object count. Model says "a man and a child on a bike" when GT says "two children ride their tricycles". |
| 2 | **Wrong Attribute** | ImageNet features are discriminative for class but not for exact color. Model emits "a little girl in a blue dress" when GT says "a young boy in pajamas tossing a red ball". |
| 3 | **Wrong Relationship** | No symbolic reasoning over subject-object interactions. Model identifies "boy + beach" correctly but the action *carrying* becomes *standing*. |
| 4 | **Generic Caption** | Model collapses rare scenes to the highest-probability caption. "A male hitchhiking with a big bag" → *"a man and a woman are walking down the street"* (6 images got this exact caption). |
| 5 | **Hallucination** | When visual evidence is weak, the language-model prior takes over. "A little boy jumping in a bowling alley" → *"a woman in a blue shirt standing in front of a group of people"* (gender, age, and activity all invented). |
| 6 | **Repetition** | GRU hidden state enters an attractor: *"a dog dog is jumping over an obstacle"*; *"traditional traditional dress with a traditional headdress"*. Repetition penalty of 1.2 mitigates but doesn't fully fix this. |

## 3. Vocabulary coverage — the "safe word" bias

The training vocabulary has **10,111 words** (frequency threshold 3 on the train split). Over the entire 1,873-image test set the model predicts only **892 unique words** — **8.8% of the vocabulary**. In contrast, the ground-truth reference captions use far more diverse vocabulary.

The top-10 predicted tokens are dominated by function words and generic nouns:

| Rank | Predicted | Count | GT | GT count |
|---:|---|---:|---|---:|
| 1 | `a` | 5021 | `a` | 15936 |
| 2 | `in` | 1617 | `.` | 8712 |
| 3 | `is` | 909 | `in` | 4837 |
| 4 | `man` | 745 | `the` | 3918 |
| 5 | `shirt` | 662 | `on` | 2694 |
| 6 | `of` | 648 | `man` | 2542 |
| 7 | `and` | 557 | `is` | 2439 |
| 8 | `on` | 488 | `and` | 2410 |
| 9 | `the` | 426 | `of` | 2171 |
| 10 | `are` | 425 | `with` | 2061 |

Only `man` and `shirt` carry real content beyond function words. The model has learned a template "a [person] is [verb] in/on a [surface]" and fills in from a small pool — despite having a 10,111-word vocabulary, only 8.8% is ever used.

## 4. Caption diversity

| metric | predicted | ground truth | ratio |
|---|---:|---:|---:|
| distinct-1 | 0.0425 | 0.0467 | 0.91× |
| distinct-2 | 0.1266 | 0.2980 | **0.42×** |
| distinct-3 | 0.2511 | 0.6065 | **0.41×** |
| mean length | 11.21 | 13.09 | 0.86× |
| length std | 3.76 | 5.18 | 0.73× |

Predicted **bigrams are ~2.4× less diverse** than ground-truth bigrams. At the unigram level the model is close to GT (0.91×), but at the phrase level the template pattern dominates — predictions settle into recurring bigrams like *"a man"*, *"is wearing"*, *"a white shirt"*, *"in the"*. The length distribution is ~14% shorter and ~27% narrower than GT, consistent with the model hitting `<end>` once the "safe" phrase is emitted.

## 5. Answers to required questions (PDF §16)

**Q1 — Why CNN for images?** Spatial locality, weight sharing, and hierarchical feature composition match the structure of visual data. Transfer learning from ImageNet is the other reason: our frozen ResNet50 spatial features (49 locations × 2048-d) gave the best val BLEU-4; unfreezing more than the last 2 blocks on 37K images caused overfitting.

**Q2 — Why LSTM/GRU for text?** Captions have long-range dependencies that a feed-forward network cannot express. Gating mechanisms (LSTM: input/forget/output gates + cell state; GRU: reset + update gates) solve the vanishing-gradient problem of vanilla RNNs. Our ablation: attention+GRU+GloVe reached the best test BLEU-4 = 0.3093, beating attention+LSTM+GloVe (0.2932) — GloVe warm-start compensates for the GRU's lack of cell state.

**Q3 — Limitations?** Six concrete ones, all shown in §2: (1) no object counting, (2) no spatial/relational reasoning, (3) safe-word bias (only 8.8% of vocab used), (4) template-like phrase diversity (2.4× less diverse bigrams than humans), (5) fixed 224×224 encoder loses fine detail, (6) exposure bias from teacher forcing (beam search gives a significant boost over greedy).

**Q4 — Why incorrect or repetitive captions?** Incorrect: language-model prior overrides weak visual evidence — hallucinations happen when the image is ambiguous and the decoder's internal n-gram statistics dominate. Repetitive: the GRU/LSTM hidden state gets stuck in an attractor where its own output is its own most-likely next input. Repetition penalty helps but can't eliminate it without hurting legitimate reuse.

**Q5 — What to improve with more time/resources?** (1) Transformer decoder instead of LSTM/GRU; (2) MSCOCO 330K dataset (40× more data) to unlock deeper fine-tuning; (3) bottom-up attention via Faster R-CNN object proposals instead of a 7×7 grid; (4) SCST — REINFORCE with CIDEr reward — to fix exposure bias; (5) CLIP ViT-L/16 vision encoder for richer patch-level features; (6) larger input resolution (384 or 448) for small-object recognition; (7) proper full-scale HPO (Hyperband or ASHA).

---

## Summary for the final report

The attention GRU + GloVe captioner reaches **BLEU-4 = 0.3093** on the merged Flickr8k+30k test set — the best of all six trained models. It beats the no-attention baseline by **+29.4% relative** (0.3093 vs 0.2390 for Baseline GRU). GloVe warm-start contributed the largest single improvement for the GRU decoder (+20% relative). The remaining failures are dominated by four systematic issues: the safe-word bias (only 8.8% of vocabulary used), template-like phrase diversity (2.4× less diverse bigrams than humans), the language-model prior overriding weak visual evidence on ambiguous images, and the absence of object counting and spatial reasoning. All four are **dataset-and-architecture limitations, not training bugs** — and they would be addressed in order by moving to MSCOCO, adding a Transformer decoder, using bottom-up attention, and applying SCST with CIDEr reward.

---

## References

- Xu et al. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML. https://arxiv.org/abs/1502.03044
- Bahdanau et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.* arXiv:1409.0473. https://arxiv.org/abs/1409.0473
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV. https://arxiv.org/abs/1610.02391
