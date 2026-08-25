# ConR — Contrastive Regularizer for Deep Imbalanced Regression

**Project context:** Significant Wave Height (SWH) estimation from coastal webcam images
**Primary use here:** ConR regularization on top of a Huber base loss, using InceptionV3 penultimate features (the project's `inceptionv3_full_raw_huber_conr` config)
**Learning goal:** Understand ConR from scratch — deeply enough to explain it to a professor, implement/debug it in PyTorch, and connect it to the project's other imbalance mechanisms (LDS, RankSim).

> Note on notation: plain text/arithmetic is used throughout instead of LaTeX-style symbols, so this reads cleanly on GitHub.

---

## 1. One-sentence idea

**ConR pulls the learned features of similar-SWH samples together, and specifically pushes apart the features of samples the model is *currently, actively confusing* — dissimilar true SWH, but similar predicted SWH — preventing rare large-wave features from collapsing into the majority small-wave cluster.**

---

## 2. Why ConR exists (from the paper's abstract)

> "Imbalanced distributions are ubiquitous in real-world data. They create constraints on Deep Neural Networks to represent the minority labels and avoid bias towards majority labels. [...] In this work, we propose ConR, a contrastive regularizer that models global and local label similarities in feature space and prevents the features of minority samples from being collapsed into their majority neighbours." — Keramati et al., *ConR: Contrastive Regularizer for Deep Imbalanced Regression*, ICLR 2024.

A base regression loss (Huber, here) only asks "is the predicted number close to the true number?" It says nothing about how the CNN's internal feature space is organized. In an imbalanced dataset, this commonly produces **feature collapse**: because rare large-wave samples are vastly outnumbered by common small-wave samples, the network can find it "easier" to represent a rare sample's features as *close to* the dominant cluster rather than learning a genuinely distinct representation for it — even while the base loss is technically still being minimized. Once that collapse happens, the regression head has very little useful signal left to recover the correct rare-SWH prediction from.

ConR directly targets this failure mode by regularizing the **feature space**, not just the final prediction.

---

## 3. ConR vs. the other imbalance mechanisms in this project

| Method | What it changes |
|---|---|
| Plain Huber / MSE / MRE | How prediction error is penalized |
| Stratified Sampler | Which samples enter each mini-batch |
| LDS | How much each sample's base loss is weighted |
| RankSim | How learned feature relationships are organized (rank agreement) |
| **ConR** | **Directly pulls/pushes feature pairs based on label similarity AND the model's own current confusions** |

ConR does not touch the DataLoader and does not replace the base loss — like RankSim, it is a regularizer **added** to the base loss:

```
total_loss = base_loss(pred, target) + lambda_weight * conr_loss(features, target, pred, rarity_weights)
```

**ConR vs. RankSim, specifically:** both operate on the penultimate feature space, but RankSim only compares *label-space* neighbor rankings to *feature-space* neighbor rankings — it never looks at what the model is currently predicting. ConR is different: it explicitly uses the model's **current predictions** to find *hard negatives* — pairs the model is confusing *right now* — and spends its push-apart effort specifically on those.

---

## 4. Notation used below

- `features` — `[B, D]` penultimate feature vectors for a batch of size B (2048-D for InceptionV3 here)
- `targets_raw` — `[B]` true SWH values (meters)
- `preds_raw` — `[B]` the model's current predicted SWH, from the *same* forward pass as `features`
- `w` — similarity window (meters): true-SWH pairs within `w` of each other count as label-similar
- `e` — push-strength growth rate for hard negatives
- `t` — temperature applied to cosine-similarity logits
- `rarity_weights` — optional per-sample weight (this project's `density_weights.py` table) that amplifies a rare sample's push-apart strength and overall loss contribution
- `lambda_weight` — ConR's weight inside the total training loss

---

## 5. Step 1 — Define positive pairs (label-similar)

For every pair of samples `(i, j)` in a batch, compare their true SWH values:

```
label_distance(i, j) = |true_SWH_i - true_SWH_j|
positive pair (i, j)  if  label_distance(i, j) <= w      (w = 0.15 m in this project)
```

A sample is never its own positive pair (self-comparisons are excluded). Positive pairs are the ones ConR wants to have **similar** features — because their true SWH values are close.

---

## 6. Step 2 — Define hard negatives (the core ConR idea)

This is what makes ConR different from a plain contrastive loss. A pair is a **hard negative** only if *both* of these are true:

```
label_distance(i, j)      >  w    (their TRUE SWH values are NOT close)
prediction_distance(i, j) <= w    (but the model's CURRENT predictions ARE close)
```

In words: two samples whose *true* SWH is genuinely different, but which the model is **right now, actively** predicting as if they were similar. A pair with dissimilar labels that the model *also* predicts as dissimilar is not pushed at all — ConR spends its entire push-apart budget only on the model's real, current confusions, not on every dissimilar pair in the batch.

This uses the model's own predictions (detached — no gradient flows back through the mask itself) purely to *decide which pairs to penalize*; the actual gradient signal still flows through the feature similarity term.

---

## 7. Step 3 — Measure feature similarity

Features are L2-normalized, then compared with cosine similarity, scaled by a temperature `t`:

```python
q = F.normalize(features, dim=1)
k = F.normalize(features, dim=1)
similarity(i, j) = (q_i dot k_j) / t          # t = 0.07 in this project
```

A smaller temperature makes the similarity distribution sharper (more confident), a common trick borrowed from contrastive learning (e.g., SimCLR-style losses).

---

## 8. Step 4 — Scale the push strength

Not every hard negative is pushed equally hard. The push weight grows with how far apart the pair's true labels actually are, and is further amplified for rare samples:

```
pushing_weight(i, j) = rarity_weight_i * exp(label_distance(i, j) * e)      # e = 0.35 here
```

Intuition: a hard negative whose true SWH values are *very* far apart (e.g., 0.20 m vs. 2.40 m) is a more serious confusion than one only slightly outside the similarity window, so it gets pushed harder. And if the anchor sample itself is rare (via the `density_weights.py` rarity table), its overall contribution to the loss — including this push-apart term — is upweighted.

---

## 9. Step 5 — Combine into a per-anchor contrastive loss

For each anchor `i` with at least one positive and at least one hard negative, ConR computes an InfoNCE-style contrastive loss: pull the positive-pair similarities up, push the hard-negative similarities down, weighted by the pushing strength from Step 8. This is averaged over all *valid* anchors in the batch (anchors with zero positives, e.g. the sole sample in a rare bin with no other close-by sample, are excluded from being an anchor themselves — but they can still act as a hard negative that other anchors are pushed away from).

The final regularizer is added to the base Huber loss:

```
total_loss = Huber(pred, target) + lambda_weight * ConR_loss
```

---

## 10. Two SWH-specific robustness fixes in this project's implementation

The project's `conr_loss.py` is ported verbatim (no algorithmic changes) from the official ConR reference implementation, with two deliberate, documented engineering fixes needed at this dataset's scale:

1. **Zero-positives guard.** The official code divides by each anchor's positive-pair count with no safety check. At this dataset's scale, a batch can easily contain an anchor with *no* other sample within `w = 0.15 m` of its own label (e.g., the sole Rough-bin sample in a batch) — that anchor is now explicitly masked out of the loss instead of causing a division problem, mirroring the zero-negatives guard the official code already had.
2. **Mean over valid anchors only**, not over all B anchors in the batch — otherwise a batch with many masked-out anchors (common at this project's data scale, rarer on the original AgeDB benchmark the paper used) would get a systematically diluted loss value.

---

## 11. Worked example — verified by actually running the project's algorithm

A toy batch of 5 samples: four form a tight calm-water cluster (true SWH near 0.20 m), and one is a rare large wave (true SWH = 2.40 m) — but the model is *currently* mispredicting it at 0.23 m, i.e., its feature vector has effectively collapsed toward the calm cluster (simulated by setting its feature vector close to the cluster's average).

| idx | true SWH | predicted SWH | role |
|---|---|---|---|
| 0 | 0.20 m | 0.20 m | calm cluster |
| 1 | 0.22 m | 0.22 m | calm cluster |
| 2 | 0.18 m | 0.18 m | calm cluster |
| 3 | 0.21 m | 0.21 m | calm cluster |
| 4 | 2.40 m | 0.23 m | rare, currently mispredicted (collapsed features) |

Running the exact `conr_loss()` function from the project on this batch (`w=0.15, e=0.35, t=0.07`):

- Samples 0-3 are all positive pairs of each other (labels all within 0.15 m).
- Sample 4 is a **hard negative** for every one of samples 0-3: its true label is far away (label_distance > 0.15), but its current prediction is close (prediction_distance <= 0.15) — exactly the "model is confusing this right now" case ConR targets.
- Sample 4 itself has no positive pairs (no other sample is within 0.15 m of its own *true* label), so it is not counted as its own anchor — but it still gets pushed, since it is a negative that anchors 0-3 are pulled away from.
- Resulting ConR loss: **7.24** (finite, well-behaved).
- Gradient norm on sample 4's feature vector: **1.24** — noticeably larger than the gradient norm on sample 0's feature vector: **0.79**. ConR is concretely pushing sample 4's collapsed feature harder than it's adjusting the already-correct cluster members.

This is exactly the mechanism the paper's abstract describes: preventing the rare sample's features from staying collapsed into its majority neighbors.

---

## 12. The actual SWH ConR configuration used in this project

For `inceptionv3_full_raw_huber_conr`:

| Setting | Value |
|---|---|
| Backbone | InceptionV3 |
| Regression target | raw SWH (meters) |
| Base loss | Huber (delta = 0.15 m) |
| Mechanism | ConR |
| w (similarity window) | 0.15 m |
| e (push-strength growth rate) | 0.35 |
| t (temperature) | 0.07 |
| lambda_weight | 1.0 |
| density_alpha (rarity table exponent) | 1.0 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Validation monitor | val_huber |

**Note on hyperparameters:** `w`, `e`, and `t` are calibrated for this project's SWH scale (0.05-2.77 m), not the official ConR code's AgeDB (age-in-years) defaults. `w = 0.15` matches Huber's own delta. `e = 0.35` replaces the official default of 0.01 — since `e` is roughly `1/label_range`, this preserves the same *relative* push-strength effect at this project's much smaller ~0-2.7 m scale. `t = 0.07` is unchanged from the official default, since temperature operates on scale-independent cosine similarities in [-1, 1].

---

## 13. Full training flow, one mini-batch

```
1. Load a normal random mini-batch of images + true SWH   (unchanged by ConR)
                 |
                 v
2. InceptionV3 forward_features()  ->  penultimate features
                 |
                 v
3. forward_head(features)  ->  predicted SWH  (SAME forward pass as features)
                 |
        +--------+--------+
        |                 |
        v                 v
4A. Base Huber loss   4B. ConR branch:
   (pred, target)        - build positive pairs (label distance <= w)
        |                - build hard negatives (label far, PREDICTION close)
        |                - cosine similarity / temperature
        |                - pull positives, push hard negatives (scaled by
        |                  label distance and rarity)
        |                        |
        +------------+-----------+
                     |
                     v
        5. total_loss = Huber + lambda_weight * ConR_loss
                     |
                     v
              6. backward()
                     |
                     v
   Gradients flow through BOTH the regression head AND the feature extractor
                     |
                     v
              7. AdamW optimizer step -> updated weights
```

---

## 14. Why this can help imbalanced regression

Base regression losses only ever see the final scalar prediction — they have no way to notice that a rare sample's *internal representation* has quietly drifted into looking like the majority cluster, as long as the regression head still manages to output roughly the right number for it (which becomes progressively harder as training continues and the majority cluster's gradient dominates). ConR gives the network a second, more direct signal: it explicitly detects *when the model is currently confusing a rare sample with the majority cluster* (via the hard-negative mask, built from the model's own live predictions) and applies a targeted, label-distance- and rarity-scaled push to separate them in feature space — attacking the imbalance problem at its representational root, not just at the level of the final number.

---

## 15. When ConR is useful

1. The regression target is continuous and prone to rare-sample feature collapse — deep imbalanced regression in general.
2. A penultimate feature representation is available (not just a black-box final scalar).
3. You want a mechanism that adapts to what the model is *actually* getting wrong right now, rather than a purely static rebalancing scheme.
4. You can afford the extra `[B, B]` pairwise computation per batch.

## 16. When ConR may be less useful

- **Very small batches** — fewer pairwise relationships, and a higher chance that some anchors have zero positives or zero hard negatives in a given batch.
- **Very early training** — predictions are essentially random early on, so the hard-negative mask (which depends on current predictions) may not yet reflect meaningful confusions.
- **Hyperparameter sensitivity** — `w`, `e`, `t`, and `lambda_weight` all interact; hyperparameters tuned for one label scale (e.g., AgeDB's age-in-years) do not transfer directly to a very different scale (e.g., SWH's 0-2.7 m range) without recalibration, as documented in Section 12 above.
- **Computational overhead** — an `[B, B]` pairwise similarity and masking computation per batch, on top of the base loss.

---

## 17. Real-world applications

**Computer vision:** age estimation (AgeDB — the original ConR paper's benchmark), depth estimation, crowd counting, medical severity scoring — anywhere continuous labels are imbalanced and a CNN produces a penultimate embedding.

**Marine and ocean engineering:** significant wave height (this project), rare extreme sea-state or storm-condition regression, wind-wave severity estimation.

**General deep imbalanced regression:** any setting where minority-label feature collapse is suspected — for example, a model whose per-bin metrics look poor on rare bins even though it appears to be a reasonably well-fit model overall.

---

## 18. Why this matters for an ML / DL engineering career

**Diagnosing feature collapse.** Recognizing that a regression model can be "technically minimizing the loss" while its internal representation for rare cases has quietly become indistinguishable from the majority is a genuinely useful diagnostic instinct — this is a common, non-obvious failure mode in imbalanced deep learning.

**Using live model state inside a loss function.** ConR's hard-negative mask depends on the model's own current predictions, detached from the gradient graph, used purely to decide *which pairs to penalize* this step. This "use the model's current behavior to decide what to regularize" pattern reappears in many advanced training techniques (curriculum learning, online hard-example mining, self-distillation).

**Contrastive learning fundamentals.** Cosine similarity, temperature scaling, positive/negative pair construction, and InfoNCE-style losses are foundational building blocks across modern representation learning (SimCLR, CLIP, and beyond) — ConR is a concrete, well-scoped example to have fully worked through by hand.

**Robustness engineering around research code.** The zero-positives guard and valid-anchor-only averaging (Section 10) are a good real example of the gap between a paper's clean mathematical description and the defensive code needed to run it safely on a new, differently-shaped dataset.

---

## 19. Questions a professor may ask

1. **What is the one-sentence motivation?** Prevent rare-sample features from collapsing into the majority cluster, by directly pulling similar-label features together and pushing apart features the model currently confuses.
2. **Does ConR replace the base loss?** No — it's added: `total_loss = Huber + lambda_weight * ConR_loss`.
3. **What makes a pair a "hard negative"?** True labels are far apart (> w) but the model's *current predictions* are close (<= w) — a live confusion.
4. **Why use the model's own predictions to build the mask?** So ConR only spends effort on the confusions the model is actually making right now, not on every dissimilar pair in the batch.
5. **How is push strength scaled?** By how far apart the true labels are (larger label distance = harder push) and by the anchor's rarity weight.
6. **How is ConR different from RankSim?** RankSim compares label-space and feature-space neighbor *rankings* only; ConR explicitly uses the model's current predictions to find and target *live confusions*.
7. **How is ConR different from LDS?** LDS reweights the base loss per sample; ConR adds an entirely separate contrastive term operating on the feature space.
8. **What happens to a sample with no positive pairs in a batch?** It's excluded as an anchor (zero-positives guard) but can still serve as a hard negative pushing other anchors away.
9. **What are w, e, and t?** w = label-similarity window (meters); e = push-strength growth rate with label distance; t = temperature on cosine similarities.
10. **Were these hyperparameters tuned for this project?** w and e were recalibrated for the SWH scale (0-2.7 m) versus the official code's AgeDB (age-in-years) defaults; t was left unchanged since it's scale-independent.

---

## 20. Questions likely in ML / DL interviews

**Concept:** What is contrastive learning? What is InfoNCE? Why does temperature scaling matter for contrastive losses? What is a "hard negative" and why do hard negatives matter more than random negatives? What is representation/feature collapse, and why can it happen even while a base loss keeps decreasing?

**PyTorch/engineering:** How would you implement pairwise label-distance and prediction-distance matrices efficiently? Why detach the predictions used to build the hard-negative mask? What's the memory cost of an all-pairs `[B, B]` computation, and how does it scale with batch size? How would you unit-test a masking function like this (edge cases: zero positives, zero negatives, batch size 1)?

**Research-engineering judgment:** How would you decide whether hyperparameters from one dataset (AgeDB) transfer to another (SWH)? What experiment would prove ConR's benefit is coming from the hard-negative mechanism specifically, and not just from adding any generic contrastive term? How would you monitor "feature collapse" directly during training (e.g., via embedding visualization or nearest-neighbor purity)?

---

## 21. Practical debugging checklist

- [ ] Confirm `features`, `targets_raw`, and `preds_raw` come from the **same forward pass** in the same batch.
- [ ] Confirm predictions used for the hard-negative mask are detached (no gradient through the mask itself).
- [ ] Confirm the positive-pair matrix excludes self-comparisons (diagonal is False).
- [ ] Confirm anchors with zero positives are excluded from the per-anchor average (not silently producing 0/0).
- [ ] Confirm the loss averages over *valid* anchors, not over all B anchors in the batch.
- [ ] Confirm ConR loss is finite (not NaN/Inf) on both typical batches and edge cases (batch size 1, all-similar batch, all-dissimilar batch).
- [ ] Confirm rarity weights (if used) are built from the training split only.
- [ ] Log the base loss and ConR loss separately during development.
- [ ] Check sensitivity to `w`, `e`, `t`, and `lambda_weight` — especially after porting hyperparameters from a different label scale.
- [ ] Compare rare-bin test metrics against the same baseline without ConR.

---

## 22. How to explain ConR aloud in about one minute

> "ConR is a contrastive regularizer for imbalanced regression. It adds a term to the base loss that operates directly on the CNN's penultimate features. For each pair of samples in a batch, if their true SWH values are close, ConR treats them as a positive pair and pulls their features together. But its key idea is how it picks negatives: instead of pushing apart every pair with different labels, it only pushes apart 'hard negatives' — pairs whose true SWH is different, but whose *current predictions* are similar, meaning the model is actively confusing them right now. The push strength scales up with how far apart the true labels actually are, and with how rare the sample is. This is added on top of my base Huber loss, and it directly targets the failure mode where a rare large-wave sample's features quietly collapse into the majority small-wave cluster, even while the base loss keeps decreasing."

---

## 23. Five things to remember

1. ConR = pull similar-label features together, push apart features the model is *currently* confusing.
2. "Hard negative" = labels far apart, but the model's current predictions are close — a live confusion, not a static rule.
3. Push strength scales with label distance and sample rarity.
4. It's a regularizer added to the base loss, not a replacement — like RankSim, unlike LDS.
5. It operates on penultimate features, using the model's own (detached) current predictions to decide what to penalize.

---

## 24. Source-grounded implementation summary

```python
l_dist = torch.abs(l_q - l_k)               # [B,B] true-label distances
p_dist = torch.abs(p_q - p_k)               # [B,B] current-prediction distances

pos_i = l_dist.le(w)                        # label-similar pairs
neg_i = (~l_dist.le(w)) & p_dist.le(w)       # label-far, PREDICTION-close: hard negatives
pos_i.fill_diagonal_(False)

prod = torch.matmul(q, k.T) / t              # cosine similarity / temperature
pos = prod * pos_i
neg = prod * neg_i

pushing_w = weights * torch.exp(l_dist * e)  # push scales with label distance + rarity
neg_exp_dot = (pushing_w * torch.exp(neg) * neg_i).sum(1)

per_anchor = (-torch.log(
    torch.exp(pos) / (torch.exp(pos).sum(1, keepdim=True) + neg_exp_dot.unsqueeze(-1) + 1e-12)
) * pos_i).sum(1) / pos_i.sum(1).clamp(min=1)

valid = pos_i.sum(1).bool() & neg_i.sum(1).bool()   # zero-positives / zero-negatives guard
loss = (weights.squeeze(-1) * per_anchor * valid).sum() / valid.float().sum().clamp(min=1)
```

Full source: `src/conr_loss.py` (ported verbatim from the official ConR reference implementation, `BorealisAI/ConR`, `agedb-dir/loss.py`, plus the two robustness fixes in Section 10).

---

## 25. References

1. Keramati, M., Meng, L., & Evans, R. D. (2024). *ConR: Contrastive Regularizer for Deep Imbalanced Regression.* International Conference on Learning Representations (ICLR).
2. Official reference implementation: `BorealisAI/ConR` GitHub repository.
3. Yang, Y., Zha, K., Chen, Y.-C., Wang, H., & Katabi, D. (2021). *Delving into Deep Imbalanced Regression.* ICML — the deep imbalanced regression benchmark setting ConR builds on.
4. Project implementation files: `src/conr_loss.py`, `src/density_weights.py`, `main.py` (`build_conr_config()`, `train_one_epoch()`), `configs/inceptionv3_full_raw_huber_conr.yaml`.

---

## Personal learning checkpoint

Before writing this up for the manuscript, make sure you can answer these without notes: What exactly counts as a hard negative, and why does it depend on the model's *current* predictions rather than just the labels? Why is a plain "push apart everything with different labels" contrastive loss not what ConR does? How does the push strength get scaled, and why? How is ConR different from RankSim (both operate on features) and from LDS (a loss-reweighting mechanism)? Why can a sample have zero positives, and what happens to it?

If these are all clear without looking back, the mechanism is understood at a strong practical level.
