# RankSim — Ranking-Similarity Regularization for Imbalanced Regression

**Project context:** Significant Wave Height (SWH) estimation from coastal webcam images
**Primary use here:** RankSim regularization on InceptionV3/ResNet-50 feature embeddings, combined with a base regression loss (Huber, in the project's `inceptionv3_full_raw_huber_ranksim` config)
**Learning goal:** Understand RankSim from scratch — deeply enough to explain it to a professor, implement/debug it in PyTorch, and connect it to broader ML engineering concepts.

> Note on notation: this note deliberately uses plain text/arithmetic instead of LaTeX-style symbols, so it reads cleanly on GitHub and is easy to follow without a math renderer.

---

## 1. One-sentence idea

**RankSim makes samples that are close in true SWH also end up close (similarly ranked) in the network's learned feature space.**

If 0.20 m is closer to 0.30 m than to 1.20 m or 2.40 m in *true SWH*, RankSim pushes the network to preserve that same closeness ordering in its *learned features*.

---

## 2. Why RankSim exists

A regression loss like MSE or Huber only asks one question per sample:

> "Is the predicted SWH numerically close to the true SWH?"

That's necessary, but it says nothing about *how the CNN organizes its internal feature space*. Two images with very different true SWH could end up with very similar internal features, and the base loss would never notice — as long as the final regression head still outputs the right number.

RankSim adds a second objective on top of the base loss:

> "The neighborhood structure in feature space should match the neighborhood structure in SWH-label space."

This is why RankSim is a **regularizer** — an extra term added to the loss — not a replacement for Huber/MSE/MRE.

---

## 3. Visual intuition

![RankSim workflow for SWH estimation](assets/RankSim_SWH_Workflow.png)

The core flow, in words:

```
Mini-batch of images + true SWH
            |
            v
      CNN feature extractor
            |
            +-----------------------------+
            |                             |
            v                             v
   Label-space ranking            Feature-space ranking
   (based on SWH distance)        (based on cosine similarity)
            |                             |
            +-------------+---------------+
                          |
                          v
                 Compare the two rankings
                          |
                          v
                    RankSim loss
                          |
                          v
Total loss = Base regression loss + gamma x RankSim loss
                          |
                          v
                 Backward pass + Optimizer (AdamW)
```

---

## 4. RankSim vs. the other imbalance mechanisms — what each one actually changes

| Method | What it changes |
|---|---|
| Plain Huber / MSE / MRE | How prediction error is penalized |
| Stratified Sampler | Which samples enter each mini-batch |
| LDS | How much each sample's base loss is weighted |
| **RankSim** | **How learned feature relationships are organized** |
| ConR | Adds a contrastive-style representation regularizer |

RankSim does **not** touch the DataLoader and does **not** replace the base loss — it works entirely inside the loss function, on the CNN's internal features.

---

## 5. Notation used below

A mini-batch (or a selected subset of it) of size M: samples `(x_i, y_i)` for `i = 1 ... M`.

- `x_i` = input image
- `y_i` = true continuous target (SWH, in meters)
- `f(x_i)` = CNN feature extractor
- `z_i = f(x_i)` = learned feature vector for sample i (2048-dimensional for InceptionV3's penultimate layer)
- `S_y(i,j)` = similarity between samples i and j, measured in *label* space
- `S_z(i,j)` = similarity between samples i and j, measured in *feature* space
- `rk(...)` = ranking operation (converts a row of similarities into rank positions)
- `L_RankSim` = the RankSim regularization loss
- `gamma` = how much weight RankSim gets in the total training loss
- `lambda` = a separate parameter used only inside the differentiable-ranking approximation (not a loss weight — see Section 14)

---

## 6. Step 1 — Build label-space relationships

Label similarity is defined as **negative absolute distance**:

```
S_y(i, j) = -( |y_i - y_j| )
```

Why negative? Because a *smaller* distance should mean *more* similar, and negating the distance flips "small distance" into "large similarity value."

### SWH toy batch (from the diagram)

```
y = [0.20, 0.30, 1.20, 2.40]  meters
```

Take 0.20 m as the anchor sample. Absolute SWH distances from the anchor:

```
|0.20 - 0.20| = 0.00
|0.20 - 0.30| = 0.10
|0.20 - 1.20| = 1.00
|0.20 - 2.40| = 2.20
```

Desired neighbor order (closest to farthest), based purely on the true labels:

```
0.20 m  ->  0.30 m  ->  1.20 m  ->  2.40 m
closest                              farthest
1st        2nd          3rd            4th
```

This ordering comes only from the labels — the network hasn't been involved yet.

---

## 7. Step 2 — Extract deep features

In the InceptionV3 RankSim model, the CNN produces a **2048-dimensional** feature vector right before the final regression head (the "penultimate" layer):

```
Image
  |
  v
InceptionV3 backbone
  |
  v
2048-D feature vector z
  |
  v
Regression head
  |
  v
Predicted SWH
```

The base regression loss (Huber, here) uses the final predicted SWH number. RankSim, in contrast, works directly on the 2048-D feature vector `z` — one step earlier in the pipeline. This is why a RankSim-enabled model needs a split forward pass:

```python
features = model.forward_features(images)   # 2048-D, used by RankSim
pred = model.forward_head(features)          # scalar SWH, used by Huber
```

---

## 8. Step 3 — Build feature-space similarities

For two feature vectors `z_i` and `z_j`, feature-space similarity is **cosine similarity**:

```
S_z(i, j) = (z_i dot z_j) / ( ||z_i|| * ||z_j|| )
```

Cosine similarity only cares about the *direction* (angle) of the vectors, not their length:

- close to +1 → very similar direction
- close to 0 → weak relationship
- close to -1 → opposite direction

Equivalent PyTorch operation (normalize each vector first, then a single matrix multiply gives every pairwise similarity at once):

```python
z = F.normalize(features, dim=1)
S_z = z @ z.T
```

For a batch of size B, this produces a `B x B` similarity matrix — one row per anchor sample.

---

## 9. Step 4 — Convert similarities into rankings

RankSim does not try to match the exact similarity *numbers*. It asks a softer question:

> "Is the *order* of neighbors correct?"

For each anchor sample, two rankings are produced: a label-space rank and a feature-space rank.

### Example

Desired label ranking for anchor 0.20 m (from Step 6): `[0.20, 0.30, 1.20, 2.40]`.

Suppose the network's *current* features instead rank the neighbors as: `[0.20, 1.20, 0.30, 2.40]`.

The network has placed 1.20 m ahead of 0.30 m — that's a disagreement with the true SWH ordering, and RankSim penalizes it.

---

## 10. Normalized ranking (what's actually compared)

The implementation converts each ranking into **normalized ranks between 0 and 1**. With 4 items in the anchor's neighbor set, the four possible normalized ranks are:

```
[0.25, 0.50, 0.75, 1.00]
```

Convention: lower normalized rank = more similar (closer); higher normalized rank = less similar (farther):

```
most similar        -> 0.25
2nd most similar     -> 0.50
3rd most similar     -> 0.75
least similar        -> 1.00
```

---

## 11. Step 5 — Calculate the RankSim loss

For each anchor, compare its label-space rank vector `r_y` and feature-space rank vector `r_z` using mean squared error:

```
L_i(RankSim) = MSE(r_z, r_y) = average of (r_z - r_y)^2, over all neighbors
```

Then average this across every anchor in the batch:

```
L_RankSim = (1 / M) * sum over all anchors i of L_i(RankSim)
```

### Numeric example

```
r_y = [0.25, 0.50, 0.75, 1.00]     (true SWH ranking)
r_z = [0.25, 0.75, 0.50, 1.00]     (current feature ranking — 2nd and 3rd swapped)

L_i = [ (0.25-0.25)^2 + (0.75-0.50)^2 + (0.50-0.75)^2 + (1.00-1.00)^2 ] / 4
    = [ 0 + 0.0625 + 0.0625 + 0 ] / 4
    = 0.03125
```

If the rankings match perfectly (`r_z = r_y`), then `L_i = 0` — no penalty.

---

## 12. Step 6 — Duplicate targets are thinned out first

Before ranking, the project's `batchwise_ranking_regularizer()` first finds the *unique* target values in the batch. If a target value repeats (e.g., two samples both labeled exactly 0.30 m), it randomly keeps at most one of them for the ranking subset.

Purpose:
- reduces ranking ties (ties make "correct order" ambiguous)
- gives rare target values relatively more influence within the ranking subset

**Important distinction:** this is *not* Stratified Sampling. It doesn't touch the original DataLoader batch — it only trims the ranking subset *inside* an already-formed batch.

---

## 13. Step 7 — Why ranking needs a special "differentiable" trick

Ordinary sorting/ranking is a problem for gradient descent: if the input values change slightly but their *order* doesn't change, the rank output doesn't change at all. That makes the gradient zero (or undefined) almost everywhere — useless for backpropagation.

RankSim solves this with a differentiable black-box ranking approximation (the project uses a `TrueRanker` operator):

**Forward pass:** compute the actual normalized ranking, exactly as described above.

**Backward pass:** instead of a zero gradient, perturb the similarity scores by a small step in the direction the loss wants, re-rank, and measure how much the ranking changed:

```
a_perturbed = a + lambda * (dL / d_rank)

dL/da  ≈  -(1 / lambda) * ( rank(a) - rank(a_perturbed) )
```

This "perturb and re-measure" trick is what allows a useful (non-zero) gradient to flow from the RankSim loss back into the CNN's features, even though ranking itself isn't naturally differentiable.

---

## 14. What `lambda` means (don't confuse with gamma)

Project value: `interp_strength_lambda = 1.0`

`lambda` is **not** a loss weight. It belongs entirely to the differentiable-ranking trick in Section 13 — it controls how big a perturbation is used to approximate the ranking gradient.

```
lambda
   |
   v
controls how the non-differentiable rank operation is approximated during backward()
```

---

## 15. What `gamma` means

Project value: `gamma = 1.0`

`gamma` is the weight given to the RankSim term inside the total training loss:

```
L_total = L_base + gamma * L_RankSim
```

- `gamma = 0` → RankSim has no effect at all
- larger `gamma` → ranking regularization contributes more strongly
- too large `gamma` → the ranking objective can start competing with, instead of supporting, the direct regression objective

---

## 16. The actual SWH RankSim configuration used in this project

For `inceptionv3_full_raw_huber_ranksim`:

| Setting | Value |
|---|---|
| Backbone | InceptionV3 |
| Input size | 299 x 299 |
| Pretrained | ImageNet |
| Penultimate feature | 2048-D |
| Regression target | raw SWH (meters) |
| Base loss | Huber |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| RankSim lambda | 1.0 |
| RankSim gamma | 1.0 |
| Validation monitor | val_huber |

**Note:** some comments inherited from earlier config templates mention MSE, but the actual YAML sets `training.loss: huber`. So for this run:

```
L_total = L_Huber + 1.0 * L_RankSim
```

---

## 17. Full training flow, one mini-batch

```
1. Load webcam images and raw SWH targets
                 |
                 v
2. InceptionV3 forward_features()
                 |
                 v
       2048-D feature vectors
          /                 \
         /                   \
        v                     v
3A. Regression head      3B. RankSim branch
        |                     |
        v                     v
Predicted SWH         Label & feature rankings
        |                     |
        v                     v
4A. Base Huber loss       RankSim loss
         \                   /
          \                 /
           v               v
     5. Total training loss
     L = Huber + gamma * RankSim
                 |
                 v
        6. backward()
                 |
                 v
Gradients flow through regression head AND feature extractor
                 |
                 v
         7. AdamW optimizer step
                 |
                 v
          Updated model weights
```

---

## 18. Is RankSim itself "trained"?

RankSim is a **loss term**, not a separate trainable network — there are no independent RankSim weights.

What actually happens:
1. RankSim loss is computed from the CNN's *current* features.
2. `backward()` computes the gradient of that loss with respect to those features.
3. That gradient propagates back into the CNN feature extractor (via the perturb-and-re-rank trick from Section 13).
4. AdamW updates the CNN's actual parameters.
5. Over training, the CNN gradually learns to produce SWH-consistent feature rankings.

So: **the model's own parameters are optimized to reduce both the base regression loss and the RankSim regularization loss, at the same time.**

---

## 19. Base loss and RankSim gradients combine

Example numbers:

```
L_Huber    = 0.080
L_RankSim  = 0.03125
gamma      = 1.0

L_total = 0.080 + 0.03125 = 0.11125
```

During backpropagation, the total gradient is simply the sum:

```
gradient(L_total) = gradient(L_Huber) + gamma * gradient(L_RankSim)
```

In words:

```
Huber gradient   ->  "Make the predicted SWH numerically correct."
RankSim gradient ->  "Make the learned representation respect the SWH ordering."
Combined         ->  "Improve the prediction while learning a more meaningful,
                       SWH-aware feature space."
```

---

## 20. Why this can help imbalanced regression

Take a rare high-SWH sample. A conventional model sees very few examples like it, and its representation for that region of the input space stays weak.

RankSim can still extract information from the *relationship* between that rare sample and other, more common samples:

```
0.30 m <-> 0.50 m <-> 0.90 m <-> 1.40 m <-> 2.30 m
```

The training signal is no longer only "predict 2.30 m correctly." It additionally says: "2.30 m should sit farther from low-SWH samples than from nearby high-SWH samples, in feature space." That's extra structure the rare sample benefits from, even without more copies of itself in the dataset.

---

## 21. Local vs. global relationships

Some imbalance-mitigation techniques mainly exploit *nearby* label relationships. RankSim explicitly considers the **full ranking of all neighbors** — both nearby and distant — for every anchor.

For an anchor at 0.30 m:

```
0.35 m  -> should be very close
0.60 m  -> somewhat close
1.20 m  -> farther
2.40 m  -> much farther
```

RankSim tries to preserve this entire relative ordering, not just "close" vs. "far."

---

## 22. When RankSim is useful

1. **The output is continuous and naturally ordered** — age, depth, wave height, temperature, price, severity scores.
2. **The target distribution is imbalanced** — many samples near common values, few near rare/extreme ones.
3. **Representation quality matters** — transfer learning, retrieval, generalization to rare labels, downstream use of the embeddings.
4. **Neighbor relationships carry real information.**
5. **Feature embeddings are available during training** (i.e., you have access to a penultimate layer, not just the final scalar output).

## 23. When RankSim may be less useful

- **Very small batches** — fewer pairwise relationships, less informative rankings.
- **Noisy targets** — if SWH measurements are noisy, the "true" label ordering can occasionally mislead the regularizer.
- **Weak semantic link between label distance and visual similarity** — RankSim assumes closeness in label space should map to meaningful structure in feature space; this may not hold equally well for every problem.
- **Computational overhead** — pairwise similarity is an `M x M` matrix, so cost grows roughly quadratically with the ranking-subset size.
- **Hyperparameter balance** — a poorly chosen gamma or lambda can reduce (or even hurt) the benefit.

---

## 24. RankSim vs. related ideas

RankSim is conceptually close to metric learning, representation learning, contrastive learning, learning-to-rank, and ordinal regression — but it's not the same as a simple contrastive loss.

A contrastive loss says something binary:

```
positive pair -> pull together
negative pair -> push apart
```

RankSim says something richer, for every anchor at once:

```
Preserve the entire relative ordering of neighbors, not just "close" vs. "far."
```

This distinction is worth having ready for an interview.

---

## 25. Real-world applications

**Computer vision:** age estimation from face images, depth estimation, crowd-density regression, medical severity estimation, body measurement estimation, material property prediction from images.

**Robotics and perception:** distance/depth prediction, terrain roughness estimation, traversability scoring, object range estimation, environmental-state estimation.

**Marine and ocean engineering:** significant wave height (this project), wave period / sea-state severity, wind-wave condition estimation, ice concentration/severity scores, underwater visibility or turbidity estimation.

**Industrial AI:** remaining useful life, equipment degradation, quality scores, manufacturing process variables, demand and price regression.

---

## 26. Why this matters for an ML / DL engineering career

**Loss design.** A strong engineer separates four distinct levers that are easy to conflate: prediction loss, sample weighting, sampling strategy, and representation regularization. RankSim is a clean example of the fourth.

**Representation learning.** Modern deep learning isn't only about the final scalar output — the *structure* of the internal embedding often controls how well a model generalizes.

**Pairwise computation.** RankSim's pairwise-similarity pattern reappears constantly: contrastive learning, triplet loss, retrieval systems, metric learning, graph construction, nearest-neighbor methods.

**Differentiability tricks.** Sorting/ranking is naturally non-differentiable. RankSim is one concrete answer to a recurring engineering question — "how do I optimize a useful but non-differentiable operation inside a neural network?" — a pattern that also shows up in top-k selection, discrete optimization, assignment problems, matching, and structured prediction.

**Tensor-shape discipline.** Reasoning through `features [B, D]` → `normalized [B, D]` → `similarity [B, B]` → `rank row [1, B]` is a core, transferable industry skill.

**Fair ablations.** When adding RankSim to a baseline, keep backbone, split, optimizer, learning rate, augmentations, and evaluation metrics fixed — only then can a performance difference be attributed to RankSim with confidence.

---

## 27. Questions a professor may ask

1. **What is the one-sentence motivation?** Samples close in SWH should have a similar neighbor ordering in learned feature space.
2. **Does RankSim replace Huber/MSE?** No — it's an added regularization term.
3. **Where does RankSim act?** On the penultimate learned features, not directly on the scalar prediction.
4. **How is label similarity measured?** Negative absolute distance between continuous target values.
5. **How is feature similarity measured?** Cosine similarity, in this implementation.
6. **What exactly is compared?** The ranking of neighbors in label space vs. the ranking of neighbors in feature space.
7. **What penalty compares the two rankings?** MSE between normalized ranking vectors.
8. **Why does RankSim need special differentiation?** Ordinary ranking is piecewise-constant and non-differentiable for standard gradient-based learning.
9. **What is gamma?** The weight of RankSim inside the total loss.
10. **What is lambda?** A parameter of the differentiable-ranking approximation — not the regularization weight.
11. **Why might RankSim help rare SWH regions?** It uses relationships between continuous labels to organize representations, so a rare target can still benefit from its relative position to other targets.
12. **Does it explicitly give higher loss weight to rare samples?** No — that's not its mechanism. It changes representation structure, not per-sample loss weight.
13. **Does it change the DataLoader?** No.
14. **Does it require bins?** No — it works directly on continuous labels.
15. **What happens with repeated target values?** The implementation keeps at most one example per unique target value in the ranking subset, to reduce ties.

---

## 28. Questions likely in ML / DL interviews

**Concept:** What is regularization? What is representation learning? Why can a good feature space improve generalization? Cosine similarity vs. Euclidean distance? Ranking loss vs. regression loss? Why is sorting non-differentiable? How can gradients pass through discrete operations? What is pairwise complexity? Why can batch size matter for contrastive/ranking losses?

**PyTorch:** What does `F.normalize(x, dim=1)` do? Why does `z @ z.T` produce pairwise cosine similarities after normalization? What is a custom `torch.autograd.Function`? What do `forward()` and `backward()` mean in custom autograd? Why call `optimizer.zero_grad()`? What happens when two losses are added before `backward()`? How do gradients from multiple objectives combine? What happens if a tensor is detached before the RankSim branch?

**Engineering:** How would you diagnose RankSim if training diverges? How would you tune gamma? How would you evaluate whether embeddings actually improved? What ablation would prove the improvement comes from RankSim specifically? How does batch size affect the number of pairwise relationships? What's the memory cost of a `B x B` similarity matrix?

---

## 29. Practical debugging checklist

- [ ] Confirm features have shape `[B, D]`.
- [ ] Confirm targets have shape `[B]`.
- [ ] Confirm features require gradients (no accidental `.detach()` before the RankSim branch).
- [ ] Confirm the cosine-similarity matrix is `[B, B]`.
- [ ] Confirm label-ranking direction and feature-ranking direction are consistent.
- [ ] Confirm RankSim loss is finite (not NaN/Inf).
- [ ] Confirm RankSim loss produces non-zero feature gradients when rankings actually disagree.
- [ ] Confirm total loss is `base_loss + gamma * ranksim_loss`.
- [ ] Log base loss and RankSim loss *separately* during development.
- [ ] Check sensitivity to gamma.
- [ ] Check sensitivity to batch size.
- [ ] Compare against the same baseline without RankSim.
- [ ] Evaluate rare/few-shot target regions separately.

---

## 30. How to explain RankSim aloud in about one minute

> "RankSim is a representation regularizer for imbalanced regression. The main idea is that continuous labels carry ordering information. For each sample in a mini-batch, we rank the other samples by how close they are in label space. We also rank them by cosine similarity of their learned features. RankSim penalizes disagreement between these two rankings. In my SWH experiment, InceptionV3 produces a 2048-dimensional feature vector, and RankSim encourages samples with similar SWH to have a consistent neighbor ordering in that feature space. The RankSim term is added to the base Huber loss, so the model is optimized both for accurate SWH prediction and for a more meaningful, SWH-aware representation."

---

## 31. Five things to remember

1. RankSim = rank alignment between label space and feature space.
2. It's a regularizer, not a replacement for the base loss.
3. Label similarity = negative absolute target distance.
4. Feature similarity = cosine similarity (in this implementation).
5. Total loss = base loss + gamma x RankSim loss.

---

## 32. Source-grounded implementation summary

```python
# Thin out duplicate labels before ranking
batch_unique_targets = torch.unique(targets)

# Pairwise cosine similarity of L2-normalized features
xxt = torch.matmul(
    F.normalize(x_flat, dim=1),
    F.normalize(x_flat, dim=1).permute(1, 0)
)

# For each anchor i:
label_ranks = rank_normalised(
    (-torch.abs(y[i] - y)).unsqueeze(0)
)

feature_ranks = TrueRanker.apply(
    xxt[i].unsqueeze(0),
    interp_strength_lambda
)

loss = loss + F.mse_loss(feature_ranks, label_ranks)

# Averaged over anchors
return loss / n
```

The ranking operator (`TrueRanker`) uses a custom autograd backward pass built on the perturb-and-re-rank approximation from Section 13.

---

## 33. References

1. Gong, Y., Mori, G., & Tung, F. (2022). *RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression.* Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162.
2. Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolinek, M. (2020). *Differentiation of Blackbox Combinatorial Solvers.* International Conference on Learning Representations (ICLR).
3. Yang, Y., Zha, K., Chen, Y.-C., Wang, H., & Katabi, D. (2021). *Delving into Deep Imbalanced Regression.* International Conference on Machine Learning (ICML).
4. Project implementation files: `src/ranksim_loss.py`, `src/ranking.py`, `src/model.py` (split `forward_features()` / `forward_head()`), `configs/inceptionv3_full_raw_huber_ranksim.yaml`.

---

## Personal learning checkpoint

Before moving to LDS, make sure you can answer these without notes: What does RankSim compare? Why is it useful for continuous labels? Why does it operate on features rather than the final prediction? Why is cosine similarity used? Why is ranking hard to differentiate? What's the difference between gamma and lambda? How does RankSim combine with Huber? How is RankSim different from Stratified Sampling and LDS?

If these are all clear without looking back, the mechanism is understood at a strong practical level.
