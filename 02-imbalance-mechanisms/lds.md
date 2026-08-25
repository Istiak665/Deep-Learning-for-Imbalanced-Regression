# LDS — Label Distribution Smoothing

**Project context:** Significant Wave Height (SWH) estimation from coastal webcam images
**Primary use here:** LDS reweighting on top of a Huber base loss (the project's `inceptionv3_full_raw_huber_lds` config)
**Learning goal:** Understand LDS from scratch — deeply enough to explain it to a professor, implement/debug it in PyTorch, and connect it to the project's other imbalance mechanisms.

> Note on notation: plain text/arithmetic is used throughout instead of LaTeX-style symbols, so this reads cleanly on GitHub.

---

## 1. One-sentence idea

**LDS keeps mini-batches completely random, but multiplies each sample's loss by a weight that is larger for rare SWH values and smaller for common ones — so rare, large-wave mistakes count for more during training.**

---

## 2. Why LDS exists

In the Great Lakes SWH dataset, small waves vastly outnumber large ones. Trained with a plain Huber loss, every sample counts equally, so the network is dominated by gradient signal from the thousands of small-wave images and barely notices the handful of large-wave ones.

Two very different fixes exist for this:

- **Change which samples the network sees** (Stratified Sampling) — rebalance the DataLoader itself.
- **Change how much each sample's mistake counts** (LDS) — keep normal random batches, but scale the loss per sample.

LDS is the second kind: a **loss-reweighting** mechanism. It never touches the DataLoader, the model architecture, or the optimizer — only the number that gets multiplied into the loss for each sample.

---

## 3. The core problem LDS actually solves: raw frequency is a noisy estimate

A naive alternative to LDS would be: count how many training samples fall in each small SWH bin, and weight each sample by `1 / count` for its bin. The LDS paper's key observation is that this naive approach is fragile — with continuous labels and limited data, a bin can have very few (or zero) samples purely by chance, even though nearby bins (which represent physically/visually similar wave conditions) are well populated. Weighting by raw per-bin frequency then either wildly over- or under-weights that bin.

LDS's fix: **smooth the empirical label histogram with a kernel before computing weights**, so a sparse bin borrows statistical strength from its neighbors. This is the "Distribution Smoothing" the method is named for — smoothing the estimated *label density*, not smoothing the loss function itself.

---

## 4. LDS vs. the other imbalance mechanisms in this project

| Method | What it changes |
|---|---|
| Plain Huber / MSE / MRE | How prediction error is penalized |
| Stratified Sampler | Which samples enter each mini-batch |
| **LDS** | **How much each sample's base loss is weighted** |
| RankSim | How learned feature relationships are organized |
| ConR | Adds a contrastive-style representation regularizer |

LDS does not rebalance the DataLoader and does not replace the base loss — it sits entirely inside the loss computation.

---

## 5. Notation used below

- `y` = a sample's true SWH value (meters)
- `bin_width` = width of each histogram bin used to estimate label density (0.01 m in this project — very fine-grained)
- `count(b)` = raw number of training samples falling in bin `b`
- `kernel` = a symmetric smoothing window (gaussian here) of size `ks`, spread controlled by `sigma`
- `density(b)` = the *smoothed* sample count for bin `b`, after convolving `count` with the kernel
- `w(b)` = the per-bin weight derived from `density(b)`
- `reweight` = the rule converting density to weight (`sqrt_inv` here: `w = 1 / sqrt(density)`)
- `max_weight` = an absolute ceiling applied to the final, normalized weight

---

## 6. Step 1 — Build the empirical label histogram (train split only)

Every training-set SWH value is binned into `bin_width`-wide bins (0.01 m in this project), covering `[0, max(train SWH) + bin_width)`. This gives a raw count per bin — a very fine-grained empirical histogram of how common each SWH value is.

Only the **training** split is used to build this histogram — LDS, like every other mechanism in this project, must never see validation or test targets.

---

## 7. Step 2 — Smooth the histogram with a kernel

The raw per-bin counts are convolved with a symmetric kernel window:

```
smoothed = convolve(raw_counts, kernel_window)
```

Project settings: `kernel = gaussian`, `ks = 9` (window covers 9 bins), `sigma = 2.0` (spread of the Gaussian). A Gaussian kernel weights the center bin most heavily and nearby bins progressively less, so each bin's smoothed density is a weighted average of itself and its close neighbors — this is exactly what lets an empty or sparse bin "borrow" density from nearby, better-populated bins.

---

## 8. Step 3 — Convert smoothed density into a weight

```
reweight == "inv"       ->  w = 1 / density
reweight == "sqrt_inv"  ->  w = 1 / sqrt(density)      (used in this project)
```

`sqrt_inv` is a gentler curve than plain `inv` — it still gives rare bins larger weights, but avoids the extreme blow-ups that plain inverse weighting can produce on the sparsest bins.

---

## 9. Step 4 — Normalize, then cap

**Normalize first:** the raw weights are rescaled so that the weights *as actually applied across the training set* (sample-weighted, not just averaged per bin — bins hold very different numbers of samples) average to exactly 1.0. This keeps the overall scale of the weighted loss comparable to an unweighted baseline, so early stopping and the LR scheduler (both watching `val_huber`, an *unweighted* validation metric) aren't reacting to a training loss that has silently drifted to a different numeric scale.

**Cap second:** after normalization, weights are clipped to `max_weight = 20.0` as an absolute safety ceiling, so no single ultra-rare bin can dominate a batch's loss and destabilize training.

**Order matters:** the project's code and its own self-test confirm capping must happen *after* normalizing, not before — capping first and then renormalizing can push already-capped, rare-bin weights right back above the intended cap.

---

## 10. Step 5 — Look up a weight at train time

For every sample in a training batch, its true SWH is snapped to the nearest fitted bin (`assign_lds_weights()`), and that bin's weight is multiplied into the loss:

```python
sample_weights = assign_lds_weights(targets, lds_weight_table, device=device)
loss = weighted_huber_loss(outputs, targets, sample_weights, delta=DELTA_DEFAULT)
```

This is the **only** place LDS plugs into the training loop (`train_one_epoch()` in `main.py`) — everything else (dataloaders, model, optimizer, scheduler, monitor metric) is byte-identical to a run with no mechanism at all.

---

## 11. Worked example — verified by actually running the project's algorithm

A small toy dataset, mimicking the real imbalance (lots of calm/smooth samples, a handful of moderate/rough ones), run through the exact `compute_lds_weights()` algorithm above (`bin_width=0.5` m for a readable toy example, `kernel=gaussian, ks=3, sigma=1.0, reweight=sqrt_inv, max_weight=20.0`):

| SWH bin | Raw sample count | Smoothed density | Final weight |
|---|---|---|---|
| 0.0–0.5 m (Calm/Smooth) | 18 | 9.23 | 0.81 |
| 0.5–1.0 m (Slight) | 4 | 7.02 | 0.93 |
| 1.0–1.5 m (Moderate) | 1 | 1.55 | 1.98 |
| 1.5–2.0 m (empty bin) | 0 | 0.55 | 3.33 |
| 2.0–2.5 m (Rough) | 1 | 0.45 | 3.67 |

Two things worth noticing:

1. The most common bin (18 samples) gets **downweighted** to 0.81; the rarest populated bin (1 sample, Rough) gets **upweighted** to 3.67 — roughly 4.5x more influence per sample.
2. The 1.5–2.0 m bin has **zero raw samples**, yet still receives a sensible, non-degenerate weight (3.33) purely because kernel smoothing let it borrow density from its populated neighbors. A naive `1/raw_count` scheme would have divided by zero here.

---

## 12. The actual SWH LDS configuration used in this project

For `inceptionv3_full_raw_huber_lds`:

| Setting | Value |
|---|---|
| Backbone | InceptionV3 |
| Regression target | raw SWH (meters) |
| Base loss | Huber (delta = 0.15 m) |
| Mechanism | LDS |
| bin_width | 0.01 m |
| kernel | gaussian |
| ks | 9 |
| sigma | 2.0 |
| reweight | sqrt_inv |
| max_weight | 20.0 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Validation monitor | val_huber |

Hyperparameters were copied verbatim from the project's earlier, already-verified `inceptionv3_full_raw_mse_lds` study — nothing retuned for this run, so LDS's behavior (what counts as "rare", how aggressively it's upweighted) lands in the same neighborhood as that earlier, trusted result.

---

## 13. Full training flow, one mini-batch

```
1. (once, before training) Build the LDS weight table from TRAIN targets only:
   histogram -> Gaussian-smooth -> 1/sqrt(density) -> normalize to mean 1.0 -> cap at 20.0
                 |
                 v
2. Load a normal RANDOM mini-batch of images + true SWH  (unchanged by LDS)
                 |
                 v
3. Forward pass: images -> InceptionV3 -> predicted SWH
                 |
                 v
4. Look up each sample's LDS weight from its true SWH
                 |
                 v
5. loss = weighted_huber_loss(pred, true, sample_weights)
                 |
                 v
6. backward() -> gradients scaled up for rare-SWH samples, scaled down for common ones
                 |
                 v
7. AdamW optimizer step -> updated weights
```

---

## 14. Why this can help imbalanced regression

Without LDS, a Huber loss treats a 1-degree prediction error on a 0.30 m sample and the same 1-degree-scale error on a 2.10 m sample identically — but the network sees the 0.30 m case thousands of times more often, so gradients from common samples dominate training. LDS doesn't change how many times the rare sample is seen (that's StratSample's job) — it changes how *loud* that sample's gradient is every time it *is* seen, so a handful of large-wave samples can meaningfully compete with a much larger pool of small-wave samples for the network's attention during training.

**Important nuance:** LDS reweights based on the *continuous* SWH value via its smoothed density — it does not use the discrete WMO bin column at all. That column is only used afterward for per-bin evaluation reporting, never for computing LDS weights.

---

## 15. When LDS is useful

1. The target is continuous, and label rarity is the main problem (not needing a different sampling strategy or a different feature-space structure).
2. You want to keep normal random mini-batches — e.g., because other parts of the training pipeline (batch norm statistics, augmentation diversity) benefit from natural batch composition.
3. You can afford one preprocessing pass over the training targets before training starts, to build the weight table.
4. The label space is dense enough that neighboring bins are meaningfully related (kernel smoothing assumes nearby label values represent similar underlying conditions).

## 16. When LDS may be less useful

- **Extremely small training sets** — a fine-grained histogram (bin_width=0.01 m) can be dominated by noise even after smoothing.
- **Multi-modal or discontinuous label distributions** — if nearby label values are *not* actually similar in the underlying physical sense, smoothing across them is not well-justified.
- **When the loss-scale shift itself is a concern** — even after normalization to mean 1.0, individual batches can have a higher or lower effective loss scale than an unweighted baseline, depending on which samples happen to be drawn.
- **Hyperparameter sensitivity** — `sigma`, `ks`, `reweight`, and `max_weight` all interact; a poor combination can under- or over-correct the imbalance.

---

## 17. Real-world applications

**Computer vision:** age estimation (rare very-young/very-old faces), rare disease severity scores in medical imaging, crowd-counting in sparse or extremely dense scenes.

**Marine and ocean engineering:** significant wave height (this project), extreme sea-state forecasting, rare high-wind or storm-condition regression tasks.

**Finance and industrial AI:** rare high-value transaction amount prediction, rare equipment-failure severity regression, demand forecasting for long-tail product categories.

**General deep imbalanced regression:** any continuous-target problem where the tail of the distribution matters (safety-critical, extreme-event, or rare-class importance) but data collection naturally favors the common cases.

---

## 18. Why this matters for an ML / DL engineering career

**Separating loss design from sampling design.** LDS is a clean example of solving imbalance purely through loss weighting, without touching the DataLoader — a distinct lever from Stratified Sampling, useful to keep conceptually separate in interviews and in your own ablations.

**Kernel density estimation intuition.** The core trick — smoothing a noisy empirical histogram with a kernel before trusting it — is a general statistics idea (related to KDE, kernel regression) that shows up far beyond this one method.

**Weight normalization discipline.** Understanding *why* weights must be normalized to mean 1.0 (so the loss scale stays comparable across configurations, so schedulers/early-stopping watching an unweighted metric behave correctly) is a transferable engineering habit whenever you add any kind of sample weighting to a loss.

**Order-of-operations bugs.** The normalize-before-cap ordering (Section 9) is a good real example of a subtle correctness bug class: two individually-reasonable operations that must be applied in a specific order to preserve their intended guarantees.

---

## 19. Questions a professor may ask

1. **What is the one-sentence motivation?** Rare SWH values get up-weighted in the loss, common ones down-weighted, without changing which samples are seen.
2. **Does LDS change the DataLoader?** No — batches stay fully random.
3. **What exactly gets smoothed?** The *empirical label histogram* (a density estimate), not the loss function or the predictions.
4. **Why smooth instead of just using raw per-bin frequency?** Raw counts are noisy, especially in sparse bins; smoothing borrows statistical strength from neighboring, physically-similar bins.
5. **What's the difference between `inv` and `sqrt_inv`?** Both upweight rare bins; `sqrt_inv` is gentler, avoiding extreme weight blow-ups on the sparsest bins.
6. **Why normalize the weights?** So the overall loss scale stays comparable to an unweighted baseline — important because early stopping/LR scheduling watch an unweighted validation metric (`val_huber`).
7. **Why cap the weights?** To prevent a single extremely rare bin from destabilizing training with an unbounded weight.
8. **Does LDS use the WMO bin column?** No — it computes its own fine-grained histogram directly from the continuous SWH value; the WMO bin column is only used for post-hoc evaluation reporting.
9. **Does LDS replace the base loss?** No — it reweights the base loss (Huber, here) per sample; it doesn't change the loss formula itself.
10. **How is LDS different from RankSim?** LDS changes per-sample loss weight; RankSim changes how feature relationships are organized. They operate at different points in the pipeline and could in principle be combined.

---

## 20. Questions likely in ML / DL interviews

**Concept:** What is kernel density estimation, at a high level? Why can naive inverse-frequency weighting be unstable? What's the difference between reweighting the loss and resampling the data? Why does weight normalization matter for training stability? What's a scatter/effective-sample-size argument for why weighting shifts the effective training distribution?

**PyTorch/engineering:** How would you implement a lookup-table-based per-sample weight efficiently in a training loop? How would you validate that LDS weights are being applied correctly (e.g., a unit test)? What would you check if training loss looked unstable after adding LDS? How would you tune `sigma`/`ks`? How would you evaluate whether LDS actually helped the rare-SWH bins specifically, not just the overall metric?

---

## 21. Practical debugging checklist

- [ ] Confirm the LDS weight table is built from the **training split only**.
- [ ] Confirm bin edges cover the full training target range (no samples silently outside range).
- [ ] Confirm smoothed density is never zero/negative before the `1/density` or `1/sqrt(density)` step.
- [ ] Confirm weights are normalized to mean 1.0 **before** capping, not after.
- [ ] Confirm the final applied weight never exceeds `max_weight`.
- [ ] Confirm a rare-SWH sample's weight is strictly larger than a common-SWH sample's weight.
- [ ] Log `bin_weights.min()` / `.max()` at the start of training and sanity-check the range.
- [ ] Compare training curves and per-bin test metrics against the same run without LDS.
- [ ] Confirm early stopping/scheduler are watching an **unweighted** validation metric (e.g., `val_huber` computed directly on the validation set, not reweighted).

---

## 22. How to explain LDS aloud in about one minute

> "LDS is a loss-reweighting mechanism for imbalanced regression. Instead of changing which images end up in a batch, it changes how much each sample's error counts toward the loss. I first build a fine-grained histogram of the training SWH values, smooth it with a Gaussian kernel so sparse regions borrow statistical strength from their neighbors, then convert that smoothed density into a per-sample weight — inversely proportional to the square root of density, so rare large-wave samples get upweighted and common small-wave samples get downweighted. I normalize the weights to average 1.0 so the loss scale stays comparable to an unweighted baseline, and cap them at 20 for stability. At training time, this weight simply multiplies into the Huber loss for each sample — nothing else about the pipeline changes."

---

## 23. Five things to remember

1. LDS = reweight the loss per sample, based on a **smoothed** estimate of label rarity.
2. It changes the *loss*, not the DataLoader and not the feature space.
3. Smoothing borrows density from neighboring bins — this is what makes it more robust than raw inverse-frequency weighting.
4. Weights are normalized to mean 1.0 **before** being capped at `max_weight`.
5. `sqrt_inv` (used here) is a gentler alternative to plain `inv` weighting.

---

## 24. Source-grounded implementation summary

```python
# 1) Empirical histogram of TRAIN targets only
counts, _ = np.histogram(train_targets, bins=bin_edges)

# 2) Smooth with a symmetric kernel (gaussian, ks=9, sigma=2.0 in this project)
smoothed = np.convolve(counts, kernel_window, mode="same")

# 3) Density -> weight
raw_w = 1.0 / np.sqrt(smoothed)          # reweight = "sqrt_inv"

# 4) Normalize to mean 1.0 over the ACTUAL training set, then cap
bin_weights = raw_w / applied_unnorm.mean()
bin_weights = np.clip(bin_weights, None, max_weight)

# 5) At train time: look up + apply
sample_weights = assign_lds_weights(targets, lds_weight_table, device=device)
loss = weighted_huber_loss(outputs, targets, sample_weights, delta=DELTA_DEFAULT)
```

---

## 25. References

1. Yang, Y., Zha, K., Chen, Y.-C., Wang, H., & Katabi, D. (2021). *Delving into Deep Imbalanced Regression.* Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139. [arXiv:2102.09554](https://arxiv.org/abs/2102.09554) — defines both LDS (Label Distribution Smoothing) and FDS (Feature Distribution Smoothing).
2. Project implementation files: `src/lds_loss.py`, `main.py` (`build_lds_weight_table()`, `train_one_epoch()`), `src/huber_loss.py` (`weighted_huber_loss`), `configs/inceptionv3_full_raw_huber_lds.yaml`.

---

## Personal learning checkpoint

Before moving to StratSample or ConR, make sure you can answer these without notes: What exactly does LDS smooth? Why is raw per-bin frequency an unreliable weight source? Why normalize before capping instead of after? Does LDS touch the DataLoader? How does LDS differ from RankSim and from Stratified Sampling? Why does the weighted training loss still get compared against an *unweighted* validation metric?

If these are all clear without looking back, the mechanism is understood at a strong practical level.
