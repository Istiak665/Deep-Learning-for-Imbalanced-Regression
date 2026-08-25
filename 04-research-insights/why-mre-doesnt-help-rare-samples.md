# Why MRE Loss Does Not Actually Help Rare (Large-SWH) Samples

**Type:** Research insight / discussion note
**Related to:** [`01-loss-functions/loss-functions-mse-huber-mre.md`](../01-loss-functions/loss-functions-mse-huber-mre.md) — Part B (MRE Loss), Section B.10 "Subtle Insight"
**Context:** SWH (significant wave height) estimation from webcam images — imbalanced regression, where most observations are small waves (Calm/Smooth) and very few are large waves (Moderate/Rough, WMO Sea State bins).

This note works through, step by step, using plain arithmetic (no LaTeX, no symbolic notation) — a claim that sounds true on the surface but is not: *"MRE (Mean Relative Error) divides by the true value, so it should automatically give more weight to small/rare targets."* For an imbalanced-regression problem where the rare targets are the LARGE SWH values (not small ones), this claim breaks down. Below is why, in five steps.

---

## Step 1 — What MRE actually computes

For one sample, with predicted value `y_hat` and true value `y_true`:

```
MRE loss for one sample = |y_hat - y_true| / (|y_true| + epsilon)
```

`epsilon` here is a small guard number (0.001 m in this project) added only so we never divide by zero when `y_true` is 0.

This is the loss VALUE. What actually drives learning (i.e., what actually pushes the network's weights up or down during backpropagation) is not the loss value itself — it is the GRADIENT of the loss, because gradient descent updates weights using the gradient, not the raw loss number.

## Step 2 — The gradient of MRE (this is the part that matters)

Taking the derivative of the MRE loss with respect to the prediction `y_hat` gives:

```
gradient = sign(y_hat - y_true) / (|y_true| + epsilon)
```

`sign(y_hat - y_true)` is just +1 or -1 (whether the prediction is too high or too low). It does not change the SIZE of the gradient, only its direction.

The SIZE (magnitude) of the gradient — how big a push this sample gives the network — is entirely controlled by this part:

```
gradient size = 1 / (|y_true| + epsilon)
```

This is a simple "1 divided by something" relationship. The next step is just basic division arithmetic, not calculus.

## Step 3 — Basic division fact this relationship relies on

This is ordinary arithmetic, nothing more:

```
1 / small number  = big number
1 / big number    = small number
```

Example:
```
1 / 0.05  = 20     (small y_true  ->  big result)
1 / 3.25  = 0.31   (big y_true    ->  small result)
```

Applied to MRE's gradient size `1 / (|y_true| + epsilon)`:

```
small y_true (small wave)  ->  LARGE gradient size  ->  BIG push on the network
large y_true (large wave)  ->  SMALL gradient size  ->  SMALL push on the network
```

So MRE's gradient is built so that **small targets automatically get bigger pushes, and large targets automatically get smaller pushes.** This is the opposite of what we need, because in this dataset the RARE class is the large-wave class.

## Step 4 — Putting real numbers on it (WMO Sea State bins, this project's data)

Using the midpoint SWH value of each WMO Sea State bin as `y_true`, and `epsilon = 0.001`:

| WMO Sea State bin | Representative SWH (m) | Gradient size = 1 / (SWH + 0.001) | Sample count in test set (n=692) |
|---|---|---|---|
| Calm | 0.05 | 19.61 | 56 |
| Smooth | 0.30 | 3.32 | 486 |
| Slight | 0.875 | 1.14 | 139 |
| Moderate | 1.875 | 0.53 | 9 |
| Rough | 3.25 | 0.31 | 2 |

Calm-to-Rough gradient-size ratio: `19.61 / 0.31 ≈ 63.7×`.

In plain words: for every one unit of "learning push" a Rough-sea sample gives the network, a Calm-sea sample of the same absolute prediction error gives it about **64 units of push**. And Calm+Smooth together are already 542 of the 692 test samples (about 78%), while Moderate+Rough together are only 11 of 692 (about 1.6%).

## Step 5 — The "double effect" — why this makes the imbalance WORSE, not better

There are two separate, independent sources of imbalance stacking on top of each other here:

1. **Count imbalance** (how many samples of each type exist): Calm/Smooth = ~78% of the data, Moderate/Rough = ~1.6%. This alone already means the network sees far more small-wave examples during training.
2. **Gradient-scale imbalance** (built into MRE's own formula): even for a single Moderate/Rough sample vs a single Calm/Smooth sample with the same-sized prediction error, MRE gives up to ~64x MORE weight to the small-wave sample, purely because of the `1 / y_true` structure.

These two effects multiply rather than cancel. MRE does not fix the count imbalance — it actively compounds it, because its own gradient math favors the already-overrepresented small-wave class.

**Contrast with LDS (Label Distribution Smoothing):** LDS is a resampling/reweighting mechanism that addresses effect #1 (count imbalance) — it changes how OFTEN the network sees each bin during training. But LDS does nothing to effect #2, because effect #2 is baked into the loss FORMULA itself, not into the sampling. So `MRE + LDS` still carries this structural bias — LDS can rebalance how many times a Rough-sea sample is shown, but it cannot change how weak that sample's gradient push is every time it IS shown.

**Contrast with Huber loss:** Huber's gradient (for residuals beyond the delta threshold) is bounded to a constant ±1, with no `1 / y_true` term at all — its gradient size does not depend on how large the true target is. This is a plausible mathematical reason why, in this project's 31-run factorial results, `Huber + ConR` wins the rarity-weighted metrics (SERA, Relevance-Weighted MAAPE) while `MRE + LDS` wins the typical-range/overall metrics (MAPE, MAAPE) — the two losses are structurally suited to different goals.

---

## One-line summary

**MRE's loss formula divides by the true value, which sounds like it should emphasize rare (in this case: large) targets — but the resulting gradient is actually inversely proportional to target size, so it pushes hardest on the already-abundant small-wave samples and weakest on the rare large-wave samples, compounding rather than fixing the dataset's imbalance.**

## References for this note

- Yang, Y., Zha, K., Chen, Y., Wang, H., & Katabi, D. (2021). *Delving into Deep Imbalanced Regression.* ICML 2021. (Defines Label Distribution Smoothing / LDS — addresses count imbalance, not loss-gradient imbalance.)
- See also: [`01-loss-functions/loss-functions-mse-huber-mre.md`](../01-loss-functions/loss-functions-mse-huber-mre.md) Part A (Huber, bounded gradient) and Part B (MRE, full derivation).
