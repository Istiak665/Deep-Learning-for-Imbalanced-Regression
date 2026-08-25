# Loss Functions in Deep Learning — MSE, Huber & MRE

*A self-learning reference — concept, mathematics, hand-verified worked examples, real code, industry relevance.*

**Prepared by:** Md Istiak Ahammed — PhD Researcher, Ocean Engineering / AI Applications, Michigan Technological University.
Built from the actual loss-function code used in a Significant Wave Height (SWH) estimation research project (webcam-image → deep-learning-regression pipeline). Every numeric example below was independently verified by running the real PyTorch code, not just derived on paper.

---

## 0. How to Use This Document

This is a self-learning reference — go back to first principles any time you forget *why* a loss function behaves the way it does, or need to explain it to an advisor, in a paper, or in a job interview. Each loss function is treated the same way, in the same order, so the pattern of thinking transfers:

1. What problem is this loss solving, and why does it exist alongside the alternatives?
2. The exact mathematical formula, with every symbol defined.
3. A tiny, fully worked numerical example — forward pass → loss → backward pass (gradient) → weight update → verification that the loss actually decreased.
4. Line-by-line connection to real project code.
5. Where this shows up in the real world, outside this one project.
6. References — where this idea comes from in the literature.
7. Why this matters for an ML/DL engineering career, and questions you may be asked about it.

This note covers MSE, Huber, and MRE, in that order (simplest to most structurally distinct). A companion note on the imbalance-mitigation mechanisms (StratSample, LDS, RankSim, ConR) is a natural next addition — see [What's Next](#e-whats-next-in-this-learning-hub).

---

## 1. Foundations — What a Loss Function Actually Does

A neural network takes an input (here: a webcam image) and produces a prediction ŷ (predicted significant wave height, in metres). We also know the true answer y (measured by a buoy). A **loss function** L(ŷ, y) converts the disagreement between prediction and truth into a single number — the smaller that number, the better the model is doing right now.

Training is a loop, repeated for every mini-batch, every epoch:

```
1. FORWARD PASS   : image  ->  backbone (CNN)  ->  regression head  ->  ŷ
2. LOSS           : L = loss_function(ŷ, y)
3. BACKWARD PASS  : compute ∂L/∂w for every weight w in the network (chain rule)
4. WEIGHT UPDATE  : w_new = w_old − learning_rate × ∂L/∂w   (gradient descent, or a variant like AdamW)
```

Steps 1–2 are what changes when you swap MSE ↔ MRE ↔ Huber — the network architecture, the forward pass, and the update mechanism (AdamW) stay identical. Everything downstream (how big the gradient is, which samples get emphasized, how outliers are treated) flows from the choice made in step 2. **Choosing a loss function is choosing what kind of mistake the model is taught to fear most.**

### 1.1 The toy network used throughout this document

To make every calculation checkable by hand, all worked examples below use the same tiny stand-in for the real regression head (a pooled 2048-d CNN feature vector, shrunk to 3 numbers for arithmetic sanity):

```
Input (toy pooled feature):      x = [0.6, 0.2, 0.9]

Hidden layer (2 neurons, ReLU):
  W1 = [[ 0.10, -0.20,  0.05],      b1 = [ 0.05, -0.02]
        [ 0.30,  0.10, -0.15]]

Output layer (1 neuron, linear -> predicted SWH):
  W2 = [0.8, 1.2]                   b2 = 0.10
```

Forward pass (identical for every loss function, computed once and reused):

```
h1_pre = 0.10(0.6) + (-0.20)(0.2) + 0.05(0.9) + 0.05  = 0.115
h2_pre = 0.30(0.6) +  0.10(0.2) + (-0.15)(0.9) - 0.02  = 0.045
ReLU: both positive -> h1 = 0.115,  h2 = 0.045
ŷ = 0.8(0.115) + 1.2(0.045) + 0.10  =  0.246   <-- SAME predicted SWH for every example below
```

Why the depth/architecture doesn't matter for understanding: the chain rule works the same way whether the network has 2 layers (this toy) or 50+ layers (a real CNN backbone like InceptionV3). Only the number of steps changes, not the mechanism.

---

## PART A — Huber Loss

### A.1 Concept — What Problem Does Huber Loss Solve?

Two well-known losses sit at extremes:

- **MSE (squared error)** — grows quadratically with the residual. A single large residual (e.g. a rare, hard-to-predict rough-sea sample) can dominate the gradient and destabilize training.
- **MAE (absolute error)** — every residual gets equal gradient magnitude (±1), robust to outliers, but the gradient is discontinuous exactly at zero error, which makes fine convergence near the optimum jagged.

**Huber loss interpolates**: quadratic (MSE-like, smooth) for small residuals, linear (MAE-like, bounded) for large residuals, switching at a threshold δ. This is exactly the compromise you want when your dataset has a long-tailed, imbalanced target distribution (as SWH does — most samples are Calm/Smooth sea states) with a few genuinely large, sometimes-noisy, high-SWH outliers you don't want to let dominate training.

### A.2 Mathematical Formula and Notation

Per-sample Huber loss:

```
L_δ(r) = { 0.5·r² / δ     if r < δ
         { r − 0.5·δ      if r ≥ δ

  where r = |ŷ − y|
```

| Symbol | Meaning | In code |
|---|---|---|
| y | True SWH (metres) — ground truth from buoy | `target` |
| ŷ | Predicted SWH from the network | `pred` |
| r = \|ŷ − y\| | Absolute residual (sign doesn't matter) | `resid = torch.abs(pred - target)` |
| δ (delta) | Threshold hyperparameter = 0.15 m in this project | `DELTA_DEFAULT = 0.15` |
| N | Batch size (32 in this project) | `loss.mean()` |

**Important implementation detail**: the quadratic branch is divided by δ (`0.5r²/δ`), not the textbook `0.5r²` alone. This makes both branches agree not just in value but in **gradient magnitude (exactly 1.0)** at the boundary r=δ — a smooth, kink-free transition (Yang et al. 2021, ICML LDS/FDS convention), rather than the older "clipped-quadratic" textbook formulation.

### A.3 Fully Worked Numerical Example

Using the toy network above (ŷ = 0.246):

**Case A — small residual (quadratic branch): target y = 0.30 m**

```
r = |0.246 − 0.300| = 0.054      (0.054 < δ=0.15 -> QUADRATIC branch)
L_A = 0.5 × 0.054² / 0.15 = 0.5 × 0.002916 / 0.15 = 0.00972
```

**Case B — large residual (linear branch): target y = 1.80 m (a rare, high-SWH sample)**

```
r = |0.246 − 1.800| = 1.554      (1.554 >> δ=0.15 -> LINEAR branch)
L_B = 1.554 − 0.5(0.15) = 1.554 − 0.075 = 1.479

Compare: IF this were (hypothetically) forced through the quadratic formula:
  0.5 × 1.554² / 0.15 = 8.05   <-- 5.4x larger! This is exactly what Huber prevents.
```

**Backward pass (chain rule), Case A — nothing skipped**

```
∂L/∂r = r/δ = 0.054/0.15 = 0.36        (quadratic-branch derivative)
∂r/∂ŷ = sign(ŷ−y) = sign(-0.054) = -1
∂L/∂ŷ = 0.36 × (-1) = -0.36

∂L/∂W2 = ∂L/∂ŷ × h = -0.36×[0.115, 0.045] = [-0.0414, -0.0162]     ∂L/∂b2 = -0.36
∂L/∂h  = ∂L/∂ŷ × W2 = -0.36×[0.8, 1.2] = [-0.288, -0.432]
ReLU active both sides -> ∂L/∂h_pre unchanged = [-0.288, -0.432]
∂L/∂W1 = outer([-0.288,-0.432], x) =
   row1: -0.288×[0.6,0.2,0.9] = [-0.1728, -0.0576, -0.2592]   ∂L/∂b1_1 = -0.288
   row2: -0.432×[0.6,0.2,0.9] = [-0.2592, -0.0864, -0.3888]   ∂L/∂b1_2 = -0.432
```

**Weight update (plain SGD, lr = 0.01) and verification**

```
w_new = w_old − lr × gradient   for every weight above

Re-running the forward pass with updated weights:
  ŷ_new = 0.2662   (was 0.246)
  new residual = |0.2662 − 0.300| = 0.0338   (was 0.054  -> ERROR DECREASED)
  L_new = 0.5×0.0338²/0.15 ≈ 0.00381          (was 0.00972  -> LOSS DROPPED ~61% in one step)
```

### A.4 Line-by-Line Code Connection

```python
# src/huber_loss.py
resid = torch.abs(pred - target)                 # r = |ŷ - y|      -> §A.2
cond  = resid < self.delta                        # branch selector -> §A.2
loss  = torch.where(cond,
            0.5 * resid.pow(2) / self.delta,       # quadratic branch
            resid - 0.5 * self.delta)               # linear branch
return loss.mean()                                 # average over the N=32-sample batch

# main.py, train_one_epoch()
outputs = model(images)          # forward pass          -> §1.1
loss = criterion(outputs, targets)                        # -> §A.2/A.3
scaler.scale(loss).backward()    # backward pass (autograd does §A.3's chain rule automatically)
scaler.step(optimizer)           # weight update, via AdamW, not plain SGD
```

δ = 0.15 m was not copied from a textbook — it was set from this project's own early-run residual distribution (roughly the p90 hardest-residual mark), and deliberately kept equal to a separate mechanism's (ConR) w=0.15 m similarity window, so both agree on what counts as a "meaningfully different" SWH value.

### A.5 Real-World Applications

Huber loss (and its close sibling, Smooth L1 loss) is one of the most widely deployed regression losses in production machine learning:

- **Object detection bounding-box regression** — Fast R-CNN, Faster R-CNN, YOLO, and SSD all use Smooth L1 (a Huber variant, δ=1) to regress box coordinates.
- **Robust statistics / outlier-heavy sensor data** — financial time series, industrial telemetry, any pipeline with occasional sensor glitches.
- **Autonomous driving / robotics** — trajectory and control regression targets.
- **Depth estimation and other dense-regression computer vision tasks.**
- **Reinforcement learning** — the default loss for the TD-error term in DQN (Deep Q-Networks) and successors.

### A.6 References

- Huber, P. J. (1964). *Robust Estimation of a Location Parameter.* Annals of Mathematical Statistics, 35(1), 73–101.
- Girshick, R. (2015). *Fast R-CNN.* ICCV. — introduces Smooth L1 loss for bounding-box regression.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature. — DQN, uses Huber-loss-clipped TD-error.
- Yang, Y., Zha, K., Chen, Y., Wang, H., & Katabi, D. (2021). *Delving into Deep Imbalanced Regression.* ICML.
- PyTorch docs: `torch.nn.HuberLoss` / `torch.nn.SmoothL1Loss`.

### A.7 Career Relevance

"Which loss function should we use and why" is one of the most common practical decisions an ML/DL engineer makes:

- **Computer vision (autonomous vehicles, robotics, AR/VR)** — nearly every detection/tracking pipeline uses Smooth L1/Huber; explaining why is a standard interview/design-review topic.
- **Applied ML / MLE roles with sensor or forecasting data** — robust loss selection is a first-line tool against noisy labels.
- **Research engineer / research scientist roles** — designing a *new* loss variant (as done here with the δ-normalized Huber) is applied-research skill.

### A.8 Likely Interview / Viva Questions

- What's the difference between MSE, MAE, and Huber loss, and when would you choose each?
- Why is Huber loss preferred over MSE for data with outliers? Can you derive the gradient of each?
- What does the δ (delta) hyperparameter control, and how would you choose its value in a new problem?
- What is Smooth L1 loss and where have you seen it used?
- Is Huber loss convex? Is it differentiable everywhere?
- How would you decide the value of δ from your own data, rather than guessing?
- Walk me through backpropagation for a simple network by hand.

---

## PART B — MRE Loss (Epsilon-Guarded Mean Relative Error)

### B.1 Concept — What Problem Does MRE Solve?

MSE and Huber both measure *absolute* error — a 0.05 m mistake counts the same whether the true SWH was 0.1 m or 2.0 m. But in many applications (including wave-height reporting, conventionally expressed as percentage error), what matters is *relative/proportional* accuracy. MRE (Mean Relative Error) directly optimizes that.

### B.2 Mathematical Formula and Notation

```
MRE = (1/N) Σᵢ  |ŷᵢ − yᵢ| / (|yᵢ| + ε)
```

| Symbol | Meaning | In code |
|---|---|---|
| yᵢ | True SWH of sample i | `target` |
| ŷᵢ | Predicted SWH of sample i | `pred` |
| \|ŷᵢ − yᵢ\| | Absolute error | `torch.abs(pred - target)` |
| ε (epsilon) | Small guard constant = 0.001 m | `MRE_EPS = 1e-3` |
| N | Batch size | `torch.mean(...)` |

Key structural difference from Huber/MSE: the absolute error is divided by the target's own magnitude, `|y|+ε`. This single division makes MRE fundamentally a *relative*, scale-aware loss. Note: MRE is a fraction (e.g. 0.18), not a percentage — MRE × 100 = MAPE.

### B.3 Why the Epsilon Guard Exists

If any sample's true SWH is exactly, or very near, zero (calm water), the denominator `|y|` would be ~0, and division by ~0 produces infinity or NaN — crashing training. Adding ε=0.001 guarantees the denominator can never be exactly zero:

```
Case D — pred=0.05, target=0.0 (a completely calm-water sample):
  WITHOUT ε guard:  0.05 / 0.0            ->  division by zero, training crashes
  WITH ε guard:     0.05 / (0.0+0.001)    =  50.0   (large but FINITE, valid number)
```

Deliberate consistency: ε=0.001 is the exact same value used in the project's evaluation-time MAPE calculation — so the training objective and the reported metric agree on how near-zero targets are handled.

### B.4 Fully Worked Numerical Example (independently verified by running the real PyTorch code)

Same toy network, same ŷ = 0.246.

**Case A — small target, y = 0.30 m**

```
r = |0.246 − 0.300| = 0.054
MRE_A = 0.054 / (0.30+0.001) = 0.054/0.301 = 0.1794   (= 17.94%)
```

**Case C — large target, y = 1.80 m, a NEARLY IDENTICAL absolute error (0.05 m)**

```
predicted ŷ=1.75, target y=1.80  ->  r = 0.05
MRE_C = 0.05 / (1.80+0.001) = 0.05/1.801 = 0.0278   (= 2.78%)
```

Core lesson: almost identical absolute error (0.054 m vs 0.05 m) produces a **6.5× different loss** (17.94% vs 2.78%) purely because of the target's magnitude.

**Direct comparison — MSE vs Huber vs MRE, same two cases (all numbers verified by running actual code)**

| Case | Residual | MSE | Huber (δ=0.15) | MRE |
|---|---|---|---|---|
| A (target 0.30 m) | 0.0540 | 0.002916 | 0.009720 | **0.1794 (17.94%)** |
| C (target 1.80 m) | 0.0500 | 0.002500 | 0.008333 | **0.0278 (2.78%)** |
| **Ratio A/C** | ~equal | 1.17× | 1.17× | **6.46×** |

MSE and Huber barely notice the difference (1.17×) — they only see the absolute residual. MRE reacts **6.46× more strongly** — it only cares about relative accuracy.

**Backward pass and gradient — a clean, single formula explains everything**

```
L = |ŷ−y| / (|y|+ε).  y is fixed during training, so let D = |y|+ε (constant).
∂L/∂ŷ = sign(ŷ−y) / D

Case A: sign(0.246-0.300) = -1, D = 0.301  ->  ∂L/∂ŷ = -1/0.301 = -3.3223
  [VERIFIED: PyTorch autograd gives -3.3222591876983643 -- matches to full float precision]

Key insight: gradient MAGNITUDE = 1/(|y|+ε) -- INVERSELY proportional to the target's size.
Small targets get amplified gradients (up to 1/ε = 1000x at y≈0); large targets get suppressed
gradients. Compare: Huber's gradient for this same case was only -0.36 -- MRE's gradient here
is about 9x larger, purely because the target (0.30m) is small.
```

**Full weight update (plain SGD, lr = 0.001) and verification**

```
∂L/∂W2 = -3.3223×[0.115,0.045] = [-0.3821, -0.1495]     ∂L/∂b2 = -3.3223
∂L/∂W1 = [[-1.5947,-0.5316,-2.3920],[-2.3920,-0.7973,-3.5880]]   ∂L/∂b1 = [-2.6578,-3.9867]
  [all four gradient arrays VERIFIED against PyTorch autograd -- exact match]

After one SGD step (lr=0.001):
  ŷ_new = 0.2646   (was 0.246)
  new residual = |0.2646-0.300| = 0.0354   (was 0.054  -> decreased)
  L_new = 0.0354/0.301 = 0.1174 = 11.74%   (was 17.94%  -> loss dropped ~35% in ONE step)
```

**Bonus check — the project's own built-in self-test, reproduced exactly**

```python
# from src/losses.py's own __main__ block:
pred=[0.30, 0.60, 1.20, 0.02],  target=[0.28, 0.55, 1.25, 0.01]  (last pair: near-zero target)
MRE loss = 0.2777443528175354          # confirmed by actually running the file
```

### B.5 Line-by-Line Code Connection

```python
# src/losses.py
class MRELoss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + self.eps))  # -> §B.2

# main.py
def build_criterion():  return MRELoss()      # eps NOT overridden from config -> module default 0.001
# early_stopping.monitor: val_mre             # loss -> monitor convention
```

### B.6 Real-World Applications

- **Demand / retail / financial forecasting** — MAPE (=MRE×100) is one of the most common business KPIs for forecast accuracy.
- **Wave-height and other geophysical/oceanographic verification literature** — percentage-error reporting is a long-standing convention.
- **Any regression problem where the target spans multiple orders of magnitude.**
- **Energy load/price forecasting, epidemiological case-count forecasting, e-commerce demand forecasting.**

### B.7 References

- de Myttenaere, A., Golden, B., Le Grand, B., & Rossi, F. (2016). *Mean Absolute Percentage Error for regression models.* Neurocomputing, 192, 38–48.
- Kim, S. & Kim, H. (2016). *A new metric of absolute percentage error for intermittent demand forecasts.* International Journal of Forecasting, 32(3), 669–679.
- Hyndman, R. J. & Koehler, A. B. (2006). *Another look at measures of forecast accuracy.* International Journal of Forecasting, 22(4), 679–688.

### B.8 Career Relevance

- **Forecasting/time-series roles** (retail, finance, energy, supply chain) — MAPE-family objectives are close to an industry default.
- **Regression tasks with heterogeneous target scales** — knowing when to reach for a relative-error loss instead of MSE/Huber.
- **Data science roles where stakeholders communicate in percentages.**

### B.9 Likely Interview / Viva Questions

- Why does MAPE/MRE blow up near zero, and how do you fix it in practice?
- If your absolute error is similar across two samples, why might MRE score them very differently?
- What's the difference between optimizing MSE and reporting MAPE afterward, vs. optimizing MRE/MAPE directly?
- Does a relative-error loss automatically fix a class-imbalanced regression problem? *(No — see §B.10.)*
- How would you decide between MSE, Huber, and MRE for a brand-new regression problem?

### B.10 The Subtle Insight Worth Remembering

A natural assumption: *"MRE optimizes relative error, so it must automatically help the rare, high-value (minority) samples in an imbalanced dataset."* The derivation in §B.4 shows the opposite: MRE's gradient magnitude is `1/(|y|+ε)`, which is **largest for small targets** and **smallest for large targets**. In an imbalanced dataset where the majority class is small-magnitude targets, MRE actually reinforces accuracy on the already-easy majority, not the rare tail. This is a good example of how the *mathematics of the loss itself*, not just an imbalance-mitigation mechanism layered on top, shapes which parts of a distribution a model ends up good at.

**Worked example — why this matters more than the count-imbalance alone:**

```
gradient-scale = 1 / (|true SWH| + ε)   at representative values across the SWH range:

  Calm    (y≈0.05 m)  -> 1/(0.05+0.001)  = 19.61
  Smooth  (y≈0.30 m)  -> 1/(0.30+0.001)  =  3.32
  Slight  (y≈0.875 m) -> 1/(0.875+0.001) =  1.14
  Moderate(y≈1.875 m) -> 1/(1.875+0.001) =  0.53
  Rough   (y≈3.25 m)  -> 1/(3.25+0.001)  =  0.31

  Calm / Rough ratio ≈ 63.7×
```

A Calm-water sample's gradient is worth ~64× a Rough-water sample's gradient, for the same-sized absolute error. Combined with a Calm/Smooth-dominated sample count, MRE's *own* gradient structure compounds with the dataset's count-imbalance rather than counteracting it. A resampling/reweighting mechanism (like LDS) can correct the *count* imbalance, but not this *structural* gradient-scaling bias baked into the loss formula itself — which is one plausible mathematical root cause behind why relative-error losses win on overall/typical-range accuracy metrics, while absolute-error losses (Huber) combined with feature-space regularizers (ConR) tend to win on rare-tail-weighted metrics.

---

## PART C — MSE Loss (Mean Squared Error)

### C.1 Concept — Why MSE Is Usually Where You Start

MSE is the oldest, most widely used regression loss, for a real mathematical reason: **minimizing MSE is exactly equivalent to Maximum Likelihood Estimation under the assumption that measurement noise is Gaussian (normally distributed).** That equivalence is why MSE is the default in almost every regression library and course — it isn't an arbitrary choice, it's the classical least-squares method (Gauss & Legendre, early 1800s) rediscovered in a deep-learning context.

### C.2 Mathematical Formula and Notation

```
MSE = (1/N) Σᵢ (ŷᵢ − yᵢ)²
```

| Symbol | Meaning |
|---|---|
| yᵢ | True SWH | 
| ŷᵢ | Predicted SWH |
| (ŷᵢ−yᵢ)² | **Squared** error — no absolute value, no branching, no guard |
| N | Batch size |

Notice: unlike Huber (δ) and MRE (ε), **MSE has no extra hyperparameter at all.** That simplicity is also its biggest weakness, shown next.

### C.3 Fully Worked Numerical Example

Same toy network, ŷ = 0.246.

**Case A — target y = 0.30 m (small residual)**

```
r = 0.246 − 0.300 = -0.054
MSE_A = (-0.054)² = 0.002916
```

**Case B — target y = 1.80 m (large residual, the same rare high-SWH case used for Huber)**

```
r = 0.246 − 1.800 = -1.554
MSE_B = (-1.554)² = 2.4149
```

**Direct comparison — this is the whole point of MSE's outlier sensitivity:**

| | Residual | MSE | Huber (δ=0.15) |
|---|---|---|---|
| Case A | 0.054 | 0.002916 | 0.00972 |
| Case B | 1.554 | **2.4149** | **1.479** |
| **Ratio (B/A)** | 28.8× | **828×** | 152× |

The residual only grew 28.8×, but MSE's loss grew **828×** (squaring amplifies the ratio itself: 28.8² ≈ 829). Huber grew far less (152×) because its linear branch caps large-residual growth. **This is MSE's classic "outlier sensitivity" problem** — one rare, possibly noisy sample can dominate an entire batch's gradient.

### C.4 Backward Pass — Gradient (hand-derived, verified against PyTorch autograd)

```
∂L/∂ŷ = 2(ŷ−y)

Case A: ∂L/∂ŷ = 2×(0.246−0.300) = 2×(-0.054) = -0.108
```

**The gradient-magnitude formula is the key structural difference across all three losses — the single most important unifying takeaway of this whole note:**

| Loss | Gradient magnitude | Behavior as residual grows |
|---|---|---|
| **MSE** | 2 × residual | **Unbounded** — grows linearly with the residual, forever |
| **Huber** | residual/δ (small residual) or fixed ±1 (large residual) | **Bounded** — caps out beyond δ |
| **MRE** | 1 / (\|target\|+ε) | **Residual-independent** — depends only on the target's size |

**Full backward pass, Case A (chain rule):**

```
∂L/∂W2 = -0.108×[0.115,0.045] = [-0.01242, -0.00486]     ∂L/∂b2 = -0.108
∂L/∂h  = -0.108×[0.8,1.2] = [-0.0864, -0.1296]
ReLU active both sides -> unchanged
∂L/∂W1 = [[-0.05184,-0.01728,-0.07776],[-0.07776,-0.02592,-0.11664]]   ∂L/∂b1 = [-0.0864,-0.1296]
  [all gradients VERIFIED against PyTorch autograd -- exact match]
```

**Weight update (lr = 0.05) and verification:**

```
ŷ_new = 0.2763   (was 0.246, target=0.300)
L_new = (0.2763-0.300)² ≈ 0.000561   (was 0.002916  -> loss dropped ~81% in ONE step)
  [VERIFIED end-to-end by re-running the forward pass with updated weights in real PyTorch code]
```

### C.5 Code Connection — the One Loss With No Custom File

Huber and MRE both needed a **custom** `src/*.py` file (extra hyperparameters: δ, ε). MSE needs none of that — it's literally PyTorch's built-in loss:

```python
# main.py, mse_only experiment folder
def build_criterion():
    return nn.MSELoss()   # PyTorch built-in -- no custom code needed

# Verified directly: nn.MSELoss()(pred=0.246, target=0.30) = 0.0029160...
# matches the §C.3 hand calculation exactly.
```

### C.6 Real-World Applications

- **The default starting point for essentially any regression problem** — nearly every ML library/course uses MSE first.
- **Linear regression / Ordinary Least Squares (OLS)** — the classical statistical foundation, in use since the early 1800s.
- **Autoencoders and image reconstruction** — the most common pixel-level reconstruction loss.
- **Kalman filtering, control systems, sensor fusion** — the theoretical basis of optimal (least-squares) estimation, directly relevant to robotics/autonomous-vehicle state estimation.
- **Generative models** — the reconstruction term in VAEs, and variants used in some GAN formulations.

### C.7 References

- Legendre, A. M. (1805) and Gauss, C. F. (1809) — the original least-squares method, MSE's historical foundation.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer — Chapter 3 formally derives the MSE ↔ Gaussian-noise Maximum Likelihood equivalence.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

### C.8 Career Relevance and Likely Interview Questions

MSE is assumed knowledge in interviews — the real questions probe its *limitations*:

- Why is MSE sensitive to outliers? Show it mathematically.
- Under what statistical assumption is minimizing MSE provably optimal (Maximum Likelihood)?
- Compare the gradient behavior of MSE, Huber, and MAE as the residual grows.
- If your data is noisy or imbalanced, what would you use instead of MSE, and why? *(This is exactly the arc of this entire note series — MSE → Huber → MRE → imbalance mechanisms.)*

---

## D. Cross-Comparison Summary (All Three Losses)

| | MSE | Huber (δ=0.15) | MRE (ε=0.001) |
|---|---|---|---|
| What it measures | Squared error | Absolute error, bounded | Relative (%) error |
| Extra hyperparameter | None | δ (threshold) | ε (guard) |
| Gradient magnitude | 2×residual (unbounded) | residual/δ, capped at ±1 | 1/(\|target\|+ε) |
| Outlier sensitivity | High (quadratic) | Bounded (linear beyond δ) | Bounded, but scaled by 1/target |
| Behavior near zero target | No issue | No issue | Needs ε guard (÷0 risk) |
| Emphasizes | Large-magnitude errors, everywhere equally | Same, but caps outlier influence | Small-target (majority-class) accuracy |
| This project's monitor metric | `val_rmse` | `val_huber` | `val_mre` |

---

## E. What's Next in This Learning Hub

This note now covers MSE, Huber, and MRE in full depth — the three loss functions used in the underlying research. Natural next additions, in the same format:

1. A companion note on the four **imbalance-mitigation mechanisms**: StratSample (stratified sampling), LDS (Label Distribution Smoothing), RankSim, and ConR — including a novel ConR+LDS combination.
2. A short primer on **AdamW** (the optimizer actually used vs. plain SGD used above for teaching clarity).

---

*Part of a growing self-study collection on deep learning for imbalanced regression, built while researching camera-based significant wave height estimation.*
