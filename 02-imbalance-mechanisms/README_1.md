# Imbalance-Mitigation Mechanisms

Notes, in the same worked-example format as [`01-loss-functions`](../01-loss-functions/loss-functions-mse-huber-mre.md):

- **StratSample** — stratified/rebalanced sampling — 🔜 Planned
- **LDS** — Label Distribution Smoothing (Yang et al., 2021, ICML) — ✅ Done — [`lds.md`](lds.md)
- **RankSim** — ranking-similarity feature-space regularizer — ✅ Done — [`ranksim.md`](ranksim.md)
- **ConR** — contrastive regularizer for imbalanced regression (Keramati et al., 2024, ICLR) — ✅ Done — [`conr.md`](conr.md)
- A novel **ConR+LDS** combination — 🔜 Planned

Each covers: why it's needed, the algorithm, a hand-verified numeric demo, real code walkthrough, and industry/interview relevance.
