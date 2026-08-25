# Deep Learning for Imbalanced Regression — Self-Study Notes

Personal, worked-from-first-principles notes on the deep learning concepts I use in my research (camera-based significant wave height estimation — inland-water/Great-Lakes deep learning for ocean engineering) — built to actually understand each mechanism well enough to explain it from scratch, not just call a library function.

Every note follows the same structure: **concept → math (every symbol defined) → a tiny hand-worked numeric example, cross-checked by actually running the code → real project code walkthrough → where it shows up in industry → references → interview-style questions.**

## Contents

| Topic | Status | Link |
|---|---|---|
| Huber Loss | ✅ Done | [`01-loss-functions/loss-functions-mse-huber-mre.md`](01-loss-functions/loss-functions-mse-huber-mre.md#part-a--huber-loss) |
| MRE Loss (epsilon-guarded) | ✅ Done | [`01-loss-functions/loss-functions-mse-huber-mre.md`](01-loss-functions/loss-functions-mse-huber-mre.md#part-b--mre-loss-epsilon-guarded-mean-relative-error) |
| MSE Loss | ✅ Done | [`01-loss-functions/loss-functions-mse-huber-mre.md`](01-loss-functions/loss-functions-mse-huber-mre.md#part-c--mse-loss-mean-squared-error) |
| StratSample / LDS / RankSim / ConR (imbalance mitigation) | 🔜 Planned | [`02-imbalance-mechanisms/`](02-imbalance-mechanisms/) |
| AdamW optimizer | 🔜 Planned | [`03-optimizers/`](03-optimizers/) |

A Word-document version of the completed notes (same content, formatted for offline reading/printing) is in [`docs/`](docs/).

## Why this repo exists

Loss-function and imbalance-handling choices are usually treated as one-line config swaps (`loss: huber` vs `loss: mre`) — but each one encodes a real mathematical assumption about which mistakes matter most. Working through the math and code by hand, with real verified numbers, is what actually sticks. This repo is where that work lives, growing alongside my research.

## About

Md Istiak Ahammed — PhD Researcher, Ocean Engineering / AI Applications, Michigan Technological University. Deep learning for wave measurement, wave-height estimation, and (upcoming) autonomous surface vehicle environment perception.
