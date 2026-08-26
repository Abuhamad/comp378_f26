# Lesson 27 — Differential Privacy and DP-SGD

Objectives:

- State the (ε, δ)-differential privacy definition.
- Compute the noise scale for the Laplace and Gaussian mechanisms.
- Explain how DP-SGD bounds each individual’s contribution.
- Implement DP-SGD training using Opacus.
- Plot ε vs test accuracy across training runs.

## Main Concepts

- $(\epsilon, \delta)$-Differential Privacy: formal definition, composition theorems, and privacy budget accounting
- Laplace and Gaussian mechanisms: sensitivity and noise calibration
- Differentially Private SGD (DP-SGD): per-sample gradient clipping and noise injection
- Privacy budget management: Rényi DP accounting and the moments accountant

## Learning Outcomes

- Student can state the $(\epsilon, \delta)$-DP definition and compute $\epsilon$ for a given number of SGD steps using the moments accountant.
- Student can implement DP-SGD training using `Opacus` with configurable clipping norm and noise multiplier.
- Student can plot the $\epsilon$ vs. test accuracy curve and identify the practical privacy-utility frontier for a given dataset.
