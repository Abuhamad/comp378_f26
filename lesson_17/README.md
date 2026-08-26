# Lesson 17 — Certified Robustness via Randomized Smoothing

Objectives:

- State the smoothed classifier definition.
- Justify Gaussian noise injection using the Neyman-Pearson lemma.
- Implement a Monte Carlo sampling procedure with Clopper-Pearson confidence intervals.
- Compute and interpret certified accuracy versus noise level trade-off curves.
- Explain abstention and when a certified prediction is declined.

## Main Concepts

- Randomized smoothing: building a smoothed classifier $g(x) = \arg\max_c \mathbb{P}[f(x+\epsilon)=c]$
- Gaussian noise injection and the isoperimetric inequality basis for $L_2$ certification
- Certified $L_2$ radius computation: Cohen et al. hypothesis testing approach
- Certified accuracy vs. noise level $\sigma$ trade-off

## Learning Outcomes

- Student can state the randomized smoothing certification guarantee and identify its assumptions.
- Student can implement randomized smoothing on a PyTorch vision model using Monte Carlo sampling.
- Student can compute certified accuracy at multiple $\sigma$ levels and produce a certified accuracy vs. $\sigma$ curve.
