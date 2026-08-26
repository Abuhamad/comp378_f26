# Lesson 08 — White-Box Evasion Attacks

Objectives:

- State the formal evasion optimization problem.
- Explain the geometric meaning of L0, L2, and L∞ norm balls.
- Derive the FGSM update rule from the first-order Taylor approximation.
- Implement FGSM from scratch in PyTorch and measure attack success rate versus epsilon.

## Main Concepts

- Mathematical formulation of evasion: $\arg\max_{\delta} \mathcal{L}(f(x+\delta), y)$ subject to $\|\delta\|_p \leq \epsilon$
- Perturbation norms: $L_0$, $L_2$, $L_\infty$ — definitions, geometric meaning, and practical use
- Fast Gradient Sign Method (FGSM): single-step gradient sign update
- Perturbation budget $\epsilon$ and its relationship to perceptibility

## Learning Outcomes

- Student can state the FGSM update rule and derive it from the linearized loss.
- Student can implement FGSM from scratch in PyTorch without using any attack library.
- Student can measure attack success rate (fooling rate) vs. $\epsilon$ and explain the perceptibility trade-off.
