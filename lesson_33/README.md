# Lesson 33 — Bias Mitigation Strategies

Objectives:

- Implement re-weighing using AIF360.
- Describe the disparate impact remover transformation.
- Implement adversarial debiasing using AIF360.
- Apply Hardt et al. equalized odds post-processing.
- Produce a side-by-side comparison of at least two mitigation approaches.

## Main Concepts

- Pre-processing mitigation: re-weighing training samples; disparate impact remover
- In-processing mitigation: adversarial debiasing — adversary penalizes demographic prediction
- Post-processing mitigation: equalized odds threshold adjustment (Hardt et al.)
- Mitigation-accuracy trade-off: measuring accuracy cost of each technique

## Learning Outcomes

- Student can implement re-weighing and adversarial debiasing using `AIF360` on the Adult Income dataset.
- Student can apply Equalized Odds post-processing and measure changes in fairness metrics before and after.
- Student can compare pre-, in-, and post-processing approaches by accuracy cost and fairness improvement across at least two fairness metrics.
