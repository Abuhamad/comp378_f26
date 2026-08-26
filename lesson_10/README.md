# Lesson 10 — Black-Box Evasion Attacks

Objectives:

- Explain how zeroth-order finite-difference estimation works without model access.
- Describe the Square Attack update strategy.
- Explain how the Boundary Attack works and iteratively reduces perturbation size.
- Implement the Square Attack against a mock black-box API.
- Compare score-based and decision-based attacks on query efficiency.

## Main Concepts

- Zeroth-order optimization: estimating gradients without model access
- Score-based black-box attacks: Natural Evolution Strategies (NES), Square Attack
- Decision-based black-box attacks: Boundary Attack — random-walk boundary exploration
- Query efficiency: attacks-per-image metrics and query budget constraints

## Learning Outcomes

- Student can explain how zeroth-order gradient estimation enables gradient-free attacks.
- Student can implement the Square Attack against a mock black-box API under a defined query budget.
- Student can compare query efficiency (queries per successful adversarial example) across score-based and decision-based attacks.
