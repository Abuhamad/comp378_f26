# Lesson 15 — Adversarial Training and Robust Optimization

Objectives:

- Derive the minimax robust optimization problem.
- Implement standard adversarial training by augmenting minibatches with FGSM examples.
- Implement PGD adversarial training with specified step size, iteration count, and restart.
- Train a PGD-adversarially trained model to convergence on CIFAR-10.
- Report and interpret the clean accuracy versus PGD-10 robust accuracy trade-off.

## Main Concepts

- Minimax formulation: $\min_\theta \max_{\|\delta\|\leq\epsilon} \mathcal{L}(f_\theta(x+\delta), y)$
- Standard adversarial training: augmenting minibatches with FGSM adversarial examples
- PGD adversarial training (Madry et al.): inner maximization via PGD; certified first-order worst case
- Accuracy-robustness trade-off: natural accuracy degradation under adversarial training

## Learning Outcomes

- Student can derive the minimax training objective and explain the role of the inner and outer loops.
- Student can implement PGD adversarial training and train a model to convergence on CIFAR-10.
- Student can report and interpret the clean accuracy vs. PGD-10 robust accuracy trade-off on a held-out test set.
