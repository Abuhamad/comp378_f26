# 08 — White-Box Evasion Attacks

## Abstract

This lesson defines the evasion problem and shows how a model misclassifies an input under a small perturbation. Students learn the norm constraints behind adversarial examples and derive the single-step FGSM update from a first-order loss approximation. The result is a clear attack model linking model gradients to a bounded input change.

## Objectives

- State the formal evasion optimization problem, identifying the objective function, the decision variable, and the constraint set.
- Explain the geometric meaning of $L_0$, $L_2$, and $L_\infty$ norm balls and give one real-world scenario for each perturbation model.
- Derive the FGSM update rule from the first-order Taylor approximation of the loss and justify why the sign function appears.
- Implement FGSM from scratch in PyTorch and measure attack success rate versus $\epsilon$ across at least four $\epsilon$ values.

## Content

### Evasion problem formulation

A model learns by minimizing training loss over its parameters. An evasion attack changes the input while the model stays fixed. The attacker chooses a perturbation $\delta$ which raises the loss for the true label $y$ while staying in a small norm ball around the original input $x$.

The formal problem is

$$
\arg\max_{\delta} \; \mathcal{L}(f(x + \delta), y) \quad \text{subject to} \quad \|\delta\|_p \leq \epsilon
$$

This objective centers on the input. The decision variable is the perturbation, not the model weights. The constraint set defines the allowed amount of input change. A successful evasion attack keeps the sample visually close to the original image while forcing the model to a wrong prediction.

An adversarial example sits in the small region around a clean sample where the model assigns high loss to the target label. This region stays small enough to keep the image nearly unchanged while the sample crosses the classifier boundary. The task is not training a stronger model. The task is finding a low-cost perturbation to change the model's decision.

### Perturbation norms

The norm $\|\delta\|_p$ determines how the attacker may move the input. Each norm defines a different geometry of allowed changes, and each geometry matches a different threat model.

$L_0$ counts the number of changed pixels. This formulation is sparse. A small set of pixel flips can fool a model, yet the optimization is combinatorial and difficult. A malicious patch or a few altered sensor values are natural examples of an $L_0$ attack.

$L_2$ measures the Euclidean distance of the perturbation vector. This model spreads changes across many pixels and produces smooth, low-energy noise. A medical image or a face image may receive a small but distributed shift under an $L_2$ constraint.

$L_\infty$ limits the maximum per-pixel change. This model caps each coordinate by a fixed amount, so every pixel stays within a small radius. Image classification commonly uses $L_\infty$ because pixel values live in a bounded range, such as $[0, 1]$ or $[0, 255]$.

A 2D geometric view clarifies the difference. An $L_0$ ball looks like a box with a few selected coordinates active. An $L_2$ ball looks like a circle. An $L_\infty$ ball looks like a square. The square captures the common image setting where each pixel can move only a little, but no single pixel dominates the entire attack.

### Fast Gradient Sign Method (FGSM)

The adversarial objective is nonlinear, yet a first-order Taylor approximation gives a direct route to the attack direction. A local expansion gives

$$
\mathcal{L}(f(x + \delta), y) \approx \mathcal{L}(f(x), y) + \nabla_x \mathcal{L}(f(x), y) \cdot \delta
$$

The attacker wants to increase the loss as much as possible, so the best perturbation points in the same direction as the gradient of the loss with respect to the input. Under an $L_\infty$ constraint, each coordinate must satisfy $|\delta_i| \leq \epsilon$.

A coordinate-wise maximization shows the largest increase comes from choosing

$$
\delta = \epsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}(f(x), y))
$$

This yields the FGSM update rule:

$$
x_{\text{adv}} = x + \epsilon \cdot \mathrm{sign}(\nabla_x \mathcal{L}(f(x), y))
$$

The sign function appears because the $L_\infty$ constraint limits the largest absolute value of each entry. The attacker wants the largest possible increase in loss per pixel, and the sign preserves the direction of steepest ascent while respecting the per-pixel bound. The update is a single backward pass with respect to the input tensor, not a gradient step over model weights.

This makes FGSM a fast white-box attack. A model exposes the gradient, the attacker computes the input gradient, and one sign step builds the adversarial image.

### Perturbation budget and perceptibility

The perturbation budget $\epsilon$ controls how much the image changes. In pixel space, $\epsilon = 2/255$ is a small change. $\epsilon = 8/255$ usually remains hard to notice in a natural image. Larger values such as $16/255$ or $32/255$ often create visible artifacts.

This budget creates the adversarial regime. The attack succeeds when the model flips a prediction, yet the perturbation stays below the threshold for human notice. A smaller $\epsilon$ preserves appearance but lowers attack success. A larger $\epsilon$ raises attack success but pushes the image into a visibly altered region.

This trade-off is central to white-box evasion. The objective is to push the input across the decision boundary while keeping the perturbation inside a visually acceptable budget.

## Lab

See the lab notebook at [./lab.ipynb](./lab.ipynb). The notebook crafts adversarial examples for a 10-class MNIST digit classifier using the single-step Fast Gradient Sign Method (FGSM), then trains a binary detector and tests whether detection generalizes across attack strengths.

## Summary

This lesson defined the evasion objective and the norm-ball constraint around the attack. Students compared $L_0$, $L_2$, and $L_\infty$ as different geometric models of allowed perturbations. The lecture ended with the FGSM derivation, where a single input-gradient step creates an adversarial example under a bounded $\epsilon$.

## Useful References and Resources

- Goodfellow, I. J., Shlens, J., and Szegedy, C. “Explaining and Harnessing Adversarial Examples.” ICLR 2015. The classic derivation of FGSM.
- Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. Chapters on differentiable loss functions and gradient-based optimization.
- PyTorch documentation for autograd, gradients, and input-gradient computation.
