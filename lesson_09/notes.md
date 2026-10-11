# 09 — Advanced Iterative White-Box Attacks

## Abstract

PGD extends FGSM by repeating a signed-gradient step inside a projected $L_\infty$ ball. C&W changes the attack into an optimization over the perturbation and the classification margin, and this method targets a small $L_2$ norm. JSMA keeps the attack sparse by choosing the most influential pixels with the input Jacobian and a strict $L_0$ budget. This lesson shows how white-box attacks move from one-step search to iterative and optimization-based attacks.

## Objectives

- Explain why PGD produces stronger adversarial examples than FGSM and describe the role of the projection step.
- State the C&W objective, explain the role of the confidence parameter $\kappa$, and implement the attack with PyTorch autograd.
- Describe how JSMA selects pixels with the input Jacobian and how the $L_0$ budget limits the attack.
- Compare PGD and C&W at matched $\epsilon$ and report success rate and mean $L_2$ magnitude.

## Content

### Projected Gradient Descent (PGD)

FGSM uses one signed gradient step. PGD repeats this step many times and projects each result back into the allowed region around the clean input. The update rule is

$$
x^{t+1} = \Pi_{\mathcal{B}_\epsilon(x)}(x^t + \alpha \cdot \text{sign}(\nabla_{x^t}\mathcal{L}(f(x^t), y)))
$$

The operator $\Pi_{\mathcal{B}_\epsilon(x)}$ is the projection onto the $L_\infty$ ball of radius $\epsilon$ around $x$. In practice, each coordinate is clipped to $[x_i - \epsilon, x_i + \epsilon]$ after every step. This projection keeps the adversarial example inside the same perturbation budget while allowing multiple updates to move deeper into the misclassification region. A random start improves the attack. The update begins from
$$
x^0 = x + u, \quad u \sim \mathcal{U}(-\epsilon, \epsilon)
$$
This random offset prevents the attack from locking onto a weak local path before the search begins. Small step size $\alpha$ and many iterations $k$ usually produce stronger attacks than one large step. PGD remains the standard white-box benchmark because the method preserves a clear search procedure inside the norm ball while raising the attack strength above a single FGSM step.

### Carlini & Wagner (C&W)

FGSM and PGD optimize a loss tied to the true label, yet neither method directly optimizes the classification margin. C&W changes the setup by solving a direct optimization problem over the perturbation itself. The objective is
$$
\min_{\delta} \|\delta\|_2 + c \cdot f(x+\delta)
$$
The term $\|\delta\|_2$ keeps the perturbation small. The function $f$ penalizes correct classification and pushes the model toward a wrong prediction. A successful adversarial example reduces the margin for the true class while keeping the perturbation compact. The box-constraint issue appears because the adversarial input must stay inside valid pixel bounds. C&W removes this issue with a change of variables. The perturbation is written as
$$
\delta = \tfrac{1}{2}(\tanh(w) + 1) - x
$$
This keeps the perturbed input inside $[0,1]$ without an explicit projection step. The search then runs over $w$ instead of raw pixels, which makes the optimization smoother and more stable. The confidence parameter $\kappa$ controls how strongly the attack pushes the model away from the true label. Larger $\kappa$ encourages a larger classification gap, which often yields a more confident misclassification but a larger perturbation. In practice, the attack searches over a penalty coefficient $c$ with binary search to find the smallest perturbation causing a prediction change under the requested confidence level. C&W is a strong $L_2$ attack for perturbation minimization. In the lab, each run evaluates 64 inputs and reports the mean $L_2$ norm across the batch. This batch view shows whether the attack stays strong across many examples rather than only on one image.

### Jacobian-based Saliency Map Attack (JSMA)

JSMA follows a different route. This method works under an $L_0$ budget, which limits the total number of pixels changed. The objective is sparse, not smooth. The attack chooses a small set of pixels with the highest effect on the model output. The key object is the input Jacobian of the model output with respect to the input. For a target class $t$, the Jacobian entry is
$$
J_{ij} = \frac{\partial F_t(x)_i}{\partial x_j}
$$
A single backward pass per class yields these derivatives. The attack then builds a saliency map from the most influential input coordinates. A common score for a candidate pixel pair $(p, q)$ is based on the sum of positive directional effects on the target class and negative effects on competing classes, which highlights the pair with the strongest change in model output. The attack chooses the pair with the highest saliency score, modifies both entries by a step $\theta$, and repeats until either the model misclassifies or the $L_0$ budget is exhausted. This makes JSMA different from $L_\infty$ attacks such as PGD. PGD spreads a small change across many pixels. JSMA changes a few pixels aggressively, which matches the $L_0$ threat model. The attack remains sparse, targeted, and bounded by a count constraint rather than a per-pixel magnitude constraint.

## Lab

See the lab notebook at [./lesson_09/lab.ipynb](./lesson_09/lab.ipynb). The notebook crafts adversarial examples for a 10-class MNIST digit classifier using the Projected Gradient Descent (PGD) white-box attack, then trains a binary detector that tries to catch those perturbations and see why it does not generalize across attack strengths.

## Summary

PGD adds iterative projection to FGSM, which raises attack strength under a fixed $L_\infty$ budget. C&W swaps the loss for a perturbation-minimization objective and reports the smallest $L_2$ change causing a wrong-class flip. JSMA keeps the perturbation sparse by selecting a few high-impact pixels. These methods form the white-box attack family for this lesson.

## Useful References and Resources

- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. “Towards Deep Learning Models Resistant to Adversarial Attacks.” ICLR 2018. The standard PGD formulation and projection-based attack benchmark.
- Carlini, N., and Wagner, D. “Towards Evaluating the Robustness of Neural Networks.” IEEE Symposium on Security and Privacy, 2017. The C&W optimization framework and confidence parameter.
- Papernot, N., McDaniel, P., Swami, A., and Harang, R. “The Limitations of Deep Learning in Adversarial Settings.” IEEE European Symposium on Security and Privacy, 2016. The JSMA formulation and saliency-map attack logic.
- PyTorch documentation for autograd, gradients, and tensor operations used in adversarial optimization.
