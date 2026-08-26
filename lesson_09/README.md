# Lesson 09 — Advanced Iterative White-Box Attacks

Objectives:

- Explain why PGD produces stronger adversarial examples than FGSM.
- State the C&W objective and explain the role of the confidence parameter kappa.
- Describe how JSMA selects pixels with the input Jacobian.
- Compare PGD and C&W at matched epsilon and report success rate and mean L2 magnitude.

## Main Concepts

- Projected Gradient Descent (PGD): multi-step FGSM with projection onto the $\epsilon$-ball
- Carlini & Wagner (C&W) attack: optimization-based, confidence-parameterized, $L_2$-minimal
- Jacobian-based Saliency Map Attack (JSMA): $L_0$-bounded feature-perturbation via input Jacobian

## Learning Outcomes

- Student can explain why PGD is a stronger attack than FGSM and what the projection step accomplishes.
- Student can implement the C&W $L_2$ attack using PyTorch's `autograd` optimization loop.
- Student can compare attack success rate and average perturbation magnitude across PGD and C&W at matched $\epsilon$.
