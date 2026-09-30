# 15 — Adversarial Training and Robust Optimization

## Abstract

This lesson covers adversarial training, the dominant empirical defense against adversarial examples. Students learn the minimax robust optimization objective, implement standard adversarial training using FGSM augmentation, and then implement the stronger PGD adversarial training algorithm from Madry et al. The lesson also examines the accuracy-robustness trade-off, a key tradeoff in deployed robust models.

## Objectives

- Derive the minimax robust optimization problem and explain the role of the inner maximization and outer minimization loops.
- Implement standard adversarial training by augmenting minibatches with FGSM adversarial examples.
- Implement PGD adversarial training with specified step size, iteration count, and random restart.
- Train a PGD-adversarially trained model to convergence on CIFAR-10.
- Report and interpret the clean accuracy versus PGD-10 robust accuracy trade-off on a held-out test set.

## Content

### Hook: Why preprocessing defenses fail

The previous lesson showed preprocessing defenses like feature squeezing, JPEG compression, and autoencoders. These methods remove or smooth features in input space before feeding data to a classifier. But adversaries can attack the preprocessing itself. An adversary generates an adversarial example, then passes it through the defense, and the defense often fails to remove the perturbation. Stronger attacks anticipating the defense outperform attacks on undefended models. Preprocessing defenses break under adaptive attacks because they rely on a single fixed transformation and do not teach the model to represent robust features.

A different approach asks: what if we train the model to resist adversarial examples? Instead of adding a separate defense stage, we embed robustness into the model itself by updating its weights to minimize loss on adversarial examples, not just clean data. This leads to the concept of adversarial training.

### Minimax robust optimization

Adversarial training optimizes a minimax objective. The formal problem is

$$min_theta max_{||delta||_p <= epsilon} L(f_theta(x + delta), y)$$

The inner maximization finds the strongest attack within an $L_p$ ball of radius epsilon. The outer minimization adjusts the model parameters theta to minimize the worst-case loss. This objective connects adversarial training to game theory. The adversary tries to cause the largest loss, and the defender (model) tries to minimize loss against the adversary's best effort. Solving the minimax problem exactly is intractable because the inner max requires enumerating all possible perturbations and the outer min must adjust theta accordingly. Practical adversarial training approximates this objective by using a finite-step inner optimization, typically FGSM or PGD, rather than solving the inner max exactly.

The minimax formulation explains why adversarial training is computationally expensive. Each gradient step on theta requires running a full attack (the inner loop) first, then computing a gradient on the adversarial examples. Standard gradient descent on clean data requires one backward pass per minibatch. Adversarial training requires an attack plus a backward pass, roughly doubling the training time.

### Standard adversarial training

Standard adversarial training uses FGSM to generate adversarial examples during each minibatch. The algorithm works as follows. For each minibatch of clean examples (x, y), compute the gradient of the cross-entropy loss with respect to x using one backward pass. Generate adversarial examples as $x_adv = x + epsilon * sign(grad_x L)$. Clip the adversarial examples to stay within the allowed perturbation ball and the valid image range. Compute the cross-entropy loss on the adversarial batch and backpropagate to update theta. This cycle repeats for every minibatch.

FGSM adversarial training improves robustness over standard training but at a cost. The computational overhead is roughly 2x the cost of standard training because each minibatch requires one forward and backward pass to compute gradients for the attack, then a second forward and backward pass on the adversarial examples. A model trained on FGSM examples achieves 60-70% robust accuracy against PGD-10 attacks, compared to 10-20% for standard clean training. Models trained on FGSM adversarial examples can be broken by PGD attacks because PGD refines the perturbation over multiple steps, whereas FGSM uses only a single step. FGSM training is stronger than no training, but still vulnerable to stronger attacks.

### PGD adversarial training

Madry et al. 2018 proposed PGD adversarial training, which replaces the single FGSM step with a full PGD inner loop. The algorithm is as follows. For each minibatch of clean examples (x, y), initialize a perturbation delta uniformly at random in the $L_p$ ball of radius epsilon. Run k steps of projected gradient descent on delta, maximizing the loss, with step size alpha and projection back to the ball after each step. This inner PGD loop generates a strong adversarial example. Compute the cross-entropy loss on the PGD adversarial example and backpropagate to update theta.

The inner PGD maximization is stronger than FGSM because it refines the attack over multiple steps. More PGD steps in the inner loop improve the quality of the adversarial examples used for training. Madry et al. recommend k=7 to k=20 steps, depending on the threat model and computational budget. Typical values for CIFAR-10 are k=7 with step size alpha=2/255 and epsilon=8/255. The random restart is crucial: starting from a random perturbation rather than zero allows the algorithm to find diverse adversarial examples and avoid settling into local maxima. This variety during training improves generalization to different attacks at test time.

PGD adversarial training provides a certified first-order worst-case guarantee. If the inner loop solves the maximization problem to first-order optimality (i.e., the gradient with respect to delta is small), then the solution is an upper bound on the true worst-case loss. This guarantee does not hold in the nonconvex setting for deep networks, but the statement still motivates the algorithm: training against stronger approximations to the worst-case attack improves robustness.

The computational cost of PGD adversarial training is higher than FGSM training. With k=7 inner steps, the training time is roughly 8x the cost of standard clean training on the same data. Models trained with PGD adversarial training achieve 70-80% robust accuracy against PGD-10 attacks on CIFAR-10, compared to 10-20% for standard training and 60-70% for FGSM training. This is the strongest empirical defense known for the L_infinity threat model without additional assumptions.

### Accuracy-robustness trade-off

Models trained with adversarial training suffer reduced clean accuracy compared to standard trained models. A standard model on CIFAR-10 achieves 90-95% clean accuracy. A model trained with PGD adversarial training achieves 80-87% clean accuracy on the same dataset. This 5-15% drop in clean accuracy is the accuracy-robustness trade-off. Every percentage point of additional robustness costs clean accuracy.

Ilyas et al. 2019 proposed a theoretical explanation for this trade-off. Standard training uses non-robust features, which are features correlated with the true label within the training set but rely on small variations at the scale of adversarial perturbations. Robust features are features whose variation remains meaningfully correlated with the label even under adversarial perturbation. Robust training must learn robust features instead of non-robust features. This shift in the feature set reduces accuracy on clean data because robust features are less informative overall, even though they generalize better under perturbation.

The practical implication is a deployment decision. Does the application require high clean accuracy or high robust accuracy? In an adversarial threat setting, robust accuracy matters more. In a benign setting, clean accuracy is the priority. Some applications may accept the trade-off and deploy a model with 85% clean accuracy and 75% robust accuracy. Others require 90%+ clean accuracy and accept the lower robust accuracy.

### Training protocol details

Practical PGD adversarial training uses several additional techniques to improve convergence and robustness. Cyclic learning rates help avoid getting stuck in local minima. Early stopping on the inner loop prevents the PGD maximization from solving too accurately, which can cause overfitting to the worst-case examples. Batch size affects training dynamics. Larger batches increase the diversity of adversarial examples in each minibatch, which improves robust generalization. Typical batch sizes for adversarial training are 256 to 512 on CIFAR-10.

Random restarts in the inner loop, described earlier, are critical for finding diverse adversarial examples. A single restart may converge to a local adversarial pattern, whereas multiple restarts explore the space of adversarial perturbations more thoroughly. Madry et al. recommend 1 random restart as a practical default, though more restarts increase robustness at higher computational cost.

## Summary

This lesson covered adversarial training, the dominant empirical defense against adversarial examples. The minimax robust optimization formulation frames the problem as a game between an adversary and the model. Standard adversarial training uses FGSM to augment minibatches. PGD adversarial training strengthens the inner maximization by running k steps of PGD. Models trained with PGD reach 70-80% robust accuracy against PGD-10 attacks on CIFAR-10, but this robustness comes at the cost of 5-15% reduction in clean accuracy. The next lesson explores certified robustness approaches that provide formal guarantees beyond empirical robustness.

## Useful References and Resources

- Goodfellow, I., Shlens, J., and Szegedy, C. Explaining and Harnessing Adversarial Examples. Introduces FGSM and standard adversarial training.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vlachyraya, A. Towards Deep Learning Models Resistant to Adversarial Attacks. Introduces PGD adversarial training and the certified first-order guarantee.
- Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B., and Madry, A. Adversarial Examples Are Not Bugs, They Are Features. Explains the robust features hypothesis and the accuracy-robustness trade-off.
- RobustBench, a benchmark for adversarially trained models on CIFAR-10, CIFAR-100, and ImageNet.
- PyTorch documentation for autograd, gradient clipping, and optimizer hyperparameters.
