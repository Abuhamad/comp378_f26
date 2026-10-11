# 11 — Adversarial Transferability & Substitute Models

## Abstract

Adversarial examples crafted on one model often fool other architecturally distinct models, a phenomenon called transferability. This lesson explores why transferability occurs, examines two leading hypotheses for the mechanism, and introduces the substitute model training pipeline, a practical black-box attack strategy exploiting transferability to generate effective adversarial perturbations without querying the target model thousands of times. Amplification techniques increase transfer success rates, enabling real-world attacks across diverse model deployments.

## Objectives

- Explain the two main hypotheses for why adversarial examples transfer across model architectures.
- Train a substitute model by querying a black-box target API and collecting synthetic labels.
- Generate transferable adversarial examples using PGD applied to the trained substitute.
- Measure the cross-architecture transfer fooling rate between at least two distinct model families.
- Articulate how MI-FGSM and input diversity improve transfer success rate over vanilla FGSM.

## Content

### Transferability Phenomenon

Szegedy et al. (2014) first observed adversarial examples generated on one deep neural network transfer to fool other, different networks. Papernot et al. (2016) demonstrated systematic cross-architecture transfer: adversarial examples crafted on ResNet-50 fool VGG-16, and examples crafted on ResNet fool InceptionV3. Transfer success varies by architecture pair but consistently exceeds chance. On CIFAR-10, a PGD example optimized against ResNet-18 with epsilon = 8/255 fools VGG-16 at roughly 73% fooling rate, compared to 95% when an attacker attacks VGG-16 directly. This gap motivates the substitute model strategy. Transferability occurs both within-domain (different ImageNet classifiers) and cross-domain (ImageNet-trained models transfer to specialized medical imaging models), though transfer rates drop in cross-domain scenarios. The phenomenon has profound security implications: an attacker without gradient access to a production model can train a substitute, attack it freely, and expect reasonable success against the real target.

### Two Transferability Hypotheses

Two competing hypotheses explain transferability. The first hypothesis posits deep networks trained on the same data distribution learn similar feature representations. During ImageNet training across many architectures, early layers converge to edge and texture detectors, mid-layers to shape and object-part features, and deeper layers to high-level semantic features. Empirical studies show 50-80% feature overlap in lower layers across ResNet and VGG architectures, enabling perturbations exploiting these shared features to generalize across models. An adversarial patch targeting "dog ears" works because ResNet and VGG both learn dog-ear features at similar locations in their feature hierarchies. The second hypothesis emphasizes aligned decision-boundary geometry. Neural networks trained on high-dimensional data create decision boundaries occupying only small-volume subspaces relative to the ambient dimension. Adversarial perturbations pushing examples across one model's boundary often cross other models' boundaries in the same region of input space, because the boundaries are locally parallel or quasi-parallel. This geometric alignment emerges naturally from the data manifold structure, independent of training procedure. Both hypotheses contribute to observed transfer rates. Adversarial training and specific architectural choices can decouple these mechanisms. Models trained on semantically disjoint datasets show lower transfer rates, suggesting the shared-features hypothesis plays a measurable role.

### Substitute Model Training Pipeline

The substitute model training pipeline consists of four steps. First, the attacker selects an initial query set, typically 100 to 2,000 unlabeled or pseudo-labeled samples from a public dataset related to the target domain (e.g., CIFAR-10 if the target classifies CIFAR-10-like images, or ImageNet if the target is a general object classifier). Second, the attacker queries the black-box target model's prediction API, collecting soft-label probability distributions for each sample in the query set. Each query consumes one API call. For 2,000 samples, the attacker makes 2,000 queries. Third, the attacker trains a surrogate model architecture (ResNet-18, VGG-16, or another open-source model) on the collected (image, soft-label) pairs using standard supervised learning with cross-entropy loss over the soft labels. This training step uses no information from the target model's gradients, only its predictions. Fourth, the attacker repeats steps 2 and 3 iteratively. After each surrogate training iteration, the attacker uses the surrogate to identify samples on which it disagrees most with the target, adds those high-disagreement samples to the query set, and retrains. Papernot et al. (2016) demonstrated convergence in 2 to 6 iterations, with surrogate accuracy improving from 65% in iteration 1 to 95% by iteration 6 on MNIST. The Jacobian-based dataset augmentation technique amplifies this: after training the surrogate, compute the Jacobian matrix (gradients of all output logits with respect to all input pixels) for each sample in the query set. Samples with high Jacobian norms (indicating high prediction sensitivity) are near the decision boundary. Small, random perturbations of these samples yield new high-value query points. This augmentation reduces the total number of API queries needed to achieve a given surrogate accuracy by approximately 30-50% depending on the data distribution.

### Transferability Amplification

Transferability of a vanilla PGD attack on the surrogate typically achieves a 50-65% fooling rate against a geometrically distinct target. Three amplification techniques improve this.

**Input diversity.** Input diversity transformations apply random resizing, padding, and crop operations to each image in the PGD loop, forcing the adversarial perturbation to remain robust across multiple scales and translations. Formally, at iteration $t$ of the PGD attack, instead of optimizing the loss on a single input image $x$, the attacker samples random affine transformations $T_i \sim \mathcal{T}$, evaluates the loss on each transformed input $T_i(x)$, and averages the resulting gradients:

$$
g_t = \frac{1}{N} \sum_{i=1}^{N} \nabla_{x^t} \mathcal{L}\bigl(f(T_i(x^t)), y\bigr), \quad T_i \sim \mathcal{T}
$$

After $N = 10$-$20$ random transformation samples per step, this averaging stabilizes the perturbation against geometric shifts. Using input diversity increases transfer rate by approximately 5-10 percentage points on CIFAR-10.

**Translation-invariant FGSM (TI-FGSM).** TI-FGSM replaces the gradient update with a kernel-convolved version. At each gradient-computation step, the gradient is convolved with a small kernel $K$ (typically a $3 \times 3$ or $5 \times 5$ Gaussian or uniform kernel) to smooth out pixel-level noise and emphasize structural features:

$$
\tilde{g}_t = K * \nabla_{x^t} \mathcal{L}(f(x^t), y)
$$

The smoothed gradient update remains effective against the surrogate while generalizing better to new architectures. TI-FGSM improves transfer rate by another 5-10 percentage points.

**Momentum iterative FGSM (MI-FGSM).** MI-FGSM accumulates gradient information across multiple attack iterations. Maintain a momentum term $m$ initialized to zero. At each iteration $t$, compute the gradient $g_t = \nabla_{x^t} \mathcal{L}(f(x^t), y)$ of the loss with respect to the current perturbation, then update the momentum as

$$
m_t = \mu \cdot m_{t-1} + (1 - \mu) \cdot g_t
$$

typically using decay $\mu = 1.0$. Apply the perturbation update using $m_t$ rather than $g_t$:

$$
x^{t+1} = x^t + \alpha \cdot \text{sign}(m_t)
$$

This momentum buffer filters out high-frequency noise and focuses on consistent descent directions, improving both attack success on the surrogate and transfer success to new models. MI-FGSM alone adds 10-15 percentage points to transfer rate.

Combining all three techniques (input diversity, TI-FGSM, momentum) typically lifts transfer rate from 55% to 75-80% against a new architecture when compared to vanilla PGD.

### Ensemble-Based Amplification

Ensemble attacks craft adversarial examples simultaneously against multiple surrogate architectures, then average or combine the resulting perturbations. An attacker trains $K$ surrogates (e.g., $K = 3$: ResNet-18, VGG-16, MobileNet) and generates $K$ independent adversarial perturbations $\delta_1, \dots, \delta_K$, one per surrogate. The ensemble perturbation is the average of the $K$ individual perturbations:

$$
\delta_{\text{ens}} = \frac{1}{K} \sum_{k=1}^{K} \delta_k
$$

or a weighted average $\delta_{\text{ens}} = \sum_{k=1}^{K} w_k \, \delta_k$ with $\sum_{k=1}^{K} w_k = 1$ when certain surrogates match the target architecture more closely. The ensemble perturbation transfers more reliably than any single surrogate's perturbation because it captures features common to multiple model families.

Empirical results show ensemble size matters:

| Ensemble size $K$ | Approximate transfer rate |
|:---:|:---:|
| 1 (single surrogate) | 55-60% |
| 2 | 65-70% |
| 3 | 72-78% |
| 4 | 75-80% |

Beyond $K = 4$, improvements plateau, suggesting diminishing returns as the ensemble approaches the full diversity of possible architectures. Selecting diverse surrogate architectures is critical. An ensemble of three ResNets yields lower improvement than an ensemble of ResNet, VGG, and MobileNet, because the three ResNets learn largely overlapping representations. The attacker can guide surrogate selection by analyzing the target's input-output behavior: if the target makes mistakes on texture-heavy images, selecting a surrogate known to rely on texture (e.g., a network trained on stylized ImageNet) increases ensemble diversity.

### Cross-Architecture Transfer Rate Measurement

To measure transfer rate, compute the fooling rate on the target model when evaluated on adversarial examples generated against a substitute model. Formally, let $S$ be the surrogate model, $T$ be the target (black-box) model, and $x_{\text{adv}} = x + \delta$ be an adversarial example with perturbation $\delta$ crafted such that $S(x + \delta)$ is misclassified but (ideally) $x + \delta$ is imperceptibly perturbed. The transfer rate is the fraction of examples in a test set $\mathcal{D}$ on which $T(x + \delta)$ is also misclassified:

$$
\text{TR} = \frac{1}{|\mathcal{D}|} \sum_{(x, y) \in \mathcal{D}} \mathbb{1}\bigl[T(x + \delta) \neq y\bigr]
$$

where $\mathbb{1}[\cdot]$ is the indicator function. On CIFAR-10 with $\epsilon = 8/255$, a ResNet-18 substitute attacked with PGD and measured against a VGG-16 target achieves a 70-75% transfer rate. The same experiment against Inception-V3 yields a 65-70% transfer rate. Differences in transfer rate across target architectures reflect differences in learned features and decision-boundary geometry. Reporting transfer rate requires specifying the surrogate architecture, the attack method, the perturbation budget $\epsilon$, the target architecture, and the test set. Comparison tables capture this: one row for "PGD on ResNet-18 surrogate, $\epsilon = 8/255$," another row for "PGD on ensemble surrogate, $\epsilon = 8/255$." A proper comparison always includes a white-box baseline (attack the target directly with full gradient access) to contextualize the transfer-rate result. If a white-box attack on VGG-16 achieves a 95% fooling rate and the transfer attack achieves 70%, the 70% value is meaningful because it shows the substitute approach recovers $70/95 \approx 74\%$ of white-box success while requiring no gradient information.

## Lab

See [lab_01.ipynb](./lab_01.ipynb) and [lab_02.ipynb](./lab_02.ipynb) for a hands-on exercise training and measuring transfer fooling rate.

## Summary

Adversarial examples transfer across models because of shared learned features and aligned decision-boundary geometry. The substitute model training pipeline enables practical black-box attacks by querying the target API a limited number of times, training a surrogate on collected soft labels, and generating attack examples on the surrogate transferring to the target. Amplification techniques (input diversity, TI-FGSM, momentum) and ensemble surrogates together lift transfer success rates from 55% to 75-80%, making the attack practical despite no gradient access to the real target.

## Useful References and Resources

- Szegedy, C., Zaremba, W., Sutskever, I., et al. (2014). "Intriguing properties of neural networks." ICLR. [Original transferability observations.]
- Papernot, N., McDaniel, P., Goodfellow, I. (2016). "Transferability in machine learning: from phenomena to black-box attacks using adversarial samples." arXiv:1605.07277. [Substitute model training pipeline, Jacobian dataset augmentation, empirical transfer rates on MNIST and CIFAR-10.]
- Dong, Y., Liao, F., Pang, T., Su, H., Yelchenko, J., Bao, J., Zhou, B., Song, S. (2018). "Momentum Iterative FGSM for Adversarial Attacks." CVPR. [MI-FGSM method and transfer rate improvements.]
- Xie, C., Zhang, Z., Zhou, Y., Song, S., Yuille, A. L. (2019). "Improving Transferability of Adversarial Examples with Input Diversity." CVPR. [Input diversity transformations and measured transfer-rate gains.]
- Li, Y., Bai, S., Zhang, Y., Xia, S. T., Tao, D. (2019). "Improving the Transferability of Adversarial Examples with Ensemble Methods." AAAI. [Ensemble-based amplification, surrogate selection, and K-value analysis.]
- Tramer, F., Papernot, N., Goodfellow, I., Boneh, D. (2016). "The space of transferable adversarial examples." arXiv:1704.03453. [Analysis of decision-boundary geometry and hypothesis testing.]
