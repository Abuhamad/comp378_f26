# 14 — Heuristic Defenses & Input Pre-Processing

## Abstract

This lesson covers practical defenses that pre-process inputs to reduce adversarial perturbations before they reach the classifier. Feature squeezing, JPEG compression, and autoencoder-based denoising reduce adversarial accuracy with minimal deployment cost. Adaptive attacks using BPDA expose fundamental limitations of these defenses, motivating the need for stronger approaches like adversarial training.

## Objectives

- Implement a feature-squeezing pipeline combining bit-depth reduction and spatial smoothing to detect or mitigate adversarial examples.
- Explain how JPEG compression removes high-frequency adversarial perturbation energy and how to tune quality factor to trade clean and adversarial accuracy.
- Measure defense efficacy (clean accuracy drop, adversarial accuracy) for each pre-processing method against FGSM and PGD at matched epsilon.

## Content

### Motivation: Why Pre-Processing Defenses

White-box attacks from m08 (FGSM, PGD) operate at epsilon = 8/255, a 3 percent pixel-value perturbation. Robust training to defend against these attacks costs 10 times more compute than standard training. Pre-processing defenses offer a faster, cheaper alternative. Feature squeezing, JPEG compression, and autoencoder denoising can be deployed immediately while robust models train in the background. This lesson explores their mechanics and their failure modes against adaptive attacks.

### Feature Squeezing

Bit-depth reduction constrains pixel values to fewer bits. A standard image uses 8 bits per channel, allowing 256 values (0 to 255). Reducing to 1 bit per channel yields only 2 values. When we add an adversarial perturbation of size epsilon = 8/255 to an 8-bit value and then quantize to 1-bit, quantization often discards the perturbation.

Spatial smoothing using a median filter removes small high-frequency structures. A 3 by 3 median filter replaces each pixel with the median of its 3 by 3 neighborhood, smoothing out thin lines and sparse noise. A 5 by 5 filter is more aggressive. Adversarial perturbations tend to spread across many pixels in sparse patterns; median filtering can suppress them while preserving edges.

Consistency detection runs the classifier on the original input and the squeezed input, comparing the predicted class. If the two outputs differ, the input is flagged as adversarial. A threshold on the difference in confidence scores can tune the detection sensitivity.

Feature squeezing trades accuracy against defense strength. Reducing to 1 bit destroys fine details, dropping clean accuracy by 20 to 30 percent on CIFAR-10. Reducing to 3 bits preserves more fidelity while still offering some defense.

### JPEG Compression Defense

JPEG compression decomposes images into frequency components via the discrete cosine transform (DCT). Each 8 by 8 pixel block is transformed into cosine basis functions: low frequencies (broad color regions) and high frequencies (edges, fine detail). Adversarial perturbations concentrate in high-frequency components because they must be small in magnitude yet change classifier predictions.

The quantization step follows DCT. High-frequency coefficients are divided by large quantization values and rounded, discarding fine-grained information. The quality factor Q, typically 1 to 100, controls quantization coarseness. Q = 100 preserves all frequencies with minimal loss. Q = 50 aggressively quantizes high frequencies. Q = 1 discards nearly everything.

When JPEG decompresses, high-frequency adversarial perturbations vanish. An adversarial example crafted at the image level cannot survive the DCT and quantization round trip.

Trade-off between robustness and accuracy exists. At Q = 75 on CIFAR-10, adversarial accuracy against FGSM improves 15 to 25 percent compared to no defense, but clean accuracy drops 5 to 10 percent. At Q = 50, the adversarial gain is larger, but clean accuracy drops 15 to 20 percent. Moderate Q values like 75 are practical middle grounds.

### Autoencoder-Based Denoising

An autoencoder is a neural network composed of an encoder and decoder. The encoder maps an input image to a lower-dimensional latent representation. The decoder reconstructs the image from the latent vector. We train the autoencoder on clean images only, minimizing reconstruction mean squared error (MSE).

A denoiser-based defense uses the reconstruction error as an anomaly score. Clean images reconstruct with low MSE. Adversarial examples, which lie outside the training distribution, often reconstruct poorly, producing higher MSE. A threshold on MSE can classify inputs as clean or adversarial.

Limitations: the decoder is trained only on clean data. If an adversarial perturbation is small in magnitude and plausible under the pixel distribution of clean images, the autoencoder may reconstruct it without issue. The denoiser does not learn adversarial robustness; instead, it learns the boundary between clean and slightly corrupted images. An adaptive attacker can craft perturbations that fall within the reconstructable region.

### Limitations of Pre-Processing Defenses

Adaptive attacks know the defense mechanism fully. An attacker who knows the feature squeezing defense can use Expectation Over Transformation (EOT) to average gradients over multiple random pre-processing passes, finding adversarial examples that remain effective after transformation. EOT essentially trains the attack to be robust to the defense.

When pre-processing is non-differentiable (e.g., a median filter or quantization operation), an adaptive attacker uses Backward Pass Differentiable Approximation (BPDA). During backpropagation, the attacker replaces the non-differentiable operation with a smooth approximation. For example, quantization can be approximated by a rectified linear function. This allows gradient-based attacks to flow through the defense, optimizing perturbations that overcome it.

Carlini and Wagner (2017) proposed nine guidelines for defense evaluation. Any defense paper must report adversarial accuracy against adaptive attacks (EOT or BPDA), not just against standard attacks unaware of the defense. Many papers fail this check, claiming defense success only because the attack lacked full knowledge.

Pre-processing defenses are heuristics, not guarantees. They work well against weak, non-adaptive attacks. Against a strong adaptive attacker, they are easily circumvented.

### Defense Evaluation Protocol

A fair evaluation compares three defense configurations and two attack methods. Defense configurations: (1) no defense baseline, (2) bit-depth reduction to 3 bits and 3 by 3 median filter, (3) full three-stage pipeline (bit-depth + median + JPEG at Q = 75). Attack methods: FGSM at epsilon = 8/255; PGD with step size alpha = 2/255, 40 iterations, and epsilon = 8/255.

For each combination, compute three accuracies: clean accuracy (unperturbed inputs), FGSM adversarial accuracy, and PGD adversarial accuracy. Report the nine numbers in a 3 by 3 table (rows are defense variants, columns are input types).

This protocol ensures all defenses are evaluated under the same threat model and computational budget (PGD at 40 steps takes more time than FGSM but allows stronger perturbations).

## Summary

Pre-processing defenses reduce adversarial accuracy through bit-depth reduction, spatial smoothing, or reconstruction-based anomaly detection. These defenses are inexpensive to deploy but can be bypassed by adaptive attacks that account for the defense mechanism. Understanding both the effectiveness and the failure modes of heuristic defenses motivates the need for adversarial training and certified robustness.

## Useful References and Resources

- Xu et al. (2017), "Feature Squeezing: detecting adversarial examples in deep neural networks", on bit-depth reduction and spatial smoothing
- Carlini and Wagner (2017), "Towards Evaluating the Robustness of Neural Networks", on BPDA and defense evaluation guidelines
- Song et al. (2019), "Defending against adversarial examples via random convolutions", on denoiser-based defenses
- DCT and JPEG quantization: standard signal processing references
- PGD attack implementation: Madry et al. (2018)
