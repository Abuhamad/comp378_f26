# Lesson 14 — Heuristic Defenses and Input Pre-Processing

Objectives:

- Implement a feature-squeezing combining bit-depth reduction and spatial smoothing.
- Explain how JPEG compression removes high-frequency adversarial perturbation energy.
- Measure defense efficacy for each pre-processing method against FGSM and PGD.

## Main Concepts

- Feature squeezing: bit-depth reduction and spatial smoothing as pre-processing
- JPEG compression defense: high-frequency perturbation removal
- Autoencoder-based denoising: reconstruction as adversarial noise removal
- Limitations of pre-processing defenses: adaptive attacks that circumvent them

## Learning Outcomes

- Student can implement a feature-squeezing filter pipeline (bit-depth + spatial smoothing) in PyTorch.
- Student can measure the defense efficacy of each pre-processing method against FGSM and PGD.
- Student can identify the conditions under which a pre-processing defense is bypassed by an adaptive adversary.
