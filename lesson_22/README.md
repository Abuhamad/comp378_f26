# Lesson 22 — Defenses Against Data Poisoning and Backdoor Detection

Objectives:

- Apply Spectral Signatures using SVD to detect and quarantine suspected poison samples.
- Implement Activation Clustering using k-means to identify and quarantine infected clusters.
- Run Neural Cleanse to reverse-engineer per-class trigger candidates.
- Apply model unlearning via fine-tuning on clean data.
- Report a complete defense evaluation using detection rate, ASR, and clean accuracy.

## Main Concepts

- Data sanitization: outlier detection in feature space to remove poison samples
- Spectral Signatures: singular value decomposition to detect poisoned feature correlations
- Activation Clustering: clustering internal activations to separate poisoned from clean samples
- Neural Cleanse: reverse-engineering minimal triggers via optimization; model unlearning post-detection

## Learning Outcomes

- Student can implement Activation Clustering to identify and quarantine poisoned samples in a training set.
- Student can run Neural Cleanse to reverse-engineer a backdoor trigger from a suspected infected model.
- Student can apply model unlearning (fine-tuning on clean data) to remove a detected backdoor and verify ASR drops below threshold.
