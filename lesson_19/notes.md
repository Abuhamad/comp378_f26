# 19 — Robustness Benchmarking & Red-Teaming Toolkits

## Abstract

Standardized evaluation of adversarial robustness requires reproducible attack protocols and systematic testing workflows. This lesson covers **RobustBench**, the **AutoAttack** ensemble, the Adversarial Robustness Toolbox (**ART**), and the structured red-teaming process that organizations use to audit models before deployment. Students learn how to run multi-attack audits, interpret vulnerability metrics, and integrate robustness monitoring into the model development lifecycle.

## Objectives

- Student can explain what AutoAttack is, why it uses four complementary attacks, and why it is preferred over single-attack evaluation for leaderboard comparisons.
- Student can use IBM ART to run a multi-attack robustness audit and produce a structured vulnerability report with per-attack metrics.
- Student can situate a model on the RobustBench leaderboard by its AutoAttack accuracy.
- Student can describe the six phases of a structured red-teaming workflow and how each phase produces a concrete deliverable.
- Student can configure a continuous monitoring integration that flags robustness regression across model updates.

## Content

### Standardized Robustness Evaluation Protocols

Robustness claims without standardized evaluation are unreliable. The RobustBench leaderboard, launched in 2020, solved a critical problem: a defense evaluated only against FGSM often fails against PGD or C&W. Two papers claiming 85% robustness might mean different things depending on the attack suite used. RobustBench maintains CIFAR-10 and ImageNet leaderboards where all submissions run the same attack protocol, making cross-paper comparisons valid.

The leaderboard accepts model checkpoints with metadata: attack method, perturbation budget (epsilon), dataset, architecture. Robustness is reported as adversarial accuracy under AutoAttack at epsilon = 8/255 on CIFAR-10 or epsilon = 4/255 on ImageNet. Submissions must include code and results that reviewers can reproduce. This reproducibility requirement prevents manipulation.

RobustBench also tracks corruption robustness via CIFAR-10-C, where models are tested on 15 types of corruptions (Gaussian blur, fog, rotation) at five severity levels. A robust model typically maintains 50-60% accuracy on CIFAR-10-C if it reaches 50% AutoAttack accuracy, showing that adversarial and corruption robustness partially overlap but are not identical.

### AutoAttack Composition

AutoAttack is an ensemble of four complementary attacks, each targeting different failure modes. Submitting a defense only against one attack (like FGSM) to a leaderboard would be rejected because single attacks miss large regions of the threat surface.

APGD-CE applies PGD with cross-entropy loss and a descending step size schedule. It works well when the model has clear loss gradients. APGD-DLR uses the Difference of Logits Ratio loss, which avoids saturation. The DLR loss is scale-invariant and still provides useful gradients even when softmax probabilities are near 0 or 1. On defenses with flat loss surfaces, DLR often finds adversarial examples when CE fails.

FAB, the Fast Adaptive Boundary attack, minimizes perturbation norm rather than maximizing loss. It searches along the decision boundary to find the closest adversarial example. FAB catches models that prioritize large margins but can be fooled at small perturbation norms. Square Attack is a score-based black-box attack that requires only access to model predictions, not gradients. It uses random search over the threat space and finds failures in some defenses by probing without gradient information.

All four attacks run sequentially on the same input batch. The final adversarial accuracy is the fraction of examples that remain correctly classified after all four attacks. This ensemble approach prevents evaluation bias and is now the industry standard for robustness leaderboards.

### Adversarial Robustness Toolbox (ART) Architecture

IBM's Adversarial Robustness Toolbox provides a unified interface for adversarial attacks and defenses. The core classes are `Estimator` (wraps a PyTorch, TensorFlow, or Keras model), `Attack` (40+ attack implementations), `Defense` (pre-processing, adversarial training, and certified defense wrappers), and `Metrics` (fooling rate, certified radius, decision boundary distance).

To audit a model, create an `Estimator` by passing a PyTorch model and a preprocessing function. Then instantiate attack objects (e.g., `FastGradientMethod` for FGSM, `ProjectedGradientDescent` for PGD, `CarliniL2Method` for C&W). Call `attack.generate(x_test)` to produce adversarial examples. ART handles batching, device placement, and gradient tracking automatically.

A minimal ART audit loop loads a trained model, wraps it in `PyTorchClassifier`, instantiates three attacks with given epsilon budgets, and runs each on test data. The output is per-attack accuracy and perturbation magnitude. Defenses are added via `Defense` classes: `AdversarialTraining` wraps training loops to include adversarial examples, and `FeatureSqueezing` applies input compression.

ART is extensible: you can subclass `Attack` and override `_compute_perturbation()` to add custom attacks. This flexibility allows researchers to prototype new defenses and audits without forking the library.

### Automated Vulnerability Auditing

Structured vulnerability audits run multiple attack families systematically and produce a report. The audit specifies a threat model (perturbation budget, norm), attack family (FGSM, PGD, AutoAttack), and metrics (fooling rate, mean perturbation magnitude, certified radius if applicable).

Per-attack metrics include fooling rate (percentage of test examples fooled), mean L2 perturbation magnitude (average Euclidean distance of successful adversarial examples), and mean L-infinity magnitude (max pixel change). A model with 50% fooling rate under FGSM at epsilon 8/255 but 90% fooling rate under PGD reveals that gradient-free or less adaptive attacks underestimate vulnerability.

A structured audit report is a table with one row per attack, recording fooling rate, mean perturbation magnitude, and a severity label (critical if fooling rate > 80%, high if 50-80%, medium if 20-50%, low if < 20%). The report also recommends a primary defense action: if fooling rate is high at small epsilon, adversarial training is likely needed. If only adaptive attacks (PGD, C&W) succeed at large epsilon, certified defenses may be necessary.

Audits scale: running 10 attacks on 10,000 test examples generates a detailed vulnerability profile. ART's `Metrics` class computes all standard measures and generates the report automatically. Saving the report to JSON or CSV enables versioning and trend analysis across model iterations.

### Red-Teaming Process

Red-teaming is a structured workflow distinct from ad-hoc testing. The six phases are:

**Phase 1:** Scope and threat model definition. Specify protected assets (the model's predictions), threat actors (untrusted users, domain adversaries), attack surface (inference only, or include training data access?), and perturbation budget (epsilon for evasion, or access to model weights for backdoors?).

**Phase 2:** Attack surface enumeration. List all known attack families that fit the threat model. Include FGSM, PGD, AutoAttack for evasion, model extraction if the attacker has query access, and poisoning if training data is controllable. This phase prevents ad-hoc attacks and ensures systematic coverage.

**Phase 3:** Multi-attack evaluation run. Execute all enumerated attacks via ART or custom code. Record per-attack fooling rate, perturbation magnitude, and query cost. This phase produces raw vulnerability data.

**Phase 4:** Finding triage. Sort findings by severity (how damaging?) and exploitability (how easy to trigger?). A high-severity, low-exploitability finding (requires thousands of queries) is lower priority than a low-severity, high-exploitability finding (requires one input). Triage focuses the response effort.

**Phase 5:** Structured report. Document findings, reproduce one representative example per attack family, and recommend defenses. The report is a contract: if recommendations are followed, the next red-team cycle should find fewer critical issues. This report feeds into the AI Impact Assessment required by governance frameworks.

**Phase 6:** Continuous monitoring. Integrate robustness checks into the CI/CD pipeline. On each model update, run the full audit and flag any regression (e.g., fooling rate increased by > 5 percentage points). MLflow, introduced in **Lesson 06**, integrates with this workflow: log robustness metrics alongside accuracy and F1, enabling automated alerts if monitored values degrade.

Red-teaming differs from ad-hoc testing in three ways. First, it is systematic. The scope, threat model, and attack family list are fixed before testing, not decided reactively. Second, it is reproducible. Every finding is backed by a code recipe. Third, it is continuous. Red-teaming does not end at deployment. It monitors robustness regressions as the model and its environment evolve.

## Lab

See [./lab-latest.ipynb](./lab-latest.ipynb). The lab exercises automated vulnerability auditing using IBM ART: loading a CIFAR-10 model into an ART `PyTorchClassifier`, running three attacks (FGSM, PGD, C&W) on test data, computing per-attack fooling rates and perturbation magnitudes, and producing a structured vulnerability report. The lab also covers situating a model on the RobustBench leaderboard by comparing its AutoAttack accuracy to published leaderboard values.

## Summary

Standardized robustness benchmarks like RobustBench use the AutoAttack ensemble to prevent single-attack evaluation bias. The Adversarial Robustness Toolbox (ART) provides a unified interface for running multi-attack audits and generating structured reports. Automated vulnerability auditing produces per-attack metrics (fooling rate, perturbation magnitude) that inform defense selection. Red-teaming codifies auditing into a six-phase workflow: scope definition, attack surface enumeration, multi-attack evaluation, finding triage, structured reporting, and continuous monitoring. This systematic approach prevents overlooked vulnerabilities and ensures robustness claims are reproducible.

## Useful References and Resources

- RobustBench leaderboard: https://robustbench.github.io/. CIFAR-10 and ImageNet leaderboards with AutoAttack evaluation criteria and model checkpoints.
- AutoAttack paper (Croce & Hein, 2020): "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks". Describes APGD-CE, APGD-DLR, FAB, and Square Attack.
- IBM Adversarial Robustness Toolbox (ART) documentation: https://adversarial-robustness-toolbox.readthedocs.io/. API reference, attack implementations, and tutorial notebooks.
- NIST AI Risk Management Framework (AI RMF) section on red-teaming: https://pages.nist.gov/AIRMF/. Governance and deployment best practices.
- MLflow documentation (from Lesson 06): https://mlflow.org/. Experiment tracking and model registry for continuous monitoring.
