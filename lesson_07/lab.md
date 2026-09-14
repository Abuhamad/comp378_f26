# Adversarial Attacks and Robustness in AI Systems

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Introduction](#introduction) | Background on adversarial ML, real-world impact, workshop scope |
| 2 | [Types of Adversarial Attacks](#2-types-of-adversarial-attacks) | Evasion, poisoning, inference, and extraction attacks |
| 3 | [Threat Models](#3-threat-models) | White-box, black-box, gray-box attacker capabilities |
| 4 | [Model Robustness](#4-model-robustness) | General vs. adversarial robustness, accuracy trade-offs |
| 5 | [Technical Features & Parameters](#5-key-parameters-in-attacks-and-defenses) | Epsilon, learning rate, batch size, regularization effects |
| 6 | [Defense Mechanisms](#6-defense-mechanisms) | Detection, postprocessing, training-based defenses |
| 7 | [Summary & References](#7-summary-of-parameters-and-their-roles) | Key takeaways and further reading |

---

## What You Will Learn

By the end of this workshop, you will be able to:

1. **Identify and classify** the major families of adversarial attacks against machine learning models (evasion, poisoning, inference, extraction).
2. **Understand threat models** that define attacker capabilities -- white-box, black-box, and gray-box -- and select appropriate defenses for each scenario.
3. **Distinguish between general robustness** (resilience to natural noise) and **adversarial robustness** (resistance to intentional manipulation).
4. **Explain key parameters** that govern attacks and defenses: epsilon (perturbation budget), learning rate, step size, iteration count, batch size, and regularization strength.
5. **Implement defense strategies** including adversarial training, defensive distillation, input preprocessing, ensemble methods, and data sanitization.
6. **Navigate the ART framework** (Adversarial Robustness Toolbox) for launching attacks and deploying defenses against ML models.

---

## About ART (Adversarial Robustness Toolbox)

The **Adversarial Robustness Toolbox (ART)** is an open-source Python library originally created by [IBM Research](https://adversarial-robustness-toolbox.github.io/) and now maintained as a [Linux Foundation AI & Data (Trusted-AI) project](https://github.com/Trusted-AI/adversarial-robustness-toolbox). It provides a comprehensive suite of tools for:

- **Attacking models**: Generate adversarial examples using established attack algorithms (`FastGradientMethod`, `ProjectedGradientDescent`, `CarliniL2Method`, `SaliencyMapMethod`, and more)
- **Defending models**: Apply defense mechanisms to harden models against attacks (adversarial training, input preprocessing, detector-based defenses)
- **Evaluating robustness**: Measure how well models resist various attack strategies

ART was designed with a modular architecture, wrapping around popular ML frameworks (TensorFlow, PyTorch, scikit-learn, XGBoost) via a concept called **estimators**. This allows researchers and practitioners to study adversarial ML without reinventing the infrastructure.

> **Note:** This document is designed as a *self-guided educational resource*. You can read through it sequentially without any external references.

### ART Architecture Overview

```
+-------------------------------------------------------------------+
|                     ART Framework                                 |
|                   (Adversarial Robustness Toolbox)                 |
+-------------------------------------------------------------------+

  +------------------+     +------------------+     +--------------+
  |   art.attacks    |     |   art.defences   |     | art.metrics  |
  |                  |     |                  |     |              |
  |  evasion         |---->|  trainer         |-->| evaluators   |
  |  poisoning       |     |  preprocessor    |     |              |
  |  extraction      |     |  detector        |     |              |
  |  inference       |     |  postprocessor   |     |              |
  +------------------+     +------------------+     +--------------+
         |                          |
    +----+----+              +------+------+
    |         |              |             |
FGSM       PGD        AdversarialTrainer  FeatureSqueezing
BIM        C&W        (adv. training)     (preprocessor)
JSMA       DeepFool   DefensiveDistill.   BinaryInputDetector
ZOO        SimBA      (transformer)       (detector)
HopSkipJump ...       ...                 ...

+-------------------------------------------------------------------+
|                     ESTIMATORS LAYER                              |
|                     (art.estimators.classification)               |
|                                                                   |
|  +-----------------+  +-----------------+  +-------------------+  |
|  | KerasClassifier |  | PyTorch         |  | TensorFlowV2      |  |
|  |                 |  | Classifier      |  | Classifier        |  |
|  +-----------------+  +-----------------+  +-------------------+  |
|  (also SklearnClassifier, XGBoostClassifier, and more)            |
|                                                                   |
|  Each estimator wraps a model in a unified ART interface          |
|  (predict, fit, loss_gradient) enabling cross-framework attacks   |
|  and defences without changing the attack/defence code.           |
+-------------------------------------------------------------------+

+-------------------------------------------------------------------+
|                       USAGE FLOW                                  |
|                                                                   |
|  1. Wrap model --> classifier = TensorFlowV2Classifier(...)      |
|  2. Instantiate attack --> attack = FastGradientMethod(          |
|                                estimator=classifier, eps=0.03)    |
|  3. Generate adversarial examples: x_adv = attack.generate(x)    |
|  4. Apply a defence (e.g. AdversarialTrainer)                    |
|  5. Evaluate robustness via ART's built-in metrics               |
+-------------------------------------------------------------------+
```

The core design principle is the **estimator abstraction**. Every supported ML model is wrapped in an ART estimator that exposes a uniform interface (`predict`, `fit`, `loss_gradient`). This means you can switch between TensorFlow and PyTorch backends without modifying your attack or defence code. Similarly, each attack and defence is pluggable: swapping `FastGradientMethod` for `ProjectedGradientDescent` requires only changing the class you instantiate.

### Minimal ART Workflow (real API)

The snippets below illustrate the actual ART API. Attacks are **not** callable objects; you build one against an estimator and then call `.generate()`.

```python
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

# 1. Wrap a trained model in an ART estimator
classifier = TensorFlowV2Classifier(
    model=model,                 # your trained tf.keras model
    nb_classes=10,
    input_shape=(28, 28, 1),
    clip_values=(0.0, 1.0),      # valid pixel range, enforced during attacks
    loss_object=loss_fn,
)

# 2. Build an attack against that estimator (note: eps, not epsilon)
attack = FastGradientMethod(estimator=classifier, eps=0.03)

# 3. Generate adversarial examples (attacks are NOT callable)
x_adv = attack.generate(x=x_clean)
```

Defences follow the same pattern. Adversarial training, for example, wraps an attack and retrains the classifier on a mix of clean and adversarial inputs:

```python
from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer

pgd = ProjectedGradientDescent(estimator=classifier, eps=0.03, eps_step=0.005, max_iter=40)
trainer = AdversarialTrainer(classifier=classifier, attacks=pgd, ratio=0.5)
trainer.fit(x_train, y_train, nb_epochs=10, batch_size=128)
```

Note the British spelling: the package is `art.defences` (not `defenses`), and ART uses `eps` / `eps_step` for the perturbation budget and step size (not `epsilon` / `alpha`).

### How ART Organizes Its Components

```
  ART Package Structure
  =====================

  art/
  +-- attacks/                    # Attack implementations
  |   +-- evasion/               # Inference-time input-manipulation attacks
  |   |                          #   (FastGradientMethod, ProjectedGradientDescent,
  |   |                          #    CarliniL2Method, SaliencyMapMethod, SimBA,
  |   |                          #    ZooAttack, HopSkipJump, AutoAttack, ...)
  |   +-- poisoning/             # Training-data manipulation attacks (backdoors)
  |   +-- extraction/            # Model replication / stealing attacks
  |   +-- inference/             # Membership inference, model inversion, attribute
  +-- defences/                   # Defence implementations (British spelling)
  |   +-- trainer/               # Training-time defences (AdversarialTrainer, ...)
  |   +-- preprocessor/          # Input-space defences (FeatureSqueezing, ...)
  |   +-- postprocessor/         # Output-space defences (ReverseSigmoid, ...)
  |   +-- detector/              # Adversarial-example detectors (BinaryInputDetector)
  |   +-- transformer/           # Model transforms (DefensiveDistillation, ...)
  +-- estimators/                # Model wrappers for different frameworks
  |   +-- classification/        # KerasClassifier, PyTorchClassifier,
  |   |                          #   TensorFlowV2Classifier, SklearnClassifier, ...
  |   +-- regression/            # Regressor wrappers
  |   +-- object_detection/      # Detector-model wrappers
  +-- metrics/                   # Robustness evaluation utilities
  +-- utils.py                   # Common helpers (clipping, distance, etc.)
```

---

## Introduction

As AI systems become integral to healthcare, autonomous vehicles, finance, and national security, the risk of malicious exploitation grows alongside their adoption. **Adversarial attacks** strategically manipulate inputs to cause incorrect predictions or decisions, ranging from imperceptible perturbations to sophisticated system-level undermining.

This workshop covers:

- Types of adversarial attacks and their impact
- Adversarial and general robustness in AI systems
- Model stability under adversarial perturbations
- Defense techniques and their implementation
- The role of key parameters (epsilon, learning rate, regularization) in attacks and defenses

---

## 1. Introduction to Adversarial Attacks

Adversarial attacks were first demonstrated by **Goodfellow et al. (2014)**, who showed that carefully crafted noise added to an image could fool a deep learning model into misclassification -- even though the perturbed image looked identical to a human observer. This revealed a fundamental vulnerability: neural networks are sensitive to subtle, targeted perturbations in input space.

### Why This Matters

```
Clean Input  -->  Model  -->  Correct Prediction
     |
     +-- Add small perturbation (invisible to human) -->  Model  -->  WRONG Prediction
```

A patch on a stop sign, noise on audio, or reworded text can cause catastrophic failures in deployed AI systems. The ethical and security implications span every domain where AI makes high-stakes decisions.

**Figure 1:** A small sticker placed on a stop sign is enough to cause an AI system to misclassify it entirely.

![Stop sign adversarial patch -- a small sticker causes misclassification](https://www.itm-p.com/wp-content/uploads/2021/03/StoppSign_evasion-1-1024x724.png)

**Figure 2:** Adversarial examples applied to traffic signs -- added noise pixels cause the model to read the wrong sign while the image remains visually unchanged to a human observer.

![Adversarial examples for traffic signs](https://www.researchgate.net/publication/369368588/figure/fig1/AS:1679281685544/Adversarial-examples-for-traffic-signs-picture-by-Chen-and-Wu-71.jpg)

### Key Takeaways

> **Section 1 Takeaways:**
>
> - Adversarial perturbations exploit the high-dimensional linearity of neural networks: small, directed changes in input space cause large output shifts.
> - The attack was first demonstrated by Goodfellow et al. (2014) and confirmed that this is a fundamental property of deep learning, not a bug in any single model.
> - Real-world impact spans autonomous vehicles (misread signs), security systems (face recognition bypass), and content moderation (adversarial text injection).

---

## 2. Types of Adversarial Attacks

Adversarial attacks fall into four principal categories, distinguished by their objective and the phase in which they strike.

### Comparison of Attack Categories

| Dimension        | Evasion Attacks                | Poisoning Attacks              | Inference Attacks              | Extraction Attacks             |
| ---------------- | ------------------------------ | ------------------------------ | ------------------------------ | ------------------------------ |
| **Attack Phase**  | Deployment (inference time)    | Training time                  | Deployment (query time)        | Deployment (query time)        |
| **Attacker Goal** | Cause misclassification of a specific input | Corrupt model learning | Recover private training data  | Replicate the target model     |
| **Data Access**   | Single query at inference      | Write access to training set   | Model output queries           | Model output queries (many)    |
| **Impact Scope**  | Individual prediction failure  | System-wide behavior change    | Privacy breach                 | Intellectual property theft    |
| **Detection Ease**| Difficult (perturbations hidden) | Moderate (anomaly detection) | Very difficult (normal outputs) | Detectable (query volume spikes)|
| **Example Use Case** | Bypass facial recognition | Corrupt fraud model training   | Determine if patient was in dataset | Steal proprietary model weights |

```
+------------------+     +----------------------+
|  Poisoning       |     |  Evasion             |
|  (Training Phase)|     |  (Deployment Phase)   |
+------------------+     +----------------------+
                            |
                    +-------+-------+
                    |                |
              White-box        Black-box

+-------------------------------------------------------------------+
|  INFERENTIAL ATTACKS (output-driven)                              |
|                                                                   |
|  Member-            Model             Extract-                    |
|  ship               Inversion           tion                      |
|  Attack             Attack            Attack                      |
|  +---------------+ +-------------+  +------------------+         |
|  | Did x_i join? | | Reconstruct |  | Build surrogate  |         |
|  | the training  | | sensitive   |  | model with       |         |
|  | set?          | | attributes  |  | identical behavior|         |
|  +---------------+ +-------------+  +------------------+         |
+-------------------------------------------------------------------+
```

### 2A. Evasion Attacks

Evasion attacks manipulate inputs at inference time to cause misclassification. The attacker modifies the input just enough that the model's decision boundary is crossed, while the change remains imperceptible or near-imperceptible to humans.

#### White-Box Evasion

The attacker has full knowledge of the model: architecture, weights, gradients, and possibly training data. This gives maximum flexibility in crafting targeted attacks against any class.

```
 Attacker
     |
     | Knows: Architecture + Weights + Gradients
     v
  [ Model Internals ]  <--- Full Access
     |
     v
  Adversarial Input --> Misclassification
```

**Key Methods** (ART classes live under `art.attacks.evasion`):

| Method | ART class | Description | Norm | Targeted? | Cost |
| ------ | --------- | ----------- | ---- | --------- | ---- |
| FGSM | `FastGradientMethod` | Single-step sign-of-gradient perturbation | L-infinity | Untargeted / targeted | Low (one forward + backward pass) |
| BIM | `BasicIterativeMethod` | Iterated FGSM without random start | L-infinity | Untargeted / targeted | Medium (T iterations) |
| PGD | `ProjectedGradientDescent` | Iterated FGSM with random start + projection back onto the epsilon-ball | L-infinity / L2 | Untargeted / targeted | Medium (T iterations) |
| C&W | `CarliniL2Method`, `CarliniLInfMethod`, `CarliniL0Method` | Min-norm optimization; no epsilon budget | L2 / L-inf / L0 | Targeted (also untargeted) | High (nested optimization) |
| JSMA | `SaliencyMapMethod` | Jacobian saliency map; changes few pixels | Sparse (L0) | Targeted | Medium |
| DeepFool | `DeepFool` | Min-norm step toward the nearest decision boundary | L2 | Untargeted | Medium |
| EAD | `ElasticNet` | Elastic-net (L1 + L2) min-norm attack | L1 / L2 | Targeted | High |
| AutoAttack | `AutoAttack` | Parameter-free ensemble; standard robustness benchmark | L-infinity / L2 | Untargeted / targeted | High |

**Figure 4:** White-box attack scenario -- the attacker has full access to model internals including architecture, weights, and gradients.

![White-box attack scenario](https://www.researchgate.net/publication/350922961/figure/fig5/AS:1015777796837376@1619191707008/White-box-attack-scenario.png)

#### Black-Box Evasion

The attacker has no knowledge of internal model parameters and can only query the model and observe outputs. Attacks in this setting rely on surrogate models (train a copy on query-response pairs), transferability phenomena, or zeroth-order optimization (estimate gradients from output differences).

```
 Attacker
     |
     | Knows: Nothing (only input/output API)
     v
  +-------------+
  |  Black Box  |  <--- Query Only
  +-------------+
     |
     v
  Adversarial Input --> Misclassification
```

Black-box attacks split by the feedback they need: **score-based** attacks read the model's output probabilities/scores, while **decision-based** attacks see only the final (hard) predicted label. ART classes below also live under `art.attacks.evasion`.

| Method | ART class | Feedback | Description | Query Complexity |
| ------ | --------- | -------- | ----------- | ---------------- |
| SimBA | `SimBA` | Score-based | "Simple Black-box Attack": gradient-free random search, adding/subtracting basis vectors (pixel or DCT) that reduce the true-class probability | Low-moderate |
| ZOO | `ZooAttack` | Score-based | "Zeroth-Order Optimization": estimates gradients from finite differences of output scores via coordinate descent | High; grouping / importance sampling reduces it |
| Square Attack | `SquareAttack` | Score-based | Query-efficient random search using localized square-shaped updates | Low-moderate |
| HopSkipJump | `HopSkipJump` | Decision-based | Estimates the gradient direction at the boundary from hard-label queries only | Moderate |
| Boundary Attack | `BoundaryAttack` | Decision-based | Random walk along the decision boundary, shrinking the perturbation toward the source image | High |

> **Section 2A Takeaways:**
>
> - White-box attacks are the strongest threat class: full gradient access enables targeted, minimal perturbations.
> - Black-box attacks come in two flavors of feedback -- *score-based* (SimBA, ZOO, Square Attack read output probabilities) and *decision-based* (HopSkipJump, Boundary Attack see only the final label) -- and they also exploit *transferability*: adversarial examples on a surrogate model often defeat the target because decision boundaries of similarly trained networks tend to align.
> - FGSM is the simplest white-box attack (one-step). PGD is its iterative generalization and is widely regarded as one of the strongest first-order attacks. Carlini & Wagner is a *min-norm* attack with no epsilon budget: it optimizes for the smallest perturbation that still misclassifies, at high computational cost, producing examples that break many defenses designed to catch FGSM/PGD.

### 2B. Poisoning Attacks

Poisoning attacks compromise the training data, causing the model to learn incorrect patterns before deployment. Unlike evasion, the attacker does not modify inputs at inference time -- instead, the damage is embedded during training and becomes active whenever a specific trigger condition is met.

```
 Training Data Pipeline
 ======================

 Normal Data  +  Malicious Data (Injected by Attacker)
        |                |
        +------v--------+
               |
           [ Training ]
               |
               v
         Backdoored Model
               |
               v
     Input with Trigger --> Predicted Label (Attacker's Choice)
```

**Key Methods:**

| Method                  | Description                                                 | Impact                    | Detection Challenge          |
| ----------------------- | ----------------------------------------------------------- | ------------------------- | ---------------------------- |
| Backdoor Poisoning      | Injects samples with a specific trigger pattern             | Model activates on trigger| Trigger may be visually subtle |
| Hidden Trigger Backdoor | Embeds covert triggers undetectable under normal inspection | Persistent backdoor       | Triggers designed to evade statistical tests |
| Label Flipping          | Changes labels of a subset of training samples              | Biased classification boundary | Statistical anomalies may be masked by noise |
| Feature Collision       | Forces clean and malicious samples to map to the same representation | Classification confusion  | Requires deep model knowledge   |

> **Section 2B Takeaways:**
>
> - Poisoning attacks shift the decision boundary at its root, making the vulnerability persistent across all future predictions.
> - Backdoor attacks are especially dangerous in practice: the model behaves normally on clean inputs and only activates when a specific trigger is present.
> - Detection requires either access to training data (for sanitization) or monitoring of training loss distributions for anomalies.

### 2C. Inference Attacks

Inference attacks exploit model outputs to recover sensitive information about the training data. These are privacy violations: the attacker never touches the model's parameters or training data but can still extract confidential information from prediction confidence scores.

```
 Model Output (confidence scores, predictions)
         |
         v
       Attacker Analysis
         |
    +----+----+
    |           |
 Membership   Model
 Inversion    Extraction
```

| Method               | Description                                                      | Privacy Impact          | Data Required             |
| -------------------- | ---------------------------------------------------------------- | ----------------------- | ------------------------- |
| Membership Inference | Determines whether a specific data point was in the training set | Reveals private inclusion | Query outputs (confidence scores) |
| Model Inversion      | Reconstructs sensitive training data from model outputs          | Full profile recovery   | Many queries, known class |

> **Section 2C Takeaways:**
>
> - Membership inference works because overfitted models behave differently on training vs. non-training inputs: they are more confident about seen examples.
> - Model inversion is most effective against classification models that output probability distributions (not just argmax labels).
> - Defenses include limiting output granularity (binary decisions only), differential privacy during training, and calibrated confidence scores.

### 2D. Extraction Attacks

Extraction attacks attempt to replicate the target model by querying it repeatedly and building a surrogate with similar behavior -- effectively stealing the model. This is an intellectual property attack: in commercial ML settings, the model weights are the product.

```
 Target Model          Attacker's Surrogate
 +-----------+    Query   +-----------+
 |           | -------->  |           |
 |  Unknown  | <--------  |  Copy     |
 |  Internals|   Output   |  Model    |
 +-----------+            +-----------+
```

> **Section 2D Takeaways:**
>
> - Extraction attacks are feasible whenever the target API does not limit query rates or output detail.
> - The surrogate model does not need to match weights exactly; it only needs to produce similar predictions on new inputs.
> - Defenses include query rate limiting, output perturbation (adding noise to confidence scores), and detecting anomalous query patterns.

---

## 3. Threat Models

A threat model defines the attacker's knowledge, access, and capabilities. It determines which attacks are feasible and which defenses are appropriate. Without a threat model, defense selection is arbitrary -- you might build expensive protections against attacks that are impossible in your deployment context while leaving realistic attack vectors wide open.

### 3A. Attacker Knowledge Spectrum

```
 Knowledge / Access
   ▲
   │
Full│      WHITE-BOX
   │     +----------------------------------------+
   │     | Architecture: Known                      |
   │     | Weights:    Known                        |
   │     | Gradients:  Available                    |
   │     | Training Data: Known                     |
Partial│    +------------------------------------+
   │     |         GRAY-BOX                       |
   │     | Architecture: Partial                    |
   │     | Weights:    Unknown                      |
   │     | Gradients:  Limited                      |
   │     | Training Data: Partial                   |
   │     +------------------------------------+
None│    |         BLACK-BOX                      |
   │     | Architecture: Unknown                  |
   │     | Weights:    Unknown                    |
   │     | Gradients:  None                       |
   │     | API Queries Only                       |
   └─────+----------------------------------------+
           Defence Requirement ▲
```

### 3B. Threat Model Comparison

| Threat Model | Attacker Access                              | Example Attack               | Appropriate Defense                                     | Feasibility in Practice | Cost to Deploy        |
| ------------ | -------------------------------------------- | ---------------------------- | ------------------------------------------------------- | ----------------------- | --------------------- |
| White-box    | Full (architecture, weights, gradients)      | FGSM, PGD, C&W               | Adversarial Training                                    | Rare (model leakage needed) | High (retraining required) |
| Gray-box     | Partial (architecture known, weights hidden) | Data Poisoning               | Data Sanitization, Robust Optimization                  | Moderate                | Medium                |
| Black-box    | None (query-only API access)                 | SimBA, ZOO, HopSkipJump, transfer attacks | Query Limitation, Ensemble Methods, Input Preprocessing | Common (APIs are public) | Variable            |

### 3C. Trade-off: Attack Complexity vs Real-World Applicability

```
 Effectiveness in Lab
   ▲
   │
High│    White-Box Attacks
   │   High effectiveness, low realism
   │
   │
   │         Gray-Box Attacks
   │
   │
   │                         Black-Box Attacks
   │                        Low effectiveness (avg),
   │                        high realism
   │
Low└──────────────────────────────────────────────► Real-World Applicability
  Low                                              High
```

Defenses must be selected to match the threat model that reflects the actual deployment environment. A defense strong against white-box attacks may be unnecessary overhead if only black-box access is realistic. Conversely, underestimating attacker knowledge leaves critical gaps.

> **Section 3 Takeaways:**
>
> - Threat models are not abstract exercises: they determine the cost-benefit analysis of every defense.
> - In practice, most deployed systems face black-box threats (public APIs), but white-box threats cannot be dismissed entirely (model theft is common in academic and industrial settings).
> - Gray-box is the hardest to defend against because partial knowledge reduces the cost of crafting targeted attacks without requiring full model access.

---

## 4. Model Robustness

Model robustness is the ability to maintain correct behavior under uncertainty -- noise, distribution shift, and adversarial manipulation. Understanding this concept requires distinguishing between resilience to natural perturbations and resistance to intentional attacks.

### 4A. General Robustness vs Adversarial Robustness

```
 Input Space
 ===========

        Natural Noise          Adversarial Perturbation
        (Random)               (Targeted)

      ~~~~~~~~~~~~~~         >>>>>> direction of gradient
     ~  ~  ~  ~  ~          >>>>>> maximizes loss
    ~  ~  x  ~  ~           >>>>>>
     ~  ~  ~  ~  ~
      ~~~~~~~~~~~~~~

 General Robustness         Adversarial Robustness
 handles random noise       handles targeted attacks
```

- **General Robustness**: Performance under natural perturbations -- sensor noise, compression artifacts, environmental variations. Measured by testing on noisy or augmented data distributions.
- **Adversarial Robustness**: Performance under intentional, optimized perturbations designed to maximize model error. Measured by attack success rate against standardized benchmarks (e.g., AutoAttack).

**Comparison:**

| Dimension        | General Robustness                 | Adversarial Robustness              |
| ---------------- | ---------------------------------- | ----------------------------------- |
| **Source of perturbation** | Nature (sensor, environment) | Attacker (optimized gradient)     |
| **Perturbation structure** | Random, unstructured         | Directed along loss gradient       |
| **Detection method**   | Accuracy on noisy test set    | Attack success rate (ASR)          |
| **Typical defense**    | Data augmentation, noise injection | Adversarial training, distillation |
| **Real-world relevance** | Everyday deployment conditions | Malicious environments           |

### 4B. The Accuracy-Robustness Trade-off

```
 Accuracy (Clean Data)
   ▲
   │
High│    Standard Training
   │   (High accuracy, low robustness)
   │
   │
   │         *
   │        *  Adversarial Training
   │       *  (Balanced accuracy and robustness)
   │      *
   │     *
   │    *
Low │   *++++++++++++
   └──────────────────────────────────────► Adversarial Robustness
     Low                  High
```

Improving adversarial robustness typically reduces clean-data accuracy. Adversarial training shifts decision boundaries to resist attack directions, which can move them away from the optimal boundaries for clean data. Finding the operating point on this curve is a central challenge in robust ML.

### 4C. Robustness vs Model Complexity

```
 Computational Cost
   ▲
   │
High│         Ensemble Methods
   │        (Multiple models, high cost)
   │
   │
   │     Adversarial Training
   │     (Iterative perturbation gen.)
   │
   │
   │  Standard Training
Low│ (Single pass, low cost)
   └──────────────────────────────────────► Robustness Level
     Low                      High
```

Robust techniques -- adversarial training, ensembles, defensive distillation -- increase computational cost during both training and inference while often reducing model interpretability. The trade-off between transparency and robustness is particularly critical in regulated domains such as healthcare and finance.

> **Section 4 Takeaways:**
>
> - General robustness and adversarial robustness address fundamentally different problems: one handles randomness, the other handles purposeful optimization against your model.
> - The accuracy-robustness trade-off is not yet fully understood -- there is no free lunch. Better defenses shift the curve rather than eliminate it.
> - Computational cost scales with defense sophistication: ensemble methods are expensive at both training and inference time, while adversarial training requires regenerating the training set iteratively.

---

## 5. Key Parameters in Attacks and Defenses

### 5A. Epsilon (eps) -- Perturbation Budget

Epsilon defines the maximum allowed perturbation magnitude. It constrains the attack to a bounded region around the original input, ensuring perturbations remain imperceptible while still being effective. A larger epsilon permits stronger attacks but also increases the risk of detection by human observers or automated sanitizers.

#### FGSM: Fast Gradient Sign Method

**Figure 5:** The classic FGSM demonstration -- a panda image perturbed with FGSM noise is misclassified as a gibbon. The formula shown illustrates the single-step gradient attack: `x_adv = x + eps * sign(grad_x[Loss])`.

![FGSM panda misclassified as gibbon](https://pytorch.org/tutorials/_images/fgsm_panda_image.png)

**Formula:**

```
x_adv = x + eps * sign(grad_x [Loss(model, x, y)])
```

**Step-by-step breakdown:**

```
 Step 1: Forward Pass
 ============================================
 Input x passes through the model.

    x --> [ Model ] --> Prediction y_hat
                  |
                  v
             Loss(L(x, y))


 Step 2: Compute Gradient
 ============================================
 Backpropagate loss w.r.t. the input.

    grad_x = d(L) / d(x)

    The gradient points in the direction
    that maximally increases the loss.


 Step 3: Take the Sign
 ============================================
 Keep only the direction, discard magnitude.

    sign(grad_x) --> vector of +1, 0, -1

    This creates uniform perturbation
    across all dimensions.


 Step 4: Scale by Epsilon and Apply
 ============================================

    x_adv = x + eps * sign(grad_x)

    Original      Perturbation       Adversarial
    +----------+     +------------+     +---------+
    |          |    +|            |     |         |
    |    x     |    +|    delta   |  =  |  x_adv  |
    |          |    +|            |     |         |
    +----------+     +------------+     +---------+

    The perturbation is bounded: ||delta||_inf <= eps
```

#### PGD: Projected Gradient Descent

**Figure 6:** PGD projects the adversarial example back onto the epsilon ball after each gradient step, ensuring the perturbation stays within bounds.

![PGD projection of adversarial examples](https://www.researchgate.net/publication/371808403/figure/fig4/AS:11431281200450802@1697939895840/The-Projected-Gradient-Descent-Attack-PGD-projects-the-adversarial-example-back-onto-a.png)

**Formula:**

```
x_0 = x + uniform_random_perturbation([-eps, +eps])

For t = 1 to T:
    x_t = x_{t-1} + alpha * sign(grad_x [Loss(model, x_{t-1}, y)])
    x_t = clip(x_t, x - eps, x + eps)    // Project onto L_infinity ball

x_adv = x_T
```

**Step-by-step breakdown:**

```
 Step 1: Random Initialization
 ============================================
 Start from a random point within the epsilon ball.

          eps ball around x
         +-------------+
        / |             |\
       |   o   x    o  |   <- x_0 chosen randomly here
        \ |             |/
         +-------------+

    This prevents the attack from getting
    stuck at poor local maxima.


 Step 2: Iterative Gradient Ascent
 ============================================
 Take multiple small steps in the gradient direction.

    alpha (step size) < eps (total budget)

    x_1 = x_0 + alpha * sign(grad_x[Loss])
    x_2 = x_1 + alpha * sign(grad_x[Loss])
    ...
    x_T = x_{T-1} + alpha * sign(grad_x[Loss])

    Direction of attack:

    x  .  .  .  .  .  .  x_T (x_adv)
       <--alpha--alpha--alpha-->


 Step 3: Projection (Clipping)
 ============================================
 After each step, project back onto the epsilon ball.

    x_t = clip(x_t, x - eps, x + eps)

    Without projection:         With projection:
    Perturbation may            Perturbation stays
    grow unbounded              within bounds

         Unbounded               Bounded (eps ball)
        +----------------+      +-------------+
       / |      x_T     |\    / |    x   | \
      | o           o  | |   | o   o   o |
     /  |         escape \|  \ | (stays) /  /
      +-----------------+   \ +-------------+
                             \

    This ensures: ||x_adv - x||_inf <= eps
```

#### Epsilon in Poisoning Attacks

**Figure 7:** The effect of varying epsilon on adversarial perturbations -- as epsilon increases, the noise grows more visible and the perturbation becomes stronger but less stealthy.

![Effect of epsilon on noise magnitude](https://media.springernature.com/lw685/springer-static/image/chp%3A10.1007%2F978-3-030-87664-7_7/MediaObjects/503908_1_En_7_Fig3_HTML.png)

In poisoning, epsilon controls the magnitude of modifications injected into training data.

```
 Training Data Point Modification
 ================================

 Original point:     x_train
 Poisoned point:     x_poisoned = x_train + delta
                      where ||delta|| <= eps

 eps small:    eps large:

   . . .           . . .
  . . o .         . * * .
   . . .    vs     . * * .
                   . . .
  Subtle,          Obvious,
  hard to detect   easy to sanitize
```

- **Small epsilon**: Subtle modifications that evade detection but may have weaker effect on the model.
- **Large epsilon**: Stronger influence on decision boundaries but more likely flagged by data sanitization.

#### Epsilon Selection Trade-off

```
 Attack Success Rate
   ▲
   │
High│
   │
   │               Increasing eps -->
   │              /
   │             /
   │            /
   │           /
   │          /___________ (may saturate or become unrealistic)
   │
Low│
   └──────────────────────────────────────► Epsilon (eps)
     Small                 Optimal         Too Large
```

### 5B. Learning Rate (eta)

The learning rate governs the step size in optimization, affecting both training stability and attack strength. In adversarial training, it controls how quickly model weights adapt to adversarial examples. During attacks like PGD, the analogous parameter is `alpha` (per-iteration step size).

```
 Training Dynamics
 =================

 Learning Rate Effect on Adversarial Training:

 eta too high:        eta optimal:         eta too low:

   Loss               Loss                Loss
    ▲                  ▲                   ▲
    |\                 | \                 |  \
    | \ oscillate      |  \ smooth         |   \ slow
    |  |/__/\          |   \              |    \
    +---+----->        +---+----->         +---+-----> epochs

 Unstable            Converges           Very slow
 may overshoot       robustly            convergence
```

| Context                   | High Learning Rate                                         | Low Learning Rate                                 |
| ------------------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| Adversarial Training      | Fast convergence, unstable, may overshoot optimal solution | Stable training, more epochs required             |
| Adversarial Attacks (PGD) | Aggressive perturbation steps, stronger attacks            | Subtle perturbations, harder to detect but weaker |

> **Section 5B Takeaways:**
>
> - The learning rate in adversarial training is critical: too high and training diverges; too low and robustness improvements stall.
> - In attack contexts, `alpha` (the per-iteration step size) and `eps` (the total perturbation budget) serve different roles: alpha controls speed of attack convergence, eps controls the stealth threshold.

### 5C. Batch Size

```
 Gradient Estimate Quality
 =========================

 Small Batch:          Optimal Batch:         Large Batch:

      /\                /\                   ____
     /  \__/\          /    \               /      \__
    /        \__/\    /      \             /
   /            \  \/        \           /
  /              \            \         /

 Noisy gradient   Balanced     Stable but slow
 Fast updates     update freq. high memory cost
```

| Context              | Small Batch                       | Large Batch                              |
| -------------------- | --------------------------------- | ---------------------------------------- |
| Adversarial Training | Noisier gradients, faster updates | Stable gradients, higher memory overhead |
| Adversarial Attacks  | Fewer examples per query batch    | More efficient generation at scale       |

### 5D. Noise Level in Training

```
 Model Performance
   ▲
   │
High│    No Noise
   │   (Overfits to clean data)
   │
   │
   │       * Optimal Noise Level
   │      ***  (Best generalization + robustness)
   │     *****
   │    *     *
   │   *       *
   │  *         *
Low │ *           *++++++++++++++
   └──────────────────────────────────────► Noise Level
     None              Too Much
```

- **Too little noise**: the model overfits to clean data and generalizes poorly.
- **Too much noise**: the model cannot learn meaningful patterns.
- **Optimal noise**: improves both generalization and robustness without degrading core performance. This principle underlies techniques such as label smoothing, mixup, and stochastic depth -- all of which inject controlled randomness during training.

### 5E. Regularization

Regularization constrains model capacity to reduce sensitivity to adversarial examples and poisoned data. Without regularization, neural networks can memorize training data with highly complex decision boundaries that are extremely fragile to small input perturbations.

```
 Without Regularization:      With L2 Regularization:
 =========================    =======================

 Decision boundary            Decision boundary
 (Complex, overfits)          (Smoother, more general)

   +  -  +  -                 +  -  +  -
  / \__/ \__/                /    /    \
 /        \__               /    /      \
/            \__           /____/        \

 Vulnerable to              More resistant to
 small perturbations        perturbations

 Penalty term:
    L_total = L_data + lambda * ||w||^2

 lambda (regularization strength):

 lambda small:   lambda optimal:   lambda large:

 Under-regularized  Balanced         Over-regularized
 (still fragile)    robustness       (underfits)
```

### 5F. Transferability of Attacks

Adversarial examples generated on one model often transfer to another model with similar architecture -- this is the foundation of black-box attacks. The phenomenon occurs because models trained on the same task with similar architectures tend to develop aligned decision boundaries in input space.

```
 Surrogate Model          Target Model
 +---------------+        +---------------+
 | Attacker's    |        | Unknown       |
 | Copy / Proxy  |        | Internals     |
 +---------------+        +---------------+
         |                         ^
         | Generates adversarial   | Same adversarial
         | example on surrogate    | example works here
         └─────────────────────────┘

 Transferability depends on:
 - Architectural similarity (more similar = higher transfer)
 - Training data overlap (more overlap = higher transfer)
 - Model capacity and depth (deeper models tend to transfer better)
 - Input representation (normalized inputs improve transfer)
```

**Defenses against transfer:**

- Ensemble models (diverse architectures reduce common vulnerability surface)
- Feature squeezing (reduce input precision, removing subtle perturbations)
- Defensive distillation (smooth output probabilities, reducing gradient information)

> **Section 5 Takeaways:**
>
> - Epsilon is the most critical parameter in any evasion attack: it directly controls the stealth-strength trade-off.
> - Learning rate must be carefully balanced in adversarial training -- it governs how quickly robustness is learned versus how stable the optimization remains.
> - Batch size affects gradient quality in adversarial training but has minimal direct impact on attacks.
> - Regularization smooths decision boundaries, which indirectly reduces adversarial susceptibility by removing the fine-grained structures that perturbations exploit.
> - Transferability enables black-box attacks: you do not need to know the target model if you can build a good surrogate or exploit the alignment of similarly-trained networks.

---

## 6. Defense Mechanisms

### 6A. Defense Classification

```
 Defence Lifecycle
 =================

 Training Phase          Inference Phase
 +----------------+      +------------------+
 | Adversarial    |      | Detection        |
 | Training       |      | Module           |
 +----------------+      +------------------+
 | Defensive      |            |
 | Distillation   |            v
 +----------------+      +------------------+
 | Data           |      | Postprocessing   |
 | Sanitization   |      | (Ensemble,       |
 +----------------+      |  Consensus)      |
                         +------------------+

 Transformation Phase
 +------------------+
 | Feature          |
 | Squeezing        |
 +------------------+
 | Input            |
 | Preprocessing    |
 +------------------+
```

### 6B. Defense Comparison

| Defense Type           | Mechanism                                           | Phase                | Targets                             | Cost (Training) | Cost (Inference) | Effectiveness Against       |
| ---------------------- | --------------------------------------------------- | -------------------- | ----------------------------------- | --------------- | ---------------- | ------------------------- |
| Adversarial Training   | Train on generated adversarial examples             | Training             | Evasion (white-box)                 | High            | None             | FGSM, PGD, C&W; moderate against unknown attacks |
| Defensive Distillation | Soften output probabilities via temperature scaling | Training + Inference | Evasion, gradient-based attacks     | Medium          | Slightly higher  | Moderate; can be bypassed with adaptive attacks |
| Data Sanitization      | Detect and remove poisoned training samples         | Training             | Poisoning                           | Low-Medium      | None             | Effective if poisoning ratio is low (<5%) |
| Adversarial Detection  | Secondary classifier flags adversarial inputs       | Inference            | Evasion                             | Low             | Moderate         | Varies; depends on detector architecture |
| Ensemble Methods       | Aggregate predictions across diverse models         | Inference            | Transfer attacks, black-box evasion | None            | High (N models)  | Reduces transferability; does not stop direct attacks |
| Feature Squeezing      | Reduce input precision before classification        | Preprocessing        | All perturbation-based attacks      | None            | Low              | Strong against small-eps attacks; weak against large eps |

**Where these live in ART** (British spelling `art.defences`):

| Defense | ART class |
| ------- | --------- |
| Adversarial Training | `art.defences.trainer.AdversarialTrainer` (also `AdversarialTrainerMadryPGD`) |
| Defensive Distillation | `art.defences.transformer.evasion.DefensiveDistillation` |
| Feature Squeezing | `art.defences.preprocessor.FeatureSqueezing` (also `SpatialSmoothing`, `JpegCompression`, `TotalVarMin`) |
| Adversarial Detection | `art.defences.detector.evasion.BinaryInputDetector` |
| Output smoothing | `art.defences.postprocessor.ReverseSigmoid` (also `GaussianNoise`, `HighConfidence`) |

There is no `art.defenses.Trainer`; adversarial training is constructed as `AdversarialTrainer(classifier=..., attacks=...)` and then `.fit(...)`, as shown in the [Minimal ART Workflow](#minimal-art-workflow-real-api) above.

### 6C. The Arms Race

```
 Time -->
 =======

 Attacker:   A1 ------ A2 ------------ A3 --->
                  \    |              /
                   \   | Stronger   /
                    v  v          v
 Defender:   D1 ---- D2 ---------- D3 --->
                    ^  ^          ^
                    |  |----------|
               Patch gap       Patch gap

 Each defense is broken. Each attack is patched.
 No single defense is permanent.
```

**Key Challenges:**

1. **No universal defense**: Each attack exploits a different vulnerability; no single mechanism covers all threat vectors. Defenses must be composed in layers (similar to cybersecurity).
2. **Robustness-accuracy trade-off**: Stronger defenses often reduce performance on clean data or increase computational cost. The operating point depends on the application's risk tolerance.
3. **Evolving threat landscape**: Defenses are reactive. New attacks emerge faster than comprehensive defenses can be deployed. Adaptive and transferable attacks (e.g., AutoAttack, meta-gradient methods) routinely break defenses that took months to design.

**Defense-by-defense attack resistance:**

| Defense / Attack   | FGSM  | PGD   | C&W   | Poisoning | Transfer | Black-Box |
| ------------------ | ----- | ----- | ----- | --------- | -------- | --------- |
| Adversarial Training    | High    | High    | Medium   | No        | Low      | Medium    |
| Defensive Distillation  | Medium  | Low     | Low      | No        | Medium   | Medium    |
| Data Sanitization       | No      | No      | No       | High      | No       | No        |
| Adversarial Detection   | Medium  | Medium  | Low      | No        | Low      | Medium    |
| Ensemble Methods          | Low     | Low     | Low      | No        | High     | Medium    |
| Feature Squeezing         | High    | Medium  | Low      | No        | Medium   | High      |

> **Section 6 Takeaways:**
>
> - Defense-in-depth is the only viable strategy: combine training-time, inference-time, and preprocessing defenses.
> - No single defense is sufficient against all attack types; each has a distinct weakness profile.
> - Adversarial training remains the most robust general-purpose defense, but it requires retraining from scratch with adversarial examples -- a cost that may be prohibitive in production settings.
> - Feature squeezing and ensemble methods are lightweight inference-only defenses that complement heavier training-based approaches.

---

## 7. Summary of Parameters and Their Roles

```
 Parameter Space in Adversarial ML
 =================================

 +----------+------------+------------------------------------+
 | Param    | Symbol     | Role                                |
 +----------+------------+------------------------------------+
 | Epsilon  | eps        | Max perturbation budget (L_inf norm)|
 +----------+------------+------------------------------------+
 | Step Sz  | alpha      | Per-iteration step in PGD           |
 +----------+------------+------------------------------------+
 | LR       | eta        | Optimization step size              |
 +----------+------------+------------------------------------+
 | Iter     | T          | Number of PGD steps                 |
 +----------+------------+------------------------------------+
 | Batch    | B          | Samples per gradient update         |
 +----------+------------+------------------------------------+
 | Reg      | lambda     | Regularization strength             |
 +----------+------------+------------------------------------+
 | Temp     | tau        | Temperature in distillation         |
 +----------+------------+------------------------------------+
```

Each parameter requires careful tuning. The interaction between them is non-trivial -- changing one often necessitates adjusting others to maintain the desired balance between robustness, accuracy, and computational feasibility. For example, increasing epsilon may require reducing the learning rate during adversarial training to compensate for steeper gradients; increasing batch size may allow higher regularization without degrading gradient quality.

### Quick Reference: Typical Parameter Ranges

| Parameter        | Typical Range       | Effect of Increasing |
| ---------------- | ------------------- | -------------------- |
| Epsilon (eps)    | 0.01 - 0.3 (image)  | Stronger attack, less stealthy |
| Step size (alpha)| 0.01 - 0.05         | Faster attack convergence, risk of overshooting |
| Iterations (T)   | 10 - 100            | Stronger attack up to saturation point |
| Learning rate    | 0.001 - 0.1         | Faster vs. stable training trade-off |
| Batch size       | 32 - 512            | Stable gradients at memory cost |
| Lambda (reg)     | 0.0001 - 0.1        | More regularization, risk of underfitting |
| Temperature (tau)| 1.0 - 10.0          | Smoother probabilities at info loss |

---

## References and Further Reading

### Foundational Papers

| # | Citation | Topic | Year |
|---|----------|-------|------|
| 1 | Szegedy et al., "Intriguing properties of neural networks" ([1312.6199](https://arxiv.org/abs/1312.6199)) | First adversarial example discovery | 2014 (ICLR) |
| 2 | Goodfellow, Shlens & Szegedy, "Explaining and Harnessing Adversarial Examples" ([1412.6572](https://arxiv.org/abs/1412.6572)) | FGSM; linear-approximation interpretation | 2015 (ICLR) |
| 3 | Kurakin et al., "Adversarial examples in the physical world" ([1607.02533](https://arxiv.org/abs/1607.02533)) | Real-world / physical adversarial examples | 2017 (ICLR Workshop) |
| 4 | Papernot et al., "The Limitations of Deep Learning in Adversarial Settings" ([1511.07528](https://arxiv.org/abs/1511.07528)) | JSMA; Jacobian saliency-map attack | 2016 (IEEE EuroS&P) |
| 5 | Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" ([1608.04644](https://arxiv.org/abs/1608.04644)) | C&W min-norm attack (L0/L2/L-inf); defense-evaluation critique | 2017 (IEEE S&P) |
| 6 | Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" ([1706.06083](https://arxiv.org/abs/1706.06083)) | PGD; adversarial training as min-max optimization | 2018 (ICLR) |
| 7 | Athalye, Carlini & Wagner, "Obfuscated Gradients Give a False Sense of Security" ([1802.00420](https://arxiv.org/abs/1802.00420)) | Why gradient masking fails; breaking defenses | 2018 (ICML) |

### ART and Tooling

| # | Resource | Description |
|---|----------|-------------|
| 8 | Nicolae et al., "Adversarial Robustness Toolbox v1.0.0" ([1807.01069](https://arxiv.org/abs/1807.01069)) | ART framework overview paper (2018) |
| 9 | [ART Documentation](https://adversarial-robustness-toolbox.readthedocs.io/) | Official API reference and tutorials |
| 10 | [ART GitHub Repository](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | Source code, examples, community |

### Advanced Topics

| # | Citation | Topic |
|---|----------|-------|
| 11 | Croce & Hein, "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks" ([2003.01690](https://arxiv.org/abs/2003.01690)) | AutoAttack benchmark suite; 2020 (ICML) |
| 12 | Cohen, Rosenfeld & Kolter, "Certified Adversarial Robustness via Randomized Smoothing" ([1902.02918](https://arxiv.org/abs/1902.02918)) | Certifiable robustness guarantees; 2019 (ICML) |
| 13 | Papernot et al., "Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks" ([1511.04508](https://arxiv.org/abs/1511.04508)) | Defensive distillation; 2016 (IEEE S&P) |

### Books and Tutorials

- **Goodfellow, Bengio, Courville** -- *Deep Learning*, Chapter 7 (Regularization), Section 7.13 "Adversarial Training" (MIT Press, 2016)
- **ART example notebooks** -- runnable tutorials in the repository's [`notebooks/` directory](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/notebooks)

> **Final Thought:** Adversarial robustness is not a destination -- it is an ongoing process. As new attacks emerge and models evolve, the defender's job is to maintain vigilance, continuously evaluate against state-of-the-art attacks, and adapt defenses accordingly.

*Workshop prepared for the SecureAI educational series.*
