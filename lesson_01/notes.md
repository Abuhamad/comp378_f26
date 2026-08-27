# 01: Introduction to Artificial Intelligence and Modern Paradigms


## Abstract

AI has passed through four historical eras, each ending when a class of problems exceeded the prior generation of methods. This lesson maps those eras, contrasts the three active paradigms, introduces a four-pillar trustworthiness taxonomy, and orients students to the four libraries used throughout this course. Students who complete this lesson have the vocabulary and working environment required for each subsequent module.

## Objectives

- Place a named AI technique or system within its correct historical era (symbolic, connectionist, statistical, foundation-model) and explain the placement.
- Articulate what distinguishes symbolic from sub-symbolic from foundation-model paradigms, giving one concrete capability difference for each transition.
- Map the four trustworthiness pillars (security, privacy, fairness, governance) to the corresponding course sections and give one real-world example of a failure in each pillar.
- Configure and verify the course toolchain (PyTorch, scikit-learn, `transformers`, ART) in a local Python environment and run a baseline tabular classifier end-to-end.

## Content

Two failures anchor the full arc of this lesson. In 1984, Digital Equipment Corporation's XCON expert system required six engineers to maintain 17,500 configuration rules, and the system failed on order variants outside those rules. In 2023, a New York law firm filed a brief in Mata v. Avianca citing six cases ChatGPT had fabricated. Both failures share a structure: a system exceeded its training or rule base and produced wrong outputs without warning. Understanding why requires tracing how AI methods have changed across four eras.

### History of AI

AI research began at the 1956 Dartmouth Summer Research Project, where John McCarthy, Marvin Minsky, and colleagues proposed a summer-long study of machine intelligence. The field has passed through four eras since then.

#### Era 1: Symbolic AI (1956 to approximately 1985)

Early programs operated on explicit symbolic rules. The Logic Theorist (1956) proved 38 of 52 theorems from Principia Mathematica. The General Problem Solver (1957) applied means-ends analysis to abstract problem structures. Expert systems extended this approach to practical domains. MYCIN (1976) encoded bacterial infection knowledge as 600 rules and achieved 69% diagnostic accuracy on bacteremia cases, matching human specialist performance. The first AI winter began around 1974 when funding bodies found general reasoning progress slower than predicted. A second wave of expert systems collapsed by the late 1980s as maintenance costs grew quadratically with rule count.

#### Era 2: Connectionist AI (1958 to approximately 1995)

Rosenblatt's Perceptron (1958) learned a binary classification boundary from labeled data. Minsky and Papert's 1969 analysis showed a single-layer perceptron cannot compute XOR, and research funding contracted. Rumelhart, Hinton, and Williams published the backpropagation algorithm in 1986, enabling multi-layer networks to learn nonlinear functions. Interest fell again by the early 1990s as training costs and dataset sizes limited practical deployment.

#### Era 3: Statistical Learning (approximately 1995 to 2012)

Cortes and Vapnik introduced support vector machines in 1995, with generalization guarantees rooted in margin theory. Breiman proposed random forests in 2001, combining bagging and feature randomness to reduce variance. These methods achieved reproducible benchmark improvements using hand-crafted features and datasets small enough to fit in memory. The transition from connectionist methods came because statistical approaches offered measurable, reproducible gains on standard tasks.

#### Era 4: Foundation Models (2012 to present)

Krizhevsky, Sutskever, and Hinton's AlexNet reduced ImageNet top-5 error from 26.2% to 15.3% in 2012, a 10.9 percentage-point drop no prior statistical method had matched. The transformer architecture (Vaswani et al., 2017) enabled training on internet-scale text. GPT-3 (2020, 175 billion parameters) demonstrated few-shot text completion, code generation, and arithmetic without task-specific training. CLIP (2021) linked images and text in a shared embedding space, enabling zero-shot image classification across 30 benchmark datasets. Each transition came when the prior era's best model could not close the remaining error on a new class of problems.

### Paradigm Comparison

Three paradigms coexist in deployed AI systems today. None has made the others obsolete.

#### Symbolic Paradigm

A symbolic system represents knowledge as explicit rules, predicates, or logical statements. The programmer encodes domain knowledge. The system applies inference rules to reach conclusions. MYCIN's rule "IF the organism is gram-negative AND the patient is immunocompromised THEN suspect Pseudomonas" is readable and auditable by a clinician. The capability boundary: symbolic systems fail when input falls outside the rule base. XCON required six full-time engineers to maintain 17,500 rules, and new product configurations still escaped coverage.

#### Sub-symbolic Paradigm

A sub-symbolic system learns distributed representations from labeled data. No human writes the features. AlexNet learned 60 million parameters from 1.2 million labeled ImageNet images. The capability gain over symbolic: the system generalizes to images outside the training set. The cost: the internal representation is not readable by inspection.

#### Foundation-Model Paradigm

A foundation model trains once on a large, unlabeled corpus, then adapts to many downstream tasks by prompting or fine-tuning. GPT-3 answered questions in domains not present in its prompt without any weight update. The capability gain over sub-symbolic: the model generalizes across tasks, not only across inputs within one task. The cost: the model requires billions of parameters and terabytes of training data, and its failure modes include confident fabrication, as in Mata v. Avianca.

The three paradigms differ on three dimensions:

- Representation: explicit rules vs. learned weights vs. learned weights with emergent structure.
- Generalization: within-rule-base vs. within-distribution vs. across-task.
- Interpretability: auditable vs. opaque vs. opaque with a natural-language interface.

### Trustworthiness Taxonomy

AI trustworthiness decomposes into four orthogonal pillars. Each has a distinct failure mode, a distinct defense class, and a designated course section.

#### Security (modules m07 to m23)

A security failure occurs when an adversary intentionally causes the model to behave incorrectly. In 2017, researchers showed adding a human-imperceptible patch to a stop sign caused a production object-detection model to classify the sign as a speed limit sign in 100% of tested cases. The defense class includes adversarial training, input certification, and architectural hardening.

#### Privacy (modules m24 to m31)

A privacy failure occurs when a model reveals information about its training data beyond what the model owner intended. In 2019, Carlini et al. extracted verbatim training sequences from GPT-2, including a person's full name, home address, and phone number, by querying the model 600,000 times. The defense class includes differential privacy, federated learning, and encrypted computation.

#### Fairness (modules m32 to m33)

A fairness failure occurs when a model produces outcomes disadvantaging a group in a way not warranted by the task. ProPublica's 2016 analysis found COMPAS classified Black defendants as high-risk at a rate 44.9 percentage points higher than white defendants among defendants who did not reoffend. The defense class includes pre-processing bias correction, in-processing constraints, and post-processing threshold adjustment.

#### Governance (modules m34 to m35)

A governance failure occurs when no accountability structure catches or corrects errors before they cause harm. The EU AI Act (2024) mandates a conformity-assessment process for all high-risk AI systems. The defense class includes model cards, audit trails, and regulatory compliance frameworks.

### Toolchain Orientation

This course uses four libraries. Each has a distinct role, and all four are required starting with the lab below.

#### scikit-learn

scikit-learn provides classical machine learning models, including logistic regression, decision trees, support vector machines, and random forests. The `fit` / `predict` API is consistent across all estimator classes. The minimum version for this course is 1.5.

#### PyTorch

PyTorch provides automatic differentiation and GPU-accelerated tensor computation for deep learning. Neural network modules, optimizers, and data loaders follow a class-based API. The minimum version is 2.3.

#### Hugging Face transformers

The `transformers` library provides pre-trained foundation models, including BERT, GPT-2, and LLaMA variants. Models load with two calls: `AutoTokenizer.from_pretrained(name)` and `AutoModel.from_pretrained(name)`. The minimum version is 4.40.

#### IBM Adversarial Robustness Toolbox (ART)

ART wraps scikit-learn and PyTorch models to expose attack and defense objects. Attacks are callable objects: they take a batch of inputs and return perturbed inputs. The minimum version is 1.17.


## Summary

AI evolved through four historical eras (symbolic, connectionist, statistical, and foundation-model), with each transition driven by a measurable failure in the prior era's methods. Three paradigms remain active today, distinguished by how they represent knowledge, how broadly they generalize, and how interpretable their outputs are. Four trustworthiness pillars, security, privacy, fairness, and governance, organize the threats and defenses covered in modules m07 through m35. A working Python environment with scikit-learn, PyTorch, `transformers`, and ART is the prerequisite for each lab in this course.

## Useful References and Resources

- Minsky, M., and Papert, S. (1969). Perceptrons. MIT Press. Chapter 1 presents the XOR limitation of single-layer networks.
- Rumelhart, D., Hinton, G., and Williams, R. (1986). "Learning representations by back-propagating errors." Nature, 323, 533–536.
- Krizhevsky, A., Sutskever, I., and Hinton, G. (2012). "ImageNet Classification with Deep Convolutional Neural Networks." NeurIPS 2012. Reports the 26.2% to 15.3% top-5 error reduction.
- Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017. Original transformer architecture paper.
- Brown, T. et al. (2020). "Language Models are Few-Shot Learners." NeurIPS 2020. Introduces GPT-3 and documents few-shot performance across 42 tasks.
- Carlini, N. et al. (2019). "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks." USENIX Security 2019. Documents verbatim extraction from GPT-2.
- Angwin, J. et al. (2016). "Machine Bias." ProPublica, May 23, 2016. Presents the COMPAS analysis including the 44.9 percentage-point disparity figure.
- European Parliament. (2024). Regulation (EU) 2024/1689, the EU Artificial Intelligence Act. Article 43: Conformity assessment for high-risk AI systems.
- scikit-learn documentation, version 1.5. "Logistic Regression," User Guide Section 1.1.11. https://scikit-learn.org/1.5/modules/linear_model.html#logistic-regression
- IBM Adversarial Robustness Toolbox. "Get Started" documentation, version 1.17. https://adversarial-robustness-toolbox.readthedocs.io/en/latest/
- Hugging Face. "Quicktour" for the `transformers` library. https://huggingface.co/docs/transformers/quicktour
