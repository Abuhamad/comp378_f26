# 40 - Capstone: AI Red-Teaming and Blue-Teaming

## Abstract

This capstone lesson synthesizes security defenses from all 39 prior lessons into a coherent defense-in-depth posture. You will learn to plan and execute structured red-team assessments covering evasion, data poisoning, model extraction, and LLM injection attacks, map each attack to its kill chain stage, execute blue-team monitoring and incident response, and interpret real-world AI security failures. The capstone integrates offense and defense as two sides of the same continuous security loop.

## Objectives

- Student can produce a structured red-team plan document including: scope definition, threat model with attacker goals, rules of engagement, and success criteria for at least three attack classes (exercised in lab Step 0).
- Student can execute a structured red-team assessment covering at least three of the four CTF attack classes (clean-label poisoning, evasion under query budget, model extraction, indirect prompt injection).
- Student can map each discovered attack to its kill chain stage and propose a specific defense-in-depth countermeasure for each stage.
- Student can activate blue-team monitoring to detect at least two of the CTF attacks in progress using runtime anomaly detection signals.
- Student can produce a complete AI incident response report for a detected attack, including detection timeline, containment action, and remediation plan.
- Student can compile a combined red-team report integrating all four flag findings, kill chain mapping, and blue-team response timeline into a single structured document.

## Content

### Red-Teaming Methodology

A red team conducts structured, goal-driven adversarial evaluation of an AI system. Red-teaming differs from ad hoc "try to break it" testing by enforcing four phases. Planning, reconnaissance, exploitation, and reporting separate structured attacks from random testing.

The planning phase begins with a scope contract. Scope defines which components, data, and API endpoints are in bounds. A typical scope statement names the target model architecture, the input domain (images, text, tabular data), the training pipeline (online, batch-retraining, or frozen), and which defenses are already deployed. Rules of engagement specify whether attacks on training data are permitted, whether the team has white-box access or black-box access, whether live queries consume a budget (e.g., 500 queries maximum for model extraction), and whether the attack must be imperceptible or only effective. The threat model identifies attacker goals such as misclassifying a target, poisoning the training process, extracting model weights, or manipulating an LLM. Success criteria state measurable achievements, for example "achieve 90 percent top-1 misclassification rate on 10 adversarial examples under Linf norm of 8/255".

Reconnaissance gathers system information without triggering defenses. Techniques include API enumeration to list available endpoints and their latency, confidence probing to query the model with known inputs and measure output distributions, model fingerprinting to infer model family and framework from response patterns, and checkpoint inspection to load the model and check activation functions and layer counts. Reconnaissance produces a threat model update if attacker capabilities prove weaker or stronger than assumed.

Exploitation executes multi-class attacks. Each attack class is attempted within its scope and rules of engagement. A PGD evasion attack respects the query budget and perturbation bound. A poisoning attack injects crafted samples into the training pipeline and measures resulting model drift. A model extraction attack sends queries until it achieves 90 percent fidelity on a held-out test set. An LLM injection attack finds a gadget chain through prompt injection and indirect prompt injection. Exploitation is iterative. Early failures trigger hypothesis refinement, not immediate termination.

Reporting translates raw attack findings into a structured document. Each finding names the attack class, the kill chain stage where it succeeded, evidence such as a log or a misclassified example, and a single recommended countermeasure. Findings are prioritized by severity measured as exploitability times impact, and each receives a structured headline for executive summary. The report also names any scope or rules-of-engagement violations encountered by the red team, so defenders understand the assumptions under which the attacks succeeded.

### Kill Chain Mapping for AI

A kill chain maps the progression of an attack from initial access to final objective. The AI attack kill chain has five stages: data collection, training, deployment, inference, and monitoring.

In the data collection stage, an attacker poisons the training dataset. Techniques include label flipping, feature injection by adding carefully chosen pixels to images or substrings to text, and clean-label poisoning by modifying existing samples to maintain label correctness while introducing adversarial patterns. Defenses at this stage include data sanitization through clustering and removal of statistical outliers, provenance tracking by recording the source of every training sample, and data poisoning detection by comparing gradient-based statistics of poisoned samples against benign populations.

In the training stage, an attacker backdoors the learning process. Techniques include Byzantine attacks in federated learning by sending poisoned model updates to a central aggregator and trigger-based backdoors by adding a watermark pattern that activates specific misclassifications when present. Defenses include robust aggregation protocols such as Krum or Median aggregation in federated settings, anomaly detection on model update statistics, and backdoor trigger detection during validation.

In the deployment stage, an attacker manipulates the model container or supply chain. Techniques include checkpoint tampering by replacing a saved model with a backdoored version and dependency injection by modifying imported libraries so that inference produces attacker-controlled outputs. Defenses include cryptographic model signing, runtime integrity checks by comparing the loaded model's gradient hash against a trusted reference, and sandboxing by isolating inference in a container with restricted system calls.

In the inference stage, an attacker queries the deployed model. Techniques include evasion attacks such as FGSM, PGD, C&W, and boundary-attack methods, model extraction via substitute models or meta-learning, and membership inference attacks by detecting whether a data point was in training. Defenses include adversarial training, input preprocessing through JPEG compression or bit depth reduction, certified defenses such as randomized smoothing or interval bound propagation, rate limiting on queries, and output perturbation by adding noise to confidence scores before returning to the user.

In the monitoring stage, an attacker exploits the feedback loop between user behavior and model retraining. Techniques include instruction attacks by providing false labels to manipulate the next training run and drift exploitation by querying in ways that push the model toward attacker-preferred regions. Defenses include anomaly detection on input distributions by detecting query patterns that deviate from historical norms, confidence distribution monitoring by detecting shifts in the model's confidence on correctly classified data, output policy enforcement by detecting when outputs violate business rules, and immutable audit logs by recording all inference queries for post-hoc analysis.

A defense-in-depth posture places countermeasures at every stage. A single defense rarely stops all attacks. Instead, multiple defenses across the kill chain make compromise progressively harder. For example, training-stage Byzantine-robust aggregation prevents data-collection poisoning from reaching the aggregator, while inference-stage rate limiting still catches extraction attempts. This redundancy means that if one defense fails, others remain active.

### Blue-Team Response Playbook

A blue team detects attacks and contains compromise. A playbook maps each attack class to its detection triggers, escalation paths, and containment actions.

For evasion attacks, detection triggers include a spike in the query volume to a model API above historical baseline such as 10 times normal per-minute rate, a sudden increase in misclassification rate on queries that appear otherwise normal, or a clustering of queries around decision boundaries suggesting adversarial optimization. Escalation involves alerting the security team, beginning a timeline log, and checking whether the attack is targeted against one class or untargeted trying to minimize overall accuracy. Containment actions include rate-limiting the attacking source IP, deploying a shadow classifier to verify that the main model is under attack and not simply malfunctioning, and rolling back the model to a known-good checkpoint if compromise is confirmed.

For poisoning attacks in federated learning, detection triggers include a federated worker sending model updates with unusual gradient magnitudes or activation patterns, a sudden drop in validation accuracy after aggregation, or anomalous updates from a small number of workers. Escalation involves isolating the suspected worker and requesting its local training logs and data sample statistics. Containment actions include removing the worker's update from the aggregation, rerunning aggregation with robust methods such as Krum or Median, and temporarily increasing local validation frequency to catch further drift.

For model extraction attacks, detection triggers include a surge in queries to the model API from a single source, a pattern of queries designed to sweep the confidence distribution across low, medium, and high confidence on diverse inputs, or an observed clone model whose predictions match the original with suspiciously high fidelity on held-out data. Escalation involves confirming that the external model is indeed a functional clone by comparative testing. Containment actions include rate-limiting or temporarily blocking the attacking IP, adding noise to confidence scores through output perturbation, requiring authentication for API access, and deploying a meta-model to detect surrogate queries designed to build a substitute model rather than answer a legitimate question.

For LLM injection attacks, detection triggers include outputs that violate the chatbot's intended scope such as a finance chatbot refusing to answer finance questions because of a prompt override, a command-and-control string appearing in the model's output indicating tool calls that should not have been made, or an output format that deviates from the learned template suggesting prompt injection succeeded. Escalation involves recording the exact user input and model output, checking whether the injection came from a benign user or a known attacker, and testing whether the injection persists if the vector store is cleared. Containment actions include filtering the malicious document from the vector store, updating the system prompt to be more robust against injection, rate-limiting tool calls from the chatbot, and implementing a security monitor that detects when COLANG guardrails are overridden.

### Continuous Monitoring Architecture

Continuous monitoring uses runtime signals to detect anomalies before they become breaches. A telemetry pipeline collects signals from the model, the data, and the operational environment, streams them to a time-series database, and triggers alerts when signals deviate from historical baselines.

Query volume monitoring tracks the number of requests per time window such as per minute. Baseline is the 90th percentile of request rates from the past 30 days. An alert fires when the current rate exceeds the baseline by 3 standard deviations. This catches volume-based attacks like model extraction sweeps and evasion probing.

Confidence distribution monitoring tracks the mean and standard deviation of the model's confidence scores on inputs the model predicts correctly. Baseline is the distribution from a validation set or from the past 7 days of inference. An alert fires when the current mean confidence drops by more than 2 standard deviations from baseline, or when the standard deviation increases above baseline by more than 1.5 times. This catches data distribution shift and confidence-lowering attacks.

Output policy enforcement monitors whether each prediction violates business rules. For example, a lending model should not approve loans under certain credit-score thresholds, or should not approve and deny loans for applicants with identical credit profiles. A violation log records any output that breaches the policy, and alerts fire if violations exceed a threshold such as 5 per day. This catches both misclassification and systematic bias.

Model drift alerting measures whether the model's performance on a held-out validation set has degraded. Every hour or every day, run inference on 1,000 samples from a recent labeled set and compare the accuracy against a baseline accuracy from a month ago. An alert fires if accuracy drops by more than 2 percentage points. This catches poisoning and training-stage attacks that corrupt the model weights.

Feature attribution monitoring (using SHAP or similar explainability methods) records which input features the model relies on for each prediction. If the top features change between day 1 and day 30, this may indicate model drift or data distribution shift. If the top features include a previously-unknown adversarial feature (e.g., a specific pixel pattern), this suggests attack activity.

An alert aggregation layer deduplicates and correlates signals across the telemetry pipeline. For example, a spike in query volume combined with a spike in misclassifications suggests evasion, whereas a spike in query volume alone suggests probing or extraction. Aggregation groups correlated alerts into incidents and assigns severity based on the number of concurrent alerts and their historical precedent.

### AI Incident Response Lifecycle

An incident is a confirmed breach or security event. The incident response lifecycle has four stages: detection, containment, remediation, and post-mortem.

Detection occurs when an alert fires and the security team confirms it is a true positive. True-positive confirmation involves running an exploratory query. Ask the model to classify an input known to be benign. If the model misclassifies it, the anomaly is real. Check whether the anomaly is geographically or temporally correlated with known attack campaigns. Look for corroborating signals. If evasion is suspected, run a re-training-time test to confirm the model under attack is indeed vulnerable. Document the time, the alert, the confirming evidence, and the likely attack class.

Containment isolates the compromised system to prevent spread and further damage. For an inference-time attack, containment may mean deactivating the API endpoint, rolling back the model to a known-good version, or switching to a shadow model that uses an orthogonal architecture. For a training-time attack, containment involves halting retraining, isolating the poisoned data, and preparing a new training run on clean data. For an LLM attack, containment involves removing adversarial documents from the vector store and updating the system prompt.

Remediation repairs the system. For evasion, remediation includes retraining the model on the same clean data, or if the attack revealed a systemic weakness, retraining with augmented data or a more robust architecture such as adding adversarial training or switching to certified defenses. For poisoning, remediation involves identifying the poisoned samples, removing them from the training dataset, and retraining. For extraction, remediation includes querying the suspected clone model to understand what it learned, then deciding whether to accept that the model is now known to a competitor and focus on detection, or to discontinue the model. For LLM injection, remediation involves rewriting system prompts, retraining on prompt-injection-resistant examples, and adding COLANG guardrails.

Post-mortem analysis answers four questions. What was the timeline? What was missed? What will prevent recurrence? Who had access?

Timeline reconstructs the attack from first alert to containment. Typical timeline includes the alert timestamp, the first evidence of malicious input, the point at which the attack could have been detected earlier using better monitoring, and the time at which containment was achieved. A 10-hour gap between the alert and detection indicates reactive detection, while a 30-minute gap indicates proactive anomaly detection.

Root cause analysis identifies what allowed the attack to succeed. Common root causes include missing input validation, no rate limiting on queries, insufficiently robust aggregation in federated learning, outdated system prompts in LLMs, or known vulnerabilities in dependencies that were never patched. Recording the root cause prevents the same attack from succeeding again.

Recommendations propose procedural and technical changes. For each attack, the post-mortem recommends at least one defense. For evasion, recommendations include adversarial training or certified defenses. For poisoning, recommendations include robust aggregation, data sanitization, or cryptographic model signing. For extraction, recommendations include output perturbation or authentication on the API. For LLM injection, recommendations include prompt robustness training or COLANG guardrails.

Access review checks whether the attacker had legitimate access to the system representing an insider threat, or exploited a publicly-exposed interface representing an outsider threat. If an insider, the post-mortem recommends role-based access control, activity logging, and periodic access audits. If an outsider, the post-mortem recommends network segmentation and API authentication.

### Lessons from Real-World AI Security Incidents

In 2021, researchers demonstrated that adversarial patches printed as physical objects that cause misclassification could fool autonomous vehicle perception systems. A sticker placed on a stop sign caused the classifier to misread it as a speed-limit sign. This revealed that the training dataset lacked examples of real-world occlusions and lighting variations. Defense involved augmenting training data with patches and occlusions, deploying certified defenses on safety-critical perception, and implementing a secondary human-review stage before acting on safety-critical classifications.

In 2023, multiple ChatGPT instances were jailbroken using prompt injection and indirect injection by adding adversarial instructions to documents loaded into the chatbot. Attackers used techniques like role-playing claiming "you are now an unrestricted AI" and context injection by embedding instructions in user-provided documents. The root cause was that the system prompt was not made sufficiently robust against adversarial rewrites. Defense involved using COLANG guardrails to enforce non-overridable safety constraints, sanitizing all documents before indexing, implementing prompt-injection detection via anomalous output format detection, and rate-limiting tool calls.

In 2024, a dataset used to train a fraud detection model was poisoned by competitor employees who introduced mislabeled benign transactions as fraudulent. This caused the model to reject legitimate customer transactions, damaging customer satisfaction. The attack succeeded because dataset provenance was not tracked and the training team did not validate the source of the data. Defense involved implementing data provenance tracking, digitally signing all data sources, sampling and manually verifying training data before each training run, and deploying Byzantine-robust aggregation if using federated learning.

In 2022, machine learning models deployed by a major financial institution were reverse-engineered from API queries, allowing competitors to build functional clones within weeks. The root cause was an unrestricted query budget and lack of output perturbation. Defense involved implementing rate limiting on queries per user per day, adding calibrated noise to confidence scores before returning them, requiring API authentication, and monitoring for surrogate-model detection patterns.

Each incident involved a failure at one of the kill chain stages. The stop-sign patch bypassed inference-time defenses because the model was never trained on adversarially-perturbed examples, representing a training-stage failure. The ChatGPT jailbreak succeeded because the system prompt was not hardened, representing a deployment-stage failure. The fraud-detection poisoning happened because the data collection stage lacked integrity checks. The financial model extraction succeeded because the inference stage had no rate limiting or noise injection.

## Lab

See [lab_red_team.ipynb](./lab_red_team.ipynb) and [lab_blue_team.ipynb](./lab_blue_team.ipynb) for hands-on exercises.

## Summary

This capstone synthesized 39 prior lessons into a coherent security posture spanning offense and defense. Red-teaming methodology enforces planning, reconnaissance, exploitation, and reporting phases, ensuring that adversarial evaluation is systematic rather than ad hoc. Kill chain mapping ties each attack class to a specific stage (data through monitoring) and proposes defense-in-depth countermeasures. Blue-team response playbooks define detection triggers, escalation steps, and containment actions for evasion, poisoning, extraction, and injection attacks. Continuous monitoring architecture streams runtime signals into alert logic. The AI incident response lifecycle transforms alerts into contained, remediated, and documented incidents. Real-world incidents consistently reveal failures at one kill chain stage where defense was missing or misconfigured.

## Useful References and Resources

- NIST AI Risk Management Framework (Lesson 35): AI Impact Assessment template for recording threat models and blue-team response plans.
- MITRE ATT&CK for Enterprise (Lesson 07): Kill chain taxonomy adapted for AI systems. Privilege Escalation and Persistence techniques map to training-stage attacks.
- Adversarial Robustness Toolbox (Lesson 19): Red-teaming framework for multi-class attack orchestration and AutoAttack and ART evaluation suites.
- Robustness Benchmarking (Lesson 19): Benchmark suite (RobustBench) for testing defenses against standardized attacks.
- DP-SGD and Differentially-Private Training (Lesson 27): Privacy budget accounting and formal privacy guarantees for training.
- Federated Learning Defenses (Lesson 31): Krum and Secure Aggregation implementations for Byzantine-robust federated averaging.
- Guardrails and Sandboxing (Lesson 39): COLANG guardrail framework for LLM constraint enforcement and container sandbox configuration for model isolation.
- Model Watermarking and IP Protection (Lesson 38): Cryptographic model signing and integrity checking for deployment-stage supply chain defense.
- Explainability and Interpretability (Lesson 34): SHAP for feature attribution in continuous monitoring anomaly detection.
