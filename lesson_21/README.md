# Lesson 21 — Backdoor and Trojan Attacks in Neural Networks

Objectives:

- State the backdoor threat model.
- Implement a BadNets-style pixel-patch backdoor and report clean accuracy and ASR.
- Describe blend and frequency-domain invisible triggers.
- Explain the Trojan weight-manipulation approach.
- Explain how fine-tuning a backdoored foundation model propagates the trigger.

## Main Concepts

- Trigger-based backdoor attacks: patch triggers, blend triggers, invisible frequency triggers
- BadNets attack: pixel-patch trigger insertion and label flipping during training
- Trojaning via internal representation modification: direct weight manipulation
- Backdoor risks in foundation models: fine-tuning propagation of pre-trained triggers

## Learning Outcomes

- Student can explain the BadNets threat model and why trigger-based backdoors survive standard training.
- Student can implement a BadNets-style backdoor attack that achieves >95% attack success rate while maintaining clean accuracy.
- Student can describe how triggers can be invisible (frequency domain, blending) and why this complicates detection.
