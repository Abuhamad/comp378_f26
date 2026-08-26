# Lesson 38 — Model Watermarking, IP Protection, and Machine Unlearning

Objectives:

- Explain the Kirchenbauer green-list/red-list token-bias watermarking scheme.
- Implement the Kirchenbauer scheme on GPT-2 output.
- Measure watermark survival rate after paraphrasing attacks.
- Describe at least one AI-generated text detection approach.
- State the GDPR Article 17 right to erasure.
- Apply gradient ascent approximate unlearning to remove a target training sample’s influence.

## Main Concepts

- LLM output watermarking: green-list/red-list token bias (Kirchenbauer et al. scheme)
- Watermark detection: statistical hypothesis test on token distribution
- Synthetic content detection: AI-generated text and image detectors
- Machine unlearning: exact unlearning vs. approximate unlearning (gradient ascent, SISA training)

## Learning Outcomes

- Student can implement the Kirchenbauer token-bias watermarking scheme and compute detection z-scores.
- Student can measure watermark detection rate after paraphrasing attacks on watermarked text.
- Student can apply approximate machine unlearning (gradient ascent) to remove a specific training example's influence and verify via MIA.
