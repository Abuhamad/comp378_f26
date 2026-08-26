# Lesson 28 — Encrypted Computation: Homomorphic Encryption and SMPC

Objectives:

- Explain additive secret sharing.
- Distinguish BFV/BGV from CKKS.
- Explain why ReLU cannot be computed exactly under HE.
- Run encrypted neural network inference using TenSEAL (CKKS).
- State the computational overhead and multiplicative-depth constraints of HE.

## Main Concepts

- Secure Multi-Party Computation (SMPC): secret sharing and garbled circuits for joint computation
- Leveled and Fully Homomorphic Encryption (FHE): addition and multiplication on ciphertexts
- Privacy-preserving inference: running a neural network on encrypted inputs without decryption
- Practical limitations: computational overhead, circuit depth constraints, noise budget

## Learning Outcomes

- Student can explain the difference between leveled HE and FHE and when each is applicable.
- Student can run encrypted neural network inference using `TenSEAL` (CKKS scheme) on encrypted image inputs.
- Student can measure and report the latency overhead of encrypted inference vs. plaintext inference.
