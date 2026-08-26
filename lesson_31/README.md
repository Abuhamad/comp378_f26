# Lesson 31 — Defenses and Privacy in Federated Learning

Objectives:

- Implement Krum aggregation and state its Byzantine fault tolerance guarantee.
- Explain the Coordinate-wise Trimmed Mean algorithm.
- Describe Multi-Krum and explain the performance-robustness trade-off.
- Describe the SecAgg double-masking protocol.
- Apply Local Differential Privacy to client gradient vectors using a Gaussian mechanism.
- Reason about the combined effect of layering robust aggregation, SecAgg, and LDP.

## Main Concepts

- Byzantine-robust aggregation: Krum, Coordinate-wise Trimmed Mean, Multi-Krum
- Secure Aggregation (SecAgg): cryptographic protocol to hide individual client updates from the server
- Local Differential Privacy (LDP): client-side noise injection before gradient upload
- Defense combination: robust aggregation + SecAgg + LDP interaction and trade-offs

## Learning Outcomes

- Student can implement Krum aggregation and explain its Byzantine fault tolerance guarantee.
- Student can describe the SecAgg cryptographic protocol and what information it hides from the server.
- Student can apply LDP to client gradients and measure the accuracy cost relative to the privacy gain ($\epsilon$).
