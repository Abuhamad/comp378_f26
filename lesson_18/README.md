# Lesson 18 — Exact and Deterministic Neural Network Verification

Objectives:

- Trace Interval Bound Propagation through a two-layer ReLU network by hand.
- Use auto_LiRPA to obtain IBP and alpha-CROWN certified bounds.
- State the completeness and soundness guarantees of SMT-based verification.
- Explain why IBP is sound but incomplete.
- Describe the scalability limits of complete verification.

## Main Concepts

- Interval Bound Propagation (IBP): layer-wise over-approximation of output range
- Linear relaxation methods: $\alpha$-CROWN and $\beta$-CROWN tighter bound propagation
- Satisfiability Modulo Theories (SMT): encoding network verification as constraint satisfaction (Reluplex)
- Verification properties: local robustness ($\epsilon$-ball safety), output reachability

## Learning Outcomes

- Student can trace IBP through a two-layer ReLU network by hand.
- Student can use `auto_LiRPA` to obtain certified bounds for a small MLP under an $L_\infty$ perturbation.
- Student can state the completeness and soundness guarantees of SMT-based vs. relaxation-based verification.
