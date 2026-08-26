# Lesson 34 — Explainable AI and Interpretability

Objectives:

- Distinguish local from global interpretability.
- Explain the Shapley value derivation as a fair contribution allocation.
- Generate SHAP waterfall and summary plots for a tabular model.
- Generate LIME superpixel explanations for an image classifier.
- Implement Integrated Gradients for a deep network.
- Measure explanation stability as the standard deviation of SHAP values.

## Main Concepts

- Local vs. global interpretability: explaining individual predictions vs. overall model behavior
- SHAP (SHapley Additive exPlanations): game-theoretic feature attribution; TreeSHAP and KernelSHAP
- LIME (Local Interpretable Model-agnostic Explanations): local linear surrogate fitting
- Integrated Gradients: attribution via path integrals in deep networks
- Explanation stability and adversarial vulnerability of explanations

## Learning Outcomes

- Student can generate SHAP waterfall and summary plots for a tabular model and interpret feature attributions.
- Student can generate LIME explanations for an image classifier and identify the most influential superpixels.
- Student can measure and report explanation stability (standard deviation of SHAP values across input perturbations).
