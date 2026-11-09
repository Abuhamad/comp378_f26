# 34 — Explainable AI and Interpretability

## Abstract

Predictions from complex models matter, but explaining why a model made a prediction is often as important as the prediction itself. This lesson covers techniques for interpreting model behavior at two levels: local explanations answer "why did this model predict X for this input" while global explanations reveal "how does this model behave across all inputs." Students learn Shapley value game theory, LIME's neighborhood sampling approach, and gradient-based Integrated Gradients, then measure explanation stability under perturbations to expose whether explanations are trustworthy for auditing and accountability.

## Objectives

- Student can distinguish local from global interpretability and state a concrete use case for each.
- Student can explain the Shapley value derivation as a fair contribution allocation across players and map it to feature attribution.
- Student can generate SHAP waterfall and summary plots for a tabular model using `shap.KernelExplainer` and interpret the top three positive and negative attributions.
- Student can generate LIME superpixel explanations for an image classifier and identify the five most influential superpixels for a given prediction.
- Student can implement Integrated Gradients for a deep network, verify that attributions satisfy the completeness axiom, and compare attributions to KernelSHAP on the same input.
- Student can measure explanation stability as the standard deviation of SHAP values across 20 perturbed copies of a single input and report whether stability is above or below a defined threshold.

## Content

### Local vs. Global Interpretability

Interpretability splits into two complementary scopes. Local interpretability explains a single prediction: given input X and output Y, which input features drove the model to output Y instead of an alternative? Local questions matter when a customer disputes a loan denial, a patient challenges a medical diagnosis, or a regulator audits a consequential decision. Global interpretability reveals overall model behavior: how does the model rank features by influence across the entire dataset? Global questions inform model debugging, feature selection, and understanding which relationships the model learned. A credit-scoring model might locally explain a specific loan denial (high debt-to-income ratio pushed score below threshold), while global analysis reveals the model relies primarily on payment history and credit utilization across 10,000 applications. Both explain the model, but to different stakeholders at different points in the decision lifecycle.

```mermaid
flowchart TB
    Q{"What do we want<br/>to explain?"}
    Q -->|"a single prediction"| L["Local interpretability<br/><i>why did the model output Y<br/>for this input X?</i>"]
    Q -->|"overall behavior"| G["Global interpretability<br/><i>which features drive the model<br/>across the whole dataset?</i>"]

    L --> LU["Use cases: disputed loan denial,<br/>challenged diagnosis,<br/>regulator audit of one decision"]
    G --> GU["Use cases: model debugging,<br/>feature selection,<br/>understanding learned relationships"]

    LU --> M["Methods in this lesson:<br/>SHAP, LIME, Integrated Gradients<br/>(local), SHAP summary plots (global)"]
    GU --> M
```

### SHAP (SHapley Additive exPlanations)

SHAP grounds feature attribution in Shapley values from cooperative game theory. In a coalition game, players contribute to a total payoff, and the Shapley value assigns each player a fair share based on their marginal contribution across all possible coalitions. For machine learning, features are players, the model is the game, and the prediction is the payoff. The Shapley value for feature $i$ is the average marginal contribution of feature $i$ when added to every possible subset of the other features.

Formally, for feature $i$ in the feature set $N$ (with $|N| = D$), the Shapley value is the weighted average marginal contribution over all subsets $S$ of the remaining features:

```math
\varphi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(D - |S| - 1)!}{D!} \left[ f(S \cup \{i\}) - f(S) \right],
```

or equivalently $`\varphi_i = \mathbb{E}\left[f(S \cup \{i\}) - f(S)\right]`$, where $S$ is the (random) set of features that precede $i$ in a uniformly random ordering of all features. This formulation satisfies two axioms. The **local accuracy** axiom states that the sum of feature attributions plus a baseline equals the model output: $`f(X) = \varphi_0 + \sum_{i=1}^{D} \varphi_i`$, where the baseline $`\varphi_0 = \mathbb{E}[f(X)]`$ is the average model output. The **consistency** axiom ensures that if a model changes so that feature $i$'s marginal contribution increases or stays the same for every subset $S$, then feature $i$'s attribution does not decrease.

Computing exact Shapley values on a feature space of $D$ dimensions requires evaluating $2^D$ coalitions, which is infeasible for large $D$. KernelSHAP approximates Shapley values by sampling coalitions. Instead of evaluating all $2^D$ coalitions, KernelSHAP samples $M$ coalitions, evaluates the model on each coalition representation, and solves a weighted least-squares problem to estimate attributions. The Shapley kernel assigns the largest weights to coalitions with very few or very many features present, because those coalitions are the most informative about each feature's individual marginal contribution. For tabular data with dozens of features, KernelSHAP produces accurate approximations with $M \approx 1000$ to $2000$ samples.

TreeSHAP exploits tree structure to compute exact Shapley values in polynomial time for tree ensembles and gradient boosted models. Trees decompose feature interactions into local structure. TreeSHAP walks the tree and computes the exact contribution of each feature along the decision path. For a 1000-tree ensemble, TreeSHAP computes explanations in seconds rather than hours, making explanation audits feasible on production models.

```mermaid
flowchart LR
    subgraph GAME["Cooperative game theory"]
        P["Players"]
        PAY["Total payoff"]
        SH["Shapley value:<br/>fair share of payoff per player"]
    end
    subgraph ML["Feature attribution"]
        F["Features"]
        PRED["Prediction f(X)"]
        PHI["Attribution φ_i per feature"]
    end
    P -. "maps to" .-> F
    PAY -. "maps to" .-> PRED
    SH -. "maps to" .-> PHI

    PHI --> EX{"How to compute?<br/>exact needs 2^D coalitions"}
    EX -->|"any model<br/>(model-agnostic)"| KS["KernelSHAP (approximate)<br/>sample M ≈ 1000–2000 coalitions,<br/>weighted least squares"]
    EX -->|"tree ensembles /<br/>gradient boosting"| TS["TreeSHAP (exact)<br/>polynomial time via<br/>tree structure"]
```

### LIME (Local Interpretable Model-agnostic Explanations)

LIME explains a single prediction by building a local surrogate model around the query input. The intuition is that globally, the model is complex and nonlinear, but locally around a single input, a linear model often approximates the decision boundary well. LIME's algorithm has three steps.

First, LIME perturbs the input in the input space (not the feature space). For images, LIME segments the image into superpixels, then samples random binary masks over superpixels, toggling each superpixel on or off. For tabular data, LIME adds noise to continuous features and uniformly samples categorical features. LIME generates $M$ perturbed instances (typically $M = 1000$ for tabular data, $M = 500$ for images) around the original input.

Second, LIME evaluates the original model on each perturbed instance and records the predictions. LIME also computes the distance from each perturbed instance to the original input: closer instances receive higher weights in a Gaussian kernel.

Third, LIME fits a weighted linear model to the perturbed instances and their predictions, where each instance's weight is the kernel proximity computed in the previous step. The weights emphasize instances similar to the query input, so the linear model captures local decision logic. The coefficients of the linear model are the LIME attributions. For an image, the coefficients tell which superpixels push the prediction up or down. For tabular data, the coefficients rank features by local influence.

LIME's main weakness is sensitivity to the sampling distribution and neighborhood definition. Small changes to the perturbation radius can shift which features appear important. LIME makes no statistical guarantee that the linear surrogate is faithful to the original model in any formal sense.

```mermaid
flowchart LR
    X["Query input X"] --> S1["1. Perturb<br/>images: toggle superpixels<br/>via random binary masks;<br/>tabular: noise / resampling<br/>→ M instances"]
    S1 --> S2["2. Evaluate & weight<br/>query the original model;<br/>Gaussian kernel on distance:<br/>closer instances → higher weight"]
    S2 --> S3["3. Fit weighted<br/>linear surrogate<br/>around X"]
    S3 --> A["Coefficients = attributions<br/>superpixels / features that push<br/>the prediction up or down"]
```

### Integrated Gradients

Integrated Gradients attributes a prediction to input features by integrating gradients along a straight-line path from a baseline $X'$ (often an all-zeros or random input) to the query input $X$. The method grounds in two axioms. **Sensitivity** states that if the input and the baseline differ only in feature $i$ and the model outputs differ, then feature $i$ must have a nonzero attribution. **Completeness** states that the sum of attributions over all features equals the model output minus the baseline output: $`\sum_i \mathrm{IG}_i(X) = f(X) - f(X')`$.

Integrated Gradients is defined as:

```math
\mathrm{IG}_i(X) = (X_i - X'_i) \int_{0}^{1} \frac{\partial f\big(X' + \alpha\,(X - X')\big)}{\partial X_i} \, d\alpha .
```

The path integral accumulates the gradient of the model output with respect to feature $i$ as we move from the baseline toward $X$. In practice, numerical integration approximates the continuous integral (e.g., a Riemann sum or the trapezoidal rule) by averaging gradients at $K$ discrete points (typically $K = 50$ to $500$) along the path:

```math
\mathrm{IG}_i(X) \approx (X_i - X'_i) \cdot \frac{1}{K} \sum_{k=1}^{K} \frac{\partial f\big(X' + \tfrac{k}{K}\,(X - X')\big)}{\partial X_i} .
```

For a deep convolutional network classifying an image, Integrated Gradients produces a per-pixel attribution map. Pixels with large positive attributions pushed the network toward the predicted class. Pixels with large negative attributions pushed away from it. Integrated Gradients satisfies completeness: summing all pixel attributions reconstructs the output difference between the input image and the baseline. This formal guarantee makes Integrated Gradients useful for auditing, unlike LIME which offers no such guarantee.

```mermaid
flowchart LR
    B["Baseline X′<br/>(all-zeros / random input)"] == "straight-line path<br/>X′ + α(X − X′), α: 0 → 1" ==> X["Query input X"]
    B -.-> G["Evaluate ∂f/∂X_i at<br/>K = 50–500 points<br/>along the path"]
    X -.-> G
    G --> ACC["Average gradients and<br/>scale by (X_i − X′_i)"]
    ACC --> IG["Attribution IG_i(X)<br/>per feature / pixel"]
    IG --> CHK["Completeness check:<br/>Σ IG_i = f(X) − f(X′)"]
```

### Explanation Stability

Explanations are stable if small perturbations to the input produce small changes to the attributions. A credit-scoring model might assign high attribution to debt-to-income ratio for a borderline applicant. If adding Gaussian noise (standard deviation $\sigma = 0.01$) to the input features changes the attribution to debt-to-income by $50\%$, the explanation is fragile and unreliable for auditing.

Measure stability by generating $K$ perturbed copies of a test input (typically $K = 20$), computing attributions for each perturbed copy, and computing the standard deviation of attributions across the $K$ copies. If the mean SHAP attribution for a feature is $\mu = 0.3$ and the standard deviation across perturbations is $\sigma = 0.15$, the coefficient of variation is $`\mathrm{CV} = \sigma / |\mu| = 0.15 / 0.3 = 0.5`$, indicating substantial instability. Flag features where the standard deviation exceeds $10\%$ of the mean absolute attribution as unstable.

Slack et al. (2019) demonstrated that explanations can be intentionally manipulated. By adding adversarial perturbations to model inputs, an attacker can shift feature attributions to hide discriminatory behavior while maintaining the same predictions. A lending model trained to discriminate by race might learn to weight race heavily in its decision. SHAP values would reveal this discrimination. But with adversarial perturbations in training data, the model learns to hide race's true contribution, and SHAP values report a false attribution. Explanation instability correlates with vulnerability to such attacks.

```mermaid
flowchart TB
    X["Test input X"] --> P["Generate K = 20 perturbed copies<br/>X + ε, with Gaussian noise ε"]
    P --> A["Compute attributions<br/>(e.g., SHAP values)<br/>for every copy"]
    A --> S["Per-feature mean μ and<br/>standard deviation σ<br/>across the K copies"]
    S --> D{"σ > 10% of |μ|?"}
    D -->|"yes"| U["Unstable explanation<br/>unreliable for auditing;<br/>correlates with vulnerability to<br/>adversarial manipulation<br/>(Slack et al., 2019)"]
    D -->|"no"| OK["Stable explanation<br/>trustworthy enough for<br/>audits and accountability"]
```

### Attention to Scope

This lesson focuses on local and global explanations through SHAP, LIME, and Integrated Gradients, and measurement of explanation stability. Fairness metrics and bias mitigation belong to m32. AI governance frameworks and formal auditing processes belong to m35. Guardrail configuration and PII handling belong to m39. Counterfactual explanations (e.g., "if income were $5,000 higher, the prediction would change") and concept-based explanations (TCAV) are outside the course scope.

## Lab

The lab notebook at [lab.ipynb](./lab.ipynb) exercises KernelSHAP on a tabular credit-scoring model, LIME on an image classifier, and measures explanation stability through perturbation analysis. Students generate waterfall and summary plots, identify influential superpixels, and compute per-feature standard deviations to flag unstable attributions.

## Summary

Explaining model predictions requires specifying scope: local explanations answer "why this prediction" while global explanations reveal model-wide tendencies. Shapley values offer a theoretically grounded, consistent foundation for local attribution. KernelSHAP and TreeSHAP make this practical on real models. LIME builds local linear surrogates through neighborhood sampling, offering model-agnostic explanations but without formal guarantees. Integrated Gradients uses gradient paths to produce pixel-level or feature-level attributions satisfying completeness. All three methods are vulnerable to instability: small input changes can shift attributions significantly, raising concerns for use in audits or policy decisions. Measuring stability through perturbation analysis reveals whether explanations are reliable enough for high-stakes applications.

## Useful References and Resources

- Lundberg, S., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. In Advances in Neural Information Processing Systems (NIPS). [Introduces SHAP and connects Shapley values to feature attribution]
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). Why Should I Trust You?: Explaining the Predictions of Any Classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. [Introduces LIME]
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. In International Conference on Machine Learning (ICML). [Introduces Integrated Gradients and completeness axiom]
- Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2019). Fooling LIME and SHAP: Adversarial Attacks on Post-Hoc Explanation Methods. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency (FAccT). [Demonstrates adversarial manipulation of explanations]
- `shap` Python package (<https://github.com/slundberg/shap>) [Implements SHAP, KernelSHAP, TreeSHAP with waterfall and summary plot visualization]
- `lime` Python package (<https://github.com/marcotcr/lime>) [Implements LIME for tabular and image data with superpixel segmentation]
- `captum` PyTorch package (<https://captum.ai/>) [Implements Integrated Gradients and other gradient-based attribution methods for neural networks]
