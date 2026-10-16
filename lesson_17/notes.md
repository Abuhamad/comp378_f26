# 17 — Certified Robustness via Randomized Smoothing

## Abstract

This lesson covers randomized smoothing, a probabilistic defense technique to certify robustness with a formal guarantee. Students learn how Gaussian noise injection yields a smoothed classifier, why the Neyman-Pearson lemma makes Gaussian noise optimal for L₂ certification, and how Cohen et al. (2019) uses hypothesis testing to compute a certified radius for each input.

## Objectives

- State the smoothed classifier definition and explain the relationship between the base classifier and the smoothed classifier.
- Justify Gaussian noise injection using the Neyman-Pearson lemma and isoperimetric inequality to derive L₂ certification.
- Implement a Monte Carlo sampling procedure with Clopper-Pearson confidence intervals to compute certified radius.
- Compute and interpret certified accuracy versus noise level trade-off curves.
- Explain abstention and when a certified prediction is declined.

## Content

### Smoothed classifier definition

A smoothed classifier $g(x) = \arg\max_c \mathbb{P}_{\varepsilon \sim \mathcal{N}(0,\sigma^2 I)}[f(x+\varepsilon)=c]$ averages predictions over Gaussian-noisy versions of the input. Here $f$ is the base classifier, $\varepsilon$ is a noise vector drawn from a standard Gaussian with variance $\sigma^2$, and $c$ ranges over all classes.

The majority-vote interpretation is intuitive. Sample noise vectors, add each to the input, pass through the base classifier, and count votes per class. The smoothed classifier picks the class with the most votes. This averaging process widens the effective decision boundary of the base classifier. A small perturbation to the input may flip a single noisy sample or a handful of them, but most samples still vote the same way, so the smoothed output stays stable.

Unlike an ensemble of diverse models, the smoothed classifier reuses one base model with different noise realizations. The smoothing trades determinism for robustness. The base classifier may be fragile to adversarial perturbations, but averaging over noise reduces this fragility for the smoothed classifier.

### Gaussian noise injection rationale

The Neyman-Pearson lemma establishes the optimal statistical test for distinguishing between two distributions. To tell apart $\mathcal{N}(\mu_1, \sigma^2 I)$ from $\mathcal{N}(\mu_2, \sigma^2 I)$, the optimal test compares the likelihood ratio against a threshold. This test has the lowest false-negative rate for any fixed false-positive rate. The optimal test depends on the difference $|\mu_1 - \mu_2|$ in the L₂ norm.

For certified robustness, the intuition is direct. An adversary perturbs an input $x$ by adding a bounded noise vector $\delta$ with $\|\delta\|_2 \leq r$. If we view the adversarial perturbation and the Gaussian noise as competing to flip the predicted class, the Neyman-Pearson lemma says Gaussian noise is the hardest-to-beat defense because it provides the tightest statistical separation.

The isoperimetric inequality for Gaussian measure refines this intuition. Among all measurable sets with the same surface area in an L₂ ball, the half-space maximizes the Gaussian measure of points outside the ball. This means shifting a half-space (decision boundary) by an L₂ distance $r$ affects the Gaussian probability less than shifting any other shape. A certifiable defense against L₂ perturbations leverages this geometric fact.

Gaussian noise is specific to L₂ certification. Laplace noise would certify L∞ robustness, but Laplace's distribution does not perform as well for the L₂ case. Randomized smoothing with Gaussian noise is the principled choice for L₂.

### Cohen et al. (2019) certification

The certification procedure runs a hypothesis test on noisy samples. Draw $n$ noise samples, evaluate the base classifier on each $x + \varepsilon_i$ for $i = 1, \ldots, n$, and count votes for the top two classes. Call the top-class vote count $N_A$ and the second-class count $N_B$.

The true probability $p_A = \mathbb{P}[f(x+\varepsilon)=c_A]$ is unknown. Use $N_A$ to estimate it. For a one-sided confidence interval at level $1 - \alpha = 0.999$, the Clopper-Pearson method computes a lower confidence bound $\underline{p}_A$ and an upper confidence bound $\overline{p}_B$ using binomial tail probabilities. These bounds hold with high probability over the random noise draws.

The certified radius formula is $R = \frac{\sigma}{2}(\Phi^{-1}(\underline{p}_A) - \Phi^{-1}(\overline{p}_B))$, where $\Phi^{-1}$ is the inverse Gaussian CDF. This radius is guaranteed by the Neyman-Pearson lemma. If no adversarial perturbation with norm below $R$ can flip the predicted class, the smoothed classifier is certified robust to any such perturbation.

Abstention occurs when $\underline{p}_A \leq 0.5$. If even the lower confidence bound on the top class does not exceed 50 percent, the smoothed classifier does not make a certified prediction. Abstaining here ensures the soundness guarantee: no claim extends beyond the confidence bound.

Choosing $n$ affects the tightness of the confidence interval. With $n = 100$, the confidence interval is wide and the certified radius may be small. With $n = 10000$, the bound tightens and the radius grows. A typical choice is $n = 1000$ to balance computation time and certificate tightness.

### Certified versus empirical accuracy

Certified accuracy at radius $r$ is the fraction of test examples for which the smoothed classifier is certified correct at radius $\geq r$. Plot this quantity across different radii to visualize the accuracy-robustness trade-off.

Noise level $\sigma$ controls this trade-off. Higher $\sigma$ increases certified radius because the Gaussian measure widens and the confidence bounds separate further. However, higher $\sigma$ also adds stronger noise to inputs during training and testing, degrading the base classifier accuracy. A small $\sigma$ like 0.12 gives tight confidence bounds and tighter certified radii. A large $\sigma$ like 0.50 gives wider certified radii but also steeper accuracy drops.

The base classifier must be trained with Gaussian data augmentation at the same $\sigma$ value used for certification. Training with Gaussian noise improves the base classifier's tolerance to noise, so it makes better decisions on noisy inputs. Mismatch between training noise and certification noise degrades performance. Plot certified accuracy versus $r$ for multiple $\sigma$ values to see how each regime trades off robustness and accuracy.

## Lab

See the lab notebook at [./lesson_17/lab.ipynb](./lab.ipynb). The notebook implements the Monte Carlo certification procedure, computes certified accuracy at multiple noise levels, and visualizes the certified accuracy versus radius trade-off.

## Summary

This lesson introduced randomized smoothing as a certified defense against adversarial perturbations. Students learned how the smoothed classifier uses Gaussian noise and majority voting, understood the Neyman-Pearson justification for Gaussian noise, and applied hypothesis testing with Clopper-Pearson bounds to compute certified radii. The trade-off between noise level, certified radius, and base classifier accuracy now becomes explicit through empirical curves.

## Useful References and Resources

- Cohen, J., Royer, E., Gowal, S., and Saleem, A. Certified Adversarial Robustness via Randomized Smoothing. ICML 2019. The foundational paper defining the certification guarantee and the hypothesis-testing procedure.
- Neyman, J., and Pearson, E. On the Problem of the Most Efficient Tests of Statistical Hypotheses. Philosophical Transactions of the Royal Society A, 1933. The theoretical underpinning of the optimal test for Gaussian distributions.
- Clopper, C., and Pearson, E. The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial. Biometrika, 1934. Reference for confidence interval construction.
- RobustBench documentation: <https://robustbench.github.io>. Models trained with randomized smoothing and baseline certified accuracy values.
