# 18 — Exact & Deterministic Neural Network Verification

## Abstract

This lesson covers formal verification of neural networks under bounded perturbations. Unlike probabilistic certification methods that hold with high probability, deterministic verification proves that a network satisfies a property for all inputs within a given radius. Three complementary approaches appear in practice: Interval Bound Propagation for speed, linear relaxation methods (CROWN family) for tightness, and SMT-based solvers (Reluplex) for completeness. This lesson provides the tooling and theory to reason about which method suits each scenario.

## Objectives

- Student can trace Interval Bound Propagation through a two-layer ReLU network by hand, propagating input interval to output interval.
- Student can use auto_LiRPA to obtain IBP and $\alpha$-CROWN certified bounds for a small neural network under an $L_\infty$ perturbation.
- Student can state the completeness and soundness guarantees of SMT-based verification (Reluplex) versus relaxation-based methods.
- Student can explain why IBP is sound but incomplete and why this matters for certification claims.
- Student can describe the scalability limits of complete verification and when incomplete methods are preferred.

## Content

### Formal Verification Problem Statement

Verification asks: given a neural network $f$, an input $x$, a perturbation radius $\varepsilon$, and a property $P$, does $f(x')$ satisfy property $P$ for all $x'$ such that $||x' - x||∞ ≤ \varepsilon$ ? The property $P$ typically encodes a safety constraint.

Example: “$f(x')$ does not misclassify to any adversarial class” or “the output logit for the true class stays above all other logits by margin $\delta$.” Formal verification converts this logical question into a yes/no answer with a mathematical proof.

Two classes of answer exist. A verification method is sound if its answer is always correct: when it says "yes, the property holds for all perturbations," that claim cannot be false. A method is complete if it can answer yes or no for any property and network in finite time: no unanswerable cases. Most practical methods trade completeness for scalability. Randomized smoothing from lesson 17 gave a probabilistic guarantee. Now we want a definitive proof.

### Interval Bound Propagation (IBP)

Interval Bound Propagation tracks the range of possible values at each neuron. Start with the input interval $[x - \varepsilon, x + \varepsilon]$ represented as lower and upper vectors $x_{under}$ and $x_{over}$. For a linear layer with weight matrix $W$ and bias $b$, the output interval satisfies:

- $z_{under}$ = W+ @ $x_{under}$ + W- @ $x_{over}$ + b$ and
- $z_{over} = W+ @ x_{over} + W- @ x_{under} + b$,

where $W+$ contains $max(W, 0)$ and $W-$ contains $min(W, 0)$. This formula accounts for the sign of each weight: positive weights grow the interval lower bound when multiplied by the lower input, while negative weights grow it when multiplied by the upper input.

For ReLU activation, the output interval is $[max(0, z_{under}), max(0, z_{over})]$ per element. If $z_{under} ≥ 0$, ReLU is inactive and passes the interval through unchanged. If $z_{over} ≤ 0$, ReLU activates fully and outputs $[0, 0]$.
If $z_{under} < 0 < z_{over}$, the neuron is unstable and the output is $[0, z_{over}]$. Repeat this propagation forward layer by layer until the output layer.

Worked example:

two-layer ReLU network with

- $w_1 = [[1, 1], [-1, 0]]$,
- $b_1 = [0, 0]$,
- $W_2 = [[1, 1]]$,
- $b_2 = [0]$.
- Input $x = [1, 2]$ with $\varepsilon$ = 0.5, so input interval is $[0.5, 1.5]$ per coordinate.

Layer 1 weights:

- $w_1+ = [[1, 1], [0, 0]]$,
- $w_1- = [[0, 0], [-1, 0]]$.
- Lower: $[1, 1] @ [0.5, 0.5] + [0, 0] @ [1.5, 1.5] = [1, 0]$.
- Upper: $[1, 1] @ [1.5, 1.5] + [0, 0] @ [0.5, 0.5] = [3, 0]$.

So

- $z_{1_{under}} = [1, 0]$,
- $z_{1_{over}} = [3, 0]$.
- After ReLU: $[max(0, 1), max(0, 0)]$ = $[1, 0]$ and $[max(0, 3), max(0, 0)]$ = $[3, 0]$.

Layer 2:

- $W_2+ = [[1, 1]]$,
- $W_2- = []$.
- Lower: $[1, 1] @ [1, 0] = 1$.
- Upper: $[1, 1] @ [3, 0] = 3$.

Final certified interval: $[1, 3]$ for the output.

IBP is sound: the true output always lies within the certified interval. IBP is incomplete: the interval can be much wider than the true range. Over-approximation accumulates as depth increases. A network with 100 layers can produce intervals so loose that they give zero information. This is the fundamental trade-off: IBP runs in one forward pass (linear time in network size) but sacrifices bound tightness.

### Linear Relaxation Methods (CROWN Family)

CROWN and its variants ($\alpha$-CROWN, $\beta$-CROWN) replace the post-ReLU intervals with linear functions that tightly bound the true network output. The core idea: at each unstable ReLU neuron, the ReLU is sandwiched between two linear functions. Instead of using the loose interval $[0, z_{over}]$ for an unstable neuron, CROWN finds linear lower and upper bounds $\ell(z)$ and $u(z)$ that better approximate ReLU behavior.

CROWN computes these bounds via a backward pass. Start at the output layer and propagate linear constraints downward. For each ReLU, maintain an upper and lower linear function. The linear parameters (slopes $\alpha$ and intercepts) are fixed by the network geometry. $\alpha$-CROWN adds learnable slope parameters $\alpha$ per neuron. Instead of using the worst-case slope (e.g., 0 for an unstable ReLU), $\alpha$-CROWN optimizes each $\alpha$ in $[0, 1]$ to tighten the bound. The optimization runs via gradient descent on the bound gap. Tighter bounds improve robustness certification: more networks pass verification as the certified margin grows.

$\beta$-CROWN extends $\alpha$-CROWN by adding branch-and-bound splits. When $\alpha$-CROWN gives a loose bound, $\beta$-CROWN selects an unstable ReLU to split.

- Split 1 forces that ReLU on (setting $post = pre$).
- Split 2 forces it off (setting $post = 0$).

Each branch recurses with tighter constraints. $\beta$-CROWN is exponential in the worst case but produces the tightest bounds in practice. For a small network and property, $\beta$-CROWN certification often completes in seconds. For large networks, the search tree grows and compute time becomes prohibitive.

Comparison: IBP is fastest but loosest (seconds, very wide bounds). $\alpha$-CROWN is slower and much tighter (minutes on modern hardware). $\beta$-CROWN is slowest but tightest (minutes to hours on modest networks). All three are sound but incomplete. None guarantees that the property holds. They only guarantee that any certificate they produce is correct. A failed verification means the method is too weak, not that an adversarial example exists.

### SMT-Based Verification (Reluplex)

SMT solvers handle satisfiability of systems of linear arithmetic constraints with select non-linear functions. Reluplex encodes the neural network as SMT constraints. Each neuron is a variable. Weights and biases are constants. Linear layers become linear arithmetic constraints. ReLU is encoded as a disjunction: either $pre ≤ $0 and $post = 0$, or $pre > 0$ and $post = pre$. This disjunction models the two possible modes of a ReLU.

Reluplex uses the SMT solver internally to explore the constraint space. It alternates between two phases: **relaxation** and **refinement**.

- In relaxation, drop the ReLU disjunction and solve the resulting linear program.
- In refinement, check which ReLU constraints are violated and add splitting constraints to repair them.

This interplay continues until either a satisfying assignment is found (adversarial example found) or the solver proves unsatisfiability (property holds).

**Soundness** and **completeness**: if Reluplex terminates with `UNSAT`, the property is proven to hold for all perturbations within $\varepsilon$. If it terminates with `SAT`, an explicit adversarial example is returned.

If the solver runs to a configurable timeout, the result is inconclusive. Reluplex is the only practical method in this lesson that combines soundness and completeness, meaning a negative result is a proof, not just a failed verification.

The Reluplex algorithm was developed by Katz et al. (2017) for neural networks with ReLU activations. Modern SMT-based verifiers like Marabou extend Reluplex and scale to networks with several hundred neurons. The tool Reluplex is open-source and distributed as part of the Marabou verifier.

### Completeness Versus Soundness

Soundness and completeness are dual guarantees. A sound method never produces a false positive: a certificate it issues is always valid. An incomplete method can fail to certify even though the property holds. All four approaches (IBP, $\alpha$-CROWN, $\beta$-CROWN, Reluplex) are sound: if they say a property holds, it truly does.

Completeness is rarer. Only Reluplex (and related SMT-based methods) achieve both soundness and completeness: they can prove a property holds or exhibit a counterexample. IBP, $\alpha$-CROWN, and $\beta$-CROWN can fail even when a property holds because their bounds over-approximate. $\beta$-CROWN is closer to complete than $\alpha$-CROWN because branch-and-bound explores more possibilities, but both remain incomplete.

The cost of completeness is computational. Reluplex explores an exponential search space in the worst case. For a network with 10,000 stable ReLUs, the worst case requires exponential time. In practice, on networks with 1,000 to 10,000 neurons, Reluplex often finishes in reasonable time. Beyond that, it times out or runs out of memory. Incomplete methods are faster but trade certainty for speed.

### Scalability Limits

Today, exact verification scales to networks with roughly 10,000 ReLU neurons using Reluplex on a single machine with a time budget of hours. State-of-the-art incomplete methods ($\beta$-CROWN) scale to networks with 100,000 neurons or more but accept the risk of false negatives. This scalability gap reflects a fundamental complexity barrier: proving robustness for deep networks is NP-complete in general.

Hybrid approaches combine strategies. One pattern: use $\beta$-CROWN to certify the first 90% of layers without splits, then apply Reluplex to the final layers where the search space is smaller. Another: use IBP for a fast lower bound on robustness, then use Reluplex only if IBP fails to certify. A third: run $\alpha$-CROWN with timeouts. If it succeeds, declare certified. If it times out, fall back to an incomplete conclusion (unverified, not proven adversarial).

For modern architectures, scalable verification remains an open problem. Transformers and vision transformers are harder to verify than MLPs due to their size and non-monotone activations (softmax, layer norm). Verification tools are part of the ecosystem for evaluating model robustness at scale, covered in Lesson 19.

## Lab

See [./lesson_18/lab.ipynb](./lab.ipynb).

This lab exercises Interval Bound Propagation and $\alpha$-CROWN using the auto_LiRPA library. Students load a pre-trained 3-layer MLP, compute certified bounds under $L_\infty$ perturbations, and compare bound tightness between the two methods.

## Summary

Formal verification of neural networks proves robustness properties hold for all inputs within a perturbation radius, eliminating the probabilistic gaps of randomized smoothing. Three sound methods trade off completeness and scalability: IBP runs in linear time but with loose bounds, $\alpha$-CROWN and $\beta$-CROWN tighten bounds iteratively at higher cost, and Reluplex is the only complete method but scales only to ~10,000 neurons. Hybrid approaches combine these methods to balance speed and proof strength. This lesson equips students to select verification tools and understand their guarantees.

## Useful References and Resources

- Katz, G., Barrett, C., Dill, D. L., Yang, K., and Harmon, S. (2017). "Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks." In Computer Aided Verification (CAV). Springer.
- Zhang, H., Weng, T. W., Chen, P. Y., Hsieh, C. J., and Daniel, L. (2018). "Efficient Neural Network Verification with Exactness Characterization." arXiv:1807.03571. Describes CROWN and $\alpha$-CROWN algorithms.
- Wang, S., Zhang, H., Cheung, K., Hsieh, C. J., and Cheng, Y. (2021). "$\beta$-CROWN: Efficient Bound Propagation for Neural Network Verification." NeurIPS.
- auto_LiRPA library (<https://github.com/Verified-Intelligence/auto_LiRPA>). Reference implementation for IBP, $\alpha$-CROWN, and β-CROWN.
- Marabou verifier (<https://github.com/Reluplex/Marabou>). Open-source SMT-based verifier extending Reluplex algorithm.
- NIST AI RMF (Governance section on robustness verification). Describes verification as part of AI risk management.
