# 10 — Black-Box Evasion Attacks

## Abstract

Black-box evasion attacks operate when a model reveals only scores or labels. Zeroth-order estimation approximates a gradient from score queries, and score-based attacks such as NES and Square Attack turn the estimate into a viable adversarial example under a query budget. This lesson covers black-box attack settings, decision-based Boundary Attack behavior, and the query-efficiency metrics used to compare attack strength.

## Objectives

- Explain how zeroth-order finite-difference estimation approximates gradients without model access, and describe why NES reduces variance compared with a single finite-difference query.
- Describe the Square Attack update strategy and explain why the method avoids gradient estimation.
- Explain how the Boundary Attack starts from a misclassified image and iteratively reduces perturbation size while staying adversarial.
- Implement the Square Attack against a mock black-box API and compare success rate vs. query budget across three query limits.
- Compare score-based and decision-based attacks on query efficiency and explain when each approach is more practical.

## Content

### Zeroth-Order Gradient Estimation

A black-box model hides weights and internals. A score-based attacker can still probe the model by sending inputs and reading the scores. The most direct estimate follows finite differences. For a loss $\mathcal{L}(x)$, the gradient along a basis direction $e_i$ follows

$$
\hat{\nabla}_x \mathcal{L} \approx \frac{\mathcal{L}(x + \mu e_i) - \mathcal{L}(x - \mu e_i)}{2\mu} e_i
$$

The estimate measures how the score moves when a small perturbation shifts one coordinate. A full estimate over a $d$-dimensional input needs roughly $2d$ score queries. This cost grows quickly as the input size grows.

Natural Evolution Strategies, or NES, lowers this cost by sampling a small population of random directions instead of measuring every coordinate separately. A large population can estimate the gradient with fewer expensive evaluations because the directions are averaged with weights tied to their score change. The estimator is a variance-reduced version of the finite-difference approach, and the final update follows a signed gradient step. The sign step gives a direct move toward a higher loss under the attack objective without requiring any internal gradient access.

### Score-Based Black-Box Attacks

Score-based black-box attacks assume the model returns class probabilities or logits. This setting matches public APIs exposing confidence scores. NES belongs to this family because the attack continually estimates a gradient from returned scores and then pushes the input toward the adversarial region.

Square Attack follows a different path. The method does not estimate a gradient. Instead, the attack samples structured square updates on the image, usually in a random location and with a fixed magnitude. A candidate patch uses a square region with value $\pm \epsilon$, and the attack accepts the update only when the target class score rises. The procedure turns the problem into a random search over structured perturbations instead of a continuous gradient descent procedure.

The square pattern matters because the geometry matches attack budgets well. Axis-aligned squares cover a large region of the image without touching every pixel. Each proposal becomes more informative than a uniform random perturbation. The attack remains simple: propose a square patch, evaluate the returned score, and keep the patch when the target score improves. Repeated proposals reach a misclassified input faster than naive full-image random search.

### Decision-Based Black-Box Attacks

Decision-based attacks assume a stricter model. The attacker sees only the final prediction label, not the rank or score. Boundary Attack fits this setting. The attack begins with an input that already crosses the decision boundary and is misclassified. In practice, the attacker starts from a large perturbation and then projects the point back into a region that leaves the sample adversarial.

The key move is a boundary walk. The attacker moves along a direction reducing the perturbation size while keeping the point on the adversarial side of the decision boundary. Each step shortens the distance to the clean input and moves the point closer to the original image. A successful boundary walk keeps the example misclassified while reducing the norm of the perturbation. The result is a small adversarial example with a wrong label.

This attack family trades access to probabilities for many more queries. The model provides only binary feedback. The attacker must search along the boundary with repeated label checks, which makes each successful example more expensive than score-based attacks.

### Query Efficiency

Query efficiency is the main metric for black-box attacks. A practical attack gets judged by success and by the number of model calls needed to achieve success. Three common measures are useful here:

- Queries per successful adversarial example, which counts the number of model calls needed for a single successful attack.
- Success rate vs. query budget, which shows how often an attack succeeds before a fixed call limit is reached.
- Median queries to first success, which summarizes the typical cost when the attack succeeds.

The comparison is clear. Score-based attacks use returned scores to focus updates and often succeed in a few hundred to a few thousand queries. Decision-based attacks need repeated label-only searches and often require tens of thousands of queries to reduce the perturbation enough for a clean-looking example. This gap matters in deployed systems, where each query incurs latency, cost, or rate limits.

## Lab

See the lab notebook at [./lab.ipynb](./lab.ipynb). The notebook implements a simplified Square Attack against a mock black-box API and reports success rate and median query cost across budgets of 1 000, 5 000, and 10 000 queries.

## Summary

Zeroth-order estimation turns a black-box model into a proxy for gradient information by querying score changes around the input. NES reduces variance by averaging many random directions, while Square Attack removes gradient estimation altogether and searches over square patches. Boundary Attack needs only labels and slowly walks along the decision boundary until the perturbation becomes small enough. Query-efficiency metrics reveal the practical gap between score-based and decision-based attacks.

## Useful References and Resources

- Ilyas, A., Engstrom, L., Athalye, A., and Lin, M. “Black-Box Adversarial Attacks with Limited Queries and Information.” ICML Workshop, 2018. The NES formulation and black-box query setting.
- Andriushchenko, M., Croce, F., Flammarion, N., and Hein, M. “Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search.” ECCV, 2020. The square-patch random-search strategy.
- Brendel, W., Rauber, J., and Bethge, M. “Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models.” ICLR, 2018. The boundary-walk method and label-only attack setup.
- Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. MIT Press. The finite-difference view of gradient estimation and optimizer behavior.
