# 05 — Trends in AI: Generative AI & Foundation Models

## Abstract
Generative models learn to produce new samples from learned structure. This lesson compares autoencoders, GANs, diffusion models, large language models, and agentic workflows. The comparison focuses on training objectives, failure modes, sample quality, stability, and tool use.

## Objectives
- Explain the autoencoder training objective and describe how the bottleneck latent space produces compressed representations.
- Describe the GAN training objective, identify the failure mode of mode collapse, and name at least one mitigation strategy.
- Compare diffusion models and GANs on sample quality vs. training stability, citing at least one concrete trade-off for each.
- Fine-tune a Hugging Face Transformer checkpoint on a new text classification task and report accuracy and F1 on a held-out test set.
- Describe the structure of an agentic workflow and give one example of a task where a tool-using agent outperforms a single-prompt LLM call.

## Content
### Autoencoders
An autoencoder has an encoder and a decoder. The encoder maps an input $x$ to a latent vector $z$. The decoder maps $z$ back to a reconstruction $\hat{x}$.

Training minimizes reconstruction loss. Mean squared error fits continuous values:

$$L_{MSE}=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2$$

Binary cross-entropy fits binary values. A bottleneck limits the latent vector's capacity. The encoder must retain information needed for reconstruction and discard repeated detail. The latent space therefore acts as a compressed representation.

Latent interpolation tests learned structure. Given latent vectors $z_a$ and $z_b$, evaluate $z(\alpha)=(1-\alpha)z_a+\alpha z_b$ for values of $\alpha$ between 0 and 1. Smooth changes in reconstructions show organized latent structure. This test does not prove perfect generation.

### Generative Adversarial Networks (GANs)
A GAN has a generator $G$ and a discriminator $D$. The generator maps random noise $z$ to a synthetic sample $G(z)$. The discriminator returns a probability for real data. Training uses a minimax objective:

$$\min_G\max_D\,\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

The discriminator learns to separate real and generated samples. The generator learns to make generated samples receive high discriminator scores. Each update changes the task faced by the other network.

Training can become unstable when the discriminator overwhelms the generator. Gradients can become too weak for useful generator updates. Mode collapse occurs when many noise vectors produce samples from one small output region. A class check, sample grid, or pairwise feature-distance check can detect repeated outputs.

Minibatch discrimination gives the discriminator information about sample diversity across a batch. The generator then receives pressure to produce varied samples. Wasserstein loss provides a named alternative objective with a smoother training signal. This lesson uses both approaches conceptually and does not derive Wasserstein GANs.

### Diffusion models
A diffusion model adds Gaussian noise to data over $T$ forward steps. The process begins with $x_0$ and ends with a nearly standard normal sample:

$$x_T\sim\mathcal{N}(0,I)$$

A denoising network learns the reverse process. Starting from noise, the network predicts a less noisy state at each step until the network recovers an estimate of $x_0$. Score matching gives the intuition: learn the direction toward higher data density at each noise level.

GANs generate a sample through a learned generator pass. Diffusion models use many reverse denoising steps. Diffusion models usually provide greater diversity and at least comparable sample quality, while training stays more stable than GAN training. Sampling takes longer because the reverse process uses multiple steps. The trade-off is diversity and stability versus sampling speed.

### Large Language Models (LLMs)
A GPT-style model predicts the next token from preceding tokens. For a token sequence $x_1,\ldots,x_T$, training minimizes:

$$L=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

Pre-training repeats this objective across large text collections. Increasing parameter count, data, and compute can produce capabilities absent at smaller scales. The lesson uses few-shot learning and chain-of-thought reasoning as examples of scale-linked emergent capabilities. A few-shot prompt supplies examples before a new task. A chain-of-thought prompt asks for intermediate reasoning before an answer.

In-context learning changes the prompt, not the model weights. The same checkpoint can perform a new task after examples or instructions enter the context. Fine-tuning changes weights through additional training. The lab uses a Hugging Face Transformer checkpoint for a two-class text classification task and reports accuracy and macro-F1 on held-out data.

### Agentic workflows
An agentic workflow repeats three stages: observe the environment, think through a plan, and act through a tool. The think stage uses an LLM call and can represent a chain-of-thought plan. The act stage can call an API or execute code. The agent observes the result and repeats the loop.

A single-prompt LLM call can answer a fixed question. A tool-using agent fits a task with changing external data or multiple dependent steps. For example, an agent can observe a request for a current sales report, call a data API, execute code to calculate totals, inspect the result, and call the API again when a required field is missing. The planning horizon sets how many dependent actions the agent attempts. Each extra action creates another point where the workflow can fail.


## Summary
Autoencoders compress inputs through a latent bottleneck and train with reconstruction loss. GANs use a generator-discriminator minimax game and can suffer from instability and mode collapse. Diffusion models trade slower sampling for diverse samples and more stable training. LLMs learn next-token prediction, support in-context learning, and can provide plans for agent loops built from observation, thought, and tool action.

## Useful References and Resources
- Goodfellow et al., "Generative Adversarial Nets," Sections 1 and 2, for the GAN objective and training roles.
- Kingma and Welling, "Auto-Encoding Variational Bayes," Sections 1 and 2, for encoder-decoder latent representations.
- Ho, Jain, and Abbeel, "Denoising Diffusion Probabilistic Models," Sections 2 and 3, for forward noise and reverse denoising.
- Hugging Face Transformers documentation, "Text classification," for checkpoint fine-tuning with `Trainer`.
- Brown et al., "Language Models are Few-Shot Learners," Abstract and Sections 1 and 2, for in-context learning.
