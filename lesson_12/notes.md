# 12 — Adversarial Attacks Beyond Vision: NLP & Audio

## Abstract

Adversarial attacks are not limited to image classifiers. Text and audio models face distinct adversarial threats, but attacking them requires different approaches. This lesson covers word-substitution attacks on NLP models and imperceptible perturbations on speech recognition systems, introducing attack algorithms tuned to the discrete and temporal nature of language and audio.

## Objectives

- Student can articulate why gradient-based image attacks do not directly apply to NLP and what makes text perturbation discrete.
- Student can describe the TextFooler word importance ranking and semantic similarity constraint.
- Student can describe how BERT-Attack uses a masked language model to generate coherent adversarial substitutions.
- Student can explain the psychoacoustic masking mechanism that makes audio adversarial perturbations imperceptible.
- Student can run TextAttack to generate word-substitution adversarial examples and report ASR, substitution rate, and USE similarity.

## Content

### Hook: Why Text and Audio Are Different

Imagine a sentiment classifier trained on movie reviews. You input a sentence: "I loved this film." The model predicts positive sentiment with 95% confidence. Now apply FGSM, the gradient-based attack from m08. Compute the gradient of the loss with respect to the input, step in that direction, and add a small continuous perturbation to each word vector.

The result is not "I hated this film." The result is gibberish. FGSM assumes you can smoothly update input coordinates and that small changes remain valid. Neither assumption holds for text. Words are discrete tokens, not points in continuous space. Adding 0.001 to the embedding of "loved" does not produce "hated" when you discretize back to token IDs.

This is the core challenge of NLP adversarial attacks: the input space is combinatorial, not continuous. You cannot use gradient descent directly. You must search over valid word substitutions, rank them by impact, and greedily select the best ones. The attacks in this lesson solve this problem with different strategies.

### Discrete Perturbation Spaces

Text input is tokenized into a sequence of discrete tokens. A tokenizer breaks "I loved this film" into tokens ["I", "loved", "this", "film"]. Each token maps to an integer ID in a vocabulary, typically 30,000 to 100,000 tokens for a large language model.

Perturbations in text space are also discrete. You cannot add 0.5 to a token ID. You can only substitute one token for another, delete a token, or insert a new one. An adversarial example is a sequence of tokens that differs from the original by a small number of substitutions, typically 1 to 10 word changes across a sentence of 10 to 20 words.

Contrast this to image space. FGSM adds continuous values to every pixel. A 224×224 RGB image has 150,528 coordinates, and FGSM can perturb all of them. A 20-word sentence has only 20 coordinates. In vision, small perturbations applied to many dimensions are imperceptible. In NLP, changing 1 word out of 20 is a 5 percent change. Humans notice word changes far more readily than they notice imperceptible pixel changes.

This difference drives the design of NLP attacks. Gradient-based optimization works poorly in discrete spaces. Instead, NLP attacks use ranking heuristics and constraint satisfaction. Two strategies dominate: deletion-based importance ranking (which word matters most to the model?) and language-model-based coherence (which substitution preserves grammaticality and meaning?).

### TextFooler Algorithm

TextFooler ranks words by deletion impact and uses semantic similarity to constrain substitutions. Here is the full procedure.

**Step 1:** Rank words by importance. For each word in the input sentence, remove it, run the model, and measure the change in output probability for the target class. A larger probability drop means the word is more important to the model's prediction. Compute an importance score for each word. Sort words by descending importance score.

**Step 2:** Generate candidate substitutions for each word. Use counter-fitted embeddings, a word embedding space trained to preserve semantic and antonymic structure. Counter-fitted embeddings place synonyms close together and antonyms far apart. For each word, find the N nearest neighbors in counter-fitted embedding space. These neighbors are candidate replacements.

**Step 3:** Rank and filter candidates. For each candidate substitution, compute semantic similarity between the original word and the candidate using cosine similarity in the counter-fitted embedding space. Discard candidates with similarity below a threshold, typically 0.5. Also discard candidates that change the part-of-speech tag (e.g., do not substitute a noun with a verb). This filtering enforces grammatical and semantic coherence.

**Step 4:** Greedy substitution. Process words in descending importance order. For each word, try each filtered candidate substitution, evaluate the resulting sentence on the model, and check if the model's prediction has flipped. If yes, perform the substitution and move to the next word. If no candidate causes a flip, move to the next word without substituting. Continue until the model prediction flips or all words have been processed.

The result is an adversarial example with the minimum number of word substitutions needed to flip the model's prediction. The algorithm guarantees that each substitution preserves semantic similarity and part-of-speech. In practice, TextFooler achieves high attack success rates (60–80 percent on sentiment classifiers) with an average of 2–3 word substitutions per example.

TextFooler fails in some scenarios. If a word is critical and the counter-fitted embeddings contain no high-similarity synonyms, TextFooler may not find a valid substitution. If a word is an out-of-vocabulary (OOV) term or a misspelling, no embeddings exist. If the model relies on multiple words with similar importance, for example "definitely" and "absolutely" both strongly signal positive sentiment, then removing one leaves the other intact.

### BERT-Attack Algorithm

BERT-Attack improves on TextFooler by using a masked language model (MLM) to generate context-aware candidate substitutions. Instead of looking up nearest neighbors in a fixed embedding space, BERT-Attack masks each word and asks a BERT model to predict what word should go there.

Here is the procedure.

**Step 1:** Identify important words. Like TextFooler, rank words by deletion impact.

**Step 2:** Generate context-aware candidates using BERT MLM. For each word, mask it in the sentence (replace it with `[MASK]`). Pass the masked sentence to BERT. BERT outputs a probability distribution over all vocabulary items for the `[MASK]` token. Sample or select the top-K alternatives. These candidates are context-aware because BERT conditions on the surrounding words.

**Step 3:** Rank and filter. Apply the same semantic similarity and POS filter as TextFooler, using counter-fitted embeddings or a separate semantic similarity model. This ensures candidates preserve meaning and grammatical function.

**Step 4:** Greedy substitution. Process words in descending importance order. For each word, try each filtered candidate, evaluate the model, and perform the substitution if the prediction flips. Otherwise, move to the next word.

BERT-Attack often outperforms TextFooler because the MLM-generated candidates are tuned to local context. Consider the sentence "The bank held a meeting." If you mask "bank", TextFooler might suggest "shore" or "edge" (nearest neighbors to "bank" in embedding space, all meanings of the word). BERT-Attack predicts that "bank" should be replaced with words like "company", "institution", or "office" (words that fit the context "The `[MASK]` held a meeting"). The context-aware candidates are more likely to preserve sentence meaning and thus pass the semantic similarity filter.

However, BERT-Attack is slower than TextFooler because running BERT for each word incurs inference latency. In practice, BERT-Attack achieves 70–85 percent attack success rates, higher than TextFooler, at the cost of 5–10x longer wall-clock time.

### Acoustic Adversarial Examples

Audio adversarial examples target automatic speech recognition (ASR) systems, which convert audio waveforms to text transcripts. The attack goal is to make the ASR model transcribe a target phrase instead of the original speech.

The threat model differs from NLP attacks. Audio is continuous, not discrete. Perturbations are small, continuous additions to the audio waveform. However, imperceptibility is the key challenge. Humans have finely tuned hearing and notice distortion readily. The attack must exploit psychoacoustic masking, the phenomenon where human ears fail to perceive certain sounds in the presence of masking signals.

Psychoacoustic masking has two components. Frequency masking occurs when a loud sound at a certain frequency makes nearby frequencies inaudible. For example, a 4000 Hz tone at high volume masks adjacent frequencies between 3500 and 4500 Hz. Temporal masking occurs when a sound makes nearby sounds in time inaudible. A loud sound can mask quieter sounds that occur within 100–200 milliseconds before or after it.

An acoustic adversarial attack exploits both types of masking. The attack adds carefully shaped perturbations to the audio waveform such that the perturbations are masked by loud regions of the original audio. Humans cannot hear the perturbations, but the ASR model processes the entire waveform and recognizes the perturbations as features.

The optimization procedure uses connectionist temporal classification (CTC) loss, the standard loss for ASR training. CTC aligns sequences of audio frames to sequences of characters or phonemes, allowing the model to learn without frame-level label alignment. An adversarial attack computes the CTC loss for a target transcript on the original audio. This loss measures how far the model is from recognizing the target. Gradient descent updates the audio waveform to minimize this loss, bringing the ASR model closer to recognizing the target phrase.

The process is: (1) define a target transcript, for example "please call 911". (2) Start with the original audio waveform. (3) Compute the CTC loss for the target transcript. (4) Compute gradients of the loss with respect to the audio samples. (5) Take a small step in the gradient direction to update the audio. (6) Clip the perturbations to ensure they remain within psychoacoustic masking thresholds. (7) Repeat until the ASR model outputs the target transcript.

The clipping step is critical. Without clipping, the attack would add unbounded perturbations and the resulting audio would be distorted. With clipping, perturbations remain within the masking thresholds learned from human hearing studies. In practice, acoustic attacks achieve 70–90 percent success rates on commercial ASR systems with perturbations that are imperceptible to human listeners.

### NLP Attack Evaluation Metrics

Evaluating NLP adversarial examples requires multiple metrics because discrete text perturbations have different properties than continuous image perturbations.

**Attack success rate (ASR)** is the fraction of examples for which the attack flipped the model's prediction to a target or untargeted misclassification. For a sentiment classifier, if 80 out of 100 examples were successfully misclassified, ASR is 0.80. This metric is model-dependent: an attack with high ASR on one classifier may fail on a more robust classifier.

**Average word substitution percentage** is the average number of words changed per example, normalized by sentence length. If an attack changed 2 words in a 10-word sentence, the substitution percentage is 0.20 or 20 percent. Across 100 examples, if the mean is 0.15, report it as 15 percent. This metric quantifies perturbation size. Smaller perturbation size (fewer word changes) is better because smaller perturbations are more stealthy and more likely to preserve sentence meaning.

**Universal Sentence Encoder (USE)** semantic similarity measures how much the adversarial sentence's meaning has changed relative to the original. USE is a deep neural network that encodes sentences into 512-dimensional embeddings. Cosine similarity between the original and adversarial embeddings ranges from -1 to 1, with 1 meaning identical meaning. Report USE similarity as a decimal between 0 and 1, for example 0.92. Higher similarity is better. It indicates the adversarial example still conveys the original meaning, conveyed in different words.

**Grammar preservation** is a binary or scored metric. Use a grammar checker (e.g., LanguageTool) to flag grammatical errors in the adversarial example. Count the number of new errors introduced and report it as a fraction of total words. Alternatively, report it as a binary: "example is grammatically correct" or "example contains grammatical errors". Higher grammar preservation (fewer new errors) is better.

Together, these metrics form a multi-dimensional success criterion. A high-quality adversarial example has high attack success rate (the attack worked), low word substitution percentage (the change was small), high USE similarity (the meaning was preserved), and low grammar-error rate (the output is readable). An attack that achieves high ASR by replacing 50 percent of words is less effective than one that achieves the same ASR with 3 percent word changes.

## Lab

See [./lab.ipynb](./lab.ipynb).

## Summary

Text and audio models face adversarial threats fundamentally different from image classifiers because their input spaces are discrete and their imperceptibility mechanisms differ. TextFooler and BERT-Attack are two approaches to ranking word importance and generating valid substitutions. BERT-Attack's use of masked language models produces more context-aware candidates at higher computational cost. Acoustic attacks exploit psychoacoustic masking to embed imperceptible perturbations into audio. Evaluation requires multiple metrics: attack success rate, word substitution percentage, semantic similarity, and grammar preservation. These metrics together measure perturbation size and semantic fidelity.

## Useful References and Resources

- Chen, P. Y., Zhang, H., Sharma, Y., Yi, J., & Hsieh, C. J. (2019). "ZOO: Zeroth Order Optimization Based Black-box Attacks to Deep Neural Networks without Training Substitute Models." Proceedings of the 2019 ACM Workshop on Artificial Intelligence and Security. Focus on Section 3 for discrete optimization in non-vision domains.
- Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment." AAAI Conference on Artificial Intelligence. Covers both TextFooler and BERT-Attack algorithms with implementation details.
- Carlini, N., Papernot, N., Mishra, P., Goodfellow, I., & McDaniel, P. (2017). "Towards Evaluating the Robustness of Neural Networks." IEEE Symposium on Security and Privacy. Section 4 extends to audio and discusses CTC-loss-based adversarial examples.
- Morris, J. X., Lifland, E., Yoo, J. Y., Bailey, J., Prabhakaran, V., & Wang, Y. (2020). "TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Reference for library usage and recipe implementations.
- TextAttack documentation: "Recipes and attack methods." Available at textattack.readthedocs.io. Provides example code and hyperparameter defaults for TextFooler and BERT-Attack.
- Carlini, N., Mishra, P., Vaidya, T., Zhang, Y., Sherr, M., Shields, C., ... & McDaniel, P. (2016). "Audio Adversarial Examples: Targeted Attacks on Speech-to-Text." Workshop on Security and Privacy in Machine Learning. Foundational work on CTC-loss attacks and psychoacoustic masking constraints.
