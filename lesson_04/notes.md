# 04 — Advanced Deep Architectures (CNNs, RNNs, Transformers)

## Abstract
This lesson covers deep architectures for grid data and sequences. Students learn how local filters encode spatial structure, how recurrent cells keep memory across time, and how attention lets a model compare all positions in a sequence at once. The lecture ends with the Transformer stack and the role of positional encoding.

## Objectives
- Explain why convolution is parameter-efficient for spatial data and calculate a receptive field size for a given network depth.
- Extract and visualize intermediate feature maps from a convolutional layer and interpret the spatial patterns each map responds to.
- Distinguish attention-based from recurrence-based sequence modeling and state at least two scenarios where Transformers outperform RNNs.
- Fine-tune a pre-trained ResNet-18 on CIFAR-10 and visualize feature maps from the first and last convolutional layers.

## Content
### Convolutional neural networks
Images and other grid-structured data contain local patterns across the field of view. A 2D convolution applies a kernel to small windows of an image and produces a feature map that records where a pattern appears. The operation uses a sliding window, so each output location combines a local region of the input with a shared filter. Shared weights give convolution parameter efficiency compared with a fully-connected layer, because one filter detects a pattern across many positions instead of using a separate weight for each pixel-location pair.

A kernel can change behavior through stride and padding. A stride larger than 1 reduces the spatial size of the output and increases the effective spacing between receptive field centers. Padding adds border values so output dimensions stay larger or remain aligned. Pooling layers then reduce spatial resolution by taking a max or average over local neighborhoods. Max pooling keeps the strongest activation in each region and gives a degree of translation invariance, while average pooling smooths local evidence.

The receptive field grows with network depth. Each new layer adds neighboring pixels from the previous layer, so a stack of convolutions can aggregate information from a broader region. Feature maps often encode richer concepts as depth increases. Early layers respond to edges, corners, and textures. Later layers combine those cues into parts, objects, and scene-level contexts. A standard CNN block follows a simple pattern: convolution, activation, and pooling, with multiple blocks stacked to build a hierarchy of features.

### Recurrent architectures
Sequence data arrives one token at a time, so a model needs memory from earlier positions. A vanilla RNN updates a hidden state with the current input and the previous state:

$$
h_t = tanh(W_x x_t + W_h h_{t-1} + b)
$$

This recurrence makes the model sensitive to order. Backpropagation through time sends gradients across many time steps, and repeated matrix multiplications can shrink the signal toward zero. This effect is the vanishing gradient problem. Early information loses influence as the sequence grows, so long-range dependencies become hard to learn.

LSTM cells address this issue by adding a dedicated cell state that acts as a gradient highway. The input gate decides which new information enters the memory, the forget gate decides which information leaves, and the output gate decides which part of the cell state affects the hidden state. This design gives the network a stable path for information to persist over many time steps. A GRU uses a lighter gate structure with update and reset gates, which reduces parameter count while keeping the core idea of selective memory.

### Attention mechanism
RNNs compress the current context into a hidden state, and this bottleneck hurts long-range reasoning. Attention solves this by allowing each position to look directly at all positions in the sequence. The scaled dot-product attention formula is

$$
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
$$

The query vectors ask which information matters, the key vectors describe what each position offers, and the value vectors carry the content passed forward. The softmax weights produce a context vector that emphasizes relevant positions and suppresses less useful ones.

Multi-head attention expands this idea by splitting the representation into multiple heads. Each head learns a separate projection of the queries, keys, and values, so the model can attend to different kinds of relationships in parallel. This design is valuable for long sequences because the model does not depend on one hidden state to summarize everything. Attention can model direct dependencies across far-apart tokens and is a core reason Transformers outperform recurrent models in many sequence tasks.

### Transformer architecture
The Transformer architecture is built from stacked blocks of attention and feed-forward computations. An encoder block applies multi-head self-attention, adds a residual connection and layer normalization, passes the result through a feed-forward network, and applies another residual connection and normalization. The output is a contextualized representation for each token that incorporates information from the full sequence.

A decoder block adds masked self-attention and cross-attention. Masking prevents a token from seeing future positions during training, so generation remains causal. Cross-attention links the decoder state to the encoder state in encoder-decoder architectures, which is useful when the decoder reads a source sequence and produces a target sequence. Positional encoding injects order information because the attention mechanism itself is permutation-invariant. The standard sinusoidal formulation uses

$$
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))\\
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
$$

This keeps the encoding relative to position and preserves a simple geometric pattern across positions.

Transformer variants differ by how they use encoder and decoder stacks. Encoder-only models attend to the full input and are used for sequence understanding tasks. Decoder-only models generate outputs one token at a time and are common in autoregressive text generation. Encoder-decoder models combine both directions and are useful for sequence-to-sequence tasks. This lesson treats the Transformer as a general architecture for attention-based reasoning across many data types and tasks.

## Summary
This lesson connected spatial modeling and sequence modeling under one deep-learning lens. Students saw how local filters create feature maps in CNNs, how recurrent cells handle temporal memory and vanishing gradients, and how attention provides direct dependencies across long sequences. The final section tied these ideas together in the Transformer stack and showed how positional encoding and attention blocks reshape modern architecture design.

## Useful References and Resources
- Goodfellow, I., Bengio, Y., and Courville, A. Deep Learning. Chapters on convolutional networks, recurrent networks, and attention models.
- Vaswani, A., Shazeer, N., Parmar, N., et al. Attention Is All You Need. The foundational Transformer paper.
- PyTorch documentation for `torchvision.models.resnet18`, `Conv2d`, and feature-map extraction via forward hooks.
- LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-Based Learning Applied to Document Recognition. Early work on convolutional feature learning and receptive fields.
