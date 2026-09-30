# 13 — Physical-World Attacks & Universal Perturbations

## Abstract

Digital adversarial attacks like FGSM and PGD fail when printed and placed in the real world because transformations such as rotation, scale, and lighting break the pixel-level adversarial pattern. This lesson covers optimization under transformation variance using Expectation Over Transformation (EOT), adversarial patch design with printability constraints, and Universal Adversarial Perturbations (UAPs) fooling classifiers across any image. You will learn to build robust physical attacks surviving the translation from screen to printed paper to camera feed.

## Objectives

- Explain how Expectation Over Transformation (EOT) makes a perturbation robust to rotation, scale, and lighting changes by optimizing over a distribution of transformations.
- Generate a printable adversarial patch using EOT optimization with Non-Printable Score (NPS) printability loss.
- Measure the fooling rate of an adversarial patch across a diverse image set.
- Describe the stop-sign misclassification attack and its physical validation methodology.
- Construct a Universal Adversarial Perturbation (UAP) achieving an above-threshold fooling rate across an entire test set.
- Compare patch-based and full-image UAP attacks by coverage, printability, and deployment mode.

## Content

### Physical-World Attack Motivation and Challenges

Digital adversarial examples break in the physical world. An FGSM perturbation fooling a ResNet in pixel space fails once the image is printed, rotated, or photographed at a distance. Why? The pixel coordinates of the adversarial pattern no longer align with the pixel coordinates seen by the camera. Lighting changes the pixel values. Viewing angle rotates the entire image. Distance shrinks or enlarges the pattern.

The gap between digital and physical attacks is transformation variance. A perturbation optimized against one image in one viewing condition generalizes poorly to the same object under rotation by 10 degrees, scale change of 20 percent, or indoor lighting differing from the lab setting. Researchers (Eykholt et al. 2018) demonstrated this with a physical stop sign: they printed stickers on a real stop sign and drove cars toward it. Autonomous vehicle classifiers trained on COCO, KITTI, or ImageNet misclassified the sign at 100 percent rate when viewed from the street, even though white-box FGSM attacks on the same sign's image fail in the physical deployment.

The core challenge: optimization must not assume fixed geometry and lighting. Instead, the adversary must optimize a perturbation remaining adversarial across a distribution of possible transformations. This distribution includes rotation (e.g., 0 to 15 degrees in each axis), scale (0.8x to 1.2x), brightness jitter (0.9x to 1.1x intensity), and viewpoint shift (if the camera angle varies). The objective shifts from minimizing loss on one image to minimizing expected loss over many transformed versions of the same image.

### Expectation Over Transformation (EOT) Formulation

Expectation Over Transformation (EOT) formulates robust optimization as an expectation over a distribution of transformations. Given a classifier $f$, a true label $y$, and an adversarial perturbation $\delta$, the EOT objective is:

$$\mathbb{E}_{t \sim T}[\mathcal{L}(f(t(x+\Delta)), y)]$$

Here, $T$ is a distribution of transformations, $t$ samples from $T$, $x$ is the original image, $x+\delta$ is the perturbed image, $t(x+\delta)$ applies the transformation to the perturbed image, $f$ is the classifier, and the loss $L$ measures how far the classifier's output diverges from the true label. The expectation averages loss over many sampled transformations.

To optimize under this expectation, the adversary samples $K$ transformations per gradient step (e.g., K=10). For each transformation $t_i$, the adversary computes the loss $L(f(t_i(x+\delta)), y)$. The gradient accumulates across all $K$ losses and updates delta. This Monte Carlo approximation makes the gradient descent algorithm work: instead of computing the exact expectation (impossible for continuous distributions), the algorithm samples a batch and takes a step in the direction of the average gradient.

Not all transformations are differentiable. Rotation by a continuous angle is differentiable through a geometric transformation matrix. Nearest-neighbor interpolation (used in some image libraries) is not. EOT implementations use differentiable resizing (bilinear or bicubic), differentiable rotation matrices, and differentiable brightness scaling. Library functions like PyTorch's affine transformation or TensorFlow's image.rotate are differentiable by default.

The distribution T typically includes four transformation types: rotation in the range [-15, 15] degrees, scale factor in [0.8, 1.2], brightness jitter in [0.9, 1.1], and sometimes translation by [-5, 5] pixels. Each transformation is sampled independently per gradient step. This forces the perturbation $\delta$ to work across the envelope of expected real-world variation.

### Adversarial Patch Design

A patch is a small, contiguous rectangular region of pixels. The adversary modifies this region. The patch is smaller than the full image (e.g., 32x32 pixels in a 224x224 image). The adversary places the patch at a random location on the target object or scene. The adversary chooses patch pixels so the overlaid patch causes misclassification.

Patch optimization introduces two constraints absent from full-image perturbation. First, only pixels inside the patch region can be modified. A binary mask $M(x,y)$ equals 1 inside the patch and 0 outside. The perturbed image is $x + \delta * M, not x + \delta$. Second, the patch must be printable. Printers cannot reproduce every color in the RGB space. When an RGB pixel (240, 50, 100) is sent to a printer, the actual printed color is often (200, 70, 110), smearing hue and saturation. The Non-Printable Score (NPS) quantifies how far a patch's colors deviate from what a printer can reproduce.

The NPS loss uses a printer simulation model. One approach fits a lookup table from natural RGB colors to printed colors by measuring actual printer output. Another fits a neural network to predict printed colors. The simplest version constrains pixel values to a safe range: clip all patch pixels to [50, 200] in each channel. Printers reproduce this range reliably.

The joint optimization objective combines adversarial loss and printability loss:

$$\text{total loss} = \mathcal{L}_{\text{adv}} + \lambda \cdot \mathcal{L}_{\text{NPS}}$$

Lambda is a weight (e.g., 0.01 to 0.1) trading off fooling power against printability. High $\lambda$ enforces strict printability at the cost of lower attack success rate. Low $\lambda$ allows harsh colors but ensures the attack works after printing.

Patch placement strategy matters. The adversary can place the patch at a fixed location (e.g., top-left corner of a stop sign) or sample a random location within the target region per iteration. Random placement during training makes the patch robust to placement variation at test time. During EOT transformations, the rotation operation transforms the patch along with the image, so the patch itself survives rotation.

The algorithm alternates three steps per iteration: (1) sample K transformations and K random patch placements, (2) compute loss for each, (3) backpropagate through the average loss and update patch pixels via gradient descent (e.g., Adam optimizer with learning rate 0.01). After each update, clip patch pixels to a safe printability range. Typical training runs for 100 to 500 iterations until the fooling rate stabilizes.

### Physical Adversarial Attack Case Studies

Eykholt et al. 2018 demonstrated the first large-scale physical adversarial attack: a modified stop sign fooling object detectors at 100 percent rate at real-world viewing distances. They printed a pattern of yellow and black stickers on a physical stop sign. A vehicle equipped with a camera drove toward the sign. The detector (YOLO v2, Faster R-CNN, SSD) failed to recognize the sign as a stop sign in 100 percent of video frames from a range of 10 meters to 1 meter away. The attack persisted across viewpoint angles spanning 15 to 75 degrees and across different lighting conditions.

The researchers validated the attack using a simulator first. They rendered the stop sign in a 3D graphics engine (CARLA, GTA), added the adversarial pattern, applied random affine transformations, and tested the detector. Once the simulator showed 100 percent misclassification, they printed the pattern and mounted it on a real sign. Real-world results matched predictions: near 100 percent failure rates.

This work established two validation methodologies. First, simulate in 3D: place the perturbed object in a 3D scene, render it from many viewpoints, apply camera blur and noise, and test the detector. Second, deploy physically: print the attack, mount it on a real target, and measure the detection rate with a real camera and detector in uncontrolled outdoor lighting.

Transferability across detectors was also high. A patch optimized against Faster R-CNN (using white-box access) misclassified objects in YOLO and SSD at similarly high rates. This cross-model transfer motivated the use of expectation over transformation: by optimizing over many transformations and placements, the patch became transferable by accident.

Face recognition evasion attacks follow a similar pattern. Sharif et al. 2016 designed eyeglass frames fooling face recognizers. Researchers printed adversarial patterns on the frames. Systems like FaceNet and VGGFace misidentify or reject people wearing the frames. The frames survive rotation (person turns their head), scale (photo taken at different distances), and lighting. These attacks used EOT-style optimization to handle the variation in head pose and photography distance.

### Universal Adversarial Perturbations (UAPs)

A universal adversarial perturbation is an image-agnostic perturbation. It fools a classifier on any input image. Whereas an adversarial patch is paired with a single target object, a UAP is a single perturbation $\delta$. When added to any image, it produces a high probability of misclassification across the image set.

The UAP is defined formally as a perturbation $\delta$ bounded in the $L_{\infty}$ norm: (e.g., $\epsilon=8/255$). When added to any natural image $x$, $f$ misclassifies the perturbed image $x+\delta$ with probability exceeding a threshold (e.g., 80 percent fooling rate). The constraint $||\delta||_{\infty} <= \epsilon$ ensures the perturbation remains small and imperceptible.

UAPs generalize across images because they exploit a global property of the classifier. Classifiers are sensitive to orthogonal feature space directions independent of specific image content. Perturbations moving representations away from correct class clusters in one image also move representations away in other images. UAPs thus capture classifier vulnerabilities transcending individual inputs.

UAP construction uses an iterative algorithm based on DeepFool. For each training image $x_i$, compute the DeepFool perturbation minimizing the $L_{\infty}$ distance to the decision boundary (requires small budget $\epsilon_i$). Then, accumulate and project: add the DeepFool perturbation from $x_i$ onto a running universal perturbation, clip to the $L_{\infty}$ ball of size $\epsilon$, and move to the next image. After iterating over the entire training set multiple times, the accumulated UAP fools images even outside the training set (good transfer) and approaches the desired fooling rate threshold.

The algorithm is:

1. Initialize universal perturbation $\delta$ to zero.
2. For each image $x_i$ in the training set, compute DeepFool perturbation $r_i$ (the minimal perturbation to cross the decision boundary).
3. Proposed update: $\delta' = \delta + r_i$.
4. Project: $\delta = clip(\delta', \epsilon$) in the $L_{\infty}$ norm.
5. Repeat over the training set until fooling rate exceeds 80 percent.

UAPs achieve high fooling rates: Moosavi-Dezfooli et al. 2017 reported 100 percent fooling on CIFAR-10 and 75+ percent on ImageNet with $\epsilon=8/255$. Unlike per-image attacks, UAPs require no knowledge of the test image: apply the same $\delta$ to any image and expect near-threshold fooling rate.

### Comparison of Patch vs. Full-Image UAP

Patches and UAPs differ in coverage, printability, and deployment.

Coverage: A patch modifies a small region (32x32 or 64x64 pixels) inside a single image or object. A full-image UAP (or UAP applied to the entire image) modifies all pixels, spreading perturbation across the entire image. To achieve comparable fooling rates, patches often require larger perturbation magnitudes (more saturated colors) than UAPs, because the perturbation is concentrated in one place.

Printability: A patch can be printed as a standalone sticker or pattern and physically deployed (glued to a sign, worn as a badge). A full-image UAP requires modifying every pixel of a target object or camera feed in real time, which is impractical for most physical scenarios. Patches are thus the preferred physical adversarial attack mode. If a UAP is printed, only the pixels within the printed material are modified, reducing it to a spatial-constrained attack.

Deployment mode: A patch is placed on a specific object (a stop sign, a person's shirt, a car). The patch is portable and reusable across scenes. A UAP is an abstract perturbation meant to be added to any image computationally. Some UAP defenses apply UAPs to entire camera feeds in real time using video overlays or augmented reality, but this requires infrastructure outside of typical autonomous system deployment.

Fooling rate: UAPs typically achieve higher global fooling rates (75 to 95 percent) because they leverage the full image space. Patches must work across a larger distribution of viewing angles and placements, so per-patch fooling rates are often lower (40 to 80 percent) unless highly optimized. However, patches are more semantically meaningful (they appear as a physical object) and transfer to real-world scenarios more reliably.

## Summary

Physical adversarial attacks succeed by optimizing perturbations across a distribution of transformations. Expectation Over Transformation (EOT) forms this as an expectation over sampled transformations, enabling gradients to propagate through rotation, scale, and lighting variation. Adversarial patches apply EOT to small, printable regions, while Universal Adversarial Perturbations extend the same principle to image-agnostic, full-image perturbations. Both attacks achieve near-perfect fooling rates in real-world deployment. The stop-sign case study validates this result. Patches and UAPs differ in coverage, printability, and deployment feasibility, with patches dominating practical physical attack scenarios.

## Useful References and Resources

- Eykholt et al. 2018. Physical Adversarial Examples for Object Detectors. Conference paper: describes the stop-sign attack, YOLO/Faster R-CNN evaluation, and real-world validation methodology.
- Brown et al. 2017. Adversarial Patch. Conference paper: introduces patch masking, Non-Printable Score (NPS) loss formulation, and joint optimization of adversarial and printability objectives.
- Moosavi-Dezfooli et al. 2017. Universal Adversarial Perturbations. Conference paper: defines UAPs, describes DeepFool-based construction algorithm, and reports fooling rates on CIFAR-10 and ImageNet.
- Sharif et al. 2016. Accessorize to a Crime: Real and Stealthy Attacks on Face Recognition. Conference paper: applies physical adversarial principles to face recognition using eyeglass frames, demonstrates cross-model transfer.
- PyTorch Affine Transforms (torchvision.transforms.functional.affine): documentation for differentiable image rotation and scale in gradient-based optimization.
- Non-Printable Score reference implementation: GitHub repository demonstrating printer color lookup tables and NPS loss computation.
