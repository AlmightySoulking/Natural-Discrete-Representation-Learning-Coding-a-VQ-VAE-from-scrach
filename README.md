# Natural-Discrete-Representation-Learning-Coding-a-VQ-VAE-from-scrach

This repository contains a clean and educational PyTorch implementation of the paper:

> **Neural Discrete Representation Learning**  
> *Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu*  
> [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)


<img width="1169" height="394" alt="image" src="https://github.com/user-attachments/assets/16755461-2012-46c6-925d-e7614ede23b5" />

The implementation demonstrates **Vector Quantized Variational AutoEncoders (VQ-VAE)** – a novel approach to learning discrete latent variables in deep generative models.

---

## 📝 Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Training](#training)
- [Results](#results)
- [Dependencies](#dependencies)
- [Paper Summary](#paper-summary)
- [To-Do](#to-do)
- [Citation](#citation)
- [License](#license)

---

## 📌 Introduction

Unlike standard VAEs that use continuous latent variables, **VQ-VAE** learns a discrete latent representation through **vector quantization**, enabling better compression and structure learning in the latent space. This model is especially useful in **image, audio, and video generation**.

---

## 🧠 Model Architecture

The model consists of:

- **Encoder**: Maps the input image to a continuous latent space.
- **Codebook (Embedding Space)**: Discrete embedding vectors that act as learned latent variables.
- **Vector Quantizer**: Replaces the continuous encoder output with the nearest codebook vector.
- **Decoder**: Reconstructs the image from the quantized latent codes.

The loss function is composed of:
- **Reconstruction Loss:** $ \log p(x|z_q(x))$ ensures our output looks like the input and we will use Mean Squared Error
- **Codebook Loss:** $||sg[z_e(x)] - e||_2^2 $ will update our codevectors in the codebook by moving them closer to the output of the encoder $z$, while not drilling down into the $min$ function by placing a stop gradient on $z$
- **Commitment Loss:** $\beta||z_e(x) - sg[e]||_2^2$ is exactly the opposite of our codebook loss, but ensures the output of the encoder is close to our codevectors, and has a weight $\beta$ to allow for some divergence.

---


Clone the repo:

```bash
git clone https://github.com/your-username/Neural-Discrete-Representation-Learning.git
cd Neural-Discrete-Representation-Learning
