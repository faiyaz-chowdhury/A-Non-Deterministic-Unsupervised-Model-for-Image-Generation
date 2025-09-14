# A-Non-Deterministic-Unsupervised-Model-for-Image-Generation

## Generative Self-Organizing Map with Attention (Gen-SOM-Attn)

This repository contains the official **PyTorch implementation** of the **Generative Self-Organizing Map with Attention (Gen-SOM-Attn)** — a **novel, non-deterministic, unsupervised neural network for image generation**.  
Developed as part of **CSE425: Neural Networks**, this project introduces a fresh perspective on generative modeling.

---

## Project Overview

The **Gen-SOM-Attn** addresses limitations of existing generative models such as the **training instability of GANs** and the **blurry outputs of VAEs**. Its novelty lies in combining:

- A **deep convolutional autoencoder**  
- A **Kohonen Self-Organizing Map (SOM)**  
- An **attention mechanism**  

### Key Characteristics
- **Unsupervised:** Learns from unlabeled Fashion-MNIST data  
- **No Error-Correction:** SOM uses competitive learning rather than backprop for latent organization  
- **Interpretable:** Topologically ordered latent space with meaningful neighborhoods  
- **Non-Deterministic:** Images sampled stochastically from the latent manifold for diverse outputs  

---

## Features
- Novel architecture: Convolutional Autoencoder + SOM + Attention  
- High-quality image generation (comparable to VAE)  
- 12×12 SOM grid for structured latent mapping  
- Stable training — no adversarial objectives  
- Benchmarked against a VAE baseline  

---

## Results and Visualizations

### Latent Space Topology
The SOM organizes data into a **2D grid**, grouping similar images together.  
Example: Shirts, trousers, and footwear occupy **contiguous regions**.  

Generated file: `som_topology_analysis.png`

### Generated Samples and Comparison
Stochastic sampling yields **new, diverse clothing items**.  
Side-by-side evaluations against a **VAE baseline**.  

Generated file: `comprehensive_model_evaluation.png`

---

## Getting Started

You can replicate training, evaluation, and visualization with the provided **Jupyter Notebook**.

<details>
  <summary><strong>Prerequisites</strong></summary>

- Python 3.8 or higher  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- scikit-learn  
</details>

<details>
  <summary><strong>Installation</strong></summary>

```bash
# Clone repository
git clone https://github.com/your-username/Gen-SOM-Attn.git
cd Gen-SOM-Attn

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn
