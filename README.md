A-Non-Deterministic-Unsupervised-Model-for-Image-Generation
Generative Self-Organizing Map with Attention (Gen-SOM-Attn)

This repository contains the official PyTorch implementation of the Generative Self-Organizing Map with Attention (Gen-SOM-Attn) — a novel, non-deterministic, unsupervised neural network for image generation.
Developed as part of CSE425: Neural Networks, this project introduces a fresh perspective on generative modeling.

Project Overview

The Gen-SOM-Attn addresses limitations of existing generative models such as the training instability of GANs and the blurry outputs of VAEs. Its novelty lies in combining:

A deep convolutional autoencoder

A Kohonen Self-Organizing Map (SOM)

An attention mechanism

Key Characteristics

Unsupervised: Learns patterns directly from unlabeled data (Fashion-MNIST).

No Error-Correction: Uses competitive learning in the SOM, avoiding backpropagation in latent space.

Interpretable: Produces a topologically ordered latent space with intuitive clustering.

Non-Deterministic: Generates diverse images by stochastically sampling from the learned latent manifold.

Features

Novel Architecture: Convolutional Autoencoder + SOM + Attention Module

High-Quality Generation: Sharp, diverse images competitive with VAEs

Structured Latent Space: 12×12 SOM grid mapping learned features

Stable Training: No adversarial dynamics like GANs

Comprehensive Evaluation: Benchmarked against a standard VAE baseline

Results and Visualizations
Latent Space Topology

The SOM organizes data into a 2D grid, grouping similar images together.
Example: Shirts, trousers, and footwear occupy contiguous regions.

(Generated: som_topology_analysis.png)

Generated Samples and Comparison

Stochastic sampling yields new, diverse clothing items.
Side-by-side evaluations against a VAE baseline.

(Generated: comprehensive_model_evaluation.png)

Getting Started

You can replicate training, evaluation, and visualization with the provided Jupyter Notebook.

Prerequisites

Python 3.8+

PyTorch

Torchvision

NumPy

Matplotlib

scikit-learn

Installation
# Clone repository
git clone https://github.com/your-username/Gen-SOM-Attn.git
cd Gen-SOM-Attn

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn

Running the Experiment
jupyter notebook "Non_Deterministic_Unsupervised_Model_for_Image_Generation_425_Modified_cellwise.ipynb"


This will automatically:

Download Fashion-MNIST

Train and evaluate Gen-SOM-Attn

Train and evaluate VAE baseline

Save trained models and results

Project Structure

Running the notebook produces the following outputs (with timestamps):

gen_som_attn_model_{timestamp}.pth: Gen-SOM-Attn weights

vae_baseline_model_{timestamp}.pth: VAE baseline weights

training_results_{timestamp}.pkl: Training history (losses, metrics)

som_topology_analysis.png: Latent space visualization

comprehensive_model_evaluation.png: Generation and evaluation plots
