# Product Context

## Project Overview
**Ambient Masked Diffusion** is a research project aiming to implement and experiment with Masked Diffusion Models (MDM) on the MNIST dataset. The primary goal is to handle "Ambient" settings where training data may have missing values (masked pixels) that are not ground-truth zeros but unknown values.

## Goals
1.  **Baseline**: Train a Masked Diffusion Model on clean, binarized MNIST data to establish a performance benchmark.
2.  **Ambient**: Train the same model architecture on MNIST data with missing values (simulated by dropping pixels) without access to the ground truth for those pixels during training.
3.  **Generation**: Generate high-quality MNIST digits from the trained models.

## Core Features
-   **Masked Diffusion**: A discrete diffusion process where pixels are masked over time and the model learns to unmask them (predict the original value).
-   **Binarized MNIST**: Data is treated as discrete tokens {0, 1}, plus a special {MASK} token.
-   **Configurable Experiments**: YAML-based configuration for easy switching between baseline (clean) and ambient (missing data) experiments.

