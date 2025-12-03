# Adversarial Attack on Neural Networks - Research Repository

A comprehensive research framework for studying adversarial robustness of neural networks on the MNIST dataset. This repository implements state-of-the-art adversarial attack methods (FGSM, PGD) and defense mechanisms (adversarial training) to evaluate and improve model robustness.

## Overview

This repository provides a complete pipeline for:
- **Training** baseline CNN models on MNIST
- **Attacking** models with adversarial examples using FGSM and PGD
- **Defending** models through adversarial training
- **Evaluating** robustness across multiple attack strategies and perturbation budgets
- **Visualizing** adversarial examples and robustness curves

## Research Background

This implementation is based on foundational papers in adversarial machine learning:

1. **"Explaining and Harnessing Adversarial Examples"** (Goodfellow et al., 2014)
   - Introduces the Fast Gradient Sign Method (FGSM)
   - Demonstrates the vulnerability of neural networks to small perturbations

2. **"Towards Deep Learning Models Resistant to Adversarial Attacks"** (Madry et al., 2017)
   - Introduces Projected Gradient Descent (PGD) as a strong attack
   - Proposes adversarial training as a defense mechanism
   - Establishes the adversarial training framework

3. **AutoAttack** (Croce & Hein, 2020)
   - Provides standardized evaluation framework for adversarial robustness
   - Includes multiple attack strategies for comprehensive evaluation

## Features

### Attack Methods
- **FGSM (Fast Gradient Sign Method)**: One-step gradient-based attack
- **PGD (Projected Gradient Descent)**: Iterative attack with projection to epsilon-ball
- **AutoAttack**: Standardized evaluation suite with multiple attack strategies

### Defense Mechanisms
- **Adversarial Training**: Training models on adversarial examples to improve robustness

### Evaluation Tools
- Epsilon sweep analysis (accuracy vs. perturbation budget)
- Multi-norm evaluation (L∞ and L2)
- Clean vs. adversarial accuracy comparison
- Visualization of adversarial examples

## Project Structure

```
AdversarialAttackNN_R1/
├── src/
│   ├── attack.py          # FGSM and PGD attack implementations
│   ├── defense.py         # Adversarial training defense
│   ├── model.py           # SimpleCNN architecture
│   ├── train.py           # Standard training loop
│   ├── evaluate.py        # Evaluation and visualization functions
│   ├── dataset.py         # MNIST data loading utilities
│   └── uti.py             # Utility functions (normalization, clamping)
├── checkpoints/           # Saved model checkpoints
│   ├── mnist_cnn.pt       # Baseline trained model
│   └── mnist_cnn_adv.pt   # Adversarially trained model
├── data/                  # MNIST dataset storage
├── run.py                 # Main script: train baseline + basic attack demo
├── run_attacks.py         # Comprehensive attack evaluation suite
└── train_adv.py           # Adversarial training script
```

## Installation

### Requirements

```bash
pip install torch torchvision matplotlib autoattack
```

Or create a `requirements.txt`:

```txt
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
autoattack>=0.3.0
```

## Usage

### 1. Train Baseline Model

Train a standard CNN on MNIST:

```bash
python run.py
```

This will:
- Train a SimpleCNN model for 3 epochs
- Save the model to `checkpoints/mnist_cnn.pt`
- Evaluate clean accuracy
- Generate and visualize a sample of adversarial examples
- Optionally perform adversarial training

### 2. Comprehensive Attack Evaluation

Run the full evaluation suite:

```bash
python run_attacks.py
```

This performs:
- Clean accuracy evaluation
- FGSM attack sweep (epsilon: 0.0 to 0.3)
- PGD attack sweep (epsilon: 0.0 to 0.3)
- Visualization of adversarial examples
- Multi-norm AutoAttack evaluation (L∞ and L2)

### 3. Adversarial Training

Train a robust model using adversarial training:

```bash
python train_adv.py
```

This will:
- Train a model using PGD adversarial examples
- Save the adversarially trained model to `checkpoints/mnist_cnn_adv.pt`
- Use epsilon=0.3, step_size=0.075, 40 PGD steps

## Key Components

### Attack Methods

#### FGSM Attack
```python
from src.attack import fgsm_attack

adversarial_examples = fgsm_attack(
    model, x, y, 
    epsilon=0.1,  # Perturbation budget (normalized space)
    device='cuda'
)
```

#### PGD Attack
```python
from src.attack import pgd_attack

adversarial_examples = pgd_attack(
    model, x, y,
    epsilon=0.1,           # Perturbation budget
    step_size=0.01,        # Step size per iteration
    num_steps=40,          # Number of iterations
    random_start=True,     # Random initialization within epsilon-ball
    device='cuda'
)
```

### Defense: Adversarial Training

```python
from src.defense import adversarial_training

robust_model = adversarial_training(
    model, train_loader,
    epsilon=0.1,           # Attack strength during training
    epochs=3,
    device='cuda'
)
```

### Evaluation

```python
from src.evaluate import test_accuracy, test_adversarial

# Clean accuracy
clean_acc = test_accuracy(model, test_loader, device='cuda')

# Adversarial accuracy
adv_acc, examples = test_adversarial(
    model, test_loader,
    attack_fn=fgsm_attack,
    attack_kwargs={'epsilon': 0.1},
    device='cuda'
)
```

## Model Architecture

The repository uses a simple CNN architecture (`SimpleCNN`):

- **Convolutional Layers**: 
  - Conv2d(1, 32, 3) → ReLU
  - Conv2d(32, 64, 3) → ReLU
  - MaxPool2d(2)
- **Fully Connected Layers**:
  - Linear(9216, 128) → ReLU
  - Linear(128, 10)

## Research Objectives

This repository aims to:

1. **Quantify Robustness**: Measure adversarial robustness across different perturbation budgets
2. **Compare Attacks**: Evaluate relative strength of FGSM vs. PGD attacks
3. **Evaluate Defenses**: Assess the effectiveness of adversarial training
4. **Establish Baselines**: Provide reproducible benchmarks for adversarial robustness research
5. **Visualize Results**: Generate visualizations of adversarial examples and robustness curves

## Expected Results

After running the evaluation suite, you should obtain:

- **Robustness Curves**: Accuracy vs. epsilon plots showing degradation under attacks
- **Comparison Metrics**: Clean accuracy vs. adversarial accuracy for baseline and robust models
- **Visual Examples**: Side-by-side comparisons of clean vs. adversarial images
- **Quantitative Benchmarks**: Standardized robustness metrics at various epsilon values

## Notes

- All perturbations are computed in **normalized space** (after MNIST normalization: mean=0.1307, std=0.3081)
- To convert pixel-space epsilon to normalized space: `epsilon_norm = epsilon_pixel / 0.3081`
- The implementation follows best practices from the referenced papers
- Models are saved in PyTorch's standard `.pt` format

## References

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*.

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2017). Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*.

3. Croce, F., & Hein, M. (2020). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. *ICML 2020*.

## License

This is a research repository. Please cite the original papers when using their methods.
