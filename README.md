# PyTorch Experiments

This repository contains my hands-on experiments with PyTorch and neural networks.

## Goal

The goal of this project is not just to use PyTorch, but to deeply understand how neural networks actually work.

I focus on connecting mathematical concepts with real implementations, including:

- Forward propagation
- Backpropagation and the chain rule
- Gradient flow through layers
- Activation functions (ReLU, Sigmoid, Softmax, GELU)
- Loss functions (BCE, CrossEntropy)
- Numerical stability (e.g. softmax stabilization)

---

## Approach

Instead of treating PyTorch as a black box, I analyze what happens inside the model:

1. **Mathematical Derivation:** Studying the calculus and linear algebra behind the operation.
2. **Manual Implementation:** Implementing the logic using basic NumPy/Python to understand the raw matrix manipulations.
3. **PyTorch Integration:** Transitioning to PyTorch to leverage its optimization and Autograd capabilities.
4. **Comparative Analysis:** Analyzing the difference between manual and automated implementations to uncover the "magic" happening under the hood.

---

## Project Structure

```text
├── experiments/
│   └── 01_single_neuron/
│       ├── images/
│       ├── README.md
│       ├── requirements.txt
│       └── train.py
├── .gitignore
├── LICENSE
└── README.md
```
