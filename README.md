# 🧪 PyTorch Experiments

This repository contains my hands-on experiments with PyTorch and neural networks.

## 🎯 Goal

The goal of this project is not just to use PyTorch, but to deeply understand how neural networks actually work.

I focus on connecting mathematical concepts with real implementations, including:

- Forward propagation
- Backpropagation and the chain rule
- Gradient flow through layers
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (BCE, CrossEntropy)
- Numerical stability (e.g. softmax stabilization)

---

## 🧠 Approach

Instead of treating PyTorch as a black box, I analyze what happens inside the model:

- How gradients are computed
- Why certain formulas simplify (e.g. \( Z - Y \))
- How matrix operations relate to theory
- How different design choices affect training

I previously implemented a neural network from scratch using NumPy, and this repository extends that understanding using PyTorch.

---

## 📁 Project Structure

```text
experiments/
├── 01_single_neuron/
├── 02_mlp/
├── 03_loss_functions/
└── 04_activations/

notes/
└── backpropagation.md
