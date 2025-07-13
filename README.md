# Nikola AI

Nikola is a self-organizing, self-learning, fractal-emergent AI built with PyTorch. It features a dynamic, hierarchical neural network with fractal scaling, Hebbian learning for connection optimization, and Model-Agnostic Meta-Learning (MAML) for rapid task adaptation. This repository is public, welcoming community contributions and feedback.

## Features
- **Fractal Structure**: Hierarchical network with exponential node growth (2^level).
- **Self-Organization**: Combines Hebbian learning and simulated annealing to dynamically adjust connections and nodes.
- **Self-Learning**: Uses backpropagation for node-level training and MAML for network adaptation.
- **Task**: Classifies 8-bit binary sequences into four classes (sum modulo 4).

## Installation
1. Clone or download the repository.
2. Install dependencies:
   ```bash
   pip install torch numpy
