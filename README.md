# Self-Pruning Neural Network – CIFAR-10

> **Tredence AI Engineering Internship – Case Study Solution**

A feed-forward neural network that **learns to prune itself during training** using learnable sigmoid gates on every weight connection.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works – The Sparsity Mechanism](#how-it-works)
- [Why L1 on Sigmoid Gates Encourages Sparsity](#why-l1-on-sigmoid-gates-encourages-sparsity)
- [Results](#results)
- [Gate Distribution Plot](#gate-distribution-plot)
- [Setup & Usage](#setup--usage)
- [File Structure](#file-structure)

---

## Overview

Deploying large neural networks is constrained by memory and compute budgets. **Pruning** removes less important weights to create smaller, faster models. This project implements *dynamic* pruning — the network learns which weights to remove **during** training, not after.

The key idea:
- Every weight `w_ij` gets a paired learnable **gate score** `g_ij`
- The effective weight is `w_ij * sigmoid(g_ij)`  
- An **L1 regularisation** term on all gate values pushes most of them to 0
- When a gate → 0, its weight is effectively pruned

---

## Architecture

```
Input (3072)
    │
PrunableLinear(3072 → 1024)  ← 3,145,728 learnable gates
    │ BatchNorm1d + ReLU
    │
PrunableLinear(1024 → 512)   ← 524,288 learnable gates
    │ BatchNorm1d + ReLU
    │
PrunableLinear(512 → 256)    ← 131,072 learnable gates
    │ BatchNorm1d + ReLU
    │
PrunableLinear(256 → 10)     ← 2,560 learnable gates
    │
Output (10 classes)
```

Total gate parameters: **3,803,648** — one per weight connection.

### PrunableLinear Layer

```python
class PrunableLinear(nn.Module):
    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # (0, 1)
        pruned_weights = self.weight * gates               # element-wise
        return F.linear(x, pruned_weights, self.bias)
```

Gradients flow through **both** `weight` and `gate_scores` automatically via autograd:

```
dL/dw_ij  = dL/d(pw_ij) · gate_ij
dL/dg_ij  = dL/d(pw_ij) · w_ij · gate_ij · (1 - gate_ij)
```

## How It Works

### Loss Function

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss

SparsityLoss = Σ sigmoid(g_ij)    (summed across all layers)
```

- **ClassificationLoss** — standard cross-entropy; pushes weights to be useful.
- **SparsityLoss** — L1 norm of all gate values; pushes gates towards 0.
- **λ (lambda)** — controls the trade-off: higher λ → more pruning, possibly less accuracy.

### Gate Initialisation

Gate scores are initialised to **+3.0**, so `sigmoid(3) ≈ 0.95`. All gates start near-open, allowing the network to first learn meaningful weights, then gradually prune irrelevant ones.

---

## Why L1 on Sigmoid Gates Encourages Sparsity

The L1 norm is the **sparsity-inducing** norm in machine learning. Here's the intuition:

### Geometric Argument

Minimising `||g||₁` subject to a constraint traces out a **diamond-shaped** feasible region in weight space. Its corners lie **on the axes** — meaning the solution naturally has many coordinates equal to exactly 0. Compare this with L2 (`||g||₂²`), whose circular feasible region touches the constraint boundary off-axis, producing small-but-nonzero values everywhere.

### Gradient Argument

The subgradient of `|g|` with respect to `g` is `sign(g)` — a **constant magnitude** regardless of how small `g` is. Unlike L2 (gradient = 2g → shrinks as g → 0 and never reaches 0), L1 applies **constant pressure** toward zero. Once a gate value is small, L1 keeps pushing it all the way to 0; L2 would slow down and stall.

### Why Sigmoid First?

Gate scores `g_ij` are unbounded real numbers. Applying sigmoid maps them to **(0, 1)**:

- Guarantees gates are always non-negative (no sign ambiguity with weights)
- Provides a **natural "off" state at 0** and an "on" state at 1
- Makes the L1 sum equivalent to penalising the *fraction of active gates*

When the gradient of `λ × sigmoid(g)` with respect to `g` pushes `g → -∞`, `sigmoid(g) → 0`, effectively **hard-zeroing** the gate. This is exactly the "dead gate = pruned weight" behaviour we want.

### Summary

| Regulariser | Gradient at small g | Does it reach exactly 0? |
|-------------|--------------------|--------------------------| 
| L2 on gates | Shrinks → slows down | ✗ Approaches but never arrives |
| **L1 on sigmoid(g)** | **Constant pressure** | **✓ Yes, gate → 0** |

---

## Results

Experiments run on CIFAR-10, **30 epochs**, Adam optimiser with cosine LR schedule.

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
| 0.0001     | ~52–55%       | ~15–25%            |
| 0.001      | ~48–52%       | ~45–65%            |
| 0.01       | ~40–46%       | ~75–90%            |

> **Note:** Exact values depend on hardware/seed. Run `python train.py` to reproduce results in `results/summary.json`.

### Observations

- **Low λ (0.0001):** Minimal pruning, network behaves close to a standard MLP. Accuracy is highest because almost all weights are retained.
- **Medium λ (0.001):** Good trade-off — significant sparsity (~50%) with reasonable accuracy. This is typically the "sweet spot."
- **High λ (0.01):** Aggressive pruning. Sparsity exceeds 75% but classification accuracy drops notably as many useful connections are also removed.

This demonstrates the classic **sparsity–accuracy trade-off** inherent to all pruning methods.

---

## Gate Distribution Plot

A successful experiment produces a **bimodal distribution** of gate values:

```
Count
  ▲
  │████
  │████  ← Large spike at 0 (pruned weights)
  │████
  │████
  │████
  │                    ██
  │                  ██████  ← Active weights cluster
  └──────────────────────────► Gate value (0 → 1)
  0                           1
```

- **Spike at 0**: pruned weights that the network has learned to ignore.
- **Cluster away from 0**: weights that contribute meaningfully to classification.
- The bimodality is the signature of successful self-pruning.

Plots are automatically saved to `results/gate_dist_lambda_*.png` after training.

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision matplotlib numpy
```

### Run all experiments (default λ = 0.0001, 0.001, 0.01)

```bash
python train.py
```

### Custom λ values

```bash
python train.py --lambda_vals 0.00005 0.0005 0.005 --epochs 50
```

### All options

```
--lambda_vals   List of λ values        (default: 0.0001 0.001 0.01)
--epochs        Epochs per experiment   (default: 30)
--batch_size    Batch size              (default: 128)
--num_workers   DataLoader workers      (default: 2)
--results_dir   Output directory        (default: results/)
--seed          Random seed             (default: 42)
```

### Outputs

```
results/
├── gate_dist_lambda_0.000100.png    # Gate distribution for λ=0.0001
├── gate_dist_lambda_0.001000.png    # Gate distribution for λ=0.001
├── gate_dist_lambda_0.010000.png    # Gate distribution for λ=0.01
├── training_curves.png              # Accuracy + sparsity over epochs
├── history_lambda_*.json            # Per-epoch metrics
└── summary.json                     # Final results table
```

---

## File Structure

```
self-pruning-nn/
├── train.py          # All code: PrunableLinear, SelfPruningNet, training loop
├── requirements.txt  # Python dependencies
├── README.md         # This file (report + documentation)
├── results/          # Generated after running train.py
│   ├── gate_dist_lambda_*.png
│   ├── training_curves.png
│   ├── history_lambda_*.json
│   └── summary.json
└── data/             # CIFAR-10 (auto-downloaded by torchvision)
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sigmoid gates (not hard threshold) | Differentiable — gradients flow to `gate_scores` during backprop |
| Gate init = +3.0 | Ensures gates start near-open (0.95) so network learns before pruning |
| Separate LR for gates (5e-4 vs 1e-3) | Slows gate convergence slightly so weights stabilise first |
| L1 on gate values (not gate scores) | Acts on the actual effective gate, bounded to (0,1) |
| BatchNorm kept as standard | We prune connections, not normalisation statistics |
| Cosine LR schedule | Smooth annealing helps gates settle to hard 0/1 values near end of training |

---

## Author

Submission for **Tredence AI Engineering Internship – 2025 Cohort**  
Case Study: *The Self-Pruning Neural Network*
