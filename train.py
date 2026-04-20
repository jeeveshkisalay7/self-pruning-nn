"""
Self-Pruning Neural Network for CIFAR-10
=========================================
Tredence AI Engineering Internship – Case Study Submission

Architecture:
  - PrunableLinear: a custom linear layer with learnable sigmoid gates.
  - SelfPruningNet: feed-forward network built from PrunableLinear layers.
  - Total Loss = CrossEntropyLoss + λ * SparsityLoss (L1 of all gate values)

Usage:
  python train.py                   # runs default experiment
  python train.py --lambda_vals 0.0001 0.001 0.01 --epochs 30
"""

import argparse
import os
import json
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns which weights to prune.

    Each weight w_ij has a corresponding gate score g_ij (a learnable scalar).
    The gate is obtained by passing g_ij through a sigmoid so it lies in (0, 1).
    The effective weight used in the forward pass is  w_ij * sigmoid(g_ij).

    During training, an L1 regularisation term on all gate values drives
    many of them towards zero, effectively pruning those connections.

    Gradient flow:
      dL/dw_ij   = dL/d(pw_ij) * gate_ij          (standard chain rule)
      dL/dg_ij   = dL/d(pw_ij) * w_ij * gate_ij*(1-gate_ij)
    Both paths are handled automatically by PyTorch autograd because every
    operation here is differentiable.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias parameters (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Gate score tensor – same shape as weight.
        # Initialised to +3 so sigmoid(3) ≈ 0.95 → all gates start near-open.
        # This lets the network first learn useful weights, then prune gradually.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 3.0))

        self._reset_parameters()

    def _reset_parameters(self):
        # Kaiming uniform, consistent with nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound  = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def get_gates(self) -> torch.Tensor:
        """Return gate values in (0, 1) via sigmoid."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = self.get_gates()                    # shape: (out, in)
        pruned_weights = self.weight * gates                # element-wise gate
        return F.linear(x, pruned_weights, self.bias)

    def sparsity_penalty(self) -> torch.Tensor:
        """L1 norm of gate values for this layer (always positive)."""
        return self.get_gates().sum()

    def active_fraction(self, threshold: float = 1e-2) -> float:
        """Fraction of gates above threshold (i.e., not pruned)."""
        gates = self.get_gates().detach()
        return (gates > threshold).float().mean().item()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Simple feed-forward network for CIFAR-10 (32×32×3 → 10 classes).

    All Linear layers are replaced with PrunableLinear.
    BatchNorm is intentionally kept as standard nn.BatchNorm1d because we
    only prune the weight connections, not the normalisation statistics.
    """

    def __init__(self):
        super().__init__()

        # Input: 32*32*3 = 3072 features
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # flatten
        return self.net(x)

    def prunable_layers(self):
        """Yield all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """Aggregate L1 penalty across all PrunableLinear layers."""
        total = sum(layer.sparsity_penalty() for layer in self.prunable_layers())
        return total

    def global_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Percentage of weights whose gate value is below `threshold`.
        A high number means most weights have been pruned.
        """
        below, total = 0, 0
        with torch.no_grad():
            for layer in self.prunable_layers():
                gates  = layer.get_gates()
                below += (gates <= threshold).sum().item()
                total += gates.numel()
        return 100.0 * below / total if total > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Return a flat numpy array of all gate values (for plotting)."""
        vals = []
        with torch.no_grad():
            for layer in self.prunable_layers():
                vals.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(vals)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=256,        shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, lambda_val, device):
    model.train()
    total_loss = cls_loss_sum = sparse_loss_sum = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss    = F.cross_entropy(logits, labels)
        sparse_loss = model.sparsity_loss()
        loss        = cls_loss + lambda_val * sparse_loss

        loss.backward()
        optimizer.step()

        total_loss      += loss.item()
        cls_loss_sum    += cls_loss.item()
        sparse_loss_sum += sparse_loss.item()

        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    n = len(loader)
    return {
        "total_loss"   : total_loss    / n,
        "cls_loss"     : cls_loss_sum  / n,
        "sparse_loss"  : sparse_loss_sum / n,
        "train_acc"    : 100.0 * correct / total,
    }


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds   = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full Experiment (one λ value)
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(lambda_val: float, epochs: int, device, train_loader, test_loader,
                   results_dir: str, verbose: bool = True):
    """Train one model with a given λ and return final metrics."""
    model = SelfPruningNet().to(device)

    # Separate learning-rate for gate_scores (slightly smaller to start pruning gradually)
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]

    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": 1e-3},
        {"params": gate_params,   "lr": 5e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    t0      = time.time()

    for epoch in range(1, epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, lambda_val, device)
        test_acc    = evaluate(model, test_loader, device)
        sparsity    = model.global_sparsity()
        scheduler.step()

        row = {**train_stats, "test_acc": test_acc, "sparsity": sparsity, "epoch": epoch}
        history.append(row)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(
                f"  λ={lambda_val:.4f} | Epoch {epoch:3d}/{epochs} | "
                f"Loss {train_stats['total_loss']:.4f} | "
                f"Train {train_stats['train_acc']:.1f}% | "
                f"Test {test_acc:.1f}% | "
                f"Sparsity {sparsity:.1f}% | "
                f"{elapsed:.0f}s"
            )

    # Save gate distribution plot
    gate_vals = model.all_gate_values()
    plot_path = os.path.join(results_dir, f"gate_dist_lambda_{lambda_val:.6f}.png")
    plot_gate_distribution(gate_vals, lambda_val, plot_path)

    # Save training history
    hist_path = os.path.join(results_dir, f"history_lambda_{lambda_val:.6f}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    final_test_acc = history[-1]["test_acc"]
    final_sparsity = history[-1]["sparsity"]

    return {
        "lambda"    : lambda_val,
        "test_acc"  : final_test_acc,
        "sparsity"  : final_sparsity,
        "history"   : history,
        "gate_vals" : gate_vals,
        "plot_path" : plot_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(gate_vals: np.ndarray, lambda_val: float, save_path: str):
    """
    Plot histogram of all gate values for a trained model.
    A successful run shows a large spike near 0 (pruned) and a cluster away from 0 (kept).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Gate Value Distribution  (λ = {lambda_val})", fontsize=14, fontweight="bold")

    # ── Full range ──
    ax = axes[0]
    ax.hist(gate_vals, bins=100, color="#4A90D9", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Full Distribution (0 → 1)")
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2, label="Prune threshold (0.01)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Zoom on near-zero spike ──
    ax = axes[1]
    near_zero = gate_vals[gate_vals < 0.1]
    pct_pruned = 100.0 * (gate_vals < 0.01).mean()
    ax.hist(near_zero, bins=80, color="#E05C5C", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_title(f"Zoom: Gates < 0.1  |  Pruned (< 0.01): {pct_pruned:.1f}%")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Gate distribution plot saved → {save_path}")


def plot_training_curves(all_results: list, results_dir: str):
    """Plot accuracy and sparsity curves for all λ values on the same axes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves Across λ Values", fontsize=14, fontweight="bold")

    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, result in enumerate(all_results):
        lam     = result["lambda"]
        history = result["history"]
        epochs  = [h["epoch"]    for h in history]
        test    = [h["test_acc"] for h in history]
        spar    = [h["sparsity"] for h in history]
        c       = colours[i % len(colours)]

        ax1.plot(epochs, test, color=c, linewidth=1.8, label=f"λ={lam}")
        ax2.plot(epochs, spar, color=c, linewidth=1.8, label=f"λ={lam}")

    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Test Accuracy");  ax1.legend();  ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch");  ax2.set_ylabel("Sparsity Level (%)")
    ax2.set_title("Sparsity Level");  ax2.legend();  ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Training curves saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Self-Pruning Neural Network – CIFAR-10")
    p.add_argument("--lambda_vals", nargs="+", type=float,
                   default=[0.0001, 0.001, 0.01],
                   help="List of λ (sparsity penalty) values to evaluate.")
    p.add_argument("--epochs",      type=int, default=30,
                   help="Training epochs per experiment.")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--results_dir", type=str, default="results",
                   help="Directory to store plots and JSON logs.")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" Self-Pruning Neural Network – CIFAR-10")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  λ values    : {args.lambda_vals}")
    print(f"{'='*60}\n")

    os.makedirs(args.results_dir, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers)

    # ── Run experiments ──────────────────────────────────────────────────────
    all_results = []
    for lam in args.lambda_vals:
        print(f"\n► Running experiment: λ = {lam}")
        result = run_experiment(
            lambda_val   = lam,
            epochs       = args.epochs,
            device       = device,
            train_loader = train_loader,
            test_loader  = test_loader,
            results_dir  = args.results_dir,
        )
        all_results.append(result)

    # ── Summary table ────────────────────────────────────────────────────────
    plot_training_curves(all_results, args.results_dir)

    print(f"\n{'='*60}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print(f"{'-'*12} {'-'*15} {'-'*16}")
    for r in all_results:
        print(f"{r['lambda']:<12.6f} {r['test_acc']:>14.2f}% {r['sparsity']:>15.1f}%")
    print(f"{'='*60}\n")

    # Save summary JSON
    summary = [{"lambda": r["lambda"], "test_acc": r["test_acc"], "sparsity": r["sparsity"]}
               for r in all_results]
    with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("All experiments complete. Results saved to:", args.results_dir)


if __name__ == "__main__":
    main()
