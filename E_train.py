

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from dgl.nn.pytorch import SAGEConv

from get_amazon import amazon_data
from get_yelp import yelp_data
from get_tfinance import tfinance_data
from get_tsocial import tsocial_data
from fast_e import local_1hop_energy_lnorm


def get_best_f1(labels, probs):
    """Find best threshold for macro-F1."""
    best_f1, best_thres = 0.0, 0.0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average="macro")
        if mf1 > best_f1:
            best_f1 = mf1
            best_thres = thres
    return best_f1, best_thres


def compute_comprehensive_metrics(labels, probs, preds):
 
    metrics = {}

    
    metrics['recall'] = recall_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds)
    metrics['macro_f1'] = f1_score(labels, preds, average="macro")

   
    metrics['auroc'] = roc_auc_score(labels, probs)
    metrics['auprc'] = average_precision_score(labels, probs)

   
    k = int(labels.sum())
    if k > 0:
        top_k_indices = np.argsort(probs)[-k:]  # indices of top-K predictions
        metrics['reck'] = labels[top_k_indices].sum() / k
    else:
        metrics['reck'] = 0.0

    return metrics


def ana_feat(all_feature_importances):
    """Analyze feature importance rankings across multiple runs."""
    num_runs = len(all_feature_importances)
    num_features = len(all_feature_importances[0])

    importances_array = np.array(
        [imp.cpu().numpy() if hasattr(imp, "cpu") else imp for imp in all_feature_importances]
    )

    rankings = np.zeros((num_runs, num_features))
    for run_idx in range(num_runs):
        sorted_indices = np.argsort(-importances_array[run_idx])  # Desc
        for rank, feat_idx in enumerate(sorted_indices):
            rankings[run_idx, feat_idx] = rank + 1

    avg_rankings = np.mean(rankings, axis=0)
    std_rankings = np.std(rankings, axis=0)

    mean_importance = np.mean(importances_array, axis=0)
    std_importance = np.std(importances_array, axis=0)

    feature_stats = {
        "mean_importance": mean_importance,
        "std_importance": std_importance,
        "avg_ranking": avg_rankings,
        "std_ranking": std_rankings,
    }
    return avg_rankings, feature_stats


class SpectralFlipGatingMLP(nn.Module):
  

    def __init__(self, in_feats=25, hidden_dim=16, gate_type="per_feature", num_layers=2):
        super().__init__()
        self.in_feats = in_feats
        self.gate_type = gate_type
        self.num_layers = num_layers

        out_dim = 1 if gate_type == "per_node" else in_feats
        input_dim = in_feats

        if num_layers == 2:
            self.gate_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid(),
            )
        elif num_layers == 3:
            self.gate_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid(),
            )
        elif num_layers == 4:
            self.gate_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"num_layers must be 2, 3, or 4, got {num_layers}")

        # Init: prefer flip at start (gate ~ 0)
        with torch.no_grad():
            final_linear = self.gate_mlp[-2]  # Linear before Sigmoid
            final_linear.weight.zero_()
            final_linear.bias.fill_(-2.0)

    def forward(self, X_for_gate, R_normal, R_flip):
        """
        Args:
            X_for_gate: [N,F] (normalized) original features to compute gates
            R_normal:   [N,F] (normalized) normal energy
            R_flip:     [N,F] (normalized) flipped energy

        Returns:
            Z:     [N,F]
            gates: [N,1] or [N,F]
        """
        gates = self.gate_mlp(X_for_gate)
        Z = gates * R_normal + (1.0 - gates) * R_flip
        return Z, gates


class FeatureWiseMLP(nn.Module):
    """Feature-wise attention on energy features."""
    def __init__(self, d_model=25, reduction_ratio=4):
        super().__init__()
        hidden_dim = max(d_model // reduction_ratio, 1)
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.Sigmoid(),
        )

    def forward(self, energy_features):
        attn = self.attention_net(energy_features)
        return energy_features * attn, attn


class GatedEnergySAGE(nn.Module):


    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_classes,
        dropout=0.5,
        aggregator_type="mean",
        attention_reduction=4,
        gate_hidden_dim=16,
        gate_type="per_feature",
        gate_num_layers=2,
    ):
        super().__init__()

        self.in_feats = in_feats
        self.gate_type = gate_type

        self.spectral_gate = SpectralFlipGatingMLP(
            in_feats=in_feats,
            hidden_dim=gate_hidden_dim,
            gate_type=gate_type,
            num_layers=gate_num_layers,
        )

        self.feature_attention = FeatureWiseMLP(
            d_model=in_feats,
            reduction_ratio=attention_reduction,
        )

        self.conv1 = SAGEConv(in_feats, hidden_dim, aggregator_type)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggregator_type)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2, aggregator_type)

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = dropout
        self.activation = nn.ReLU()

        # caches (analysis)
        self.cached_energy_normal = None
        self.cached_energy_flip = None
        self.cached_energy_mixed = None
        self.cached_gates = None
        self.cached_attention_weights = None

    @staticmethod
    def _zscore(x, mean, std):
        return (x - mean) / std.clamp_min(1e-8)

    def forward(self, graph, features, edge_index, train_mask=None, return_analysis=False):
        # ---- Energy (frozen) ----
        with torch.no_grad():
            R_normal = local_1hop_energy_lnorm(
                X=features,
                edge_index=edge_index,
                edge_weight=None,
                eps=1e-8,
                deg_eps=1e-12,
            )
            R_flip = 2.0 - R_normal

        self.cached_energy_normal = R_normal.detach()
        self.cached_energy_flip = R_flip.detach()

        # ---- Normalize X for gating (TRAIN stats) ----
        if train_mask is not None and self.training:
            x_mean = features[train_mask].mean(dim=0, keepdim=True)
            x_std = features[train_mask].std(dim=0, keepdim=True).clamp_min(1e-8)
            self.train_x_mean = x_mean.detach()
            self.train_x_std = x_std.detach()
        elif hasattr(self, "train_x_mean"):
            x_mean = self.train_x_mean
            x_std = self.train_x_std
        else:
            x_mean = features.mean(dim=0, keepdim=True)
            x_std = features.std(dim=0, keepdim=True).clamp_min(1e-8)

        Xn = self._zscore(features, x_mean, x_std)

        # ---- Normalize R (TRAIN stats), apply same to R_flip ----
        if train_mask is not None and self.training:
            r_mean = R_normal[train_mask].mean(dim=0, keepdim=True)
            r_std = R_normal[train_mask].std(dim=0, keepdim=True).clamp_min(1e-8)
            self.train_r_mean = r_mean.detach()
            self.train_r_std = r_std.detach()
        elif hasattr(self, "train_r_mean"):
            r_mean = self.train_r_mean
            r_std = self.train_r_std
        else:
            r_mean = R_normal.mean(dim=0, keepdim=True)
            r_std = R_normal.std(dim=0, keepdim=True).clamp_min(1e-8)

        Rn = self._zscore(R_normal, r_mean, r_std)
        Rf = self._zscore(R_flip, r_mean, r_std)

        # ---- Gate + mix (TRAINABLE) ----
        Z_mixed, gates = self.spectral_gate(Xn, Rn, Rf)

        self.cached_energy_mixed = Z_mixed.detach()
        self.cached_gates = gates.detach()

        # One-time debug
        if not hasattr(self, "_debug_printed"):
            print("\n[DEBUG] First forward pass statistics:")
            print(f"  X(raw):        mean={features.mean():.4f}, std={features.std():.4f}")
            print(f"  Xn(zscore):    mean={Xn.mean():.4f}, std={Xn.std():.4f}")
            print(f"  R_normal(raw): mean={R_normal.mean():.4f}, std={R_normal.std():.4f}, min={R_normal.min():.4f}, max={R_normal.max():.4f}")
            print(f"  Rn(zscore):    mean={Rn.mean():.4f}, std={Rn.std():.4f}")
            print(f"  Gates:         mean={gates.mean():.4f}, std={gates.std():.4f}, min={gates.min():.4f}, max={gates.max():.4f}")
            print(f"  Z_mixed:       mean={Z_mixed.mean():.4f}, std={Z_mixed.std():.4f}, min={Z_mixed.min():.4f}, max={Z_mixed.max():.4f}")
            self._debug_printed = True

        # ---- Normalize Z (TRAIN stats) ----
        if train_mask is not None and self.training:
            z_mean = Z_mixed[train_mask].mean(dim=0, keepdim=True)
            z_std = Z_mixed[train_mask].std(dim=0, keepdim=True).clamp_min(1e-8)
            self.train_z_mean = z_mean.detach()
            self.train_z_std = z_std.detach()
        elif hasattr(self, "train_z_mean"):
            z_mean = self.train_z_mean
            z_std = self.train_z_std
        else:
            z_mean = Z_mixed.mean(dim=0, keepdim=True)
            z_std = Z_mixed.std(dim=0, keepdim=True).clamp_min(1e-8)

        energy_normalized = self._zscore(Z_mixed, z_mean, z_std)  # IMPORTANT: do not overwrite

        # ---- Attention ----
        weighted_energy, attention_weights = self.feature_attention(energy_normalized)
        self.cached_attention_weights = attention_weights.detach()

        # ---- GraphSAGE ----
        h = self.conv1(graph, weighted_energy)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(graph, h)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(graph, h)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h)

        if return_analysis:
            analysis = {
                "gates": gates,
                "energy_normal": R_normal,
                "energy_flip": R_flip,
                "energy_mixed": Z_mixed,
                "attention_weights": attention_weights,
            }
            return logits, analysis

        return logits

    def get_feature_importance(self):
        if self.cached_attention_weights is None:
            return None
        return self.cached_attention_weights.mean(dim=0)

    def get_gate_statistics(self):
        if self.cached_gates is None:
            return None
        gates = self.cached_gates
        stats = {
            "mean_gate": gates.mean().item(),
            "std_gate": gates.std().item(),
            "min_gate": gates.min().item(),
            "max_gate": gates.max().item(),
        }
        if self.gate_type == "per_feature":
            stats["mean_gate_per_feature"] = gates.mean(dim=0)
        return stats


def train(args, data=None, print_config=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if print_config:
        print("=" * 70)
        print("GATED ENERGY-BASED GRAPHSAGE - COMPREHENSIVE METRICS")
        print("=" * 70)
        print(f"\nDevice: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("\nConfiguration:")
        print(f"  Dataset: {args.dataset.capitalize()} (undirected={args.undirected})")
        print(f"  Train/Val/Test ratio: {args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio}")
        print(f"  Hidden dimension: {args.hidden_dim}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Attention reduction ratio: {args.attention_reduction}")
        print(f"  Gate type: {args.gate_type}")
        print(f"  Gate layers: {args.gate_num_layers}")
        print(f"  Gate hidden dim: {args.gate_hidden_dim}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Random seed: {args.seed}")
        print("\nEvaluation Metrics:")
        print("  - Macro-F1: Classification performance with optimal threshold")
        print("  - AUROC: Overall discrimination ability")
        print("  - AUPRC: Precision-Recall curve (better for imbalanced data)")
        print("  - RecK: Recall at top-K (retrieval performance)")

    # Load dataset
    if data is None:
        print("\n" + "=" * 70)
        print("LOADING DATASET")
        print("=" * 70)

        if args.dataset == "amazon":
            data = amazon_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=args.seed,
                homo=True,
                undirected=args.undirected,
                verbose=True,
            )
        elif args.dataset == "yelp":
            data = yelp_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=args.seed,
                homo=True,
                undirected=args.undirected,
                verbose=True,
            )
        elif args.dataset == "tfinance":
            data = tfinance_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=args.seed,
                undirected=args.undirected,
                verbose=True,
            )
        elif args.dataset == "tsocial":
            data = tsocial_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=args.seed,
                undirected=args.undirected,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    else:
        data.regenerate_masks(args.train_ratio, args.val_ratio, args.seed)

    graph = data.graph.to(device)
    features = data.features.to(device)
    labels = data.labels.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    # edge_index in PyG format (dst, src)
    src, dst = graph.edges()
    edge_index = torch.stack([dst, src], dim=0).to(device)

    in_feats = features.shape[1]
    num_classes = data.num_classes

    print("\nDataset splits:")
    print(f"  Train: {train_mask.sum().item():,} nodes")
    print(f"  Val:   {val_mask.sum().item():,} nodes")
    print(f"  Test:  {test_mask.sum().item():,} nodes")

    weight = (1 - labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print(f"\nClass weight (fraud): {weight:.2f}")

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = GatedEnergySAGE(
        in_feats=in_feats,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        aggregator_type="mean",
        attention_reduction=args.attention_reduction,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_type=args.gate_type,
        gate_num_layers=args.gate_num_layers,
    ).to(device)

    print(f"\nUsing Spectral-Flip Gating with {args.gate_type} gates")
    print(f"Gate MLP: {args.gate_num_layers} layers, hidden dim: {args.gate_hidden_dim}")
    print("Gate initialization: weight=0, bias=-2.0 (prefers flip at start)")
    print(f"Attention reduction ratio: {args.attention_reduction}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"  Spectral Gate: {sum(p.numel() for p in model.spectral_gate.parameters()):,}")
    print(f"  Feature Attention: {sum(p.numel() for p in model.feature_attention.parameters()):,}")
    print(f"  GraphSAGE Layers: {sum(p.numel() for p in [*model.conv1.parameters(), *model.conv2.parameters(), *model.conv3.parameters()]):,}")
    print(f"  Classifier: {sum(p.numel() for p in model.classifier.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, verbose=True, min_lr=1e-6
    )

    best_val_f1 = 0.0
    best_test_metrics = {}
    best_model_state = None

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    time_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        logits = model(graph, features, edge_index, train_mask=train_mask)

        loss = F.cross_entropy(
            logits[train_mask],
            labels[train_mask],
            weight=torch.tensor([1.0, weight], device=device),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features, edge_index, train_mask=None)
            probs = logits.softmax(1)

            # Validation: find best threshold
            val_f1, val_thres = get_best_f1(
                labels[val_mask].cpu().numpy(),
                probs[val_mask].cpu().numpy(),
            )

            # Test: apply threshold and compute all metrics
            preds = np.zeros_like(labels.cpu().numpy())
            preds[probs[:, 1].cpu().numpy() > val_thres] = 1

            test_labels = labels[test_mask].cpu().numpy()
            test_preds = preds[test_mask.cpu().numpy()]
            test_probs = probs[test_mask][:, 1].cpu().numpy()

            # Compute comprehensive metrics
            test_metrics = compute_comprehensive_metrics(test_labels, test_probs, test_preds)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_test_metrics = {
                    "recall": test_metrics['recall'],
                    "precision": test_metrics['precision'],
                    "macro_f1": test_metrics['macro_f1'],
                    "auroc": test_metrics['auroc'],
                    "auprc": test_metrics['auprc'],
                    "reck": test_metrics['reck'],
                    "threshold": val_thres,
                    "epoch": epoch + 1,
                }
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            scheduler.step(val_f1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Val F1: {val_f1:.4f} (best: {best_val_f1:.4f}) | "
                f"Test F1: {test_metrics['macro_f1']:.4f} | "
                f"AUROC: {test_metrics['auroc']:.4f} | "
                f"AUPRC: {test_metrics['auprc']:.4f}"
            )

    time_end = time.time()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_test_metrics['epoch']}")

    # Gate analysis
    print("\n" + "=" * 70)
    print("SPECTRAL-FLIP GATE ANALYSIS")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        _ = model(graph, features, edge_index, train_mask=None)
        gate_stats = model.get_gate_statistics()

        if gate_stats is not None:
            print("\nGate statistics (0 = pure flip, 1 = pure normal):")
            print(f"  Mean gate: {gate_stats['mean_gate']:.4f}")
            print(f"  Std gate:  {gate_stats['std_gate']:.4f}")
            print(f"  Min gate:  {gate_stats['min_gate']:.4f}")
            print(f"  Max gate:  {gate_stats['max_gate']:.4f}")

            if args.gate_type == "per_feature" and "mean_gate_per_feature" in gate_stats:
                per_feat_gates = gate_stats["mean_gate_per_feature"]
                sorted_indices = torch.argsort(per_feat_gates, descending=True)

                print("\nTop 5 features preferring NORMAL energy (gate → 1):")
                for i, idx in enumerate(sorted_indices[:5]):
                    print(f"    {i+1}. Feature {idx.item():2d}: {per_feat_gates[idx].item():.4f}")

                print("Top 5 features preferring FLIPPED energy (gate → 0):")
                for i, idx in enumerate(sorted_indices[-5:]):
                    print(f"    Feature {idx.item():2d}: {per_feat_gates[idx].item():.4f}")

    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        _ = model(graph, features, edge_index, train_mask=None)
        feature_importance = model.get_feature_importance()

        if feature_importance is not None:
            feature_importance = feature_importance.squeeze()
            if feature_importance.dim() == 1 and feature_importance.size(0) > 1:
                sorted_indices = torch.argsort(feature_importance, descending=True)
                top_k = min(10, feature_importance.size(0))
                bottom_k = min(5, feature_importance.size(0))

                print("\nLearned feature importance weights (average across all nodes):")
                print(f"\nTop {top_k} most important energy dimensions:")
                for i, idx in enumerate(sorted_indices[:top_k]):
                    print(f"  {i+1}. Feature {idx.item():2d}: {feature_importance[idx].item():.4f}")

                print(f"\nBottom {bottom_k} least important energy dimensions:")
                for i, idx in enumerate(sorted_indices[-bottom_k:]):
                    print(f"  Feature {idx.item():2d}: {feature_importance[idx].item():.4f}")

    # Final results with all metrics
    print("\n" + "=" * 70)
    print("FINAL RESULTS - COMPREHENSIVE METRICS")
    print("=" * 70)
    print(f"\nTraining completed in {time_end - time_start:.2f}s")
    print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_test_metrics['epoch']}")
    print(f"Optimal threshold: {best_test_metrics['threshold']:.4f}")

    print("\n" + "-" * 70)
    print("Test Results (at best validation epoch):")
    print("-" * 70)

    print("\nClassification Metrics (with optimal threshold):")
    print(f"  Recall:    {best_test_metrics['recall']*100:.2f}%")
    print(f"  Precision: {best_test_metrics['precision']*100:.2f}%")
    print(f"  Macro F1:  {best_test_metrics['macro_f1']*100:.2f}%")

    print("\nRanking Metrics (probability-based):")
    print(f"  AUROC:     {best_test_metrics['auroc']*100:.2f}%")
    print(f"  AUPRC:     {best_test_metrics['auprc']*100:.2f}%")
    print(f"  RecK:      {best_test_metrics['reck']*100:.2f}%")

    print("\nMetric Descriptions:")
    print("  - AUROC: Overall ability to distinguish positive from negative")
    print("  - AUPRC: Precision-Recall trade-off (better for imbalanced data)")
    print("  - RecK:  % of positives captured in top-K predictions (K = # positives)")

    best_test_metrics["feature_importance"] = feature_importance
    best_test_metrics["gate_stats"] = gate_stats
    return best_test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GatedEnergySAGE with Comprehensive Metrics")

    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="yelp", choices=["amazon", "yelp", "tfinance", "tsocial"])
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--undirected", action="store_true")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--attention_reduction", type=int, default=10)

    # Gating parameters
    parser.add_argument("--gate_type", type=str, default="per_feature", choices=["per_node", "per_feature"])
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--gate_num_layers", type=int, default=2, choices=[2, 3, 4])

    # Training parameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=47)

    # Multiple runs
    parser.add_argument("--runs", type=int, default=10)

    args = parser.parse_args()

    if args.runs == 1:
        train(args)
    else:
        print("\n" + "=" * 70)
        print(f"RUNNING {args.runs} EXPERIMENTS")
        print("=" * 70)

        print("\n" + "=" * 70)
        print("LOADING DATASET (SHARED ACROSS ALL RUNS)")
        print("=" * 70)

        initial_seed = 2
        if args.dataset == "amazon":
            data = amazon_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=initial_seed,
                homo=True,
                undirected=args.undirected,
                verbose=True,
            )
        elif args.dataset == "yelp":
            data = yelp_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=initial_seed,
                homo=True,
                undirected=args.undirected,
                verbose=True,
            )
        elif args.dataset == "tfinance":
            data = tfinance_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=initial_seed,
                undirected=args.undirected,
                verbose=True,
            )
        elif args.dataset == "tsocial":
            data = tsocial_data(
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                random_state=initial_seed,
                undirected=args.undirected,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        all_metrics = {
            "recall": [],
            "precision": [],
            "macro_f1": [],
            "auroc": [],
            "auprc": [],
            "reck": []
        }
        all_feature_importances = []
        all_gate_stats = []

        for run in range(args.runs):
            print(f"\n{'='*70}")
            print(f"RUN {run+1}/{args.runs}")
            print("=" * 70)

            args.seed = 2 + run
            metrics = train(args, data=data, print_config=False)

            for k in all_metrics:
                all_metrics[k].append(metrics[k])

            if metrics.get("feature_importance", None) is not None:
                all_feature_importances.append(metrics["feature_importance"])
            if metrics.get("gate_stats", None) is not None:
                all_gate_stats.append(metrics["gate_stats"])

        print("\n" + "=" * 70)
        print("AGGREGATED RESULTS - COMPREHENSIVE METRICS")
        print("=" * 70)

        print("\nClassification Metrics:")
        for metric_name in ["recall", "precision", "macro_f1"]:
            values = all_metrics[metric_name]
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            print(f"  {metric_name.replace('_', ' ').title():12s}: {mean_val:.2f} ± {std_val:.2f}")

        print("\nRanking Metrics:")
        for metric_name in ["auroc", "auprc", "reck"]:
            values = all_metrics[metric_name]
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            print(f"  {metric_name.upper():12s}: {mean_val:.2f} ± {std_val:.2f}")

        if len(all_gate_stats) > 0:
            print("\n" + "=" * 70)
            print("GATE STATISTICS ACROSS ALL RUNS")
            print("=" * 70)
            mean_gates = [s["mean_gate"] for s in all_gate_stats]
            print(f"\nAverage gate value: {np.mean(mean_gates):.4f} ± {np.std(mean_gates):.4f}")
            print("  (0 = pure flipped energy, 1 = pure normal energy)")

        if len(all_feature_importances) > 0:
            print("\n" + "=" * 70)
            print("FEATURE IMPORTANCE ANALYSIS ACROSS ALL RUNS")
            print("=" * 70)

            avg_rankings, feature_stats = ana_feat(all_feature_importances)
            sorted_by_rank = np.argsort(avg_rankings)

            print(f"\nAnalyzed {len(all_feature_importances)} runs")
            print(f"Total features: {len(avg_rankings)}")

            print("\nTop 10 most consistently important features (by average ranking):")
            for i, feat_idx in enumerate(sorted_by_rank[:10]):
                print(
                    f"  {i+1}. Feature {feat_idx:2d}: "
                    f"Avg Rank {avg_rankings[feat_idx]:.2f} ± {feature_stats['std_ranking'][feat_idx]:.2f} | "
                    f"Avg Importance {feature_stats['mean_importance'][feat_idx]:.4f} ± {feature_stats['std_importance'][feat_idx]:.4f}"
                )

            print("\nBottom 5 least important features (by average ranking):")
            for i, feat_idx in enumerate(sorted_by_rank[-5:]):
                print(
                    f"  Feature {feat_idx:2d}: "
                    f"Avg Rank {avg_rankings[feat_idx]:.2f} ± {feature_stats['std_ranking'][feat_idx]:.2f} | "
                    f"Avg Importance {feature_stats['mean_importance'][feat_idx]:.4f} ± {feature_stats['std_importance'][feat_idx]:.4f}"
                )


if __name__ == "__main__":
    main()
