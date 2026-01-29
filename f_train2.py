

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import time

try:
    import dgl
except ImportError:
    raise ImportError("DGL is required.")

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
)

from f_model2 import FastTemporalGatedEnergySAGE
from dataloader_semi import load_semi_supervised_data


def get_best_f1(labels, probs):
    """Find best threshold for macro-F1 on validation set."""
    best_f1, best_thres = 0.0, 0.0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1
        mf1 = f1_score(labels, preds, average="macro")
        if mf1 > best_f1:
            best_f1 = mf1
            best_thres = thres
    return best_f1, best_thres


def compute_metrics(labels, probs, preds):
    """Compute comprehensive metrics."""
    metrics = {}
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['macro_f1'] = f1_score(labels, preds, average="macro", zero_division=0)
    metrics['binary_f1'] = f1_score(labels, preds, zero_division=0)

    try:
        metrics['auroc'] = roc_auc_score(labels, probs)
    except ValueError:
        metrics['auroc'] = 0.0

    try:
        metrics['auprc'] = average_precision_score(labels, probs)
    except ValueError:
        metrics['auprc'] = 0.0

    k = int(labels.sum())
    if k > 0:
        top_k_indices = np.argsort(probs)[-k:]
        metrics['reck'] = labels[top_k_indices].sum() / k
    else:
        metrics['reck'] = 0.0

    return metrics



def batch_inference(model, graph, features, edge_index, batch_size, device=None):
    """Run inference in batches, moving each batch to GPU on-the-fly."""
    model.eval()
    all_logits = []
    num_samples = features.shape[0]
    if device is None:
        device = next(model.parameters()).device

    for i in range(0, num_samples, batch_size):
        batch_features = features[i:i+batch_size].to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(graph, batch_features, edge_index)
        all_logits.append(logits.cpu())

    return torch.cat(all_logits, dim=0)




def train_tegnn(args, data=None, verbose=True):
    """
    Train TEGNN with proper normalization (no data leak).
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print("=" * 70)
        print("TEGNN V2 - WITH NORMALIZATION (NO DATA LEAK)")
        print("=" * 70)
        print(f"\nDevice: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load Data ----
    if data is None:
        data = load_semi_supervised_data(
            args.dataset,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            random_state=args.seed,
            config={'slide_win': args.slide_win, 'slide_stride': args.slide_stride},
            verbose=verbose,
        )

    # Only move edge_index and masks to GPU; keep features/labels on CPU
    edge_index = data.edge_index.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    # Create DGL graph
    src, dst = data.edge_index[0], data.edge_index[1]
    graph = dgl.graph((src.numpy(), dst.numpy()), num_nodes=data.num_nodes)
    graph = graph.to(device)

    if verbose:
        print(f"\nModel Configuration:")
        print(f"  Num nodes: {data.num_nodes}")
        print(f"  Window size: {data.window_size}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Gate type: {args.gate_type}")
        print(f"  Dropout: {args.dropout}")
        print(f"  WITH Z-SCORE NORMALIZATION (training stats only)")

    # ---- Create Model ----
    model = FastTemporalGatedEnergySAGE(
        num_nodes=data.num_nodes,
        window_size=data.window_size,
        hidden_dim=args.hidden_dim,
        num_classes=data.num_classes,
        dropout=args.dropout,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_type=args.gate_type,
        gate_num_layers=args.gate_num_layers,
    ).to(device).float()

    # ---- Compute normalization stats from TRAINING data only ----
    train_mask_cpu = data.train_mask.cpu()
    train_features = data.features[train_mask_cpu].float()
    if verbose:
        print(f"\nComputing normalization stats from {train_features.shape[0]} training samples...")
    model.set_normalization_stats(train_features.to(device))
    if verbose:
        print("  Stats computed and cached.")

    if verbose:
        print(f"\nTotal parameters: {model.get_num_parameters():,}")

    # ---- Class Weight ----
    train_labels_np = data.labels[train_mask_cpu].numpy()
    n_normal = (train_labels_np == 0).sum()
    n_anomaly = (train_labels_np == 1).sum()
    weight = n_normal / n_anomaly if n_anomaly > 0 else 1.0
    if verbose:
        print(f"Class weight (anomaly): {weight:.2f}")

    # ---- Optimizer and Scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.patience,
        verbose=verbose, min_lr=1e-6
    )

    # ---- Data Loaders (all features stay on CPU, moved to GPU per-batch) ----
    train_features_cpu = data.features[train_mask_cpu].float()
    train_labels_cpu = data.labels[train_mask_cpu].long()
    train_dataset = TensorDataset(train_features_cpu, train_labels_cpu)

    num_workers = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    val_mask_cpu = data.val_mask.cpu()
    test_mask_cpu = data.test_mask.cpu()
    val_features = data.features[val_mask_cpu].float()
    val_labels = data.labels[val_mask_cpu].long()
    test_features = data.features[test_mask_cpu].float()
    test_labels = data.labels[test_mask_cpu].long()

    class_weight = torch.tensor([1.0, weight], device=device, dtype=torch.float32)
    num_batches = len(train_loader)

    if verbose:
        print(f"\nTraining with batch_size={args.batch_size}, {num_batches} batches/epoch")
        print("\n" + "=" * 70)
        print("TRAINING")
        print("=" * 70)

    # ---- Training Loop ----
    best_val_f1 = 0.0
    best_test_metrics = {}
    best_model_state = None

    time_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=True).float()
            batch_labels = batch_labels.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            logits = model(graph, batch_features, edge_index)
            loss = F.cross_entropy(logits, batch_labels, weight=class_weight)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches

        # ---- Evaluation ----
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # Validation
                val_logits = batch_inference(
                    model, graph, val_features, edge_index, args.batch_size, device
                )
                val_probs = val_logits.softmax(1).numpy()
                val_labels_np = val_labels.numpy()

                val_f1, val_thres = get_best_f1(val_labels_np, val_probs)

                # Test
                test_logits = batch_inference(
                    model, graph, test_features, edge_index, args.batch_size, device
                )
                test_probs = test_logits.softmax(1).numpy()
                test_labels_np = test_labels.numpy()

                test_preds = (test_probs[:, 1] > val_thres).astype(int)
                test_metrics = compute_metrics(test_labels_np, test_probs[:, 1], test_preds)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_test_metrics = {
                        **test_metrics,
                        "threshold": val_thres,
                        "epoch": epoch + 1,
                    }
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                scheduler.step(val_f1)

            if verbose and ((epoch + 1) % args.log_every == 0 or epoch == 0):
                print(
                    f"Epoch {epoch+1:3d}/{args.epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Val F1: {val_f1:.4f} (best: {best_val_f1:.4f}) | "
                    f"Test F1: {test_metrics['macro_f1']:.4f} | "
                    f"AUROC: {test_metrics['auroc']:.4f}"
                )

    time_end = time.time()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ---- Final Results ----
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"\nTraining completed in {time_end - time_start:.2f}s")
        print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_test_metrics.get('epoch', 'N/A')}")
        print(f"Optimal threshold: {best_test_metrics.get('threshold', 'N/A'):.4f}")

        print("\n" + "-" * 70)
        print("Test Results:")
        print("-" * 70)

        print("\nClassification Metrics:")
        print(f"  Macro F1:  {best_test_metrics['macro_f1']*100:.2f}%")
        print(f"  Binary F1: {best_test_metrics['binary_f1']*100:.2f}%")
        print(f"  Recall:    {best_test_metrics['recall']*100:.2f}%")
        print(f"  Precision: {best_test_metrics['precision']*100:.2f}%")

        print("\nRanking Metrics:")
        print(f"  AUROC:     {best_test_metrics['auroc']*100:.2f}%")
        print(f"  AUPRC:     {best_test_metrics['auprc']*100:.2f}%")
        print(f"  RecK:      {best_test_metrics['reck']*100:.2f}%")

    return best_test_metrics




def run_experiments(args):
    """Run multiple experiments and aggregate results."""
    print("\n" + "=" * 70)
    print(f"RUNNING {args.runs} EXPERIMENTS (WITH NORMALIZATION)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    data = load_semi_supervised_data(
        args.dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        config={'slide_win': args.slide_win, 'slide_stride': args.slide_stride},
        verbose=True,
    )

    all_metrics = {
        "macro_f1": [], "binary_f1": [], "auroc": [],
        "auprc": [], "reck": [], "recall": [], "precision": [],
    }

    for run in range(args.runs):
        print(f"\n{'='*70}")
        print(f"RUN {run+1}/{args.runs}")
        print("=" * 70)

        args.seed = args.base_seed + run
        data.regenerate_masks(args.train_ratio, args.val_ratio, args.seed)

        metrics = train_tegnn(args, data=data, verbose=(run == 0))

        for k in all_metrics:
            if k in metrics:
                all_metrics[k].append(metrics[k])

    # ---- Aggregated Results ----
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (WITH NORMALIZATION)")
    print("=" * 70)

    print("\nClassification Metrics:")
    for metric_name in ["macro_f1", "binary_f1", "recall", "precision"]:
        values = all_metrics[metric_name]
        if len(values) > 0:
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            label = metric_name.replace('_', ' ').title()
            print(f"  {label:16s}: {mean_val:.2f} ± {std_val:.2f}")

    print("\nRanking Metrics:")
    for metric_name in ["auroc", "auprc", "reck"]:
        values = all_metrics[metric_name]
        if len(values) > 0:
            mean_val = np.mean(values) * 100
            std_val = np.std(values) * 100
            print(f"  {metric_name.upper():12s}: {mean_val:.2f} ± {std_val:.2f}")

    return all_metrics




def main():
    parser = argparse.ArgumentParser(description="Train TEGNN V2 (with normalization)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="msl", choices=["msl", "swat", "wadi"])
    parser.add_argument("--train_ratio", type=float, default=0.4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--slide_win", type=int, default=15)
    parser.add_argument("--slide_stride", type=int, default=1)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)

    # Gating
    parser.add_argument("--gate_type", type=str, default="per_feature", choices=["per_node", "per_feature"])
    parser.add_argument("--gate_hidden_dim", type=int, default=16)
    parser.add_argument("--gate_num_layers", type=int, default=2, choices=[2, 3])

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=5)

    # Experiment
    parser.add_argument("--seed", type=int, default=1542)
    parser.add_argument("--base_seed", type=int, default=1542)
    parser.add_argument("--runs", type=int, default=1)

    args = parser.parse_args()

    if args.runs == 1:
        train_tegnn(args)
    else:
        run_experiments(args)


if __name__ == "__main__":
    main()
