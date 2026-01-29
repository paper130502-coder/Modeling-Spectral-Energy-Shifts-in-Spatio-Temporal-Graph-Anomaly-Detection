"""
Semi-supervised Data Loader for Temporal Graph Anomaly Detection.

This module provides data loading utilities for time series anomaly detection
with graph structure. It supports:
- MSL (Mars Science Laboratory) dataset
- SWaT (Secure Water Treatment) dataset
- WADI (Water Distribution) dataset

The data is organized as:
- Features: sliding window of time series values per node
- Labels: binary anomaly labels (0=normal, 1=anomaly)
- Graph: fully connected graph or custom adjacency

Caching: Preprocessed data is saved as .npy files for fast loading.
Run `python dataloader_semi.py --preprocess swat wadi` to create cache.

Usage:
    data = load_semi_supervised_data("msl", train_ratio=0.4, val_ratio=0.2)
    print(data.features.shape)  # (num_samples, num_nodes, window_size)
"""

import torch
import numpy as np
import pandas as pd
import os
import hashlib
from sklearn.preprocessing import MinMaxScaler


def get_cache_path(data_dir, dataset_name, window_size, stride):
    """Get cache file path for preprocessed data."""
    cache_dir = os.path.join(data_dir, dataset_name, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = f"w{window_size}_s{stride}"
    return os.path.join(cache_dir, cache_name)


def save_to_cache(cache_path, features, labels, num_nodes):
    """Save preprocessed data to .npy files."""
    np.save(f"{cache_path}_features.npy", features)
    np.save(f"{cache_path}_labels.npy", labels)
    np.save(f"{cache_path}_meta.npy", np.array([num_nodes]))
    print(f"  Cached to: {cache_path}_*.npy")


def load_from_cache(cache_path):
    """Load preprocessed data from .npy files."""
    features = np.load(f"{cache_path}_features.npy")
    labels = np.load(f"{cache_path}_labels.npy")
    meta = np.load(f"{cache_path}_meta.npy")
    num_nodes = int(meta[0])
    return features, labels, num_nodes


def cache_exists(cache_path):
    """Check if cache files exist."""
    return (os.path.exists(f"{cache_path}_features.npy") and
            os.path.exists(f"{cache_path}_labels.npy") and
            os.path.exists(f"{cache_path}_meta.npy"))


class TemporalGraphData:
    """
    Container for temporal graph data with train/val/test splits.

    Attributes:
        features: (num_samples, num_nodes, window_size) tensor
        labels: (num_samples,) tensor of binary labels
        edge_index: (2, num_edges) tensor of edge indices
        train_mask, val_mask, test_mask: boolean masks for splits
        num_nodes: number of nodes in the graph
        window_size: size of the sliding window
        num_classes: number of classes (2 for binary)
    """

    def __init__(
        self,
        features,
        labels,
        edge_index,
        train_mask,
        val_mask,
        test_mask,
        num_nodes,
        window_size,
    ):
        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.num_classes = 2

    def to(self, device):
        """Move all tensors to device."""
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self

    def get_class_weight(self):
        """Compute class weight for imbalanced data."""
        train_labels = self.labels[self.train_mask].cpu().numpy()
        n_normal = (train_labels == 0).sum()
        n_anomaly = (train_labels == 1).sum()
        if n_anomaly == 0:
            return 1.0
        return n_normal / n_anomaly

    def regenerate_masks(self, train_ratio, val_ratio, random_state):
        """Regenerate train/val/test masks with a new random seed."""
        n_samples = len(self.labels)
        indices = np.arange(n_samples)

        np.random.seed(random_state)
        np.random.shuffle(indices)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        device = self.features.device

        self.train_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
        self.val_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
        self.test_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)

        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True


def create_fully_connected_edge_index(num_nodes):
    """Create edge index for a fully connected graph (no self-loops)."""
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def create_sliding_windows(data, labels, window_size, stride=1):
    """
    Create sliding window samples from time series data.

    Args:
        data: (time_steps, num_nodes) array
        labels: (time_steps,) array of labels
        window_size: size of sliding window
        stride: step size between windows

    Returns:
        features: (num_samples, num_nodes, window_size)
        sample_labels: (num_samples,) - 1 if any label in window is 1
    """
    time_steps, num_nodes = data.shape
    num_samples = (time_steps - window_size) // stride + 1

    features = []
    sample_labels = []

    for i in range(0, time_steps - window_size + 1, stride):
        window = data[i : i + window_size, :]  # (window_size, num_nodes)
        window = window.T  # (num_nodes, window_size)
        features.append(window)

        # Label is 1 if any point in the window (or next point) is anomaly
        window_labels = labels[i : i + window_size]
        sample_labels.append(1 if window_labels.max() > 0 else 0)

    features = np.stack(features)  # (num_samples, num_nodes, window_size)
    sample_labels = np.array(sample_labels)

    return features, sample_labels


def load_msl_data(data_dir, config, verbose=True):
    """Load MSL dataset."""
    train_path = os.path.join(data_dir, "msl", "train.csv")
    test_path = os.path.join(data_dir, "msl", "test.csv")
    list_path = os.path.join(data_dir, "msl", "list.txt")

    # Read feature names
    with open(list_path, "r") as f:
        feature_names = [line.strip() for line in f if line.strip()]

    # Read data
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)

    # Extract features in correct order
    train_data = train_df[feature_names].values
    test_data = test_df[feature_names].values
    test_labels = test_df["attack"].values if "attack" in test_df.columns else np.zeros(len(test_df))

    # Normalize
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Create train labels (all normal)
    train_labels = np.zeros(len(train_data))

    if verbose:
        print(f"MSL Dataset:")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        print(f"  Num features/nodes: {len(feature_names)}")
        print(f"  Test anomaly ratio: {test_labels.mean()*100:.2f}%")

    return train_data, train_labels, test_data, test_labels, len(feature_names)


def load_swat_data(data_dir, config, verbose=True):
    """Load SWaT dataset."""
    # Check for preprocessed files first
    train_path = os.path.join(data_dir, "swat", "train.csv")
    test_path = os.path.join(data_dir, "swat", "test.csv")
    list_path = os.path.join(data_dir, "swat", "list.txt")

    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(list_path):
        # Use preprocessed files
        with open(list_path, "r") as f:
            feature_names = [line.strip() for line in f if line.strip()]

        train_df = pd.read_csv(train_path, index_col=0)
        test_df = pd.read_csv(test_path, index_col=0)

        train_data = train_df[feature_names].values
        test_data = test_df[feature_names].values
        test_labels = test_df["attack"].values if "attack" in test_df.columns else np.zeros(len(test_df))
    else:
        # Load raw SWaT files
        normal_path = os.path.join(data_dir, "swat", "normal.csv")
        merged_path = os.path.join(data_dir, "swat", "merged.csv")

        if verbose:
            print("Loading raw SWaT files...")

        # Read training data (normal only)
        train_df = pd.read_csv(normal_path)
        train_df.columns = [c.strip() for c in train_df.columns]

        # Read merged data for testing
        # Note: merged.csv = normal.csv + attack.csv concatenated
        # This gives ~3.79% anomaly ratio (not the standard 11-12%)
        # For standard benchmark, need original SWaT test file from iTrust
        test_df = pd.read_csv(merged_path)
        test_df.columns = [c.strip() for c in test_df.columns]

        if verbose:
            print(f"  Using merged.csv for test ({len(test_df)} rows)")

        # Get feature columns (exclude Timestamp and Normal/Attack)
        feature_names = [c for c in train_df.columns if c not in ['Timestamp', 'Normal/Attack']]

        # Extract features
        train_data = train_df[feature_names].values
        test_data = test_df[feature_names].values

        # Extract labels from Normal/Attack column
        if 'Normal/Attack' in test_df.columns:
            test_labels = (test_df['Normal/Attack'].str.strip() == 'Attack').astype(int).values
        else:
            test_labels = np.zeros(len(test_df))

        # Handle missing values
        train_data = np.nan_to_num(train_data, nan=0.0)
        test_data = np.nan_to_num(test_data, nan=0.0)

    # Normalize
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Create train labels (all normal)
    train_labels = np.zeros(len(train_data))

    if verbose:
        print(f"SWaT Dataset:")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        print(f"  Num features/nodes: {len(feature_names)}")
        print(f"  Test anomaly ratio: {test_labels.mean()*100:.2f}%")

    return train_data, train_labels, test_data, test_labels, len(feature_names)


def load_wadi_data(data_dir, config, verbose=True):
    """Load WADI dataset."""
    # Check for preprocessed files first
    train_path = os.path.join(data_dir, "wadi", "train.csv")
    test_path = os.path.join(data_dir, "wadi", "test.csv")
    list_path = os.path.join(data_dir, "wadi", "list.txt")

    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(list_path):
        # Use preprocessed files (same format as MSL/SWaT)
        with open(list_path, "r") as f:
            feature_names = [line.strip() for line in f if line.strip()]

        train_df = pd.read_csv(train_path, index_col=0)
        test_df = pd.read_csv(test_path, index_col=0)

        train_data = train_df[feature_names].values
        test_data = test_df[feature_names].values
        test_labels = test_df["attack"].values if "attack" in test_df.columns else np.zeros(len(test_df))
    else:
        # Load raw WADI files
        raw_train_path = os.path.join(data_dir, "wadi", "WADI_14days.csv")
        raw_test_path = os.path.join(data_dir, "wadi", "WADI_attackdata.csv")

        if verbose:
            print("Loading raw WADI files (this may take a while)...")

        # Read raw data, skip metadata rows
        train_df = pd.read_csv(raw_train_path, skiprows=4)
        test_df = pd.read_csv(raw_test_path)

        # Get feature columns (exclude Row, Date, Time)
        feature_cols = [c for c in train_df.columns if c not in ['Row', 'Date', 'Time']]

        # Clean column names (remove the long prefix)
        clean_names = []
        for col in feature_cols:
            # Extract the sensor name from the path
            name = col.split('\\')[-1] if '\\' in col else col
            clean_names.append(name)

        # Rename columns
        rename_map = dict(zip(feature_cols, clean_names))
        train_df = train_df.rename(columns=rename_map)
        test_df = test_df.rename(columns=rename_map)

        feature_names = clean_names

        # Extract features
        train_data = train_df[feature_names].values
        test_data = test_df[feature_names].values

        # Handle missing values
        train_data = np.nan_to_num(train_data, nan=0.0)
        test_data = np.nan_to_num(test_data, nan=0.0)

        # Remove constant columns (zero variance in training data)
        stds = train_data.std(axis=0)
        non_constant = stds > 0
        if verbose:
            print(f"  Removing {(~non_constant).sum()} constant columns out of {len(feature_names)}")
        train_data = train_data[:, non_constant]
        test_data = test_data[:, non_constant]
        feature_names = [f for f, keep in zip(feature_names, non_constant) if keep]

        # WADI attack labels derived from attack_description.xlsx
        # Data collection: 9-Oct-2017 18:00:00 to 11-Oct-2017 18:00:00 (1 sample/sec)
        # Row indices are 0-based (seconds elapsed since start)
        test_labels = np.zeros(len(test_data))

        attack_periods = [
            (5100, 6616),      # Attack 1:  Oct 9  19:25:00 - 19:50:16
            (59050, 59640),    # Attack 2:  Oct 10 10:24:10 - 10:34:00
            (60900, 62640),    # Attack 3:  Oct 10 10:55:00 - 11:24:00
            (61666, 61935),    # Attack 4:  Oct 10 11:07:46 - 11:12:15
            (63040, 63890),    # Attack 5:  Oct 10 11:30:40 - 11:44:50
            (70770, 71440),    # Attack 6:  Oct 10 13:39:30 - 13:50:40
            (74897, 75595),    # Attack 7:  Oct 10 14:48:17 - 14:59:55
            (75224, 75632),    # Attack 7b: Oct 10 14:53:44 - 15:00:32
            (85200, 85780),    # Attack 8:  Oct 10 17:40:00 - 17:49:40
            (147300, 147387),  # Attack 9:  Oct 11 10:55:00 - 10:56:27
            (148674, 149480),  # Attack 10: Oct 11 11:17:54 - 11:31:20
            (149791, 150420),  # Attack 11: Oct 11 11:36:31 - 11:47:00
            (151140, 151500),  # Attack 12: Oct 11 11:59:00 - 12:05:00
            (151650, 151852),  # Attack 13: Oct 11 12:07:30 - 12:10:52
            (152160, 152736),  # Attack 14: Oct 11 12:16:00 - 12:25:36
            (163590, 164220),  # Attack 15: Oct 11 15:26:30 - 15:37:00
        ]

        for start, end in attack_periods:
            if end < len(test_labels):
                test_labels[start:end+1] = 1

    # Normalize
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Create train labels (all normal)
    train_labels = np.zeros(len(train_data))

    if verbose:
        print(f"WADI Dataset:")
        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape: {test_data.shape}")
        print(f"  Num features/nodes: {len(feature_names)}")
        print(f"  Test anomaly ratio: {test_labels.mean()*100:.2f}%")

    return train_data, train_labels, test_data, test_labels, len(feature_names)


def load_semi_supervised_data(
    dataset_name,
    train_ratio=0.4,
    val_ratio=0.2,
    random_state=42,
    config=None,
    verbose=True,
    data_dir=None,
    use_cache=True,
):
    """
    Load dataset for semi-supervised anomaly detection.

    The data is split as:
    - Training: train_ratio of test data (for learning patterns)
    - Validation: val_ratio of test data (for threshold tuning)
    - Test: remaining test data (for final evaluation)

    Args:
        dataset_name: "msl", "swat", or "wadi"
        train_ratio: fraction of data for training
        val_ratio: fraction of data for validation
        random_state: random seed for reproducibility
        config: dict with 'slide_win' and 'slide_stride'
        verbose: print loading info
        data_dir: path to data directory (default: ./data or ../data)
        use_cache: use cached .npy files if available (default: True)

    Returns:
        TemporalGraphData object
    """
    if config is None:
        config = {"slide_win": 15, "slide_stride": 1}

    window_size = config.get("slide_win", 15)
    stride = config.get("slide_stride", 1)

    # Find data directory
    if data_dir is None:
        if os.path.exists("./data"):
            data_dir = "./data"
        elif os.path.exists("../data"):
            data_dir = "../data"
        else:
            # Try relative to this file
            this_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(this_dir, "data")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(os.path.dirname(this_dir), "data")

    if verbose:
        print(f"\nLoading {dataset_name.upper()} dataset from {data_dir}")
        print(f"Window size: {window_size}, Stride: {stride}")

    # Check for cached data
    cache_path = get_cache_path(data_dir, dataset_name.lower(), window_size, stride)

    if use_cache and cache_exists(cache_path):
        if verbose:
            print(f"Loading from cache: {cache_path}")
        test_features, test_sample_labels, num_nodes = load_from_cache(cache_path)
        if verbose:
            print(f"  Loaded {len(test_features)} samples from cache")
    else:
        # Load raw data
        if verbose and use_cache:
            print("Cache not found, loading from raw files...")

        if dataset_name.lower() == "msl":
            train_data, train_labels, test_data, test_labels, num_nodes = load_msl_data(
                data_dir, config, verbose
            )
        elif dataset_name.lower() == "swat":
            train_data, train_labels, test_data, test_labels, num_nodes = load_swat_data(
                data_dir, config, verbose
            )
        elif dataset_name.lower() == "wadi":
            train_data, train_labels, test_data, test_labels, num_nodes = load_wadi_data(
                data_dir, config, verbose
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: msl, swat, wadi")

        # Create sliding windows for test data (which contains anomalies)
        test_features, test_sample_labels = create_sliding_windows(
            test_data, test_labels, window_size, stride
        )

        # Save to cache
        if use_cache:
            if verbose:
                print("Saving to cache for faster loading next time...")
            save_to_cache(cache_path, test_features, test_sample_labels, num_nodes)

    if verbose:
        print(f"\nAfter sliding window:")
        print(f"  Test samples: {len(test_features)}")
        print(f"  Sample shape: {test_features.shape}")
        print(f"  Anomaly samples: {test_sample_labels.sum()} ({test_sample_labels.mean()*100:.2f}%)")

    # Create train/val/test split
    n_samples = len(test_features)
    indices = np.arange(n_samples)

    np.random.seed(random_state)
    np.random.shuffle(indices)

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create masks
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Convert to tensors - ensure float32 for features
    features = torch.from_numpy(test_features).float()
    labels = torch.from_numpy(test_sample_labels).long()

    # Create fully connected graph
    edge_index = create_fully_connected_edge_index(num_nodes)

    if verbose:
        print(f"\nData splits:")
        print(f"  Train: {train_mask.sum().item()} samples")
        print(f"  Val:   {val_mask.sum().item()} samples")
        print(f"  Test:  {test_mask.sum().item()} samples")
        print(f"\nGraph:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {edge_index.shape[1]} (fully connected)")

    return TemporalGraphData(
        features=features,
        labels=labels,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
        window_size=window_size,
    )


def preprocess_and_cache(datasets, window_size=15, stride=1, data_dir=None):
    """
    Preprocess datasets and save to cache for fast loading.

    Args:
        datasets: list of dataset names to preprocess
        window_size: sliding window size
        stride: sliding window stride
        data_dir: path to data directory
    """
    print("=" * 70)
    print("PREPROCESSING DATASETS")
    print("=" * 70)
    print(f"Window size: {window_size}, Stride: {stride}")

    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"Preprocessing {ds.upper()}...")
        print("=" * 70)

        try:
            # Force reload (don't use cache) to create fresh cache
            data = load_semi_supervised_data(
                ds,
                config={"slide_win": window_size, "slide_stride": stride},
                data_dir=data_dir,
                use_cache=False,  # Force reload from raw files
                verbose=True,
            )

            # Now save to cache
            if data_dir is None:
                if os.path.exists("./data"):
                    data_dir_resolved = "./data"
                elif os.path.exists("../data"):
                    data_dir_resolved = "../data"
                else:
                    this_dir = os.path.dirname(os.path.abspath(__file__))
                    data_dir_resolved = os.path.join(this_dir, "data")
            else:
                data_dir_resolved = data_dir

            cache_path = get_cache_path(data_dir_resolved, ds.lower(), window_size, stride)
            save_to_cache(
                cache_path,
                data.features.numpy(),
                data.labels.numpy(),
                data.num_nodes
            )
            print(f"✓ {ds.upper()} preprocessed successfully!")

        except Exception as e:
            print(f"✗ Error preprocessing {ds}: {e}")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data loader with preprocessing support")
    parser.add_argument("--preprocess", nargs="+", default=None,
                       help="Datasets to preprocess (e.g., --preprocess swat wadi)")
    parser.add_argument("--window", type=int, default=15, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--test", type=str, default=None,
                       help="Test loading a specific dataset")

    args = parser.parse_args()

    if args.preprocess:
        preprocess_and_cache(args.preprocess, args.window, args.stride)
    elif args.test:
        print(f"Testing {args.test} data loading...")
        data = load_semi_supervised_data(
            args.test,
            config={"slide_win": args.window, "slide_stride": args.stride},
            verbose=True
        )
        print(f"\nFeatures shape: {data.features.shape}")
        print(f"Labels shape: {data.labels.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
    else:
        # Default: test MSL loading
        print("Testing MSL data loading...")
        data = load_semi_supervised_data("msl", verbose=True)
        print(f"\nFeatures shape: {data.features.shape}")
        print(f"Labels shape: {data.labels.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
