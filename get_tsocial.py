"""
TSocial dataset loader for graph-based anomaly detection.
Provides a single class 'tsocial_data' to access train/val/test datasets.
"""

import os
os.environ['DGLBACKEND'] = 'pytorch'

import torch
import dgl
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split


class tsocial_data:
    """
    TSocial fraud detection dataset in homogeneous graph format.

    Usage:
        # Load as homogeneous graph
        data = tsocial_data()

        # Access the graph and features
        graph = data.graph
        features = data.features
        labels = data.labels

        # Access train/val/test splits
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        # Training configuration
        num_classes = data.num_classes
        class_weight = data.class_weight


    """

    def __init__(self, train_ratio=0.4, val_ratio=0.2, random_state=2, undirected=False, verbose=False):
        """
        Initialize and prepare the TSocial dataset.

        Args:
            train_ratio: Ratio of nodes to use for training (default: 0.4)
            val_ratio: Ratio of nodes to use for validation (default: 0.2)
            random_state: Random seed for reproducibility (default: 2)
            undirected: Whether to convert to undirected graph (default: False)
            verbose: Whether to print loading information (default: False)
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.undirected = undirected
        self.verbose = verbose
        self._one_hop_cache = None
        self._one_hop_cache_device = None
        self._one_hop_subgraph_cache = None  # Cache 1-hop subgraph info per node

        # Load and prepare the dataset
        self._load_dataset()
        self._create_splits()

    def _load_dataset(self):
        """Load the TSocial dataset from the binary file."""
        if self.verbose:
            print("Loading TSocial dataset...")

        # Load the graph from the binary file
        graph_list, label_dict = load_graphs('dataset/tsocial')
        self.graph = graph_list[0]

        # Convert labels to proper format
        labels = self.graph.ndata['label']

        # Check if labels are one-hot encoded or already scalar
        if labels.dim() > 1 and labels.size(1) > 1:
            # One-hot format: [1, 0] -> 0, [0, 1] -> 1
            self.labels = labels.argmax(1).long()
        else:
            # Already scalar format
            self.labels = labels.long().squeeze()

        # Add self-loops
        self.graph = dgl.add_self_loop(self.graph)

        if self.verbose:
            print(f"Graph created: {self.graph.num_nodes():,} nodes, {self.graph.num_edges():,} edges")

        # Convert to undirected if requested
        if self.undirected:
            self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)
            if self.verbose:
                print(f"Converted to undirected: {self.graph.num_edges():,} edges (with reverse edges)")

        # Extract node features
        self.num_classes = 2
        self.features = self.graph.ndata['feature'].float()  # Ensure float32

        if self.verbose:
            print(f"Features shape: {self.features.shape}")
            print(f"Features dtype: {self.features.dtype}")
            print(f"Number of anomalies: {self.labels.sum().item():,}")
            print(f"Number of normal: {(self.labels == 0).sum().item():,}")

    def _create_splits(self):
        """Create train/val/test splits from all nodes."""
        # All nodes are labeled in TSocial
        all_indices = list(range(len(self.labels)))
        all_labels = self.labels

        # Calculate test ratio
        test_ratio = 1.0 - self.train_ratio - self.val_ratio

        # First split: train vs rest
        idx_train, idx_rest, _, y_rest = train_test_split(
            all_indices,
            all_labels,
            stratify=all_labels,
            train_size=self.train_ratio,
            random_state=self.random_state,
            shuffle=True
        )

        # Second split: val vs test
        val_size_of_rest = self.val_ratio / (self.val_ratio + test_ratio)
        idx_val, idx_test, _, _ = train_test_split(
            idx_rest,
            y_rest,
            stratify=y_rest,
            train_size=val_size_of_rest,
            random_state=self.random_state,
            shuffle=True
        )

        # Create masks for all nodes
        self.train_mask = torch.zeros([len(self.labels)]).bool()
        self.val_mask = torch.zeros([len(self.labels)]).bool()
        self.test_mask = torch.zeros([len(self.labels)]).bool()

        self.train_mask[idx_train] = 1
        self.val_mask[idx_val] = 1
        self.test_mask[idx_test] = 1

        # Save masks to graph
        self.graph.ndata['label'] = self.labels
        self.graph.ndata['train_mask'] = self.train_mask
        self.graph.ndata['val_mask'] = self.val_mask
        self.graph.ndata['test_mask'] = self.test_mask

        # Calculate class weight for training (for handling class imbalance)
        self.class_weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()

        if self.verbose:
            print(f"\nSplit sizes:")
            print(f"  Train: {self.train_mask.sum().item():,} nodes")
            print(f"  Val:   {self.val_mask.sum().item():,} nodes")
            print(f"  Test:  {self.test_mask.sum().item():,} nodes")
            print(f"\nClass weight for loss: {self.class_weight:.2f}")

    def regenerate_masks(self, train_ratio, val_ratio, random_state):
        """
        Regenerate train/val/test masks with a new random seed.
        This allows reusing the same graph and cached data with different splits.

        Args:
            train_ratio: Ratio of labeled nodes to use for training
            val_ratio: Ratio of labeled nodes to use for validation
            random_state: Random seed for reproducibility
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

        # Recreate splits with new seed
        self._create_splits()

    def get_train_data(self):
        """Get training data."""
        return {
            'features': self.features[self.train_mask],
            'labels': self.labels[self.train_mask],
            'mask': self.train_mask
        }

    def get_val_data(self):
        """Get validation data."""
        return {
            'features': self.features[self.val_mask],
            'labels': self.labels[self.val_mask],
            'mask': self.val_mask
        }

    def get_test_data(self):
        """Get test data."""
        return {
            'features': self.features[self.test_mask],
            'labels': self.labels[self.test_mask],
            'mask': self.test_mask
        }

    def precompute_graph_info(self):
        """
        Precompute and cache graph information for fast training access.
        Call this once before training to avoid repeated computation.
        """
        print("Precomputing graph information for fast training access...")

        # Precompute degrees (very fast to access during training)
        if self.undirected:
            # For undirected graphs, in_degree = out_degree
            self.degrees = self.graph.in_degrees()
        else:
            # For directed graphs, store both
            self.in_degrees = self.graph.in_degrees()
            self.out_degrees = self.graph.out_degrees()
            self.degrees = self.in_degrees + self.out_degrees  # Total degree

        # Cache adjacency matrix for fast subgraph extraction
        self.adj_matrix = self.graph.adjacency_matrix()

        print(f"✓ Precomputed degrees for {self.graph.num_nodes():,} nodes")
        print(f"✓ Cached adjacency matrix")
        # Optionally cache 1-hop PyTorch tensors for faster reuse (lazy creation)
        self._one_hop_cache = None
        self._one_hop_cache_device = None

    def cache_one_hop_torch(self, device=None):
        """
        Convert the DGL graph to PyTorch tensors once and cache 1-hop info.

        Returns:
            dict with:
                edge_index: [2, num_edges] LongTensor (PyTorch style)
                adj: sparse adjacency tensor (float, coalesced)
                degrees: in-degree per node (float tensor)
        """
        if self._one_hop_cache is not None:
            if device is None or self._one_hop_cache_device == device:
                return self._one_hop_cache

        src, dst = self.graph.edges()
        edge_index = torch.stack([src, dst], dim=0)
        num_nodes = self.graph.num_nodes()

        values = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()
        degrees = self.graph.in_degrees().float()

        cache = {
            'edge_index': edge_index,
            'adj': adj,
            'degrees': degrees
        }

        if device is not None:
            cache = {k: v.to(device) for k, v in cache.items()}
            self._one_hop_cache_device = device
        else:
            self._one_hop_cache_device = adj.device

        self._one_hop_cache = cache
        return cache

    def cache_1hop_subgraphs(self):
        """
        Precompute and cache 1-hop subgraph information for ALL nodes (fast version).

        For each node, stores:
        - neighbor_ids: List of node IDs in 1-hop neighborhood (including center)
        - adj_indices: Edge indices for the subgraph (COO format)
        - node_mapping: Mapping from global node ID to local subgraph index

        This allows fast extraction of subgraph Laplacians during energy computation.
        """
        if self._one_hop_subgraph_cache is not None:
            return self._one_hop_subgraph_cache

        print("Caching 1-hop subgraphs for all nodes...")
        num_nodes = self.graph.num_nodes()

        # Extract edges as tensors
        src, dst = self.graph.edges()

        print(f"  Processing {num_nodes:,} nodes and {src.shape[0]:,} edges...")

        # Build edge list per node using list of lists (faster for iteration)
        from collections import defaultdict
        node_edges = defaultdict(list)

        # Store edge indices for each node
        for idx, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
            node_edges[s].append(idx)
            node_edges[d].append(idx)

        src_np = src.numpy()
        dst_np = dst.numpy()

        cache = []

        for node_id in range(num_nodes):
            # Get all edge indices involving this node
            edge_indices = node_edges[node_id]

            if not edge_indices:
                # Isolated node
                cache.append({
                    'neighbor_ids': [node_id],
                    'adj_indices': torch.zeros((2, 0), dtype=torch.long),
                    'num_nodes': 1,
                    'center_local_idx': 0
                })
                continue

            # Get unique neighbors from these edges
            neighbors = set([node_id])  # Include self
            for idx in edge_indices:
                neighbors.add(src_np[idx])
                neighbors.add(dst_np[idx])

            neighbor_list = sorted(list(neighbors))
            neighbor_set = set(neighbor_list)
            num_subgraph_nodes = len(neighbor_list)

            # Filter edges to only those within the subgraph
            subgraph_edges_src = []
            subgraph_edges_dst = []
            for idx in edge_indices:
                s, d = src_np[idx], dst_np[idx]
                if s in neighbor_set and d in neighbor_set:
                    subgraph_edges_src.append(s)
                    subgraph_edges_dst.append(d)

            # Create node mapping
            node_mapping = {global_id: local_idx for local_idx, global_id in enumerate(neighbor_list)}

            # Convert to local indices
            if len(subgraph_edges_src) > 0:
                local_src = torch.tensor([node_mapping[s] for s in subgraph_edges_src], dtype=torch.long)
                local_dst = torch.tensor([node_mapping[d] for d in subgraph_edges_dst], dtype=torch.long)
                adj_indices = torch.stack([local_src, local_dst], dim=0)
            else:
                adj_indices = torch.zeros((2, 0), dtype=torch.long)

            cache.append({
                'neighbor_ids': neighbor_list,
                'adj_indices': adj_indices,
                'num_nodes': num_subgraph_nodes,
                'center_local_idx': node_mapping[node_id]
            })

            # Progress indicator
            if (node_id + 1) % 5000 == 0:
                print(f"    Processed {node_id + 1:,}/{num_nodes:,} nodes...")

        self._one_hop_subgraph_cache = cache
        print(f"✓ Cached 1-hop subgraphs for {num_nodes:,} nodes")
        return cache

    def get_khop_subgraph(self, node_ids, k=2):
        """
        Get k-hop subgraph for given nodes.

        Args:
            node_ids: Node IDs (can be a single int or tensor/list)
            k: Number of hops (default: 2)

        Returns:
            subgraph: DGL subgraph
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]

        # Use DGL's efficient k-hop sampling
        subgraph = dgl.khop_in_subgraph(self.graph, node_ids, k=k)[0]

        return subgraph

    def get_node_degree(self, node_id):
        """
        Get degree of a node (instant access if precomputed).

        Args:
            node_id: Node ID (int or tensor)

        Returns:
            degree: Degree of the node
        """
        if not hasattr(self, 'degrees'):
            raise RuntimeError("Call precompute_graph_info() first for fast degree access")

        return self.degrees[node_id]

    @staticmethod
    def get_degree_matrix(subgraph):
        """
        Get degree matrix for a subgraph (undirected).
        Similar to subgraph.adjacency_matrix().

        Args:
            subgraph: DGL subgraph

        Returns:
            degree_matrix: Diagonal degree matrix (sparse tensor)

        Usage:
            adj = subgraph.adjacency_matrix()  # Get adjacency
            deg = data.get_degree_matrix(subgraph)  # Get degree matrix
        """
        # Get degrees (undirected: in_degree = out_degree)
        degrees = subgraph.in_degrees().float()

        # Create diagonal matrix
        n = subgraph.num_nodes()
        idx = torch.arange(n)
        indices = torch.stack([idx, idx])
        degree_matrix = torch.sparse_coo_tensor(indices, degrees, (n, n))

        return degree_matrix

    def __repr__(self):
        return (f"tsocial_data(nodes={self.graph.num_nodes():,}, "
                f"edges={self.graph.num_edges():,}, "
                f"train={self.train_mask.sum().item():,}, "
                f"val={self.val_mask.sum().item():,}, "
                f"test={self.test_mask.sum().item():,})")


if __name__ == "__main__":
    data = tsocial_data(undirected=True, verbose=True)

    # 1. Get training dataset
    train_data = data.get_train_data()
    print(f"\nTrain features shape: {train_data['features'].shape}")
    print(f"Train labels shape: {train_data['labels'].shape}")
    print(f"Number of train nodes: {train_data['mask'].sum().item():,}")

    # 2. Get validation dataset
    val_data = data.get_val_data()
    print(f"\nVal features shape: {val_data['features'].shape}")
    print(f"Val labels shape: {val_data['labels'].shape}")
    print(f"Number of val nodes: {val_data['mask'].sum().item():,}")

    # 3. Get test dataset
    test_data = data.get_test_data()
    print(f"\nTest features shape: {test_data['features'].shape}")
    print(f"Test labels shape: {test_data['labels'].shape}")
    print(f"Number of test nodes: {test_data['mask'].sum().item():,}")

    print(f"\n{data}")
