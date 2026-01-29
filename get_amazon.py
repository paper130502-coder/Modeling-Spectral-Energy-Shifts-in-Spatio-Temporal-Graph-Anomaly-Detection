"""
Amazon dataset loader for graph-based anomaly detection.
Provides a single class 'amazon_data' to access train/val/test datasets.
"""

import os
os.environ['DGLBACKEND'] = 'pytorch'

import torch
import dgl
from dgl.data import FraudAmazonDataset
from sklearn.model_selection import train_test_split


class amazon_data:
    """
    Amazon fraud detection dataset in homogeneous or heterogeneous graph format.

    Usage:
        # Load as homogeneous graph (default)
        data = amazon_data(homo=True)

        # Load as heterogeneous graph
        data = amazon_data(homo=False)

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

    def __init__(self, train_ratio=0.4, val_ratio=0.2, random_state=2, homo=True, undirected=False, verbose=False):
        """
        Initialize and prepare the Amazon dataset.

        Args:
            train_ratio: Ratio of labeled nodes to use for training (default: 0.4)
            val_ratio: Ratio of labeled nodes to use for validation (default: 0.2)
            random_state: Random seed for reproducibility (default: 2)
            homo: Whether to convert to homogeneous graph (default: True)
            undirected: Whether to convert to undirected graph (default: False)
                       Note: undirected conversion only works with homo=True
            verbose: Whether to print loading information (default: False)
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.homo = homo
        self.undirected = undirected
        self.verbose = verbose
        self._one_hop_cache = None
        self._one_hop_cache_device = None
        self._one_hop_subgraph_cache = None  # Cache 1-hop subgraph info per node

        # Load and prepare the dataset
        self._load_dataset()
        self._create_splits()

    def _load_dataset(self):
        """Load the Amazon dataset and optionally convert to homogeneous graph."""
        if self.verbose:
            print("Loading Amazon dataset...")

        dataset = FraudAmazonDataset()
        graph_hetero = dataset[0]

        if self.homo:
            # Convert to homogeneous graph
            self.graph = dgl.to_homogeneous(
                graph_hetero,
                ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask']
            )
            self.graph = dgl.add_self_loop(self.graph)

            if self.verbose:
                print(f"Homogeneous graph created: {self.graph.num_nodes():,} nodes, {self.graph.num_edges():,} edges")

            # Convert to undirected if requested
            if self.undirected:
                self.graph = dgl.to_bidirected(self.graph, copy_ndata=True)
                if self.verbose:
                    print(f"Converted to undirected: {self.graph.num_edges():,} edges (with reverse edges)")
        else:
            # Keep heterogeneous graph as-is
            self.graph = graph_hetero

            if self.undirected:
                raise ValueError("undirected=True only works with homo=True. Set homo=True to use undirected graphs.")

            if self.verbose:
                print(f"Heterogeneous graph loaded:")
                print(f"  Node types: {self.graph.ntypes}")
                print(f"  Edge types: {self.graph.etypes}")

        # Extract node data
        self.num_classes = 2
        self.features = self.graph.ndata['feature']
        self.labels = self.graph.ndata['label'].long().squeeze(-1)

        if self.verbose:
            print(f"Features shape: {self.features.shape}")

    def _create_splits(self):
        """Create train/val/test splits from labeled nodes."""
        # Exclude first 3305 unlabeled nodes
        labeled_indices = list(range(3305, len(self.labels)))
        labeled_labels = self.labels[labeled_indices]

        # Calculate test ratio
        test_ratio = 1.0 - self.train_ratio - self.val_ratio

        # First split: train vs rest
        idx_train, idx_rest, _, y_rest = train_test_split(
            labeled_indices,
            labeled_labels,
            stratify=labeled_labels,
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

        # Extract all edges ONCE (this is fast)
        src, dst = self.graph.edges()
        src = src.tolist()
        dst = dst.tolist()

        # Build adjacency lists for all nodes at once (store edges, not just neighbors)
        from collections import defaultdict
        adjacency = defaultdict(list)  # Store edges as (src, dst) pairs
        for s, d in zip(src, dst):
            adjacency[s].append((s, d))
            adjacency[d].append((s, d))  # Both nodes need to know about this edge

        cache = []
        for node_id in range(num_nodes):
            # Get 1-hop neighbors (including self)
            neighbors = set([node_id])
            edges_in_subgraph = set()

            # Collect edges involving this node
            for s, d in adjacency[node_id]:
                neighbors.add(s)
                neighbors.add(d)
                edges_in_subgraph.add((s, d))

            neighbor_ids = sorted(list(neighbors))
            num_subgraph_nodes = len(neighbor_ids)

            # Create mapping from global ID to local subgraph index
            node_mapping = {global_id: local_idx for local_idx, global_id in enumerate(neighbor_ids)}

            # Convert edges to local indices
            edge_list = []
            for s, d in edges_in_subgraph:
                edge_list.append([node_mapping[s], node_mapping[d]])

            # Convert to tensor indices (COO format)
            if len(edge_list) > 0:
                adj_indices = torch.tensor(edge_list, dtype=torch.long).t()  # [2, num_edges]
            else:
                adj_indices = torch.zeros((2, 0), dtype=torch.long)

            cache.append({
                'neighbor_ids': neighbor_ids,
                'adj_indices': adj_indices,
                'num_nodes': num_subgraph_nodes,
                'center_local_idx': node_mapping[node_id]
            })

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
        return (f"amazon_data(nodes={self.graph.num_nodes():,}, "
                f"edges={self.graph.num_edges():,}, "
                f"train={self.train_mask.sum().item():,}, "
                f"val={self.val_mask.sum().item():,}, "
                f"test={self.test_mask.sum().item():,})")


if __name__ == "__main__":
    data = amazon_data(undirected=True, verbose=True)

    # 1. Get training dataset
    train_data = data.get_train_data()
    # print(f"   Train features shape: {train_data['features'].shape}")
    # print(f"   Train labels shape: {train_data['labels'].shape}")
    # print(f"   Number of train nodes: {train_data['mask'].sum().item():,}")

    # 2. Get validation dataset
    val_data = data.get_val_data()
    # print(f"   Val features shape: {val_data['features'].shape}")
    # print(f"   Val labels shape: {val_data['labels'].shape}")
    # print(f"   Number of val nodes: {val_data['mask'].sum().item():,}")

    # 3. Get test dataset
    test_data = data.get_test_data()
    # print(f"   Test features shape: {test_data['features'].shape}")
    # print(f"   Test labels shape: {test_data['labels'].shape}")
    # print(f"   Number of test nodes: {test_data['mask'].sum().item():,}")

    # 4. Get one node info
    node_id = train_data['mask'].nonzero(as_tuple=True)[0][0].item()
    # print(f"   Node ID: {node_id}")
    # print(f"   Features: {data.features[node_id][:5]}... (showing first 5)")
    # print(f"   Label: {data.labels[node_id].item()}")

    # 5. Get adjacency matrix for subgraph
    subgraph = data.get_khop_subgraph(node_id, k=1)
    adj = subgraph.adjacency_matrix()
    # print(f"   Subgraph nodes: {subgraph.num_nodes()}")
    # print(f"   Subgraph edges: {subgraph.num_edges()}")
    # print(f"   Adjacency matrix shape: {adj.shape}")

    # 6. Get degree matrix for subgraph
    deg = data.get_degree_matrix(subgraph)
    # print(f"   Degree matrix shape: {deg.shape}")
    # print(f"   Degree matrix type: {type(deg)}")

  
