

import torch
import torch.nn as nn
import torch.nn.functional as Func



@torch.no_grad()
def batched_energy_fully_connected(X, eps=1e-8):
    """
    Vectorized energy for fully connected graph.
    d_i = N-1 for all nodes (uniform degree).
    """
    B, N, T = X.shape
    d = N - 1
    inv_sqrt_d = 1.0 / (d ** 0.5)
    inv_d = 1.0 / d

    X_norm = X * inv_sqrt_d

    sum_X_norm = X_norm.sum(dim=1, keepdim=True)
    sum_X_norm_sq = (X_norm ** 2).sum(dim=1, keepdim=True)

    sum_others = sum_X_norm - X_norm
    sum_sq_others = sum_X_norm_sq - X_norm ** 2

    num = d * (X_norm ** 2) - 2 * X_norm * sum_others + sum_sq_others

    sum_X_sq = (X ** 2).sum(dim=1, keepdim=True)
    sum_sq_others_unnorm = sum_X_sq - X ** 2
    den = (X ** 2 + sum_sq_others_unnorm) * inv_d

    return num / (den + eps)


class FullyConnectedSAGELayer(nn.Module):


    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim, bias=True)
        self.W_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h, num_nodes):
        h_sum = h.sum(dim=1, keepdim=True)
        h_neigh = (h_sum - h) / (num_nodes - 1)
        return self.W_self(h) + self.W_neigh(h_neigh)




class SpectralFlipGate(nn.Module):
    """Learnable gating: Z = gate * Rn + (1 - gate) * Rf"""

    def __init__(self, in_feats, hidden_dim=16, gate_type="per_feature", num_layers=2):
        super().__init__()
        out_dim = 1 if gate_type == "per_node" else in_feats

        layers = [nn.Linear(in_feats, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.extend([nn.Linear(hidden_dim, out_dim), nn.Sigmoid()])

        self.gate_mlp = nn.Sequential(*layers)

        # Initialize to prefer flip (gate ≈ 0.12)
        with torch.no_grad():
            self.gate_mlp[-2].weight.zero_()
            self.gate_mlp[-2].bias.fill_(-2.0)

    def forward(self, Xn, Rn, Rf):
        """Gate computed from normalized X, applied to normalized energies."""
        gate = self.gate_mlp(Xn)
        return gate * Rn + (1.0 - gate) * Rf



class TemporalGatedEnergySAGE(nn.Module):
    """
    TEGNN with z-score normalization using training statistics only.

    During training: compute and cache stats from training samples
    During inference: use cached training stats
    """

    def __init__(
        self,
        num_nodes,
        window_size,
        hidden_dim=128,
        num_classes=2,
        dropout=0.5,
        gate_hidden_dim=16,
        gate_type="per_feature",
        gate_num_layers=2,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Spectral flip gating
        self.spectral_gate = SpectralFlipGate(
            in_feats=window_size,
            hidden_dim=gate_hidden_dim,
            gate_type=gate_type,
            num_layers=gate_num_layers,
        )

        # 3-layer vectorized SAGE
        self.sage1 = FullyConnectedSAGELayer(window_size, hidden_dim)
        self.sage2 = FullyConnectedSAGELayer(hidden_dim, hidden_dim)
        self.sage3 = FullyConnectedSAGELayer(hidden_dim, hidden_dim // 2)

        # Classifier
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # Normalization stats (will be set during training)
        self.register_buffer('x_mean', None)
        self.register_buffer('x_std', None)
        self.register_buffer('r_mean', None)
        self.register_buffer('r_std', None)
        self.register_buffer('z_mean', None)
        self.register_buffer('z_std', None)
        self.stats_initialized = False

    def set_normalization_stats(self, train_features):
        """
        Compute and cache normalization statistics from training data.
        Call this ONCE before training with all training features.

        Args:
            train_features: [N_train, num_nodes, window_size] training samples
        """
        with torch.no_grad():
            # X stats
            self.x_mean = train_features.mean(dim=(0, 1), keepdim=True)  # [1, 1, T]
            self.x_std = train_features.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

            # R stats (compute energy on training data)
            R_train = batched_energy_fully_connected(train_features)
            self.r_mean = R_train.mean(dim=(0, 1), keepdim=True)
            self.r_std = R_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

            # Z stats (need to compute gated output on training data)
            Xn = (train_features - self.x_mean) / self.x_std
            Rn = (R_train - self.r_mean) / self.r_std
            Rf = (2.0 - R_train - self.r_mean) / self.r_std

            # Use initial gate (before training) to estimate Z stats
            Z_train = self.spectral_gate(Xn, Rn, Rf)
            self.z_mean = Z_train.mean(dim=(0, 1), keepdim=True)
            self.z_std = Z_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)

            self.stats_initialized = True

    def forward(self, graph, features, edge_index):
        
        if not self.stats_initialized:
            raise RuntimeError("Call set_normalization_stats() before forward pass!")

        B, N, T = features.shape
        features = features.float()

        # 1. Energy computation (frozen)
        R_normal = batched_energy_fully_connected(features)
        R_flip = 2.0 - R_normal

        # 2. Z-score normalize using TRAINING stats
        Xn = (features - self.x_mean) / self.x_std
        Rn = (R_normal - self.r_mean) / self.r_std
        Rf = (R_flip - self.r_mean) / self.r_std  # Use same stats as R_normal

        # 3. Gated mixture
        Z = self.spectral_gate(Xn, Rn, Rf)

        # 4. Normalize Z using training stats
        Zn = (Z - self.z_mean) / self.z_std

        # 5. GraphSAGE layers
        h = self.sage1(Zn, N)
        h = Func.relu(h)
        h = Func.dropout(h, p=self.dropout_rate, training=self.training)

        h = self.sage2(h, N)
        h = Func.relu(h)
        h = Func.dropout(h, p=self.dropout_rate, training=self.training)

        h = self.sage3(h, N)
        h = Func.relu(h)
        h = Func.dropout(h, p=self.dropout_rate, training=self.training)

        # 6. Global mean pooling
        h_pooled = h.mean(dim=1)

        # 7. Classification
        return self.classifier(h_pooled)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gate_statistics(self):
        return None


# Alias
FastTemporalGatedEnergySAGE = TemporalGatedEnergySAGE
