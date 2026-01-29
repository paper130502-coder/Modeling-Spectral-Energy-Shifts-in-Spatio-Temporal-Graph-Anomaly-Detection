

import torch
from typing import Optional

@torch.no_grad()
def local_1hop_energy_lnorm(
    X: torch.Tensor,                  # [N, F]
    edge_index: torch.Tensor,          # [2, E]  (dst=i, src=j), i <- j
    edge_weight: Optional[torch.Tensor] = None,  # [E] or None
    eps: float = 1e-8,
    deg_eps: float = 1e-12,
    chunk_size: int = 10_000_000,
    empty_cache_every: int = 5,        # clear CUDA cache every N chunks (0 disables)
) -> torch.Tensor:
    
    if X.dim() != 2:
        raise ValueError(f"X must be [N, F], got shape {tuple(X.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got shape {tuple(edge_index.shape)}")
    if edge_index.dtype != torch.long:
        raise TypeError(f"edge_index dtype must be torch.long, got {edge_index.dtype}")

    device = X.device
    dtype = X.dtype
    N, F = X.shape
    E = edge_index.size(1)

    dst = edge_index[0]
    src = edge_index[1]

    # Build edge weights
    if edge_weight is None:
        w = torch.ones(E, device=device, dtype=dtype)
    else:
        if edge_weight.dim() != 1 or edge_weight.numel() != E:
            raise ValueError(f"edge_weight must be [E], got shape {tuple(edge_weight.shape)}")
        w = edge_weight.to(device=device, dtype=dtype)

    # Degree (incoming to dst): d_i = sum_j w_ij
    deg = torch.zeros(N, device=device, dtype=dtype)
    deg.index_add_(0, dst, w)

    # Stabilize for isolated nodes
    deg_safe = deg.clamp_min(deg_eps)
    inv_sqrt_deg = deg_safe.rsqrt()      # 1/sqrt(d)
    inv_deg = deg_safe.reciprocal()      # 1/d

    # Accumulators
    num = torch.zeros((N, F), device=device, dtype=dtype)
    neigh_sum = torch.zeros((N, F), device=device, dtype=dtype)

    # Chunked processing
    n_chunks = (E + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, E)

        d = dst[start:end]
        s = src[start:end]
        w_c = w[start:end]

        # Normalized signals on edges
        Xi = X[d] * inv_sqrt_deg[d].unsqueeze(1)   # [e, F]
        Xj = X[s] * inv_sqrt_deg[s].unsqueeze(1)   # [e, F]

        # Numerator messages
        diff2 = (Xi - Xj).square()
        diff2.mul_(w_c.unsqueeze(1))
        num.index_add_(0, d, diff2)

        # Denominator neighbor term: sum_j Xj^2 / d_j
        neigh = X[s].square()
        neigh.mul_(inv_deg[s].unsqueeze(1))
        neigh_sum.index_add_(0, d, neigh)

        if empty_cache_every and device.type == "cuda" and (c + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()

    # Denominator self term: Xi^2 / d_i
    self_sq_over_deg = X.square() * inv_deg.unsqueeze(1)
    den = self_sq_over_deg + neigh_sum

    return num / (den + eps)


def local_1hop_energy_lnorm_flip(
    X: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    deg_eps: float = 1e-12,
    chunk_size: int = 10_000_000,
    empty_cache_every: int = 5,
) -> torch.Tensor:
    """
    Spectral-inverted version (equivalent to 2I - L_norm at the Rayleigh quotient level):
        R_flip = 2 - R
    """
    R = local_1hop_energy_lnorm(
        X, edge_index, edge_weight=edge_weight, eps=eps, deg_eps=deg_eps,
        chunk_size=chunk_size, empty_cache_every=empty_cache_every
    )
    return 2.0 - R
