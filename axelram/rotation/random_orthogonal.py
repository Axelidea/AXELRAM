# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Haar-distributed random orthogonal matrix via SVD.

For a Gaussian matrix G ~ N(0,1)^{d x d}, the matrices U and V from
SVD(G) = U S V^T are each Haar-distributed on O(d). The product U V^T
is a uniformly random orthogonal matrix with det = +1.

This is used as an ablation baseline; the paper's main configuration
uses the Hadamard rotation instead.

Reference: G. W. Stewart, "The Efficient Generation of Random
Orthogonal Matrices with an Application to Condition Estimators",
SIAM J. Numer. Anal., 1980.
"""

import torch


def make_random_orthogonal(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    """Generate a d x d Haar-distributed random orthogonal matrix.

    Uses SVD of a Gaussian matrix:  G = U S V^T  →  return U V^T.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    g = torch.randn(d, d, generator=rng)
    u, _, vt = torch.linalg.svd(g)
    return (u @ vt).to(device)
