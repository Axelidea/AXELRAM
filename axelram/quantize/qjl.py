# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Quantized Johnson-Lindenstrauss (QJL) 1-bit residual correction.

Given a quantization residual r = k - Q(k), QJL stores the sign of a
random linear projection to correct the inner-product bias introduced
by scalar quantization.

The unbiased estimator (Mao et al., arXiv:2406.03482, Theorem 1):

    <q, k> ≈ <q, Q(k)> + ||r||_2 · C_m · sign(S r)^T (S q)

where C_m = sqrt(pi / 2) / m and S is an m-by-d i.i.d. N(0,1) matrix.
"""

import math
import torch
from typing import NamedTuple


class QJLState(NamedTuple):
    """Compressed residual: 1-bit signs plus scalar norm."""
    sign_bits: torch.Tensor   # (..., m)  values in {-1, +1}
    residual_norm: torch.Tensor  # (...)


def random_gaussian_matrix(dim_in: int, dim_out: int, seed: int,
                           device: str = "cpu") -> torch.Tensor:
    """Sample an i.i.d. N(0,1) matrix of shape (dim_out, dim_in)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mat = torch.randn(dim_out, dim_in, generator=gen)
    return mat.to(device)


def encode(residual: torch.Tensor, mat_s: torch.Tensor) -> QJLState:
    """Encode quantization residual into QJL 1-bit representation.

    Args:
        residual: (..., d) the difference between original and quantized vector
        mat_s:    (m, d)   random Gaussian projection

    Returns:
        QJLState with sign_bits (..., m) and residual_norm (...)
    """
    # Norm of the full residual (scalar per vector)
    nrm = torch.linalg.vector_norm(residual, dim=-1)

    # Project and binarise
    proj = torch.einsum("...d, md -> ...m", residual, mat_s)
    bits = proj.sign()
    # Convention: map exact zero (probability-zero event) to +1
    bits = torch.where(bits == 0, torch.ones_like(bits), bits)

    return QJLState(sign_bits=bits, residual_norm=nrm)


def decode_correction(query: torch.Tensor, state: QJLState,
                      mat_s: torch.Tensor) -> torch.Tensor:
    """Compute the additive correction for the inner-product estimate.

    Returns the scalar correction per (query, key) pair.

    Args:
        query:  (..., d)
        state:  QJLState from encode()
        mat_s:  (m, d) same matrix used in encode()

    Returns:
        correction (...,) to be added to the MSE inner product
    """
    dim_out = mat_s.shape[0]
    coeff = math.sqrt(math.pi * 0.5) / dim_out

    # Project query through the same random matrix
    q_proj = torch.einsum("...d, md -> ...m", query, mat_s)

    # Signed dot product between projected query and stored sign bits
    signed_ip = torch.einsum("...m, ...m -> ...", q_proj, state.sign_bits)

    return state.residual_norm * coeff * signed_ip
