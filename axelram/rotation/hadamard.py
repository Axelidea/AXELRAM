# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Hadamard rotation via the Fast Walsh-Hadamard Transform (FWHT).

The normalized Hadamard matrix H_d (d = 2^n) is an orthogonal matrix
constructed by Sylvester's recursive rule.  FWHT computes H_d @ x in
O(d log d) additions/subtractions with zero multiplications.

Combined with a random sign vector s in {-1, +1}^d, the randomized
Hadamard transform R = H_d @ diag(s) provides sufficient mixing for
the concentration-of-measure property to hold.
"""

import math
import torch
from typing import Optional


def fwht(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Apply the Fast Walsh-Hadamard Transform along the last dimension.

    Implements the iterative butterfly algorithm: for each stage k,
    pairs of elements at distance 2^k are combined as (a+b, a-b).

    Args:
        x: (..., d) where d must be a power of two
        normalize: if True, scale output by 1/sqrt(d)

    Returns:
        transformed tensor of same shape
    """
    d = x.shape[-1]
    assert d >= 1 and (d & (d - 1)) == 0, f"d must be a power of 2, got {d}"

    result = x.clone()
    half = 1
    while half < d:
        # Reshape so the butterfly pairs sit in a dedicated axis
        step = half * 2
        result = result.view(*result.shape[:-1], d // step, 2, half)
        top = result[..., 0, :]   # elements at even positions
        bot = result[..., 1, :]   # elements at odd positions
        result = torch.stack([top + bot, top - bot], dim=-2)
        result = result.view(*x.shape)
        half *= 2

    if normalize:
        result = result / math.sqrt(d)
    return result


def make_sign_vector(d: int, seed: int) -> torch.Tensor:
    """Generate a deterministic random sign vector in {-1, +1}^d.

    Uses a hash-style seed to avoid collisions across layers:
    each layer should use a distinct seed.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    bits = torch.randint(0, 2, (d,), generator=rng)
    return bits.float() * 2.0 - 1.0


def hadamard_rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Apply randomized Hadamard transform: y = FWHT(x * signs) / sqrt(d).

    The sign flip is element-wise multiplication (no FP multiplier needed
    in hardware -- it is a conditional negation on the sign bit).
    """
    return fwht(x * signs, normalize=True)
