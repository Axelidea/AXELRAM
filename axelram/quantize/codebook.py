# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Lloyd-Max optimal scalar quantizer.

Solves the Lloyd-Max conditions (Max 1960, Lloyd 1982) for the Gaussian
distribution N(0, 1/d) that arises from orthogonal rotation of unit vectors
in d dimensions (concentration of measure).

The codebook depends only on (d, b) and is independent of input data.

Academic reference:
  Zandieh et al., "TurboQuant", ICLR 2026 -- first applied Lloyd-Max
  to the post-rotation coordinate distribution for KV cache quantization.
"""

import math
import torch
from scipy.stats import norm as scipy_norm
from scipy.integrate import quad
from typing import Tuple


def _target_pdf(x: float, sigma: float) -> float:
    """PDF of N(0, sigma^2) evaluated at x."""
    return math.exp(-0.5 * (x / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def _conditional_mean(a: float, b: float, sigma: float) -> float:
    """Compute E[X | a < X <= b] for X ~ N(0, sigma^2).

    Uses scipy.integrate.quad for numerical integration.
    Returns the midpoint of [a, b] if the probability mass is negligible.
    """
    mass, _ = quad(_target_pdf, a, b, args=(sigma,))
    if mass < 1e-30:
        return (a + b) * 0.5
    weighted, _ = quad(lambda x: x * _target_pdf(x, sigma), a, b)
    return weighted / mass


def _cell_distortion(centroid: float, a: float, b: float, sigma: float) -> float:
    """Compute E[(X - centroid)^2 | a < X <= b] * P(a < X <= b)."""
    val, _ = quad(lambda x: (x - centroid) ** 2 * _target_pdf(x, sigma), a, b)
    return val


def solve_lloyd_max(d: int, bits: int, max_iter: int = 500) -> Tuple[list, list]:
    """
    Find the Lloyd-Max optimal scalar quantizer for N(0, 1/d).

    Initialization: CDF quantile-based (each cell has equal probability mass).
    Iteration: alternating boundary update and centroid update until convergence.

    Args:
        d: vector dimension (sigma = 1/sqrt(d))
        bits: quantization bit-width
        max_iter: iteration limit

    Returns:
        (centroids, boundaries) as plain Python lists of floats
    """
    n_levels = 1 << bits
    sigma = 1.0 / math.sqrt(d)
    rv = scipy_norm(loc=0.0, scale=sigma)

    # Quantile-based initialization: place boundaries at equal-probability splits
    boundaries = [float(rv.ppf((k + 1) / n_levels)) for k in range(n_levels - 1)]

    # Compute initial centroids as conditional means
    neg_inf = -8.0 * sigma  # effectively -inf for Gaussian
    pos_inf = 8.0 * sigma
    edges = [neg_inf] + boundaries + [pos_inf]
    centroids = [_conditional_mean(edges[i], edges[i + 1], sigma)
                 for i in range(n_levels)]

    for _ in range(max_iter):
        # Update boundaries: optimal boundary = midpoint of adjacent centroids
        boundaries = [(centroids[k] + centroids[k + 1]) * 0.5
                      for k in range(n_levels - 1)]

        edges = [neg_inf] + boundaries + [pos_inf]

        # Update centroids: conditional expectation within each cell
        new_centroids = [_conditional_mean(edges[i], edges[i + 1], sigma)
                         for i in range(n_levels)]

        # Check convergence by maximum centroid movement
        shift = max(abs(new_centroids[k] - centroids[k]) for k in range(n_levels))
        centroids = new_centroids
        if shift < 1e-12:
            break

    # Final boundary pass
    boundaries = [(centroids[k] + centroids[k + 1]) * 0.5
                  for k in range(n_levels - 1)]

    return centroids, boundaries


def compute_distortion(d: int, bits: int, centroids: list, boundaries: list) -> float:
    """Total expected MSE distortion per coordinate."""
    sigma = 1.0 / math.sqrt(d)
    n_levels = len(centroids)
    neg_inf = -8.0 * sigma
    pos_inf = 8.0 * sigma
    edges = [neg_inf] + list(boundaries) + [pos_inf]
    return sum(_cell_distortion(centroids[i], edges[i], edges[i + 1], sigma)
               for i in range(n_levels))


def verify_optimality(d: int, bits: int, centroids: list, boundaries: list) -> dict:
    """Verify that a codebook satisfies the Lloyd-Max necessary conditions.

    Condition 1: boundaries are midpoints of adjacent centroids.
    Condition 2: centroids equal the conditional mean of their Voronoi cell.
    """
    sigma = 1.0 / math.sqrt(d)
    n_levels = len(centroids)
    neg_inf = -8.0 * sigma
    pos_inf = 8.0 * sigma
    edges = [neg_inf] + list(boundaries) + [pos_inf]

    # Check midpoint condition
    midpoint_err = max(
        abs(boundaries[k] - (centroids[k] + centroids[k + 1]) * 0.5)
        for k in range(n_levels - 1)
    )

    # Check centroid condition
    centroid_err = max(
        abs(centroids[k] - _conditional_mean(edges[k], edges[k + 1], sigma))
        for k in range(n_levels)
    )

    return {
        "midpoint_max_error": midpoint_err,
        "centroid_max_error": centroid_err,
        "optimal": midpoint_err < 1e-9 and centroid_err < 1e-9,
    }


# ── Cached access ──

_cache = {}


def get_codebook(d: int, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, boundaries) as float32 tensors, cached."""
    if (d, bits) not in _cache:
        c, b = solve_lloyd_max(d, bits)
        _cache[(d, bits)] = (
            torch.tensor(c, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
        )
    return _cache[(d, bits)]
