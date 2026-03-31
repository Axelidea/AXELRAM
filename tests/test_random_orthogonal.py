# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""Tests for SVD-based random orthogonal rotation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from axelram.rotation.random_orthogonal import make_random_orthogonal


def test_orthogonality():
    """R^T R = I for the generated matrix."""
    for d in [16, 64, 128]:
        R = make_random_orthogonal(d, seed=42)
        eye = R.T @ R
        err = (eye - torch.eye(d)).abs().max().item()
        assert err < 1e-5, f"d={d}: orthogonality error = {err}"
        print(f"  d={d}: R^T R - I max error = {err:.2e}")


def test_determinant_positive():
    """det(R) = +1 (proper rotation, not reflection)."""
    for d in [16, 64, 128]:
        R = make_random_orthogonal(d, seed=42)
        det = torch.linalg.det(R).item()
        assert abs(abs(det) - 1.0) < 1e-4, f"d={d}: |det| = {abs(det)}"
        print(f"  d={d}: det = {det:.6f}")


def test_deterministic():
    """Same seed → same matrix."""
    R1 = make_random_orthogonal(64, seed=123)
    R2 = make_random_orthogonal(64, seed=123)
    assert torch.allclose(R1, R2, atol=1e-7)
    R3 = make_random_orthogonal(64, seed=456)
    assert not torch.allclose(R1, R3, atol=0.1)
    print("  determinism: PASS")


def test_inner_product_preservation():
    """<Rx, Ry> = <x, y>."""
    d = 128
    R = make_random_orthogonal(d, seed=42)
    x = torch.randn(50, d)
    y = torch.randn(50, d)
    ip_orig = (x * y).sum(dim=-1)
    ip_rot = ((x @ R.T) * (y @ R.T)).sum(dim=-1)
    err = (ip_orig - ip_rot).abs().max().item()
    assert err < 1e-3, f"inner product error = {err}"
    print(f"  d={d}: inner product error = {err:.2e}")


if __name__ == "__main__":
    print("=== Orthogonality ===")
    test_orthogonality()
    print("\n=== Determinant ===")
    test_determinant_positive()
    print("\n=== Determinism ===")
    test_deterministic()
    print("\n=== Inner Product Preservation ===")
    test_inner_product_preservation()
    print("\nAll tests passed.")
