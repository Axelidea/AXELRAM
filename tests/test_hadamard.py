# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""Tests for Hadamard rotation (FWHT)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from axelram.rotation.hadamard import fwht, make_sign_vector, hadamard_rotate


def _explicit_hadamard(d: int) -> torch.Tensor:
    """Build explicit normalized Hadamard matrix by Sylvester recursion."""
    import math
    h = torch.tensor([[1.0]])
    while h.shape[0] < d:
        h = torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0)
    return h / math.sqrt(d)


def test_fwht_matches_explicit():
    """FWHT must produce the same result as explicit matrix multiplication."""
    for d in [4, 8, 16, 32, 64, 128]:
        H = _explicit_hadamard(d)
        x = torch.randn(10, d)
        expected = x @ H.T
        actual = fwht(x, normalize=True)
        err = (expected - actual).abs().max().item()
        assert err < 1e-5, f"d={d}: max error = {err}"
        print(f"  d={d}: max error = {err:.2e}")


def test_self_inverse():
    """Normalized Hadamard is self-inverse: FWHT(FWHT(x)) = x."""
    for d in [16, 64, 128]:
        x = torch.randn(5, d)
        reconstructed = fwht(fwht(x, normalize=True), normalize=True)
        err = (x - reconstructed).abs().max().item()
        assert err < 1e-5, f"d={d}: self-inverse error = {err}"
        print(f"  d={d}: self-inverse error = {err:.2e}")


def test_orthogonality_preserves_norm():
    """Orthogonal transform preserves L2 norm."""
    d = 128
    x = torch.randn(100, d)
    y = fwht(x, normalize=True)
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    err = (norm_x - norm_y).abs().max().item()
    assert err < 1e-4, f"norm preservation error = {err}"
    print(f"  d={d}: norm preservation error = {err:.2e}")


def test_inner_product_preservation():
    """<Rx, Ry> = <x, y> for orthogonal R."""
    d = 128
    signs = make_sign_vector(d, seed=42)
    x = torch.randn(50, d)
    y = torch.randn(50, d)
    ip_original = (x * y).sum(dim=-1)
    rx = hadamard_rotate(x, signs)
    ry = hadamard_rotate(y, signs)
    ip_rotated = (rx * ry).sum(dim=-1)
    err = (ip_original - ip_rotated).abs().max().item()
    assert err < 1e-3, f"inner product error = {err}"
    print(f"  d={d}: inner product preservation error = {err:.2e}")


def test_sign_vector_deterministic():
    """Same seed → same sign vector."""
    s1 = make_sign_vector(128, seed=999)
    s2 = make_sign_vector(128, seed=999)
    assert torch.equal(s1, s2)
    # Different seed → different vector
    s3 = make_sign_vector(128, seed=1000)
    assert not torch.equal(s1, s3)
    print("  determinism: PASS")


if __name__ == "__main__":
    print("=== FWHT vs Explicit Matrix ===")
    test_fwht_matches_explicit()
    print("\n=== Self-Inverse ===")
    test_self_inverse()
    print("\n=== Norm Preservation ===")
    test_orthogonality_preserves_norm()
    print("\n=== Inner Product Preservation ===")
    test_inner_product_preservation()
    print("\n=== Sign Vector Determinism ===")
    test_sign_vector_deterministic()
    print("\nAll tests passed.")
