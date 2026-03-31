# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""Tests for QJL residual correction."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
from axelram.quantize.qjl import random_gaussian_matrix, encode, decode_correction


def test_unbiasedness():
    """The QJL estimator should be approximately unbiased over many trials.

    E[correction] ≈ <q, r> where r is the residual.
    """
    d, m = 128, 128
    n_trials = 5000
    torch.manual_seed(0)

    mat_s = random_gaussian_matrix(d, m, seed=7)
    q = torch.randn(d)
    r = torch.randn(d) * 0.01  # small residual

    true_ip = (q * r).sum().item()

    estimates = []
    for t in range(n_trials):
        # Use different random residuals each time but same projection
        r_t = r + torch.randn(d) * 0.001
        true_ip_t = (q * r_t).sum().item()
        state = encode(r_t.unsqueeze(0), mat_s)
        correction = decode_correction(q.unsqueeze(0), state, mat_s)
        estimates.append(correction.item())

    mean_est = sum(estimates) / len(estimates)
    mean_true = true_ip
    # Allow generous tolerance since this is statistical
    rel_err = abs(mean_est - mean_true) / (abs(mean_true) + 1e-6)
    print(f"  true <q,r> ≈ {mean_true:.6f}, mean QJL estimate = {mean_est:.6f}, rel_err = {rel_err:.4f}")
    assert rel_err < 0.5, f"Bias too large: {rel_err}"


def test_sign_bits_are_binary():
    """Encoded sign bits must be exactly {-1, +1}."""
    d, m = 64, 64
    mat_s = random_gaussian_matrix(d, m, seed=1)
    r = torch.randn(10, d)
    state = encode(r, mat_s)
    unique_vals = state.sign_bits.unique().tolist()
    assert set(unique_vals).issubset({-1.0, 1.0}), f"unexpected values: {unique_vals}"
    print(f"  sign_bits unique values: {unique_vals}")


def test_named_tuple_structure():
    """QJLState must be a NamedTuple with sign_bits and residual_norm."""
    d, m = 32, 32
    mat_s = random_gaussian_matrix(d, m, seed=2)
    r = torch.randn(5, d)
    state = encode(r, mat_s)
    assert hasattr(state, 'sign_bits')
    assert hasattr(state, 'residual_norm')
    assert state.sign_bits.shape == (5, m)
    assert state.residual_norm.shape == (5,)
    print(f"  shapes: sign_bits={state.sign_bits.shape}, norm={state.residual_norm.shape}")


def test_deterministic():
    """Same inputs → same output."""
    d, m = 64, 64
    mat_s = random_gaussian_matrix(d, m, seed=3)
    r = torch.randn(3, d)
    s1 = encode(r, mat_s)
    s2 = encode(r, mat_s)
    assert torch.equal(s1.sign_bits, s2.sign_bits)
    assert torch.equal(s1.residual_norm, s2.residual_norm)
    print("  determinism: PASS")


if __name__ == "__main__":
    print("=== Sign Bits Binary ===")
    test_sign_bits_are_binary()
    print("\n=== NamedTuple Structure ===")
    test_named_tuple_structure()
    print("\n=== Determinism ===")
    test_deterministic()
    print("\n=== Unbiasedness ===")
    test_unbiasedness()
    print("\nAll tests passed.")
