# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""Tests for Lloyd-Max codebook solver."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from axelram.quantize.codebook import solve_lloyd_max, verify_optimality, compute_distortion


def test_optimality_conditions():
    """Every computed codebook must satisfy Lloyd-Max necessary conditions."""
    for d in [64, 128, 256]:
        for bits in [2, 3, 4]:
            c, b = solve_lloyd_max(d, bits)
            result = verify_optimality(d, bits, c, b)
            assert result["optimal"], (
                f"d={d}, bits={bits}: midpoint_err={result['midpoint_max_error']:.2e}, "
                f"centroid_err={result['centroid_max_error']:.2e}"
            )
            print(f"  d={d}, b={bits}: PASS (midpoint_err={result['midpoint_max_error']:.2e}, "
                  f"centroid_err={result['centroid_max_error']:.2e})")


def test_symmetry():
    """N(0, sigma^2) is symmetric → codebook must be exactly symmetric."""
    for d in [64, 128]:
        for bits in [2, 3, 4]:
            c, b = solve_lloyd_max(d, bits)
            n = len(c)
            for i in range(n // 2):
                sym_err = abs(c[i] + c[n - 1 - i])
                assert sym_err < 1e-10, f"d={d}, bits={bits}: symmetry error {sym_err}"
            # Middle boundary must be zero (for even n_levels)
            if len(b) % 2 == 1:
                assert abs(b[len(b) // 2]) < 1e-10
            print(f"  d={d}, b={bits}: symmetric (max_err < 1e-10)")


def test_distortion_decreases_with_bits():
    """More bits → lower distortion."""
    for d in [128]:
        prev_dist = float("inf")
        for bits in [2, 3, 4, 5]:
            c, b = solve_lloyd_max(d, bits)
            dist = compute_distortion(d, bits, c, b)
            assert dist < prev_dist, f"d={d}: distortion at {bits}bit >= {bits-1}bit"
            print(f"  d={d}, b={bits}: distortion={dist:.8f}")
            prev_dist = dist


def test_known_values_d128():
    """Cross-check codebook values for d=128 against known results.

    The Lloyd-Max solution for N(0, 1/128) is unique, so any correct
    solver must produce these values (up to numerical precision).
    """
    c, b = solve_lloyd_max(128, 3)
    # 8 centroids for b=3, expected to be symmetric around 0
    assert len(c) == 8
    assert len(b) == 7
    # Check that the outermost centroid is approximately 0.19
    assert abs(abs(c[-1]) - 0.1902) < 0.001, f"outer centroid = {c[-1]}"
    # Check zero boundary
    assert abs(b[3]) < 1e-10, f"middle boundary = {b[3]}"
    print(f"  d=128, b=3: centroids = {[round(x, 4) for x in c]}")
    print(f"              boundaries = {[round(x, 4) for x in b]}")


if __name__ == "__main__":
    print("=== Optimality Conditions ===")
    test_optimality_conditions()
    print("\n=== Symmetry ===")
    test_symmetry()
    print("\n=== Distortion Monotonicity ===")
    test_distortion_decreases_with_bits()
    print("\n=== Known Values (d=128) ===")
    test_known_values_d128()
    print("\nAll tests passed.")
