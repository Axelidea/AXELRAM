# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
AXELRAM hardware performance model.

Computes analytical metrics for the AXELRAM smart SRAM macro architecture.
All results are labeled [FACT] (mathematically provable from the algorithm)
or [ESTIMATE] (dependent on technology assumptions from published literature).

No fabrication or synthesis has been performed.
"""

import math
import json
from dataclasses import dataclass, asdict


@dataclass
class ArchParams:
    """Architecture parameters."""
    d: int = 128          # head dimension
    bits: int = 3         # quantization bit-width
    T: int = 4096         # context length (tokens)
    n_layers: int = 32    # transformer layers
    n_kv_heads: int = 8   # KV heads per layer


def multiplication_analysis(p: ArchParams) -> dict:
    """[FACT] Count multiplications for one query over T keys.

    Conventional: Q^T K for each key → d multiplications × T keys
    AXELRAM: pre-compute table (d × 2^b mults) + norm scaling (T mults)
    """
    n_levels = 1 << p.bits
    conventional = p.T * p.d
    precompute = p.d * n_levels   # once per query
    norm_scaling = p.T            # one mult per key
    axelram = precompute + norm_scaling
    return {
        "type": "FACT",
        "conventional_mults": conventional,
        "axelram_precompute_mults": precompute,
        "axelram_norm_mults": norm_scaling,
        "axelram_total_mults": axelram,
        "reduction_factor": round(conventional / axelram, 1),
    }


def memory_analysis(p: ArchParams) -> dict:
    """[FACT] Memory footprint analysis."""
    n_levels = 1 << p.bits

    # Per-row storage: b bits × d coords + 16 bits norm
    bits_per_row = p.bits * p.d + 16
    bytes_per_row = math.ceil(bits_per_row / 8)

    # Codebook ROM
    n_centroids = n_levels
    n_boundaries = n_levels - 1
    codebook_values = n_centroids + n_boundaries
    codebook_bytes = codebook_values * 2  # FP16

    # Pre-computation table SRAM
    table_entries = p.d * n_levels
    table_bytes = table_entries * 2  # FP16

    # KV cache per head
    kv_bytes_per_head = bytes_per_row * p.T

    # Total KV cache (all layers, all heads, K+V)
    total_kv_mb = (kv_bytes_per_head * p.n_kv_heads * p.n_layers * 2) / (1024 ** 2)

    # FP16 equivalent
    fp16_per_row = p.d * 2  # bytes
    fp16_total_mb = (fp16_per_row * p.T * p.n_kv_heads * p.n_layers * 2) / (1024 ** 2)

    return {
        "type": "FACT",
        "bits_per_row": bits_per_row,
        "bytes_per_row": bytes_per_row,
        "codebook_values": codebook_values,
        "codebook_bytes": codebook_bytes,
        "table_entries": table_entries,
        "table_bytes": table_bytes,
        "kv_cache_total_mb": round(total_kv_mb, 1),
        "fp16_total_mb": round(fp16_total_mb, 1),
        "compression_ratio": round(fp16_total_mb / total_kv_mb, 1),
    }


def operation_count(p: ArchParams) -> dict:
    """[FACT] Detailed operation counts for write and read paths."""
    log2d = int(math.log2(p.d))
    n_levels = 1 << p.bits

    # Write path (per token)
    fwht_addsub = log2d * (p.d // 2)          # butterfly stages
    sign_flips = p.d                            # conditional negation
    comparisons = p.d * (n_levels - 1)         # comparator tree
    write_mults = 0                             # zero multipliers

    # Read path: pre-computation (per query)
    precomp_mults = p.d * n_levels

    # Read path: per key
    table_lookups = p.d
    adder_tree_adds = p.d - 1
    norm_mults = 1

    # Total for T keys
    total_mults = precomp_mults + p.T * norm_mults

    # Inverse transform savings
    inverse_transform_saved = p.T * fwht_addsub

    return {
        "type": "FACT",
        "write_path": {
            "fwht_add_sub": fwht_addsub,
            "sign_flips": sign_flips,
            "comparisons": comparisons,
            "multiplications": write_mults,
            "fwht_stages": log2d,
        },
        "read_path_per_query": {
            "precompute_mults": precomp_mults,
        },
        "read_path_per_key": {
            "table_lookups": table_lookups,
            "additions": adder_tree_adds,
            "norm_mult": norm_mults,
        },
        "total_mults_for_T_keys": total_mults,
        "inverse_transform_add_sub_saved": inverse_transform_saved,
    }


def run_full_analysis(p: ArchParams = None) -> dict:
    """Run all analyses and return combined results."""
    if p is None:
        p = ArchParams()
    results = {
        "params": asdict(p),
        "multiplication": multiplication_analysis(p),
        "memory": memory_analysis(p),
        "operations": operation_count(p),
    }
    return results


if __name__ == "__main__":
    results = run_full_analysis()
    print(json.dumps(results, indent=2))

    p = results["params"]
    m = results["multiplication"]
    mem = results["memory"]

    print(f"\n{'=' * 50}")
    print(f"AXELRAM Hardware Analysis (d={p['d']}, b={p['bits']}, T={p['T']})")
    print(f"{'=' * 50}")
    print(f"[FACT] Multiplication reduction: {m['reduction_factor']}x")
    print(f"       ({m['conventional_mults']:,} → {m['axelram_total_mults']:,})")
    print(f"[FACT] Codebook ROM: {mem['codebook_bytes']} bytes")
    print(f"[FACT] KV cache: {mem['kv_cache_total_mb']} MB "
          f"(FP16: {mem['fp16_total_mb']} MB, "
          f"{mem['compression_ratio']}x compression)")
