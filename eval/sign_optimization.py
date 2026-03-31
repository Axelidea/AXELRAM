# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Sign Pattern Optimization for Hadamard Rotation.

Selects per-layer sign patterns that minimize the deviation of
post-rotation coordinate distributions from N(0, 1/d), thereby
reducing quantization error with the fixed codebook.

This is a lightweight one-time calibration (not per-token, not per-vector).
Storage cost: d bits per layer (16 bytes/layer for d=128).
"""

import sys, os, time, json, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from axelram.rotation.hadamard import fwht, make_sign_vector, hadamard_rotate
from axelram.quantize.codebook import get_codebook


def collect_kv_activations(model, input_ids, device, max_tokens=2048):
    """Collect KV cache activations from a forward pass."""
    chunk = input_ids[:, :max_tokens].to(device)
    with torch.no_grad():
        outputs = model(chunk, use_cache=True)
    cache = outputs.past_key_values
    n_layers = len(cache.layers) if hasattr(cache, 'layers') else len(cache)

    activations = []
    for li in range(n_layers):
        if hasattr(cache, 'layers'):
            k = cache.layers[li].keys  # (B, H, S, D)
        else:
            k, _ = cache[li]
        activations.append(k.float())
    return activations


def quantization_mse(vectors, signs, head_dim, bits):
    """Compute MSE of quantize-dequantize with given sign pattern.

    vectors: (N, D) float tensors (already unit-normalized)
    signs: (D,) sign pattern
    """
    centroids, _ = get_codebook(head_dim, bits)
    centroids = centroids.to(vectors.device)

    rotated = hadamard_rotate(vectors, signs.to(vectors.device))
    diffs = (rotated.unsqueeze(-1) - centroids).abs()
    indices = diffs.argmin(dim=-1)
    reconstructed = centroids[indices]

    return (rotated - reconstructed).pow(2).mean().item()


def optimize_signs_for_layer(activations, head_dim, bits, n_candidates=200,
                              base_seed=42):
    """Find the best sign pattern for a layer by evaluating candidates.

    activations: (B, H, S, D) key activations for one layer
    Returns: best sign pattern (D,) and its MSE
    """
    B, H, S, D = activations.shape
    flat = activations.reshape(-1, D)

    # Normalize to unit norm
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit = flat / norms

    # Subsample for speed (max 2000 vectors)
    if unit.shape[0] > 2000:
        idx = torch.randperm(unit.shape[0])[:2000]
        unit = unit[idx]

    best_mse = float('inf')
    best_signs = None

    for c in range(n_candidates):
        seed = base_seed * 10000 + c
        signs = make_sign_vector(D, seed)
        mse = quantization_mse(unit, signs, head_dim, bits)

        if mse < best_mse:
            best_mse = mse
            best_signs = signs.clone()
            best_seed = seed

    return best_signs, best_mse, best_seed


def optimize_all_layers(model, tokenizer, bits=3, n_candidates=200,
                         n_calib_samples=8, device="cuda"):
    """Optimize sign patterns for all layers using calibration data."""
    from datasets import load_dataset

    print(f"Loading calibration data ({n_calib_samples} samples)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    all_ids = tokenizer(text, return_tensors="pt").input_ids

    mc = model.config
    head_dim = mc.hidden_size // mc.num_attention_heads
    n_layers = mc.num_hidden_layers

    # Collect activations from multiple calibration samples
    print("Collecting activations...")
    all_activations = [[] for _ in range(n_layers)]

    chunk_size = 2048
    for s in range(n_calib_samples):
        start = s * chunk_size
        if start + chunk_size > all_ids.shape[1]:
            break
        acts = collect_kv_activations(model, all_ids[:, start:start+chunk_size],
                                       device, chunk_size)
        for li in range(n_layers):
            all_activations[li].append(acts[li])

    # Concatenate per layer
    for li in range(n_layers):
        all_activations[li] = torch.cat(all_activations[li], dim=2)  # concat along seq

    # Optimize per layer
    print(f"\nOptimizing sign patterns ({n_candidates} candidates/layer)...")
    results = {"bits": bits, "head_dim": head_dim, "n_layers": n_layers,
               "n_candidates": n_candidates, "layers": []}

    for li in range(n_layers):
        t0 = time.time()
        # Also compute baseline MSE with default seed
        flat = all_activations[li].reshape(-1, head_dim).float()
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = flat / norms
        if unit.shape[0] > 2000:
            idx = torch.randperm(unit.shape[0])[:2000]
            unit_sub = unit[idx]
        else:
            unit_sub = unit

        # Baseline (seed=42 + layer*1000, matching old convention)
        baseline_signs = make_sign_vector(head_dim, 42 + li * 1000)
        baseline_mse = quantization_mse(unit_sub, baseline_signs, head_dim, bits)

        # Optimized
        best_signs, best_mse, best_seed = optimize_signs_for_layer(
            all_activations[li], head_dim, bits, n_candidates,
            base_seed=42 + li * 1000)

        improvement = (baseline_mse - best_mse) / baseline_mse * 100

        elapsed = time.time() - t0
        print(f"  Layer {li:2d}: baseline_MSE={baseline_mse:.6f} -> "
              f"optimized_MSE={best_mse:.6f} ({improvement:+.1f}%) "
              f"[{elapsed:.1f}s]")

        results["layers"].append({
            "layer": li,
            "baseline_mse": round(baseline_mse, 8),
            "optimized_mse": round(best_mse, 8),
            "improvement_pct": round(improvement, 2),
            "best_seed": best_seed,
        })

    return results


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16)
    model.eval()

    results = optimize_all_layers(model, tokenizer, bits=args.bits,
                                   n_candidates=args.candidates, device=args.device)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")
