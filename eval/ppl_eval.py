# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Perplexity evaluation for AXELRAM KV cache quantization.

Reproduces all configurations in the paper:
  A1: Random rotation + QJL    (TurboQuant baseline)
  A2: Random rotation + noQJL
  B:  Hadamard + QJL
  C:  Hadamard + noQJL         (AXELRAM main config)
  D:  Hadamard + noQJL + adaptive scaling

Method: sliding-window PPL on WikiText-2.
  1. Forward pass to produce KV cache + logits
  2. Quantize KV cache in place (rotate → quantize → dequantize → unrotate)
  3. Re-decode suffix tokens using quantized prefix cache
  4. Measure cross-entropy loss on stride window
"""

import sys
import os
import math
import json
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

# Allow importing axelram from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from axelram.quantize.codebook import get_codebook
from axelram.quantize import qjl as qjl_mod
from axelram.rotation.hadamard import make_sign_vector, hadamard_rotate, fwht
from axelram.rotation.random_orthogonal import make_random_orthogonal


# ── Configuration ──

@dataclass
class QuantConfig:
    rotation: str = "none"      # "none", "random", "hadamard"
    bits: int = 3               # total bit budget per coordinate
    use_qjl: bool = False       # steal 1 bit for QJL sign
    seed: int = 42


# ── Rotation helpers ──

def _build_rotation(kind: str, d: int, seed: int, device: str):
    """Return a rotation callable and its inverse, or (None, None)."""
    if kind == "random":
        mat = make_random_orthogonal(d, seed, device)
        fwd = lambda x: x @ mat.T
        inv = lambda x: x @ mat
        return fwd, inv
    elif kind == "hadamard":
        signs = make_sign_vector(d, seed).to(device)
        fwd = lambda x: hadamard_rotate(x, signs)
        # Inverse of normalised Hadamard is itself; undo sign flip after
        inv = lambda x: fwht(x, normalize=True) * signs
        return fwd, inv
    return None, None


# ── Core quantize / dequantize ──

@torch.no_grad()
def quantize_dequantize(
    tensor: torch.Tensor,
    rotate_fwd,
    rotate_inv,
    centroids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Quantize-then-dequantize a KV tensor in place.

    tensor: (batch, heads, seq_len, head_dim)
    Returns: same shape, quantized-then-dequantized (simulates hardware).
    """
    B, H, S, D = tensor.shape
    flat = tensor.reshape(-1, D).float()

    # Separate norm (stored as FP16 in hardware)
    norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit = flat / norms

    # Rotate
    rotated = rotate_fwd(unit) if rotate_fwd is not None else unit

    # Scalar quantize: find nearest centroid per coordinate
    # Shape: (N, D, 1) - (n_levels,) → (N, D, n_levels)
    diffs = (rotated.unsqueeze(-1) - centroids.to(device)).abs()
    indices = diffs.argmin(dim=-1)            # (N, D)
    reconstructed = centroids.to(device)[indices]  # (N, D)

    # Inverse rotate
    unrotated = rotate_inv(reconstructed) if rotate_inv is not None else reconstructed

    # Restore norm
    result = unrotated * norms
    return result.to(tensor.dtype).reshape(B, H, S, D)


# ── Dataset ──

def load_wikitext2(tokenizer):
    """Load WikiText-2 test set as a single token sequence."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    return tokenizer(text, return_tensors="pt").input_ids


# ── PPL evaluation ──

@torch.no_grad()
def evaluate_ppl(
    model,
    input_ids: torch.Tensor,
    config: Optional[QuantConfig] = None,
    stride: int = 512,
    max_length: int = 2048,
    device: str = "cuda",
) -> tuple:
    """Evaluate perplexity with optional KV cache quantization.

    Returns (ppl, n_tokens).
    """
    seq_len = input_ids.size(1)
    nlls = []
    n_tokens = 0

    # Pre-build per-layer rotation pairs
    rotate_fwd_k = {}
    rotate_inv_k = {}
    rotate_fwd_v = {}
    rotate_inv_v = {}

    if config is not None and config.rotation != "none":
        mc = model.config
        head_dim = mc.hidden_size // mc.num_attention_heads
        n_layers = mc.num_hidden_layers
        for li in range(n_layers):
            seed_k = config.seed + li * 1000
            seed_v = config.seed + (li + 500) * 1000
            rotate_fwd_k[li], rotate_inv_k[li] = _build_rotation(
                config.rotation, head_dim, seed_k, device)
            rotate_fwd_v[li], rotate_inv_v[li] = _build_rotation(
                config.rotation, head_dim, seed_v, device)

    # Determine effective MSE bits
    mse_bits = config.bits if config is not None else 3
    if config is not None and config.use_qjl:
        mse_bits = max(config.bits - 1, 1)

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        if chunk.shape[1] < 2:
            continue

        if config is None:
            # FP16 baseline
            outputs = model(chunk)
            logits = outputs.logits
        else:
            # Forward pass to get KV cache
            outputs = model(chunk, use_cache=True)
            cache = outputs.past_key_values
            n_layers_cache = (len(cache.layers) if hasattr(cache, 'layers')
                              else len(cache))

            head_dim = model.config.hidden_size // model.config.num_attention_heads

            # Separate codebooks for keys (mse_bits) and values (full bits)
            key_centroids, _ = get_codebook(head_dim, mse_bits)
            val_centroids, _ = get_codebook(head_dim, config.bits)

            # Quantize each layer's KV cache
            for li in range(n_layers_cache):
                if hasattr(cache, 'layers'):
                    k = cache.layers[li].keys
                    v = cache.layers[li].values
                else:
                    k, v = cache[li]

                k_q = quantize_dequantize(
                    k, rotate_fwd_k.get(li), rotate_inv_k.get(li),
                    key_centroids, device)
                v_q = quantize_dequantize(
                    v, rotate_fwd_v.get(li), rotate_inv_v.get(li),
                    val_centroids, device)

                if hasattr(cache, 'layers'):
                    cache.layers[li].keys = k_q
                    cache.layers[li].values = v_q

            # Re-decode suffix with quantized prefix
            n_decode = min(stride, chunk.shape[1] - 1)
            n_prefix = chunk.shape[1] - n_decode

            if n_prefix > 0 and hasattr(cache, 'layers'):
                for li in range(n_layers_cache):
                    cache.layers[li].keys = cache.layers[li].keys[:, :, :n_prefix, :]
                    cache.layers[li].values = cache.layers[li].values[:, :, :n_prefix, :]
                out_q = model(chunk[:, n_prefix:], past_key_values=cache)
                logits = torch.cat(
                    [outputs.logits[:, :n_prefix, :], out_q.logits], dim=1)
            else:
                logits = outputs.logits

        # Cross-entropy loss on stride window
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        if begin > 0:
            shift_logits = shift_logits[:, stride - 1:, :]
            shift_labels = shift_labels[:, stride - 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        n_tokens += shift_labels.numel()
        nlls.append(loss.item())

        if end >= seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


# ── Main runner ──

def run_all_configs(model_name: str, device: str = "cuda"):
    """Run the full evaluation matrix matching the paper's Table 1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'=' * 70}")
    print(f"AXELRAM PPL Evaluation: {model_name}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    input_ids = load_wikitext2(tokenizer)
    print(f"Tokens: {input_ids.shape[1]}")

    mc = model.config
    head_dim = mc.hidden_size // mc.num_attention_heads
    n_layers = mc.num_hidden_layers
    n_kv = getattr(mc, "num_key_value_heads", mc.num_attention_heads)
    print(f"layers={n_layers}, head_dim={head_dim}, kv_heads={n_kv}")

    stride = 512
    max_len = 2048
    results = {
        "model": model_name,
        "stride": stride,
        "max_length": max_len,
        "configs": {},
    }

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    t0 = time.time()
    ppl_fp16, ntok = evaluate_ppl(
        model, input_ids, config=None, stride=stride,
        max_length=max_len, device=device)
    print(f"  PPL: {ppl_fp16:.4f}  ({ntok} tokens, {time.time()-t0:.1f}s)")
    results["fp16_ppl"] = round(ppl_fp16, 4)
    results["fp16_tokens"] = ntok

    # All paper configurations
    configs = []
    for rot in ["random", "hadamard"]:
        for use_qjl in [True, False]:
            for bits in [2, 3, 4]:
                label = f"{rot}_{bits}bit_{'QJL' if use_qjl else 'noQJL'}"
                configs.append((label, QuantConfig(
                    rotation=rot, bits=bits, use_qjl=use_qjl, seed=42)))

    for label, cfg in configs:
        mse_bits = max(cfg.bits - 1, 1) if cfg.use_qjl else cfg.bits
        print(f"\n--- {label} (MSE={mse_bits}bit) ---")
        t0 = time.time()
        ppl, ntok = evaluate_ppl(
            model, input_ids, config=cfg, stride=stride,
            max_length=max_len, device=device)
        delta = ppl - ppl_fp16
        print(f"  PPL: {ppl:.4f}  (delta={delta:+.4f}, {time.time()-t0:.1f}s)")
        results["configs"][label] = {
            "ppl": round(ppl, 4),
            "delta": round(delta, 4),
            "rotation": cfg.rotation,
            "bits": cfg.bits,
            "use_qjl": cfg.use_qjl,
            "key_mse_bits": mse_bits,
        }

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"FP16 baseline: {results['fp16_ppl']:.4f}")
    print(f"\n{'Config':<35} {'2-bit':>10} {'3-bit':>10} {'4-bit':>10}")
    print("-" * 70)
    for rot in ["random", "hadamard"]:
        for ql, qn in [("QJL", True), ("noQJL", False)]:
            row = f"{rot}+{ql}"
            vals = []
            for b in [2, 3, 4]:
                key = f"{rot}_{b}bit_{ql}"
                d = results["configs"].get(key, {}).get("delta")
                vals.append(f"{d:>+10.4f}" if d is not None else f"{'N/A':>10}")
            print(f"{row:<35} {vals[0]} {vals[1]} {vals[2]}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AXELRAM PPL evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = run_all_configs(args.model, args.device)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
