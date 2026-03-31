# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Comprehensive evaluation: Default vs Optimized sign patterns.
3 models × 3 bit-widths × 10 seeds × 2 conditions.
"""

import sys, os, json, time, math, gc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from axelram.rotation.hadamard import fwht, make_sign_vector, hadamard_rotate
from axelram.quantize.codebook import get_codebook
from eval.ppl_eval import load_wikitext2, evaluate_ppl, QuantConfig
from eval.sign_optimization import optimize_all_layers, collect_kv_activations
from eval.ppl_with_optimized_signs import evaluate_ppl_optimized

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BITS = [2, 3, 4]
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-8B",
]


def run_model_evaluation(model_name, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'#' * 70}")
    print(f"# MODEL: {model_name}")
    print(f"# Seeds: {SEEDS}")
    print(f"# Bits: {BITS}")
    print(f"{'#' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16)
    model.eval()
    input_ids = load_wikitext2(tokenizer)

    mc = model.config
    head_dim = mc.hidden_size // mc.num_attention_heads
    n_layers = mc.num_hidden_layers
    print(f"head_dim={head_dim}, layers={n_layers}")

    # FP16 baseline
    print("\n--- FP16 Baseline ---")
    ppl_fp16, ntok = evaluate_ppl(model, input_ids, config=None, device=device)
    print(f"  FP16: {ppl_fp16:.4f} ({ntok} tokens)")

    results = {
        "model": model_name,
        "head_dim": head_dim,
        "n_layers": n_layers,
        "fp16_ppl": round(ppl_fp16, 4),
        "seeds": SEEDS,
        "bits": BITS,
        "default": {},     # bits -> seed -> delta
        "optimized": {},   # bits -> delta (single deterministic value)
        "optimization_info": {},  # bits -> per-layer MSE improvement
    }

    # ── Step 1: Sign optimization for each bit-width ──
    for bits in BITS:
        print(f"\n{'=' * 50}")
        print(f"Sign Optimization: b={bits}")
        print(f"{'=' * 50}")
        opt_results = optimize_all_layers(
            model, tokenizer, bits=bits, n_candidates=200,
            n_calib_samples=8, device=device)
        sign_seeds_opt = {l["layer"]: l["best_seed"] for l in opt_results["layers"]}

        # Store optimization info
        layer0 = opt_results["layers"][0]
        results["optimization_info"][str(bits)] = {
            "layer0_baseline_mse": layer0["baseline_mse"],
            "layer0_optimized_mse": layer0["optimized_mse"],
            "layer0_improvement_pct": layer0["improvement_pct"],
            "avg_improvement_pct": round(
                sum(l["improvement_pct"] for l in opt_results["layers"]) / len(opt_results["layers"]), 2),
        }

        # ── Step 2: Evaluate with optimized signs ──
        print(f"\n  Optimized signs, b={bits}:")
        t0 = time.time()
        ppl_opt, _ = evaluate_ppl_optimized(
            model, input_ids, bits=bits, sign_seeds=sign_seeds_opt, device=device)
        delta_opt = ppl_opt - ppl_fp16
        print(f"    PPL={ppl_opt:.4f}, delta={delta_opt:+.4f} ({time.time()-t0:.1f}s)")
        results["optimized"][str(bits)] = round(delta_opt, 4)

        # ── Step 3: Evaluate with default signs for each seed ──
        results["default"][str(bits)] = {}
        for seed in SEEDS:
            print(f"\n  Default seed={seed}, b={bits}:")
            # Build sign seeds from this base seed
            default_seeds = None  # Will use seed convention: seed + li*1000
            t0 = time.time()

            # Custom evaluation with specified base seed
            ppl_def, _ = evaluate_ppl_optimized(
                model, input_ids, bits=bits,
                sign_seeds={li: seed + li * 1000 for li in range(n_layers)},
                device=device)
            delta_def = ppl_def - ppl_fp16
            print(f"    PPL={ppl_def:.4f}, delta={delta_def:+.4f} ({time.time()-t0:.1f}s)")
            results["default"][str(bits)][str(seed)] = round(delta_def, 4)
            sys.stdout.flush()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'=' * 70}")
    print(f"FP16: {ppl_fp16:.4f}")
    print(f"\n{'Bits':<6} {'Default (mean±std)':<25} {'Optimized':<15} {'Default range':<25}")
    print("-" * 70)
    for bits in BITS:
        deltas = [results["default"][str(bits)][str(s)] for s in SEEDS]
        mean_d = sum(deltas) / len(deltas)
        std_d = (sum((d - mean_d)**2 for d in deltas) / len(deltas)) ** 0.5
        min_d, max_d = min(deltas), max(deltas)
        opt_d = results["optimized"][str(bits)]
        print(f"b={bits:<4} {mean_d:+.4f} ± {std_d:.4f}       {opt_d:+.4f}         [{min_d:+.4f}, {max_d:+.4f}]")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for model_name in args.models:
        safe_name = model_name.replace("/", "_")
        results = run_model_evaluation(model_name, args.device)
        all_results[model_name] = results

        # Save per-model
        out_path = os.path.join(args.output_dir, f"full_comparison_{safe_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")

    # Save combined
    combined_path = os.path.join(args.output_dir, "full_comparison_all.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved: {combined_path}")
