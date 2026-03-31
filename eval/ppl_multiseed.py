# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""
Multi-seed PPL evaluation for statistically robust comparison.

Runs each configuration with multiple random seeds and reports
mean +/- std of PPL delta, ensuring fair comparison between
rotation methods (Hadamard vs Random).
"""

import sys, os, json, time, math
from dataclasses import dataclass
from typing import Optional
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.ppl_eval import evaluate_ppl, load_wikitext2, QuantConfig


SEEDS = [42, 123, 456, 789, 1024]


def run_multiseed(model_name: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'=' * 70}")
    print(f"Multi-Seed PPL Evaluation: {model_name}")
    print(f"Seeds: {SEEDS}")
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
        dtype=torch.float16,
    )
    model.eval()

    input_ids = load_wikitext2(tokenizer)
    stride, max_len = 512, 2048

    # FP16 baseline (seed-independent)
    print("\n--- FP16 Baseline ---")
    t0 = time.time()
    ppl_fp16, ntok = evaluate_ppl(model, input_ids, config=None,
                                   stride=stride, max_length=max_len, device=device)
    print(f"  PPL: {ppl_fp16:.4f}  ({ntok} tokens, {time.time()-t0:.1f}s)")

    # Define configs (without seed - will be set per iteration)
    config_defs = []
    for rot in ["random", "hadamard"]:
        for use_qjl in [True, False]:
            for bits in [2, 3, 4]:
                label = f"{rot}_{bits}bit_{'QJL' if use_qjl else 'noQJL'}"
                config_defs.append((label, rot, bits, use_qjl))

    # Collect results: {label: [delta_seed0, delta_seed1, ...]}
    all_deltas = {label: [] for label, _, _, _ in config_defs}

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*50}")
        print(f"Seed {si+1}/{len(SEEDS)}: {seed}")
        print(f"{'='*50}")

        for label, rot, bits, use_qjl in config_defs:
            cfg = QuantConfig(rotation=rot, bits=bits, use_qjl=use_qjl, seed=seed)
            mse_bits = max(bits - 1, 1) if use_qjl else bits
            t0 = time.time()
            ppl, _ = evaluate_ppl(model, input_ids, config=cfg,
                                   stride=stride, max_length=max_len, device=device)
            delta = ppl - ppl_fp16
            all_deltas[label].append(delta)
            elapsed = time.time() - t0
            print(f"  {label:<30} delta={delta:+.4f}  ({elapsed:.1f}s)")
            sys.stdout.flush()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY: Mean +/- Std over {len(SEEDS)} seeds")
    print(f"{'=' * 80}")
    print(f"FP16 baseline: {ppl_fp16:.4f}")
    print(f"\n{'Config':<30} {'Mean delta':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)

    results = {
        "model": model_name,
        "fp16_ppl": round(ppl_fp16, 4),
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "configs": {},
    }

    for label, rot, bits, use_qjl in config_defs:
        deltas = all_deltas[label]
        mean_d = sum(deltas) / len(deltas)
        std_d = (sum((d - mean_d) ** 2 for d in deltas) / len(deltas)) ** 0.5
        min_d = min(deltas)
        max_d = max(deltas)
        print(f"{label:<30} {mean_d:>+12.4f} {std_d:>10.4f} {min_d:>+10.4f} {max_d:>+10.4f}")
        results["configs"][label] = {
            "mean_delta": round(mean_d, 4),
            "std_delta": round(std_d, 4),
            "min_delta": round(min_d, 4),
            "max_delta": round(max_d, 4),
            "all_deltas": [round(d, 4) for d in deltas],
            "rotation": rot,
            "bits": bits,
            "use_qjl": use_qjl,
        }

    # Paper-style table
    print(f"\n{'=' * 80}")
    print("Paper Table Format (mean delta)")
    print(f"{'=' * 80}")
    print(f"{'Config':<25} {'2-bit':>15} {'3-bit':>15} {'4-bit':>15}")
    print("-" * 70)
    for rot in ["random", "hadamard"]:
        for ql, qn in [("QJL", True), ("noQJL", False)]:
            row = f"{rot}+{ql}"
            vals = []
            for b in [2, 3, 4]:
                key = f"{rot}_{b}bit_{ql}"
                m = results["configs"][key]["mean_delta"]
                s = results["configs"][key]["std_delta"]
                vals.append(f"{m:+.2f}+/-{s:.2f}")
            print(f"{row:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    # TQ best vs C improvement
    print(f"\n{'=' * 80}")
    print("Improvement: C (Hadamard+noQJL) vs TQ best")
    print(f"{'=' * 80}")
    for bits in [2, 3, 4]:
        a1 = results["configs"][f"random_{bits}bit_QJL"]["mean_delta"]
        a2 = results["configs"][f"random_{bits}bit_noQJL"]["mean_delta"]
        tq = min(a1, a2)
        c = results["configs"][f"hadamard_{bits}bit_noQJL"]["mean_delta"]
        if tq > 0 and c < tq:
            improv = (tq - c) / tq * 100
            print(f"  {bits}-bit: TQ_best={tq:+.2f}, C={c:+.2f}, Improvement={improv:.0f}%")
        else:
            print(f"  {bits}-bit: TQ_best={tq:+.2f}, C={c:+.2f}, C is worse")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = run_multiseed(args.model, args.device)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")
