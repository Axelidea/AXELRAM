# MIT License - Copyright (c) 2026 Yasushi Nishida, Axelidea Inc.
"""PPL evaluation using optimized sign patterns."""

import sys, os, json, time, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from axelram.rotation.hadamard import fwht, make_sign_vector, hadamard_rotate
from axelram.quantize.codebook import get_codebook
from eval.ppl_eval import load_wikitext2, QuantConfig


@torch.no_grad()
def evaluate_ppl_optimized(model, input_ids, bits=3,
                            sign_seeds=None,  # dict: layer_idx -> seed
                            stride=512, max_length=2048, device="cuda"):
    """Evaluate PPL with per-layer optimized sign patterns."""
    seq_len = input_ids.size(1)
    nlls, n_tokens = [], 0
    mc = model.config
    head_dim = mc.hidden_size // mc.num_attention_heads
    n_layers = mc.num_hidden_layers

    # Build per-layer sign vectors
    signs_k = {}
    signs_v = {}
    for li in range(n_layers):
        if sign_seeds and li in sign_seeds:
            seed_k = sign_seeds[li]
        else:
            seed_k = 42 + li * 1000  # default
        seed_v = 42 + (li + 500) * 1000  # values always default
        signs_k[li] = make_sign_vector(head_dim, seed_k).to(device)
        signs_v[li] = make_sign_vector(head_dim, seed_v).to(device)

    centroids, _ = get_codebook(head_dim, bits)
    centroids_dev = centroids.to(device)

    def quantize_dequantize(tensor, signs):
        B, H, S, D = tensor.shape
        flat = tensor.reshape(-1, D).float()
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = flat / norms
        rotated = hadamard_rotate(unit, signs)
        diffs = (rotated.unsqueeze(-1) - centroids_dev).abs()
        indices = diffs.argmin(dim=-1)
        reconstructed = centroids_dev[indices]
        unrotated = fwht(reconstructed, normalize=True) * signs
        return (unrotated * norms).to(tensor.dtype).reshape(B, H, S, D)

    for begin in range(0, seq_len - 1, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        if chunk.shape[1] < 2:
            continue

        outputs = model(chunk, use_cache=True)
        cache = outputs.past_key_values
        n_lc = len(cache.layers) if hasattr(cache, 'layers') else len(cache)

        for li in range(n_lc):
            if hasattr(cache, 'layers'):
                k = cache.layers[li].keys
                v = cache.layers[li].values
                cache.layers[li].keys = quantize_dequantize(k, signs_k[li])
                cache.layers[li].values = quantize_dequantize(v, signs_v[li])

        n_decode = min(stride, chunk.shape[1] - 1)
        n_prefix = chunk.shape[1] - n_decode
        if n_prefix > 0 and hasattr(cache, 'layers'):
            for li in range(n_lc):
                cache.layers[li].keys = cache.layers[li].keys[:, :, :n_prefix, :]
                cache.layers[li].values = cache.layers[li].values[:, :, :n_prefix, :]
            out_q = model(chunk[:, n_prefix:], past_key_values=cache)
            logits = torch.cat([outputs.logits[:, :n_prefix, :], out_q.logits], dim=1)
        else:
            logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()
        if begin > 0:
            shift_logits = shift_logits[:, stride - 1:, :]
            shift_labels = shift_labels[:, stride - 1:]

        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), reduction="sum")
        n_tokens += shift_labels.numel()
        nlls.append(loss.item())
        if end >= seq_len:
            break

    return math.exp(sum(nlls) / n_tokens), n_tokens


if __name__ == "__main__":
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--sign-opt", required=True, help="sign optimization JSON")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load optimized seeds
    with open(args.sign_opt) as f:
        opt = json.load(f)
    sign_seeds = {l["layer"]: l["best_seed"] for l in opt["layers"]}
    print(f"Loaded optimized signs for {len(sign_seeds)} layers")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"),
        device_map="auto", dtype=torch.float16)
    model.eval()
    input_ids = load_wikitext2(tokenizer)

    # FP16 baseline
    from eval.ppl_eval import evaluate_ppl
    print("--- FP16 Baseline ---")
    ppl_fp16, _ = evaluate_ppl(model, input_ids, config=None, device=args.device)
    print(f"  FP16: {ppl_fp16:.4f}")

    # Default signs (seed=42)
    print(f"\n--- Default Signs (seed=42), b={args.bits} ---")
    ppl_default, _ = evaluate_ppl_optimized(
        model, input_ids, bits=args.bits, sign_seeds=None, device=args.device)
    print(f"  PPL: {ppl_default:.4f} (delta={ppl_default-ppl_fp16:+.4f})")

    # Optimized signs
    print(f"\n--- Optimized Signs, b={args.bits} ---")
    ppl_opt, _ = evaluate_ppl_optimized(
        model, input_ids, bits=args.bits, sign_seeds=sign_seeds, device=args.device)
    print(f"  PPL: {ppl_opt:.4f} (delta={ppl_opt-ppl_fp16:+.4f})")

    improvement = (ppl_default - ppl_opt) / (ppl_default - ppl_fp16) * 100
    print(f"\n  Improvement: {improvement:.1f}% of the quantization gap eliminated")

    results = {
        "model": args.model, "bits": args.bits,
        "fp16_ppl": round(ppl_fp16, 4),
        "default_ppl": round(ppl_default, 4),
        "default_delta": round(ppl_default - ppl_fp16, 4),
        "optimized_ppl": round(ppl_opt, 4),
        "optimized_delta": round(ppl_opt - ppl_fp16, 4),
        "gap_reduction_pct": round(improvement, 1),
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.output}")
