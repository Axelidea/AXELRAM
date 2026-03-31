# AXELRAM Paper — Tables and Narrative (Draft)

## TABLE I: Sign Pattern Sensitivity (Default, 10 seeds)

PPL increase (Δ) over FP16 baseline. Mean ± std over 10 seeds.

| Model | FP16 | 2-bit | 3-bit | 4-bit |
|-------|------|-------|-------|-------|
| LLaMA-3.1-8B | 6.98 | +0.10 ± 0.01 | +0.02 ± 0.00 | +0.00 ± 0.00 |
| Qwen2.5-3B | 8.29 | +8.13 ± 16.88 | +9.64 ± 14.86 | +1.10 ± 0.85 |
| Qwen3-8B | 9.07 | +12.23 ± 2.51 | +0.48 ± 0.29 | +0.04 ± 0.01 |

Key observations:
- LLaMA: stable across all seeds (std < 0.01)
- Qwen2.5-3B: catastrophic variance (std up to 16.88, worst delta +58.43)
- Qwen3-8B: 2-bit consistently poor; 3-4 bit moderate variance

## TABLE II: Sign Pattern Optimization Effect

| Model | Bits | Default (mean) | Default (worst) | Optimized | Spike eliminated? |
|-------|------|----------------|-----------------|-----------|-------------------|
| LLaMA-3.1-8B | 2 | +0.10 | +0.11 | +0.10 | N/A (no spike) |
| LLaMA-3.1-8B | 3 | +0.02 | +0.02 | +0.02 | N/A |
| LLaMA-3.1-8B | 4 | +0.00 | +0.01 | +0.00 | N/A |
| Qwen2.5-3B | 2 | +8.13 | +58.43 | **+0.82** | **Yes (99%)** |
| Qwen2.5-3B | 3 | +9.64 | +51.00 | **+0.58** | **Yes (99%)** |
| Qwen2.5-3B | 4 | +1.10 | +2.98 | **+0.25** | **Yes (92%)** |
| Qwen3-8B | 2 | +12.23 | +17.45 | +9.26 | Partial (47%) |
| Qwen3-8B | 3 | +0.48 | +1.09 | **+0.35** | **Yes (68%)** |
| Qwen3-8B | 4 | +0.04 | +0.05 | **+0.01** | **Yes (72%)** |

## TABLE III: Comparison with SpinQuant's Findings

| | SpinQuant (ICLR 2025) | This work |
|---|---|---|
| Domain | Weight + activation (W4A4) | **KV cache** |
| Models tested | LLaMA-2-7B | LLaMA-3.1-8B, Qwen2.5-3B, Qwen3-8B |
| Rotation types | Random orthogonal, Random Hadamard | Random Hadamard (sign pattern) |
| Max variance (Hadamard) | 6 points (accuracy) | **>50 points (PPL delta)** |
| Catastrophic failure | Not reported | **Δ > 50 on Qwen2.5-3B** |
| Model dependency | Not analyzed | **Correlated with layer norm heterogeneity** |
| Mitigation | Cayley optimization (gradient-based) | **Candidate selection (gradient-free, 200 candidates)** |
| Mitigation cost | Backprop through full model | **Forward-only, 8 calibration samples** |

## Narrative Structure

### Section 1: Introduction
- AXELRAM smart SRAM macro for dequantization-free attention
- 102.4x multiplication reduction (mathematical fact)
- Key insight: design-time fixed codebook enables ROM storage

### Section 2: Architecture
- Write path: FWHT + fixed codebook quantization
- Read path: query-side rotation + pre-computation table + adder tree
- No inverse transform (asymmetric path)

### Section 3: Sign Pattern Sensitivity (NEW - main contribution)

Motivation: SpinQuant (ICLR 2025) demonstrated that random rotation
matrices induce up to 6-point accuracy variance in W4A4 weight quantization.
We investigate whether analogous sensitivity exists in KV cache quantization,
which operates in a fundamentally different regime: online quantization of
activation vectors (not offline weight quantization).

Finding 1: The sensitivity is dramatically worse for KV cache quantization.
On Qwen2.5-3B, sign pattern choice causes >50 PPL points of degradation
(vs SpinQuant's 6 accuracy points for W4A4).

Finding 2: The sensitivity is model-dependent. LLaMA-3.1-8B shows near-zero
variance (std < 0.01) while Qwen2.5-3B shows catastrophic variance (std > 14).
This correlates with layer-wise norm heterogeneity: Qwen layer 0 has 7.8x
norm ratio vs LLaMA's ~1.5x.

Finding 3: A gradient-free sign pattern selection (200 candidates, 8 calibration
samples, one-time) eliminates catastrophic spikes, reducing worst-case delta
from +58.43 to +0.82 on Qwen2.5-3B (99% reduction).

### Section 4: Evaluation
- TABLE I: 10-seed evaluation across 3 models
- TABLE II: Optimization effect
- TABLE III: Comparison with SpinQuant

### Section 5: Hardware Architecture
- AXELRAM macro design (unchanged)
- Sign pattern stored in ROM/eFuse alongside fixed codebook
- Additional cost: 576 bytes/model (vs 30 bytes for codebook alone)

### Section 6: Related Work
- SpinQuant: MUST CITE prominently. "SpinQuant (Huynh et al., ICLR 2025)
  first demonstrated rotation seed sensitivity in W4A4 quantization.
  We extend this finding to KV cache quantization, where the effect is
  qualitatively different: catastrophic spikes (Δ>50) occur on specific
  models, and the severity correlates with layer-wise norm heterogeneity."
- TurboQuant: foundation for fixed codebook
- LOOKAT: concurrent ADC-based attention (data-dependent codebook)
- QuaRot, QuIP#: rotation for weight quantization

### Section 7: Conclusion
- AXELRAM architecture: 102.4x multiplication reduction (fact)
- New finding: sign pattern sensitivity in KV cache quantization
- Lightweight mitigation: 200-candidate selection, one-time calibration
- Honest limitation: Qwen3-8B 2-bit remains poor despite optimization
  (fundamental precision limit, not sign pattern issue)

## Citations to add
- SpinQuant: L. Huynh et al., "SpinQuant: LLM Quantization with Learned Rotations", ICLR 2025
- ParoQuant: (acknowledges multi-seed sampling without variance analysis)
