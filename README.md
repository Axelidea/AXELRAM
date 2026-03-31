# AXELRAM: Quantize Once, Never Dequantize

A smart SRAM macro architecture that computes attention scores directly from quantized KV cache indices without dequantization.

**Paper**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) (to appear)

**Author**: Yasushi Nishida ([Axelidea Inc.](https://axelidea.com))

## Overview

AXELRAM integrates orthogonal-transform-based quantization with table-lookup attention in a single SRAM macro. The key enabler is a **design-time fixed codebook**: the optimal quantizer depends only on dimension `d` and bit-width `b`, not on input data or model weights.

- **Write path**: FWHT + comparator-based quantization (zero multipliers)
- **Read path**: Pre-computed table lookup + adder tree (1 mult per key)
- **Asymmetric design**: No inverse transform on read (102.4x mult reduction)
- **Fixed codebook**: 30 bytes shared across all dimensions, storable in ROM

## Repository Structure

```
axelram/
  quantize/
    codebook.py           # Lloyd-Max optimal quantizer (independent implementation)
    qjl.py                # QJL 1-bit residual correction (for ablation)
  rotation/
    hadamard.py           # Randomized Hadamard Transform (FWHT)
    random_orthogonal.py  # SVD-based Haar rotation (ablation baseline)
eval/
  ppl_eval.py             # Perplexity evaluation (reproduces all paper results)
  hardware_model.py       # Hardware performance model ([FACT] labeled)
tests/
  test_codebook.py        # Lloyd-Max optimality & symmetry verification
  test_hadamard.py        # FWHT correctness & orthogonality tests
  test_random_orthogonal.py
```

## Quick Start

```bash
pip install -r requirements.txt

# Run unit tests (no GPU required)
python tests/test_codebook.py
python tests/test_hadamard.py

# Run hardware analysis (no GPU required)
python eval/hardware_model.py

# Reproduce PPL evaluation (requires GPU)
python eval/ppl_eval.py --model Qwen/Qwen2.5-3B-Instruct --output results/qwen25_3b.json
python eval/ppl_eval.py --model meta-llama/Llama-3.1-8B-Instruct --output results/llama31_8b.json
```

## Acknowledgments

Part of the computational work in this study was performed using the TSUBAME4.0 supercomputer at Institute of Science Tokyo.

## Citation

```bibtex
@article{nishida2026axelram,
  title={AXELRAM: Quantize Once, Never Dequantize},
  author={Nishida, Yasushi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
