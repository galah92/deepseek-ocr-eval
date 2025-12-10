# DeepSeek-OCR Evaluation

Evaluation tools for measuring DeepSeek-OCR's compression ratio and accuracy on document images.

## Overview

DeepSeek-OCR is a vision-language model that compresses document images into a small number of vision tokens while maintaining high OCR accuracy. This tool evaluates the model's performance by:

1. Feeding a document image to the model
2. Measuring how many vision tokens are used
3. Comparing OCR output to ground truth text
4. Calculating compression ratio and precision

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

**Requirements:** Python 3.10+, CUDA GPU with 16GB+ VRAM

## Usage

### OCR Evaluation

```bash
# Basic inference
uv run python eval.py --image document.png --mode small

# With ground truth for accuracy calculation
uv run python eval.py --image document.png --ground-truth gt.txt --mode small

# Dry run (no GPU needed) - shows token calculations only
uv run python eval.py --image document.png --dry-run
```

### Resolution Modes

| Mode | Resolution | Vision Tokens | Use Case |
|------|------------|---------------|----------|
| `tiny` | 512×512 | 64 | Maximum compression |
| `small` | 640×640 | 100 | Good balance |
| `base` | 1024×1024 | 256 | Higher accuracy |
| `large` | 1280×1280 | 400 | Best accuracy |
| `gundam` | Dynamic | Variable | Ultra-high resolution |

### Long-Document Experiments

```bash
# QuALITY dataset (multiple-choice QA)
uv run python -m experiment.quality_experiment --mode base --num-articles 10

# FineWiki dataset (language modeling)
uv run python -m experiment.finewiki_experiment --mode base --num-articles 20
```

## Evaluation Methodology

### Compression Ratio

```
Compression Ratio = Output Text Tokens / Vision Tokens
```

For example, if the model outputs 1000 text tokens from 100 vision tokens, the compression ratio is 10×.

### Precision (Accuracy)

Precision is calculated using **Levenshtein (edit) distance**:

```
Normalized Edit Distance = Levenshtein(output, ground_truth) / max(len(output), len(ground_truth))
Precision = 1 - Normalized Edit Distance
```

### Expected Results (from paper)

| Compression Ratio | Expected Precision |
|-------------------|-------------------|
| < 10× | ~97% (near lossless) |
| 10-12× | ~90% |
| 15-17× | ~80% |
| ~20× | ~60% |

## Project Structure

```
deepseek-ocr-eval/
├── eval.py                      # Main OCR evaluation script
├── pyproject.toml               # Dependencies
└── experiment/
    ├── utils.py                 # Shared utilities (model loading, rendering)
    ├── quality_experiment.py    # QuALITY long-document QA experiment
    ├── finewiki_experiment.py   # FineWiki language modeling experiment
    ├── quality_data/            # Cached rendered articles (QuALITY)
    ├── finewiki_data/           # Cached rendered articles (FineWiki)
    └── results/                 # Experiment results (JSON)
```

## Experiment Results

### QuALITY Long-Document QA (10 articles, 50 questions)

| Mode | Vision Tokens | Text Accuracy | Vision Accuracy | Compression |
|------|---------------|---------------|-----------------|-------------|
| **Tiny** | 64 | 36.0% (18/50) | 26.0% (13/50) | **38.51x** |
| Base | 256 | 36.0% (18/50) | 30.0% (15/50) | 17.74x |
| **Large** | 400 | 36.0% (18/50) | **34.0% (17/50)** | 12.63x |

**Key finding**: Large mode achieves near-parity with text (34% vs 36%) while using **12.6x fewer tokens**.

### FineWiki Language Modeling (20 articles)

| Metric | Text Condition | Vision Condition |
|--------|----------------|------------------|
| **Avg Word Overlap** | 7.9% | **14.5%** |
| First Word Match | 5.0% | 0.0% |
| Tokens Used | 15,709 | 6,120 |
| **Compression Ratio** | -- | **2.57x** |

### Critical Perspective

This work relates to the Lee et al. (2024) critique of optical context compression:

**Their key findings:**
1. Simpler methods (mean pooling, learned encoders) match or beat vision at the same compression ratios
2. For language modeling tasks, optical compression doesn't outperform simply truncating text
3. Good OCR accuracy doesn't translate to good downstream performance

**How our experiments relate:**
- QuALITY results show vision matches truncated text at large mode (12.6x compression)
- FineWiki results show vision outperforms text on continuation overlap, but this may be due to the model's tendency to OCR and paraphrase
- We did not compare against text-based compression methods (mean pooling, learned encoders)

## References

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [QuALITY Dataset](https://arxiv.org/abs/2112.08608)
- [FineWiki Dataset](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
- [Optical Context Compression Critique (Lee et al.)](https://arxiv.org/abs/2512.03643)

## Citation

```bibtex
@article{wei2024deepseekocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2024}
}
```
