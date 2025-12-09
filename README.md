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

The Levenshtein distance counts the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another.

### Expected Results (from paper)

| Compression Ratio | Expected Precision |
|-------------------|-------------------|
| < 10× | ~97% (near lossless) |
| 10-12× | ~90% |
| 15-17× | ~80% |
| ~20× | ~60% |

## Dataset

This evaluation uses the **Fox Benchmark** dataset, which contains diverse document images with ground truth OCR annotations.

### Download

```bash
# Download from Hugging Face
wget https://huggingface.co/datasets/ucaslcl/Fox_benchmark_data/resolve/main/focus_benchmark_test.zip
unzip focus_benchmark_test.zip
```

### Dataset Structure

```
focus_benchmark_test/
├── en_pdf_png/          # English document images
├── cn_pdf_png/          # Chinese document images
├── en_page_ocr.json     # English page-level OCR ground truth
├── cn_page_ocr.json     # Chinese page-level OCR ground truth
└── ...                  # Other annotation files
```

### Ground Truth Format

The `en_page_ocr.json` file contains:

```json
[
  {
    "image": "en_1.png",
    "len": 826,
    "conversations": [
      {"from": "human", "value": "<image>\nOCR this image: "},
      {"from": "gpt", "value": "FREEDOM OF INFORMATION ACT..."}
    ]
  }
]
```

- `image`: Filename of the document image
- `len`: Number of tokens in ground truth text
- `conversations[1].value`: Ground truth OCR text

## Example Results

Using Fox Benchmark `en_1.png` (826 ground truth tokens):

| Mode | Vision Tokens | Compression | Edit Distance | Precision |
|------|---------------|-------------|---------------|-----------|
| Tiny | 64 | 16.12× | 220 | 95.46% |
| Small | 100 | 10.29× | 143 | 97.05% |
| Base | 197 | 5.26× | 77 | 98.41% |

## References

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Fox Benchmark](https://github.com/ucaslcl/Fox)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

## Citation

```bibtex
@article{wei2024deepseekocr,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2024}
}

@article{liu2024focus,
  title={Focus Anywhere for Fine-grained Multi-page Document Understanding},
  author={Liu, Chenglong and Wei, Haoran and others},
  journal={arXiv preprint arXiv:2405.14295},
  year={2024}
}
```
