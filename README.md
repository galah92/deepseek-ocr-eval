# DeepSeek-OCR Evaluation

Evaluation tools for measuring DeepSeek-OCR's compression ratio and accuracy on document images.

## Installation

```bash
uv sync
```

**Requirements:** Python 3.10+, CUDA GPU with 16GB+ VRAM

## Usage

### OCR Evaluation

```bash
# Basic inference
uv run python eval.py ocr --image document.png --mode small

# With ground truth for accuracy calculation
uv run python eval.py ocr --image document.png --ground-truth gt.txt

# Dry run (no GPU needed) - shows token calculations only
uv run python eval.py ocr --image document.png --dry-run
```

### Long-Document Experiments

```bash
# QuALITY dataset (multiple-choice QA)
uv run python eval.py quality --mode base --num-articles 10

# FineWiki dataset (language modeling)
uv run python eval.py finewiki --mode base --num-articles 20
```

### Resolution Modes

| Mode | Resolution | Vision Tokens | Use Case |
|------|------------|---------------|----------|
| `tiny` | 512×512 | 64 | Maximum compression |
| `small` | 640×640 | 100 | Good balance |
| `base` | 1024×1024 | 256 | Higher accuracy |
| `large` | 1280×1280 | 400 | Best accuracy |

## Project Structure

```
deepseek-ocr-eval/
├── eval.py          # All evaluation and experiments
├── results/         # Experiment results (JSON)
└── .cache/          # Cached rendered articles
```

## Results

### QuALITY Long-Document QA

| Mode | Text Accuracy | Vision Accuracy | Compression |
|------|---------------|-----------------|-------------|
| Tiny | 36.0% | 26.0% | 38.5x |
| Base | 36.0% | 30.0% | 17.7x |
| Large | 36.0% | 34.0% | 12.6x |

### FineWiki Language Modeling

| Metric | Text | Vision |
|--------|------|--------|
| Avg Word Overlap | 7.9% | 14.5% |
| Compression | -- | 2.57x |

## References

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [QuALITY Dataset](https://arxiv.org/abs/2112.08608)
- [FineWiki Dataset](https://huggingface.co/datasets/HuggingFaceFW/finewiki)
