# DeepSeek-OCR Evaluation

Evaluation tools for measuring DeepSeek-OCR's compression ratio and accuracy on document images.

## Overview

DeepSeek-OCR is a vision-language model that compresses document images into a small number of vision tokens while maintaining high OCR accuracy. This tool evaluates the model's performance by:

1. Feeding a document image to the model
2. Measuring how many vision tokens are used
3. Comparing OCR output to ground truth text
4. Calculating compression ratio and precision

**Requirements:** Python 3.10+, CUDA GPU with 16GB+ VRAM

## Usage

### OCR Evaluation

```bash
# Basic inference
uv run python eval.py ocr --image document.png --mode small

# With ground truth for accuracy calculation
uv run python eval.py ocr --image document.png --ground-truth gt.txt --mode small

# Dry run (no GPU needed) - shows token calculations only
uv run python eval.py ocr --image document.png --dry-run
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
uv run python eval.py quality --mode base --num-articles 10

# FineWiki dataset (language modeling)
uv run python eval.py finewiki --mode base --num-articles 20
```

## Hypotheses & Theoretical Framework

This evaluation suite is designed to test two core hypotheses regarding optical context compression:

### Hypothesis 1: Image as Context Compression (The "Zip" Theory)

*   **Claim:** Textual information can be "zipped" into an image representation (vision tokens) and decoded or used as context by an LLM with minimal information loss, achieving high compression ratios (10x-100x).
*   **Testing Method:**
    *   **Flow 1 (Text-to-Image):** We render raw text into images using optimal settings (Dark Mode, Monospace) and feed them to the model.
    *   **Metric:** We measure if `Vision Accuracy` $\approx$ `Text Accuracy` while `Vision Tokens` $\ll$ `Text Tokens`.
    *   **Experiments:** `QuALITY` (QA) and `FineWiki` (Language Modeling) use this flow to isolate the compression efficiency from image quality issues.

### Hypothesis 2: Visual Projection Improves Data Usage (The "Modality" Theory)

*   **Claim:** Projecting information into the visual modality allows the model to utilize the data more effectively than linear text tokens, particularly for structured data where spatial relationships matter.
*   **Reasoning:**
    *   **Structure:** Images preserve 2D layouts (tables, forms, code blocks) that are flattened in text.
    *   **Attention:** Visual encoders may attend to salient features (headers, bold text) more naturally.
*   **Testing Method:**
    *   **Flow 2 (Image-to-Text):** We feed real-world document images (scans, PDFs) directly to the model.
    *   **Experiments:** `OmniDocBench` (and standard `OCR`) use this flow to validate the model's robustness and ability to leverage visual cues in complex layouts.

## Reproducing Paper Results

To verify the claims of the DeepSeek-OCR paper, this suite includes a reproduction command that runs all three key experiments:

1.  **OmniDocBench (OCR):** Evaluates OCR precision and compression on diverse real-world documents.
2.  **QuALITY (QA):** Tests long-document comprehension using vision vs. text context.
3.  **FineWiki (Language Modeling):** Measures next-sentence prediction capability.

### Running Reproduction

Run the full suite (Note: Requires high VRAM, e.g., A100 40GB+ or multiple GPUs):

```bash
uv run python eval.py reproduce
```

Run a fast verification (1 document per task, skips memory-intensive modes):

```bash
uv run python eval.py reproduce --fast
```

### OmniDocBench

You can also run the OmniDocBench experiment individually:

```bash
uv run python eval.py omnidocbench --mode base --num-articles 10
```

*Note: This will automatically download the necessary annotations and images from Hugging Face.*

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
├── eval.py          # All evaluation and experiments (ocr, quality, finewiki subcommands)
├── results/         # Experiment results (JSON)
└── .cache/          # Cached rendered articles for experiments
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

### Text Rendering Configuration Study

We tested how different text-to-image rendering configurations affect OCR accuracy when using vision tokens as context.

**Configurations tested:**
- **Fonts**: Monospace, Serif, Sans-serif
- **Sizes**: Small (14pt), Medium (20pt), Large (28pt)
- **Colors**: Default (black on white), Dark mode, Sepia, Blue, Low contrast

| Font | Vision Accuracy |
|------|-----------------|
| **Monospace** | **86.7%** |
| Serif | 83.3% |
| Sans-serif | 80.0% |

| Size | Vision Accuracy |
|------|-----------------|
| **Small (14pt)** | **86.7%** |
| **Medium (20pt)** | **86.7%** |
| Large (28pt) | 76.7% |

| Color Scheme | Vision Accuracy |
|--------------|-----------------|
| **Dark mode** | **93.3%** |
| Blue | 90.0% |
| Default (B&W) | 86.7% |
| Sepia | 80.0% |
| Low contrast | 70.0% |

**Key findings:**
1. **Dark mode is optimal** — light text on dark background achieves parity with text-only condition (93.3%)
2. **Monospace fonts** outperform serif and sans-serif for OCR readability
3. **Smaller fonts** (14-20pt) work better than large fonts (more content fits in fixed resolution)
4. **Avoid** low contrast schemes and sepia tones

Based on these results, we use **dark mode + monospace + 12pt font** for all text-to-image rendering.

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

## Datasets

- **[QuALITY](https://huggingface.co/datasets/emozilla/quality)**: Long-document multiple-choice QA dataset. Articles are 3,000-8,000 words with challenging comprehension questions.
- **[FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)**: High-quality Wikipedia articles for language modeling evaluation.

Both datasets are automatically downloaded from HuggingFace when running experiments.

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
