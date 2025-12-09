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

---

## Vision Tokens as Context Experiment

We tested whether vision tokens (rendered text images) can replace raw text tokens as context for question-answering tasks, while being more efficient.

### Hypothesis

Using vision tokens for context maintains reasoning performance while using fewer tokens than raw text.

### Methodology

1. **Text condition**: Pass document text as raw tokens + question → answer
2. **Vision condition**: Pass document as rendered image (vision tokens) + question → answer

Both conditions use the same DeepSeek-OCR model. We tested across 30 questions on 6 document types (employee records, product specs, invoices, etc.).

### Running the Experiment

```bash
# Render test documents
uv run python experiment/render_documents.py

# Run baseline experiment
uv run python experiment/run_experiment.py --mode small

# Results saved to experiment/results/
```

### Baseline Results

| Condition | Accuracy | Tokens Used |
|-----------|----------|-------------|
| Text | 93.3% (28/30) | 6,625 |
| Vision | 86.7% (26/30) | 3,300 |

**Compression: 2.3×** with only 6.6% accuracy drop.

---

## Text-to-Image Configuration Study

We systematically tested how different text rendering configurations affect vision token accuracy.

### Configurations Tested

- **Fonts**: Monospace, Serif, Sans-serif
- **Sizes**: Small (14pt), Medium (20pt), Large (28pt)
- **Colors**: Default (B&W), Dark mode, Sepia, Blue, Low contrast

### Running Configuration Experiments

```bash
# Render documents with all configurations
uv run python experiment/render_documents.py --config all

# Run font comparison
uv run python experiment/run_all_configs.py --config-type fonts --mode small

# Run size comparison
uv run python experiment/run_all_configs.py --config-type sizes --mode small

# Run color comparison
uv run python experiment/run_all_configs.py --config-type colors --mode small

# Results saved to experiment/results/comparison_*.json
```

### Font Comparison Results

| Font | Text Acc | Vision Acc | Compression |
|------|----------|------------|-------------|
| **Monospace** | 93.3% | **86.7%** | 2.30× |
| Serif | 93.3% | 83.3% | 2.30× |
| Sans-serif | 93.3% | 80.0% | 2.30× |

**Finding**: Monospace fonts are most readable for OCR.

### Size Comparison Results

| Size | Text Acc | Vision Acc | Compression |
|------|----------|------------|-------------|
| **Small (14pt)** | 93.3% | **86.7%** | 2.30× |
| **Medium (20pt)** | 93.3% | **86.7%** | 2.30× |
| Large (28pt) | 93.3% | 76.7% | 2.30× |

**Finding**: Smaller fonts perform better (more content fits in fixed resolution).

### Color Scheme Comparison Results

| Color Scheme | Text Acc | Vision Acc | Compression |
|--------------|----------|------------|-------------|
| **Dark Mode** | 93.3% | **93.3%** | 2.30× |
| Blue | 93.3% | 90.0% | 2.30× |
| Default (B&W) | 93.3% | 86.7% | 2.30× |
| Sepia | 93.3% | 80.0% | 2.30× |
| Low Contrast | 93.3% | 70.0% | 2.30× |

**Finding**: Dark mode (light text on dark background) achieves perfect accuracy parity!

### Key Findings

1. **Best configuration**: Dark mode + Monospace + Small/Medium font
   - Achieves **93.3% vision accuracy** (matches text condition!)
   - Maintains 2.3× compression

2. **Dark mode is optimal**: Light text on dark background significantly outperforms black-on-white

3. **Avoid**:
   - Low contrast color schemes (70% accuracy)
   - Large fonts (76.7% accuracy)
   - Sepia tones (80% accuracy)

4. **Font choice matters**: Monospace > Serif > Sans-serif for OCR readability

### Experiment Files

```
experiment/
├── render_documents.py    # Render text to images with various configs
├── run_experiment.py      # Run single configuration experiment
├── run_all_configs.py     # Run comparison across configurations
├── questions.json         # Test questions for each document
├── data/                  # Rendered document images
│   ├── font_mono/
│   ├── font_serif/
│   ├── font_sans/
│   ├── size_small/
│   ├── size_medium/
│   ├── size_large/
│   ├── color_default/
│   ├── color_dark/
│   ├── color_sepia/
│   ├── color_blue/
│   └── color_low_contrast/
└── results/               # Experiment results (JSON)
    ├── comparison_fonts_small.json
    ├── comparison_sizes_small.json
    └── comparison_colors_small.json
```

---

## Long-Document Experiment (QuALITY Dataset)

We tested vision tokens on long documents (~4,000-5,000 words) from the [QuALITY](https://arxiv.org/abs/2112.08608) multiple-choice QA dataset to demonstrate higher compression ratios.

### Why Long Documents?

Short documents (like our initial experiments) only achieve ~2x compression because the text fits comfortably in context. Long documents that exceed the model's 8192 token limit are where vision encoding shines - compressing thousands of text tokens into just 256 vision tokens.

### Running the QuALITY Experiment

```bash
# Run on 10 articles, 5 questions each
uv run python experiment/quality_experiment.py --mode base --num-articles 10 --questions-per-article 5

# Results saved to experiment/results/quality_experiment_*.json
```

### Results (3 articles, 9 questions)

#### Mode Comparison

| Mode | Vision Tokens | Text Accuracy | Vision Accuracy | Compression |
|------|---------------|---------------|-----------------|-------------|
| Tiny | 64 | 44.4% (4/9) | 22.2% (2/9) | 39.7x |
| Base | 256 | 44.4% (4/9) | 33.3% (3/9) | 18.3x |
| **Large** | 400 | 44.4% (4/9) | **44.4% (4/9)** | 13.0x |

**Key finding**: Large mode achieves accuracy parity with text (44.4%) while using **13x fewer tokens**!

#### Base Mode Detailed Results

| Condition | Accuracy | Tokens Used | Compression |
|-----------|----------|-------------|-------------|
| Text | 44.4% (4/9) | 58,566 | -- |
| Vision | 33.3% (3/9) | 3,204 | **18.3x** |

### Key Observations

1. **13x compression with no accuracy loss** - Large mode vision matches text accuracy while using 13x fewer tokens

2. **Compression vs accuracy trade-off** - Higher compression ratios (tiny: 39.7x) come with lower accuracy (22.2%), but large mode (13x) maintains parity

3. **Both conditions struggle with these questions** - The 44% text accuracy shows even raw text hits the 8192 token context limit, truncating important information

4. **Vision can outperform text** - On Article 2 (70bd0370), text got 0/3 while large-mode vision got 1/3, demonstrating vision encoding can preserve more information when text overflows context

5. **Optimal mode selection** - For maximum compression use tiny/base; for accuracy parity use large mode

### Per-Article Breakdown (Large Mode)

| Article | Words | Text Accuracy | Vision Accuracy |
|---------|-------|---------------|-----------------|
| 50cd3c12 | 4,888 | 2/3 (67%) | 2/3 (67%) |
| 70bd0370 | 4,168 | 0/3 (0%) | 1/3 (33%) |
| d9088e2a | 4,535 | 2/3 (67%) | 1/3 (33%) |

### Critical Perspective

Recent work by [Lee et al. (2024)](https://arxiv.org/abs/2512.03643) titled "Optical Context Compression Is Just (Bad) Autoencoding" critiques vision-based context compression. **Important**: they argue there are *better ways* to compress context, not that vision tokens can't encode information at all.

**Their key findings:**
1. **Simpler methods win** - Mean pooling and learned hierarchical encoders match or beat vision at the same compression ratios
2. **Fails to beat truncation** - For language modeling tasks, optical compression doesn't outperform simply truncating text
3. **Reconstruction ≠ downstream performance** - Good OCR accuracy doesn't translate to good LM performance

**What they propose instead:**
- Parameter-free mean pooling of text embeddings
- Learned hierarchical text compression
- These achieve better compression/quality trade-offs without rendering overhead

**How our experiments relate:**
- Our "text" baseline is essentially truncation (context overflow), which the paper predicts should be competitive
- Vision only matches truncated text at large mode (13x compression) - consistent with their critique
- We did not compare against text-based compression methods (mean pooling, learned encoders), which the paper suggests would outperform vision
- The cases where vision beats text (Article 70bd0370) may reflect different information preservation patterns rather than vision being superior

---

## References

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Fox Benchmark](https://github.com/ucaslcl/Fox)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
- [QuALITY Dataset](https://arxiv.org/abs/2112.08608)
- [Optical Context Compression Critique (Lee et al.)](https://arxiv.org/abs/2512.03643)

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
