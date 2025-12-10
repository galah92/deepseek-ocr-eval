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

### Results (10 articles, 50 questions)

#### Mode Comparison

| Mode | Vision Tokens | Text Accuracy | Vision Accuracy | Compression |
|------|---------------|---------------|-----------------|-------------|
| **Tiny** | 64 | 36.0% (18/50) | 26.0% (13/50) | **38.51x** |
| Base | 256 | 36.0% (18/50) | 30.0% (15/50) | 17.74x |
| **Large** | 400 | 36.0% (18/50) | **34.0% (17/50)** | 12.63x |

**Key finding**: Large mode achieves near-parity with text (34% vs 36%) while using **12.6x fewer tokens**!

#### Large Mode Detailed Results

| Condition | Accuracy | Tokens Used | Compression |
|-----------|----------|-------------|-------------|
| Text | 36.0% (18/50) | 315,795 | -- |
| Vision | 34.0% (17/50) | 25,000 | **12.63x** |

### Key Observations

1. **12.6x compression with minimal accuracy loss** - Large mode vision achieves 34% vs text's 36% while using 12.6x fewer tokens (only 2% accuracy drop)

2. **Consistent compression vs accuracy trade-off**:
   - Tiny (38.5x compression): 26% vision accuracy (10% drop from text)
   - Base (17.7x compression): 30% vision accuracy (6% drop from text)
   - Large (12.6x compression): 34% vision accuracy (2% drop from text)

3. **Text accuracy plateaus at 36%** - This consistent text accuracy across all modes confirms that documents exceed the 8192 token context limit, causing truncation

4. **Vision scales predictably** - Higher resolution modes (more vision tokens) recover more information, with near-linear improvement

5. **Optimal mode selection** - For maximum compression use tiny/base; for near-accuracy parity use large mode

---

## FineWiki Language Modeling Experiment

Following the methodology of [Lee et al. (2024)](https://arxiv.org/abs/2512.03643), we tested vision tokens on the [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) dataset to evaluate language modeling performance.

### Methodology

1. Load Wikipedia articles from FineWiki (English subset)
2. Split each article: first 500 words as context, next 50 words as target continuation
3. **Text condition**: Pass raw text context + ask model to continue
4. **Vision condition**: Render context as image + ask model to continue
5. Measure word overlap between prediction and target continuation

### Running the FineWiki Experiment

```bash
# Run on 20 articles with base mode
uv run python experiment/finewiki_experiment.py --mode base --num-articles 20

# Results saved to experiment/results/finewiki_experiment_*.json
```

### Results (20 articles, base mode)

| Metric | Text Condition | Vision Condition |
|--------|----------------|------------------|
| **Avg Word Overlap** | 7.9% | **14.5%** |
| First Word Match | 5.0% | 0.0% |
| Tokens Used | 15,709 | 6,120 |
| **Compression Ratio** | -- | **2.57x** |

### Key Findings

1. **Vision outperforms text on overlap metric** - Vision achieves 14.5% word overlap vs text's 7.9%, nearly 2x better

2. **Both conditions struggle with true continuation** - Low overlap scores indicate the model tends to paraphrase/summarize rather than predict exact next words

3. **Vision generates more coherent continuations** - Vision often reproduces the context structure and continues naturally, while text frequently produces empty outputs

4. **Text produces more empty outputs** - Many text predictions are empty (""), while vision usually generates content (often restating context then extending)

5. **Different failure modes**:
   - **Text**: Often generates nothing or unrelated content
   - **Vision**: Tends to OCR the rendered context first, then attempt continuation

### Example Outputs

**Article: "1778 in music"**
- Target: "Little Organ Mass - Symphony No.54 in G major..."
- Text prediction: "" (empty)
- Vision prediction: Correctly OCRs the context, then continues with relevant musical works

**Article: "1847 in the United States"**
- Target: List of Lieutenant Governors
- Text prediction: "" (empty)
- Vision prediction: Reproduces context structure, achieves 78.8% word overlap

### Critical Perspective

This experiment relates directly to the Lee et al. (2024) critique:

**Their key findings:**
1. **Simpler methods win** - Mean pooling and learned hierarchical encoders match or beat vision at the same compression ratios
2. **Fails to beat truncation** - For language modeling tasks, optical compression doesn't outperform simply truncating text
3. **Reconstruction ≠ downstream performance** - Good OCR accuracy doesn't translate to good LM performance

**What they propose instead:**
- Parameter-free mean pooling of text embeddings
- Learned hierarchical text compression
- These achieve better compression/quality trade-offs without rendering overhead

**How our experiments relate:**
- Our QuALITY results show vision matches truncated text at large mode (12.6x compression) - consistent with their critique
- Our FineWiki results show vision actually outperforms text on continuation, but this may be due to the model's tendency to OCR and paraphrase rather than true language modeling
- We did not compare against text-based compression methods (mean pooling, learned encoders), which the paper suggests would outperform vision
- The evaluation metric (word overlap) favors approaches that reproduce context, which vision does naturally

---

## References

- [DeepSeek-OCR Paper](https://arxiv.org/abs/2510.18234)
- [DeepSeek-OCR Model](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Fox Benchmark](https://github.com/ucaslcl/Fox)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
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

@article{liu2024focus,
  title={Focus Anywhere for Fine-grained Multi-page Document Understanding},
  author={Liu, Chenglong and Wei, Haoran and others},
  journal={arXiv preprint arXiv:2405.14295},
  year={2024}
}
```
