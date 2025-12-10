# Vision Tokens as LLM Context: An Empirical Investigation

**Research into the utility of optical context compression for large language models.**

## Research Question

Can vision tokens serve as an efficient alternative to text tokens for providing context to LLMs? Under what conditions does "optical context"—encoding text as images and processing through a vision encoder—outperform or complement traditional text tokenization?

This project investigates these questions using DeepSeek-OCR as a case study, a vision-language model that compresses document images into a small number of vision tokens while maintaining high OCR accuracy.

## Motivation

Traditional LLM context is expensive: long documents consume thousands of tokens, limiting what can fit in a context window. Vision encoders offer a potential compression mechanism—a rendered page of text might compress to ~100-400 vision tokens regardless of text length. But is this compression lossless for downstream tasks? When is it beneficial?

**Requirements:** Python 3.10+, CUDA GPU with 16GB+ VRAM

## Theoretical Framework

This research investigates two core hypotheses:

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
    *   **Experiments:** `OmniDocBench` uses this flow to validate the model's robustness on complex layouts.

### Hypothesis 3: Augmented Optical Context

Beyond compression, we hypothesize that encoding text as an image can capture **meta-features** and **global patterns** that autoregressive text tokenizers miss:

1.  **Visual Gestalt:** Vision encoders process data spatially/hierarchically, potentially recognizing genre and structure (e.g., "this looks like a contract") faster than linear text processing.
2.  **Spatial Indexing:** Visual tokens map to 2D patches, potentially preserving location information better than semantic vector pooling.
3.  **Robustness:** Vision may be less brittle to character-level noise (typos, encoding errors) than text tokenizers.
4.  **Rich Injection:** We can inject metadata into the visual channel—fonts, colors, highlighting—to guide attention in ways impossible with text tokens.

## Related Work & Critical Context

### The Case Against Optical Compression (Lee et al., 2024)

"[Optical Context Compression Is Just (Bad) Autoencoding](https://arxiv.org/abs/2512.03643)" presents a rigorous critique of DeepSeek-OCR's claims. Their key findings:

**1. Vision doesn't beat truncation.** For language modeling tasks, simply keeping the most recent N text tokens outperforms vision-based compression at matched token budgets. This is a damning result—if truncation works better, why use vision at all?

**2. Simpler baselines match or exceed vision:**
| Method | Parameters | Reconstruction | Language Modeling |
|--------|------------|----------------|-------------------|
| Mean Pooling | ~0 | ≈ Vision | Fails vs truncation |
| Hierarchical Encoder | Learned | > Vision | Beats truncation |
| Vision (DeepSeek-OCR) | Large | Baseline | Fails vs truncation |

**3. Reconstruction ≠ downstream utility.** DeepSeek-OCR's impressive OCR accuracy doesn't translate to language modeling gains. Good reconstruction is necessary but not sufficient.

**4. The "autoencoding" framing.** Vision encoders act as lossy autoencoders that discard information needed for downstream tasks while preserving information needed for reconstruction (pixel-level details).

### Prior Work on Visual Text Processing

The concept of processing text visually predates DeepSeek-OCR:
- **Pixel-based Language Models** (Rust et al., ICLR 2023) — character-level rendering
- **Pix2Struct** (Lee et al., ICML 2023) — screenshot parsing
- **CLIPPO** (Google, CVPR 2023) — vision-language from pixels only
- **Vision-Centric Token Compression** (Xin et al., 2024; Wang et al., 2024)

---

## Experiments

### Completed

#### 1. QuALITY: Long-Document QA (Hypothesis 1)

Tests whether vision tokens can replace text context for comprehension tasks.

| Mode | Vision Tokens | Text Accuracy | Vision Accuracy | Compression |
|------|---------------|---------------|-----------------|-------------|
| **Tiny** | 64 | 36.0% (18/50) | 26.0% (13/50) | **38.51x** |
| Base | 256 | 36.0% (18/50) | 30.0% (15/50) | 17.74x |
| **Large** | 400 | 36.0% (18/50) | **34.0% (17/50)** | 12.63x |

**Finding**: Large mode achieves near-parity with text (34% vs 36%) at **12.6x compression**.

#### 2. FineWiki: Language Modeling (Hypothesis 1)

Tests next-sentence prediction using vision vs text context.

| Metric | Text Condition | Vision Condition |
|--------|----------------|------------------|
| **Avg Word Overlap** | 7.9% | **14.5%** |
| Tokens Used | 15,709 | 6,120 |
| **Compression Ratio** | -- | **2.57x** |

**Finding**: Vision condition shows higher word overlap, but this may reflect the model's tendency to OCR and paraphrase rather than true comprehension.

#### 3. Text Rendering Configuration Study

Identified optimal rendering settings for text-to-image conversion:

| Factor | Optimal Setting | Accuracy |
|--------|-----------------|----------|
| Color Scheme | **Dark mode** | 93.3% |
| Font | **Monospace** | 86.7% |
| Size | **12-20pt** | 86.7% |

All subsequent experiments use **dark mode + monospace + 12pt**.

### Honest Assessment: Our Results vs. The Critique

Our experimental findings **partially confirm** Lee et al.'s critique:

| Their Claim | Our Evidence | Verdict |
|-------------|--------------|---------|
| Vision doesn't beat text for LM tasks | QuALITY: Text 36% vs Vision 34% (large mode) | **Confirmed** — text wins, though margin is small |
| Compression comes at accuracy cost | Tiny mode: 38x compression but only 26% accuracy | **Confirmed** — severe trade-off at high compression |
| Reconstruction ≠ downstream utility | FineWiki: Higher word overlap for vision, but likely due to OCR+paraphrase, not comprehension | **Plausible** — our metric may be flawed |

**What we did NOT test (gaps in our work):**
- We didn't compare against mean pooling or hierarchical encoder baselines
- We didn't compare against simple truncation at matched token budgets
- We used DeepSeek-OCR's model, not independent vision encoders

**Where our results diverge:**
- At 12.6x compression (large mode), vision achieves 94% of text accuracy (34/36). This is better than "fails vs truncation" suggests—but we didn't directly compare to truncation.
- The compression is real: 400 vision tokens vs ~5000 text tokens for the same article.

### Research Gap: What Lee et al. Didn't Test

The critique focused on **clean text → compression → language modeling**. They did not evaluate:

1. **Noisy/degraded input** — Does vision's robustness to character-level noise change the calculus?
2. **Structured data** — Do tables, forms, and code benefit from 2D spatial encoding?
3. **Augmented rendering** — Can visual formatting (colors, fonts, highlighting) carry semantic signal that text cannot?
4. **Real-world documents** — Scanned PDFs with layout, figures, and mixed content

These gaps motivate our proposed experiments.

### Proposed Experiments

Each experiment targets a gap in Lee et al.'s analysis, seeking conditions where vision **does** outperform text:

#### Experiment A: Robustness Boundary (Noise Injection)

**Gap addressed:** Lee et al. tested clean text only. Real-world text is noisy.

| Condition | Text Tokenizer | Vision Encoder |
|-----------|----------------|----------------|
| Clean text | ✓ Optimal | Baseline |
| Typos (5%) | Degraded | ? |
| OCR errors (10%) | Severely degraded | ? |
| Encoding corruption | Fails | ? |

*   **Hypothesis:** Vision encoders, trained on diverse image corruptions, may degrade more gracefully than text tokenizers that expect clean input.
*   **Method:** Progressively corrupt input text → Render to image → Compare degradation curves (text vs vision) on QuALITY QA task.
*   **Success criterion:** Find noise threshold where vision accuracy > text accuracy.
*   **Contribution:** Identifies practical scenarios (OCR'd documents, user-generated content, historical texts) where optical context is preferable.

#### Experiment B: Structured Data (Tables & Code)

**Gap addressed:** Lee et al. tested prose. Structured data has 2D semantics that linearization destroys.

*   **Hypothesis:** For tasks requiring spatial reasoning ("What is in row 3, column 2?"), vision preserves structure that text flattening loses.
*   **Method:**
    - Dataset: TableBench or WikiTableQuestions
    - Task: Structural queries (cell lookup, row/column operations) vs semantic queries
    - Compare: Vision vs text (markdown table) vs text (linearized)
*   **Success criterion:** Vision outperforms text on structural queries while matching on semantic queries.
*   **Contribution:** Identifies task categories where modality matters.

#### Experiment C: Augmented Rendering (Visual Metadata Injection)

**Gap addressed:** Lee et al. treated vision as pure compression. Vision can carry **additional** signal.

*   **Hypothesis:** Visual formatting (bold, color, size) can encode semantic information (entity types, importance, relationships) that improves downstream tasks.
*   **Method:**
    - Baseline: Plain text rendering (dark mode, monospace)
    - Treatment: Semantic rendering (entities bolded, keywords highlighted, section headers enlarged)
    - Task: QuALITY QA or entity-centric questions
*   **Success criterion:** Augmented rendering > plain rendering > text-only (at matched compression).
*   **Contribution:** Novel demonstration that vision is not just compression but a **richer encoding channel**.

#### Experiment D: Truncation Baseline (Addressing the Core Critique)

**Gap addressed:** We didn't directly compare to truncation—the baseline Lee et al. showed vision fails against.

*   **Method:** For QuALITY, compare:
    - Full text (if fits in context)
    - Truncated text (last N tokens matching vision token budget)
    - Vision (rendered full article)
*   **Purpose:** Establish whether our results replicate Lee et al.'s truncation finding, or whether task/model differences yield different conclusions.
*   **Contribution:** Direct engagement with the core critique.

---

## Methodology

### Metrics

**Compression Ratio** = Text Tokens / Vision Tokens

**Precision** (for OCR tasks) uses Levenshtein distance:
```
Precision = 1 - (EditDistance / max(len(output), len(ground_truth)))
```

### Resolution Modes

| Mode | Resolution | Vision Tokens | Use Case |
|------|------------|---------------|----------|
| `tiny` | 512×512 | 64 | Maximum compression |
| `small` | 640×640 | 100 | Balanced |
| `base` | 1024×1024 | 256 | Higher fidelity |
| `large` | 1280×1280 | 400 | Best accuracy |

### Running Experiments

```bash
# QuALITY QA experiment
uv run python eval.py quality --mode base --num-articles 10

# FineWiki language modeling
uv run python eval.py finewiki --mode base --num-articles 20

# OmniDocBench OCR benchmark
uv run python eval.py omnidocbench --mode base --num-articles 10

# Full reproduction suite
uv run python eval.py reproduce [--fast]
```

---

## References

- Wei et al. (2024). [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
- Lee et al. (2024). [Optical Context Compression Critique](https://arxiv.org/abs/2512.03643)
- Pang et al. (2022). [QuALITY: Question Answering with Long Input Texts](https://arxiv.org/abs/2112.08608)

## Datasets

- [QuALITY](https://huggingface.co/datasets/emozilla/quality) — Long-document QA (3,000-8,000 word articles)
- [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) — High-quality Wikipedia for LM evaluation
- [OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) — Document OCR benchmark
