"""
DeepSeek-OCR Evaluation: Compression Ratio & Accuracy

Evaluates the DeepSeek-OCR model's ability to compress document images into
vision tokens while maintaining OCR accuracy.

Usage:
    # OCR evaluation
    uv run python eval.py ocr --image document.png --mode small
    uv run python eval.py ocr --image document.png --ground-truth gt.txt

    # QuALITY long-document QA experiment
    uv run python eval.py quality --mode base --num-articles 10

    # FineWiki language modeling experiment
    uv run python eval.py finewiki --mode base --num-articles 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont
from transformers import AutoModel, AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

# Temporary paths for experiments
TMP_OUTPUT_PATH = "/tmp/deepseek_ocr_output"
TMP_BLANK_IMAGE = "/tmp/blank_32x32.png"

# Token overhead for prompts in experiments
PROMPT_TOKEN_OVERHEAD = 100
CONTINUATION_TOKEN_OVERHEAD = 50


class ModeSettings:
    """Settings for a resolution mode."""

    def __init__(
        self, base_size: int, image_size: int, crop_mode: bool, tokens: int | None
    ):
        self.base_size = base_size
        self.image_size = image_size
        self.crop_mode = crop_mode
        self.tokens = tokens  # None for dynamic modes


MODE_SETTINGS: dict[str, ModeSettings] = {
    "tiny": ModeSettings(512, 512, False, 64),
    "small": ModeSettings(640, 640, False, 100),
    "base": ModeSettings(1024, 1024, False, 256),
    "large": ModeSettings(1280, 1280, False, 400),
    "gundam": ModeSettings(1024, 640, True, None),
}

# Global model cache
_model: AutoModel | None = None
_tokenizer: AutoTokenizer | None = None


MONO_FONT_PATH = font_manager.findfont("monospace")


def get_font(size: int = 14) -> FreeTypeFont:
    """Load a monospace font using matplotlib's font manager."""
    return ImageFont.truetype(MONO_FONT_PATH, size)


# ============================================================================
# Model Loading
# ============================================================================


def load_model() -> tuple[AutoModel, AutoTokenizer]:
    """Load the DeepSeek-OCR model (cached after first load)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch

    model_name = "deepseek-ai/DeepSeek-OCR"
    print(f"Loading model from {model_name}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    _model = _model.eval().cuda()
    return _model, _tokenizer


# ============================================================================
# Image Rendering
# ============================================================================

# Default rendering settings (dark mode for optimal OCR - see README)
DEFAULT_FONT_SIZE = 12
DEFAULT_BG_COLOR = "#1e1e1e"
DEFAULT_FG_COLOR = "#d4d4d4"

# Log font config at module load
_font = get_font(DEFAULT_FONT_SIZE)
print(
    f"[Render config] font={_font.getname()[0]}, size={DEFAULT_FONT_SIZE}pt, "
    f"bg={DEFAULT_BG_COLOR}, fg={DEFAULT_FG_COLOR}, path={MONO_FONT_PATH}"
)
del _font


def render_text_to_image(
    text: str,
    output_path: str,
    font_size: int = DEFAULT_FONT_SIZE,
    max_width: int = 1200,
    padding: int = 30,
    line_spacing: int = 4,
    bg_color: str = DEFAULT_BG_COLOR,
    fg_color: str = DEFAULT_FG_COLOR,
) -> tuple[int, int, int]:
    """Render text to image (dark mode for optimal OCR).

    Returns:
        Tuple of (image_width, image_height, num_lines).
    """
    font = get_font(font_size)

    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = font.getbbox(test_line)
            if bbox[2] > max_width - 2 * padding:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))

    line_height = font_size + line_spacing
    img_height = len(lines) * line_height + 2 * padding
    img_width = max_width

    img = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=fg_color)
        y += line_height

    img.save(output_path)
    return img_width, img_height, len(lines)


# ============================================================================
# Token Calculation
# ============================================================================


def calculate_valid_vision_tokens(
    width: int, height: int, settings: ModeSettings
) -> int:
    """Calculate valid vision tokens based on image dimensions and mode settings."""
    base_size = settings.base_size
    image_size = settings.image_size
    crop_mode = settings.crop_mode
    ratio = 1 - ((max(width, height) - min(width, height)) / max(width, height))

    if crop_mode:
        if base_size == 1024:
            valid_tokens = int(256 * ratio)
        elif base_size == 1280:
            valid_tokens = int(400 * ratio)
        else:
            valid_tokens = 0

        if width > 640 or height > 640:
            num_crops = calculate_num_crops(width, height, image_size)
            if image_size == 640:
                valid_tokens += num_crops * 100
            elif image_size == 1024:
                valid_tokens += num_crops * 256
    else:
        if base_size == 1024:
            valid_tokens = int(256 * ratio)
        elif base_size == 1280:
            valid_tokens = int(400 * ratio)
        elif base_size == 640:
            valid_tokens = 100
        elif base_size == 512:
            valid_tokens = 64
        else:
            valid_tokens = 0

    return valid_tokens


def calculate_num_crops(
    width: int,
    height: int,
    tile_size: int = 640,
    min_crops: int = 2,
    max_crops: int = 6,
) -> int:
    """Calculate number of image crops for dynamic resolution mode."""
    aspect_ratio = width / height
    target_ratios = set(
        (i, j)
        for n in range(min_crops, max_crops + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_crops <= i * j <= max_crops
    )

    best_ratio = (1, 1)
    best_diff = float("inf")

    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio

    return best_ratio[0] * best_ratio[1]


def tokenize_text(text: str, tokenizer: AutoTokenizer | None = None) -> int:
    """Count tokens in text using the model's tokenizer or approximation."""
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


# ============================================================================
# OCR Evaluation
# ============================================================================


def calculate_edit_distance(output: str, ground_truth: str) -> dict[str, int | float]:
    """Calculate edit distance metrics using Levenshtein distance.

    Returns:
        Dict with edit_distance, normalized_ed, and precision (as percentage).
    """
    import Levenshtein

    edit_dist = Levenshtein.distance(output, ground_truth)
    max_len = max(len(output), len(ground_truth))
    normalized_ed = edit_dist / max_len if max_len > 0 else 0
    precision = 1 - normalized_ed

    return {
        "edit_distance": edit_dist,
        "normalized_ed": round(normalized_ed, 4),
        "precision": round(precision * 100, 2),
    }


def run_ocr(
    image_path: str, mode: str, prompt: str = "<image>\nFree OCR."
) -> tuple[str, int, int]:
    """Run OCR inference using the DeepSeek-OCR model.

    Returns:
        Tuple of (ocr_output, vision_tokens, output_tokens).
    """
    model, tokenizer = load_model()
    settings = MODE_SETTINGS[mode]

    img = Image.open(image_path)
    width, height = img.size

    vision_tokens = calculate_valid_vision_tokens(width, height, settings)

    print(f"Running inference (mode={mode}, vision_tokens={vision_tokens})...")
    output = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=TMP_OUTPUT_PATH,
        base_size=settings.base_size,
        image_size=settings.image_size,
        crop_mode=settings.crop_mode,
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )

    output_tokens = tokenize_text(output, tokenizer)
    return output, vision_tokens, output_tokens


def cmd_ocr(args: argparse.Namespace) -> None:
    """Run OCR evaluation on an image, optionally comparing to ground truth."""
    ground_truth = None
    if args.ground_truth:
        if os.path.isfile(args.ground_truth):
            with open(args.ground_truth, "r", encoding="utf-8") as f:
                ground_truth = f.read()
        else:
            ground_truth = args.ground_truth

    if args.dry_run:
        settings = MODE_SETTINGS[args.mode]
        img = Image.open(args.image)
        width, height = img.size
        vision_tokens = calculate_valid_vision_tokens(width, height, settings)

        print("=" * 60)
        print("DeepSeek-OCR Evaluation (Dry Run)")
        print("=" * 60)
        print(f"\n[INPUT]\n  Image: {args.image}\n  Dimensions: {width} x {height}")
        print(f"  Mode: {args.mode}")
        print(f"\n[VISION TOKENS]\n  Valid vision tokens: {vision_tokens}")

        if ground_truth:
            gt_tokens = tokenize_text(ground_truth)
            compression = gt_tokens / vision_tokens if vision_tokens > 0 else 0
            print(f"\n[GROUND TRUTH]\n  Text length: {len(ground_truth)} characters")
            print(f"  Approx tokens: {gt_tokens}")
            print(f"\n[COMPRESSION]\n  Compression ratio: {compression:.2f}x")
        return

    print("=" * 60)
    print("DeepSeek-OCR Evaluation")
    print("=" * 60)

    output, vision_tokens, output_tokens = run_ocr(args.image, args.mode, args.prompt)
    compression = output_tokens / vision_tokens if vision_tokens > 0 else 0

    print(f"\n[RESULTS]\n  Vision tokens: {vision_tokens}")
    print(f"  Output tokens: {output_tokens}\n  Compression ratio: {compression:.2f}x")

    if ground_truth:
        metrics = calculate_edit_distance(output, ground_truth)
        print(f"\n[ACCURACY]\n  Edit distance: {metrics['edit_distance']} characters")
        print(
            f"  Normalized ED: {metrics['normalized_ed']}\n  Precision: {metrics['precision']}%"
        )

    if args.show_output:
        print(f"\n[OCR OUTPUT]\n{'-' * 60}")
        print(output[:2000] + ("..." if len(output) > 2000 else ""))


# ============================================================================
# QuALITY Experiment
# ============================================================================


def _ensure_blank_image() -> str:
    """Ensure a blank image exists for text-only inference and return its path."""
    if not os.path.exists(TMP_BLANK_IMAGE):
        Image.new("RGB", (32, 32), color="white").save(TMP_BLANK_IMAGE)
    return TMP_BLANK_IMAGE


def _infer_with_text(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    """Run inference with text context (using blank image)."""
    return model.infer(
        tokenizer,
        prompt=prompt,
        image_file=_ensure_blank_image(),
        output_path=TMP_OUTPUT_PATH,
        base_size=512,
        image_size=512,
        crop_mode=False,
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )


def _infer_with_vision(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    image_path: str,
    settings: ModeSettings,
) -> str:
    """Run inference with image context."""
    return model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=TMP_OUTPUT_PATH,
        base_size=settings.base_size,
        image_size=settings.image_size,
        crop_mode=settings.crop_mode,
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )


def cmd_quality(args: argparse.Namespace) -> None:
    """Run QuALITY long-document QA experiment comparing text vs vision accuracy."""
    from datasets import load_dataset

    root_dir = Path(__file__).parent
    data_dir = root_dir / ".cache" / "quality"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading QuALITY dataset...")
    try:
        ds = load_dataset("emozilla/quality", split="validation")
    except Exception as e:
        print(f"Error loading QuALITY dataset: {e}", file=sys.stderr)
        return

    # Group questions by article
    articles = {}
    for item in ds:
        article_hash = hashlib.md5(item["article"][:100].encode()).hexdigest()[:8]
        if article_hash not in articles:
            articles[article_hash] = {"article": item["article"], "questions": []}
        articles[article_hash]["questions"].append(
            {
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
            }
        )

    print(f"Found {len(articles)} unique articles")
    article_items = list(articles.items())[: args.num_articles]

    model, tokenizer = load_model()
    settings = MODE_SETTINGS[args.mode]

    results = {"mode": args.mode, "articles": {}, "summary": {}}
    stats = {
        "text_correct": 0,
        "vision_correct": 0,
        "text_tokens": 0,
        "vision_tokens": 0,
        "total": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"QUALITY EXPERIMENT: Mode={args.mode}, Articles={args.num_articles}")
    print("=" * 70)

    for article_hash, article_data in article_items:
        article = article_data["article"]
        questions = article_data["questions"][: args.questions_per_article]

        print(
            f"\n{'─' * 70}\nArticle: {article_hash} ({len(article.split())} words)\n{'─' * 70}"
        )

        img_path = data_dir / f"{article_hash}.png"
        if not img_path.exists():
            render_text_to_image(article, str(img_path))

        article_results = {"questions": [], "text_correct": 0, "vision_correct": 0}

        for qa in questions:
            question, options, expected = qa["question"], qa["options"], qa["answer"]
            options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))

            # Text condition
            text_prompt = f"<image>\n{article}\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer with just the option number (0, 1, 2, or 3):"
            text_output = _infer_with_text(model, tokenizer, text_prompt)
            text_tokens = (
                len(tokenizer.encode(article, add_special_tokens=False))
                + PROMPT_TOKEN_OVERHEAD
            )

            # Vision condition
            vision_prompt = f"<image>\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer with just the option number (0, 1, 2, or 3):"
            vision_output = _infer_with_vision(
                model, tokenizer, vision_prompt, str(img_path), settings
            )
            vision_tokens = settings.tokens + PROMPT_TOKEN_OVERHEAD

            # Parse answers
            text_pred = next((int(c) for c in text_output.strip() if c in "0123"), -1)
            vision_pred = next(
                (int(c) for c in vision_output.strip() if c in "0123"), -1
            )

            text_correct = text_pred == expected
            vision_correct = vision_pred == expected

            print(f"  Q: {question[:60]}...")
            print(
                f"    Text: {text_pred} {'✓' if text_correct else '✗'}, Vision: {vision_pred} {'✓' if vision_correct else '✗'}"
            )

            if text_correct:
                article_results["text_correct"] += 1
                stats["text_correct"] += 1
            if vision_correct:
                article_results["vision_correct"] += 1
                stats["vision_correct"] += 1

            stats["text_tokens"] += text_tokens
            stats["vision_tokens"] += vision_tokens
            stats["total"] += 1

        results["articles"][article_hash] = article_results

    # Summary
    n = stats["total"]
    text_acc = round(stats["text_correct"] / n * 100, 1) if n > 0 else 0
    vision_acc = round(stats["vision_correct"] / n * 100, 1) if n > 0 else 0
    compression = (
        stats["text_tokens"] / stats["vision_tokens"]
        if stats["vision_tokens"] > 0
        else 0
    )

    results["summary"] = {
        "total_questions": n,
        "text_accuracy": text_acc,
        "vision_accuracy": vision_acc,
        "compression_ratio": round(compression, 2),
    }

    output_path = results_dir / f"quality_{args.mode}_{args.num_articles}articles.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n{'=' * 70}\nRESULTS: Text={text_acc}%, Vision={vision_acc}%, Compression={compression:.1f}x"
    )
    print(f"Saved to: {output_path}")


# ============================================================================
# FineWiki Experiment
# ============================================================================


def cmd_finewiki(args: argparse.Namespace) -> None:
    """Run FineWiki language modeling experiment comparing text vs vision continuation."""
    from datasets import load_dataset

    root_dir = Path(__file__).parent
    data_dir = root_dir / ".cache" / "finewiki"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading FineWiki dataset...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/finewiki", name="en", split="train", streaming=True
        )
    except Exception as e:
        print(f"Error loading FineWiki dataset: {e}", file=sys.stderr)
        return

    # Find articles with sufficient length
    articles = []
    min_words = args.context_words + args.continuation_words + 100
    print(f"Finding {args.num_articles} articles with {min_words}+ words...")

    for item in ds:
        text = item["text"]
        words = text.split()
        if len(words) >= min_words:
            articles.append({"title": item.get("title", "Untitled"), "text": text})
            print(f"  Found: {item.get('title', 'Untitled')[:50]}...")
        if len(articles) >= args.num_articles:
            break

    model, tokenizer = load_model()
    settings = MODE_SETTINGS[args.mode]

    results = {"mode": args.mode, "articles": [], "summary": {}}
    stats = {
        "text_overlap": 0,
        "vision_overlap": 0,
        "text_tokens": 0,
        "vision_tokens": 0,
        "total": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"FINEWIKI EXPERIMENT: Mode={args.mode}, Articles={args.num_articles}")
    print("=" * 70)

    for i, article in enumerate(articles):
        words = article["text"].split()
        context = " ".join(words[: args.context_words])
        continuation = " ".join(
            words[args.context_words : args.context_words + args.continuation_words]
        )

        print(f"\n{'─' * 70}\nArticle {i + 1}: {article['title'][:50]}...\n{'─' * 70}")

        img_path = data_dir / f"article_{i}.png"
        if not img_path.exists():
            render_text_to_image(context, str(img_path))

        # Text condition
        text_prompt = (
            f"<image>\n{context}\n\nContinue this text with the next sentence:"
        )
        text_output = _infer_with_text(model, tokenizer, text_prompt)
        text_tokens = (
            len(tokenizer.encode(context, add_special_tokens=False))
            + CONTINUATION_TOKEN_OVERHEAD
        )

        # Vision condition
        vision_prompt = "<image>\n\nContinue this text with the next sentence:"
        vision_output = _infer_with_vision(
            model, tokenizer, vision_prompt, str(img_path), settings
        )
        vision_tokens = settings.tokens + CONTINUATION_TOKEN_OVERHEAD

        # Compute overlap scores
        target_words = set(continuation.lower().split())
        text_overlap = (
            len(set(text_output.lower().split()) & target_words) / len(target_words)
            if target_words
            else 0
        )
        vision_overlap = (
            len(set(vision_output.lower().split()) & target_words) / len(target_words)
            if target_words
            else 0
        )

        print(
            f"  Text overlap: {text_overlap:.2f}, Vision overlap: {vision_overlap:.2f}"
        )

        stats["text_overlap"] += text_overlap
        stats["vision_overlap"] += vision_overlap
        stats["text_tokens"] += text_tokens
        stats["vision_tokens"] += vision_tokens
        stats["total"] += 1

        results["articles"].append(
            {
                "title": article["title"],
                "text_overlap": round(text_overlap, 3),
                "vision_overlap": round(vision_overlap, 3),
            }
        )

    # Summary
    n = stats["total"]
    if n > 0:
        text_avg = stats["text_overlap"] / n
        vision_avg = stats["vision_overlap"] / n
        compression = (
            stats["text_tokens"] / stats["vision_tokens"]
            if stats["vision_tokens"] > 0
            else 0
        )

        results["summary"] = {
            "total_articles": n,
            "text_avg_overlap": round(text_avg, 3),
            "vision_avg_overlap": round(vision_avg, 3),
            "compression_ratio": round(compression, 2),
        }

        output_path = (
            results_dir / f"finewiki_{args.mode}_{args.num_articles}articles.json"
        )
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(
            f"\n{'=' * 70}\nRESULTS: Text={text_avg:.3f}, Vision={vision_avg:.3f}, Compression={compression:.1f}x"
        )
        print(f"Saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepSeek-OCR evaluation and experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # OCR command
    ocr_parser = subparsers.add_parser("ocr", help="Run OCR evaluation on an image")
    ocr_parser.add_argument(
        "--image", type=str, required=True, help="Path to input image"
    )
    ocr_parser.add_argument(
        "--mode", type=str, default="small", choices=list(MODE_SETTINGS.keys())
    )
    ocr_parser.add_argument(
        "--ground-truth", type=str, help="Ground truth text or file path"
    )
    ocr_parser.add_argument(
        "--dry-run", action="store_true", help="Show calculations only"
    )
    ocr_parser.add_argument("--prompt", type=str, default="<image>\nFree OCR.")
    ocr_parser.add_argument(
        "--show-output", action="store_true", help="Show OCR output"
    )

    # QuALITY command
    quality_parser = subparsers.add_parser(
        "quality", help="Run QuALITY long-document QA experiment"
    )
    quality_parser.add_argument(
        "--mode", type=str, default="base", choices=["tiny", "small", "base", "large"]
    )
    quality_parser.add_argument("--num-articles", type=int, default=10)
    quality_parser.add_argument("--questions-per-article", type=int, default=5)

    # FineWiki command
    finewiki_parser = subparsers.add_parser(
        "finewiki", help="Run FineWiki language modeling experiment"
    )
    finewiki_parser.add_argument(
        "--mode", type=str, default="base", choices=["tiny", "small", "base", "large"]
    )
    finewiki_parser.add_argument("--num-articles", type=int, default=20)
    finewiki_parser.add_argument("--context-words", type=int, default=500)
    finewiki_parser.add_argument("--continuation-words", type=int, default=50)

    args = parser.parse_args()

    if args.command == "ocr":
        cmd_ocr(args)
    elif args.command == "quality":
        cmd_quality(args)
    elif args.command == "finewiki":
        cmd_finewiki(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
