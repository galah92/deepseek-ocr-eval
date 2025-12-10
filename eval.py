"""
DeepSeek-OCR Evaluation: Compression Ratio & Accuracy

Evaluates the DeepSeek-OCR model's ability to compress document images into
vision tokens while maintaining OCR accuracy.

Usage:
    uv run python eval.py --image document.png --mode small
    uv run python eval.py --image document.png --ground-truth gt.txt --mode small
    uv run python eval.py --image document.png --dry-run
"""

import argparse
import os


def calculate_valid_vision_tokens(
    width: int, height: int, base_size: int, image_size: int, crop_mode: bool
) -> int:
    """
    Calculate valid vision tokens based on image dimensions and mode.

    Padding reduces valid tokens - this formula accounts for aspect ratio.
    From the paper: N_valid = N_actual * [1 - ((max(w,h) - min(w,h)) / max(w,h))]
    """
    ratio = 1 - ((max(width, height) - min(width, height)) / max(width, height))

    if crop_mode:
        # Global view tokens
        if base_size == 1024:
            valid_tokens = int(256 * ratio)
        elif base_size == 1280:
            valid_tokens = int(400 * ratio)
        else:
            valid_tokens = 0

        # Local view tokens (if image needs cropping)
        if width > 640 or height > 640:
            num_crops = calculate_num_crops(width, height, image_size)
            if image_size == 640:
                valid_tokens += num_crops * 100
            elif image_size == 1024:
                valid_tokens += num_crops * 256
    else:
        # Native resolution mode (no cropping)
        if base_size == 1024:
            valid_tokens = int(256 * ratio)
        elif base_size == 1280:
            valid_tokens = int(400 * ratio)
        elif base_size == 640:
            valid_tokens = 100  # No ratio adjustment for small sizes
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


def get_mode_settings(mode: str) -> dict:
    """
    Get resolution settings for each mode.

    From the paper (Table 1):
    - Tiny:  512x512,  64 tokens
    - Small: 640x640,  100 tokens
    - Base:  1024x1024, 256 tokens
    - Large: 1280x1280, 400 tokens
    - Gundam: dynamic (local + global views)
    """
    modes = {
        "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False, "tokens": 64},
        "small": {
            "base_size": 640,
            "image_size": 640,
            "crop_mode": False,
            "tokens": 100,
        },
        "base": {
            "base_size": 1024,
            "image_size": 1024,
            "crop_mode": False,
            "tokens": 256,
        },
        "large": {
            "base_size": 1280,
            "image_size": 1280,
            "crop_mode": False,
            "tokens": 400,
        },
        "gundam": {
            "base_size": 1024,
            "image_size": 640,
            "crop_mode": True,
            "tokens": "dynamic",
        },
    }
    return modes.get(mode, modes["small"])


def calculate_edit_distance(output: str, ground_truth: str) -> dict:
    """
    Calculate edit distance metrics using Levenshtein distance.

    This is the same metric used in OmniDocBench evaluation.

    Returns:
        edit_distance: Raw number of character edits needed
        normalized_ed: Edit distance / max(len(output), len(ground_truth))
        precision: 1 - normalized_ed (higher is better)
    """
    try:
        import Levenshtein
    except ImportError:
        print("Error: python-levenshtein not installed. Run: uv add python-levenshtein")
        return {"edit_distance": -1, "normalized_ed": -1, "precision": -1}

    edit_dist = Levenshtein.distance(output, ground_truth)
    max_len = max(len(output), len(ground_truth))
    normalized_ed = edit_dist / max_len if max_len > 0 else 0
    precision = 1 - normalized_ed

    return {
        "edit_distance": edit_dist,
        "normalized_ed": round(normalized_ed, 4),
        "precision": round(precision * 100, 2),
    }


def tokenize_text(text: str, tokenizer=None) -> int:
    """Count tokens in text using the model's tokenizer or approximation."""
    if tokenizer:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    else:
        # Approximation: ~1.3 tokens per word for subword tokenizers
        return int(len(text.split()) * 1.3)


def run_inference(image_path: str, mode: str, prompt: str = "<image>\nFree OCR."):
    """
    Run OCR inference using the DeepSeek-OCR model.

    Returns: (output_text, vision_tokens, output_tokens)
    """
    from transformers import AutoModel, AutoTokenizer
    import torch
    from PIL import Image

    model_name = "deepseek-ai/DeepSeek-OCR"
    settings = get_mode_settings(mode)

    print(f"Loading model from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)

    img = Image.open(image_path)
    width, height = img.size

    vision_tokens = calculate_valid_vision_tokens(
        width,
        height,
        settings["base_size"],
        settings["image_size"],
        settings["crop_mode"],
    )

    print(f"Running inference (mode={mode}, vision_tokens={vision_tokens})...")
    output = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path="/tmp/deepseek_ocr_output",
        base_size=settings["base_size"],
        image_size=settings["image_size"],
        crop_mode=settings["crop_mode"],
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )

    output_tokens = tokenize_text(output, tokenizer)
    return output, vision_tokens, output_tokens


def dry_run(image_path: str, mode: str, ground_truth: str = None):
    """Show calculations without running the model (no GPU needed)."""
    from PIL import Image

    settings = get_mode_settings(mode)
    img = Image.open(image_path)
    width, height = img.size

    vision_tokens = calculate_valid_vision_tokens(
        width,
        height,
        settings["base_size"],
        settings["image_size"],
        settings["crop_mode"],
    )

    print("=" * 60)
    print("DeepSeek-OCR Evaluation (Dry Run)")
    print("=" * 60)
    print("\n[INPUT]")
    print(f"  Image: {image_path}")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Mode: {mode}")
    print(
        f"  Settings: base_size={settings['base_size']}, crop_mode={settings['crop_mode']}"
    )
    print("\n[VISION TOKENS]")
    print(f"  Valid vision tokens: {vision_tokens}")

    if ground_truth:
        gt_tokens = tokenize_text(ground_truth)
        compression_ratio = gt_tokens / vision_tokens if vision_tokens > 0 else 0

        print("\n[GROUND TRUTH]")
        print(f"  Text length: {len(ground_truth)} characters")
        print(f"  Approx tokens: {gt_tokens}")
        print("\n[COMPRESSION]")
        print(
            f"  Compression ratio: {gt_tokens} / {vision_tokens} = {compression_ratio:.2f}x"
        )

        if compression_ratio < 10:
            expected = "~97% (near lossless)"
        elif compression_ratio < 12:
            expected = "~90%"
        elif compression_ratio < 15:
            expected = "~80%"
        else:
            expected = "~60% or lower"
        print(f"  Expected precision (from paper): {expected}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek-OCR compression ratio and accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no GPU needed)
  uv run python eval.py --image doc.png --dry-run

  # Full inference
  uv run python eval.py --image doc.png --mode small

  # With ground truth for accuracy calculation
  uv run python eval.py --image doc.png --ground-truth gt.txt --mode base
        """,
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--mode",
        type=str,
        default="small",
        choices=["tiny", "small", "base", "large", "gundam"],
        help="Resolution mode (default: small)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="Ground truth text or path to text file",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show calculations without running model"
    )
    parser.add_argument(
        "--prompt", type=str, default="<image>\nFree OCR.", help="Prompt for OCR"
    )
    parser.add_argument(
        "--show-output", action="store_true", help="Show OCR output text"
    )

    args = parser.parse_args()

    # Load ground truth
    ground_truth = None
    if args.ground_truth:
        if os.path.isfile(args.ground_truth):
            with open(args.ground_truth, "r", encoding="utf-8") as f:
                ground_truth = f.read()
        else:
            ground_truth = args.ground_truth

    if args.dry_run:
        dry_run(args.image, args.mode, ground_truth)
        return

    # Full inference
    print("=" * 60)
    print("DeepSeek-OCR Evaluation")
    print("=" * 60)

    output, vision_tokens, output_tokens = run_inference(
        args.image, args.mode, args.prompt
    )

    compression_ratio = output_tokens / vision_tokens if vision_tokens > 0 else 0

    print("\n[RESULTS]")
    print(f"  Vision tokens: {vision_tokens}")
    print(f"  Output tokens: {output_tokens}")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    if ground_truth:
        metrics = calculate_edit_distance(output, ground_truth)
        print("\n[ACCURACY]")
        print(f"  Edit distance: {metrics['edit_distance']} characters")
        print(f"  Normalized ED: {metrics['normalized_ed']}")
        print(f"  Precision: {metrics['precision']}%")

    if args.show_output:
        print("\n[OCR OUTPUT]")
        print("-" * 60)
        print(output[:2000] + ("..." if len(output) > 2000 else ""))


if __name__ == "__main__":
    main()
