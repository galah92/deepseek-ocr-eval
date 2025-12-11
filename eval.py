"""Evaluate DeepSeek-OCR compression ratio and accuracy on document images."""

import argparse
import hashlib
import json
import logging
import random
import re
import string
import warnings
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoTokenizer

# Suppress specific transformers warning
warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`. However, `temperature` is set to `0.0`",
    category=UserWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("deepseek_ocr.log", mode="w"),
    ],
)
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Temporary paths for experiments
TMP_OUTPUT_PATH = Path("/tmp/deepseek_ocr_output")
TMP_BLANK_IMAGE = Path("/tmp/blank_32x32.png")

# Token overhead for prompts in experiments
PROMPT_TOKEN_OVERHEAD = 100
CONTINUATION_TOKEN_OVERHEAD = 50
MODEL_CONTEXT_LIMIT = 8192  # DeepSeek-OCR context window


@dataclass(frozen=True)
class ModeSettings:
    """Settings for a resolution mode."""

    base_size: int
    image_size: int
    crop_mode: bool
    tokens: int | None  # None for dynamic modes


MODE_SETTINGS: dict[str, ModeSettings] = {
    "tiny": ModeSettings(512, 512, False, 64),
    "small": ModeSettings(640, 640, False, 100),
    "base": ModeSettings(1024, 1024, False, 256),
    "large": ModeSettings(1280, 1280, False, 400),
    "gundam": ModeSettings(1024, 640, True, None),
}

EXPERIMENT_MODES = ["tiny", "small", "base", "large"]

# Global model cache
_model: AutoModel | None = None
_tokenizer: AutoTokenizer | None = None


MONO_FONT_PATH = font_manager.findfont("monospace")


def load_model() -> tuple[AutoModel, AutoTokenizer]:
    """Load the DeepSeek-OCR model (cached after first load)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch

    model_name = "deepseek-ai/DeepSeek-OCR"
    logger.info(f"Loading model from {model_name}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        model_name,
        attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    _model = _model.eval().to(device)
    return _model, _tokenizer


# Default rendering settings (dark mode for optimal OCR - see README)
FONT_SIZE = 12
BG_COLOR = "#1e1e1e"
FG_COLOR = "#d4d4d4"
FONT = ImageFont.truetype(MONO_FONT_PATH, FONT_SIZE)

logger.info(
    f"font={FONT.getname()[0]}, size={FONT_SIZE}pt, bg={BG_COLOR}, fg={FG_COLOR}"
)


def render_text_to_image(
    text: str,
    output_path: str,
    max_width: int = 1200,
    padding: int = 30,
    line_spacing: int = 4,
) -> None:
    """Render text to an image with dark mode settings for optimal OCR.

    The function handles line wrapping based on `max_width` and `padding`.

    Args:
        text: The input text to render.
        output_path: The path where the generated image will be saved.
        max_width: The maximum width of the output image.
        padding: The padding around the text content within the image.
        line_spacing: Additional spacing between lines of text.
    """

    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = FONT.getbbox(test_line)
            if bbox[2] > max_width - 2 * padding:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))

    line_height = FONT_SIZE + line_spacing
    img_height = len(lines) * line_height + 2 * padding
    img_width = max_width

    img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=FONT, fill=FG_COLOR)
        y += line_height

    img.save(output_path)


# ============================================================================
# Augmented Rendering Functions (Experiment C: Visual Metadata Injection)
# ============================================================================

# Semantic highlighting colors (dark mode friendly)
HIGHLIGHT_COLORS = {
    "entity": "#4FC3F7",  # Light blue for named entities
    "number": "#81C784",  # Green for numbers/dates
    "keyword": "#FFB74D",  # Orange for keywords
    "quote": "#BA68C8",  # Purple for quoted text
}


def identify_highlights(text: str) -> list[tuple[int, int, str]]:
    """Identify spans to highlight with their types.

    Returns list of (start, end, highlight_type) tuples.
    """
    highlights = []

    # Numbers and dates (years, quantities, percentages)
    for match in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b|\b\d{4}\b", text):
        highlights.append((match.start(), match.end(), "number"))

    # Quoted text
    for match in re.finditer(r'"[^"]+"|\'[^\']+\'', text):
        highlights.append((match.start(), match.end(), "quote"))

    # Capitalized words (likely proper nouns/entities) - not at sentence start
    for match in re.finditer(
        r"(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text
    ):
        # Filter out common words
        word = match.group()
        if word.lower() not in {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
            "they",
            "them",
            "their",
            "he",
            "she",
            "him",
            "her",
            "his",
            "hers",
            "we",
            "us",
            "our",
            "you",
            "your",
            "i",
            "me",
            "my",
        }:
            highlights.append((match.start(), match.end(), "entity"))

    return sorted(highlights, key=lambda x: x[0])


def render_augmented_text_to_image(
    text: str,
    output_path: str,
    max_width: int = 1200,
    padding: int = 30,
    line_spacing: int = 4,
) -> None:
    """Render text with semantic highlighting to an image.

    Highlights:
    - Named entities (capitalized words) in blue
    - Numbers and dates in green
    - Quoted text in purple

    Uses same dark mode settings as render_text_to_image for consistency.
    """
    # Get highlights for the full text
    highlights = identify_highlights(text)

    # Create highlight lookup: char_index -> color
    char_colors = {}
    for start, end, htype in highlights:
        color = HIGHLIGHT_COLORS.get(htype, FG_COLOR)
        for i in range(start, end):
            char_colors[i] = color

    # Word wrap while tracking original positions
    lines_with_positions = []  # List of [(word, start_idx, end_idx), ...]
    current_pos = 0

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines_with_positions.append([])
            current_pos += 1  # for newline
            continue

        words = []
        for match in re.finditer(r"\S+", paragraph):
            word_start = current_pos + match.start()
            word_end = current_pos + match.end()
            words.append((match.group(), word_start, word_end))

        # Line wrapping
        current_line = []
        for word_info in words:
            word, start, end = word_info
            test_line = " ".join(w[0] for w in current_line + [word_info])
            bbox = FONT.getbbox(test_line)
            if bbox[2] > max_width - 2 * padding:
                if current_line:
                    lines_with_positions.append(current_line)
                current_line = [word_info]
            else:
                current_line.append(word_info)
        if current_line:
            lines_with_positions.append(current_line)

        current_pos += len(paragraph) + 1  # +1 for newline

    line_height = FONT_SIZE + line_spacing
    img_height = len(lines_with_positions) * line_height + 2 * padding
    img_width = max_width

    img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    y = padding
    for line_words in lines_with_positions:
        if not line_words:  # Empty line
            y += line_height
            continue

        x = padding
        for word, start, end in line_words:
            # Determine color for this word (use first char's highlight)
            color = char_colors.get(start, FG_COLOR)
            draw.text((x, y), word, font=FONT, fill=color)
            bbox = FONT.getbbox(word + " ")
            x += bbox[2]

        y += line_height

    img.save(output_path)


def calculate_valid_vision_tokens(
    width: int, height: int, settings: ModeSettings
) -> int:
    """Calculate valid vision tokens based on image dimensions and mode settings."""
    long_side = max(width, height)
    short_side = min(width, height)
    ratio = 1 - ((long_side - short_side) / long_side)

    if settings.base_size == 1280:
        valid_tokens = int(400 * ratio)
    elif settings.base_size == 1024:
        valid_tokens = int(256 * ratio)
    elif settings.base_size == 640:
        valid_tokens = 100
    elif settings.base_size == 512:
        valid_tokens = 64
    else:
        valid_tokens = 0

    if settings.crop_mode and (width > 640 or height > 640):
        num_crops = calculate_num_crops(width, height, settings.image_size)
        crop_tokens = 256 if settings.image_size == 1024 else 100
        valid_tokens += num_crops * crop_tokens

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
    target_ratios = (
        (i, j)
        for n in range(min_crops, max_crops + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_crops <= i * j <= max_crops
    )

    best_ratio = min(target_ratios, key=lambda r: abs(aspect_ratio - (r[0] / r[1])))
    return best_ratio[0] * best_ratio[1]


def tokenize_text(text: str, tokenizer: AutoTokenizer | None = None) -> int:
    """Count tokens in text using the model's tokenizer or approximation."""
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return int(len(text.split()) * 1.3)


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


# ============================================================================
# Noise Injection Functions (Experiment A: Robustness Boundary)
# ============================================================================

# Common OCR confusion pairs (visually similar characters)
OCR_CONFUSIONS = {
    "O": "0",
    "0": "O",
    "l": "1",
    "1": "l",
    "I": "l",
    "S": "5",
    "5": "S",
    "B": "8",
    "8": "B",
    "g": "9",
    "9": "g",
    "Z": "2",
    "2": "Z",
    "rn": "m",
    "m": "rn",
    "cl": "d",
    "d": "cl",
    "vv": "w",
    "w": "vv",
}

# Keyboard proximity for typo simulation (QWERTY layout)
KEYBOARD_NEIGHBORS = {
    "a": "sqwz",
    "b": "vghn",
    "c": "xdfv",
    "d": "sfecx",
    "e": "wrsdf",
    "f": "dgrtcv",
    "g": "fhtybn",
    "h": "gjuynm",
    "i": "uojk",
    "j": "hkunim",
    "k": "jlomi",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iplk",
    "p": "ol",
    "q": "wa",
    "r": "etdf",
    "s": "awedxz",
    "t": "rfyg",
    "u": "yihj",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tugh",
    "z": "asx",
}


def inject_typos(text: str, rate: float, seed: int = 42) -> str:
    """Inject typos by substituting characters with keyboard neighbors.

    Args:
        text: Input text to corrupt
        rate: Fraction of characters to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Corrupted text with simulated typos
    """
    random.seed(seed)
    chars = list(text)
    num_to_corrupt = int(len(chars) * rate)

    # Get indices of alphabetic characters (don't corrupt spaces/punctuation)
    alpha_indices = [i for i, c in enumerate(chars) if c.isalpha()]
    if not alpha_indices:
        return text

    # Randomly select indices to corrupt
    indices_to_corrupt = random.sample(
        alpha_indices, min(num_to_corrupt, len(alpha_indices))
    )

    for idx in indices_to_corrupt:
        char = chars[idx].lower()
        if char in KEYBOARD_NEIGHBORS:
            neighbors = KEYBOARD_NEIGHBORS[char]
            replacement = random.choice(neighbors)
            # Preserve case
            if chars[idx].isupper():
                replacement = replacement.upper()
            chars[idx] = replacement

    return "".join(chars)


def inject_ocr_errors(text: str, rate: float, seed: int = 42) -> str:
    """Inject OCR-style errors using visually similar character substitutions.

    Args:
        text: Input text to corrupt
        rate: Fraction of eligible characters to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Corrupted text with simulated OCR errors
    """
    random.seed(seed)

    # First handle multi-character confusions
    result = text
    for original, replacement in [("rn", "m"), ("cl", "d"), ("vv", "w")]:
        if random.random() < rate:
            result = result.replace(original, replacement)

    # Then handle single-character confusions
    chars = list(result)
    single_confusions = {k: v for k, v in OCR_CONFUSIONS.items() if len(k) == 1}

    eligible_indices = [i for i, c in enumerate(chars) if c in single_confusions]

    num_to_corrupt = int(len(eligible_indices) * rate)
    indices_to_corrupt = (
        random.sample(eligible_indices, min(num_to_corrupt, len(eligible_indices)))
        if eligible_indices
        else []
    )

    for idx in indices_to_corrupt:
        chars[idx] = single_confusions[chars[idx]]

    return "".join(chars)


def inject_deletions(text: str, rate: float, seed: int = 42) -> str:
    """Inject errors by randomly deleting characters.

    Args:
        text: Input text to corrupt
        rate: Fraction of characters to delete (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Corrupted text with random deletions
    """
    random.seed(seed)
    chars = list(text)
    num_to_delete = int(len(chars) * rate)

    # Don't delete spaces to preserve word boundaries
    non_space_indices = [i for i, c in enumerate(chars) if c != " "]
    if not non_space_indices:
        return text

    indices_to_delete = set(
        random.sample(non_space_indices, min(num_to_delete, len(non_space_indices)))
    )

    return "".join(c for i, c in enumerate(chars) if i not in indices_to_delete)


def inject_insertions(text: str, rate: float, seed: int = 42) -> str:
    """Inject errors by randomly inserting characters.

    Args:
        text: Input text to corrupt
        rate: Fraction of positions to insert at (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Corrupted text with random insertions
    """
    random.seed(seed)
    chars = list(text)
    num_to_insert = int(len(chars) * rate)

    # Insert random lowercase letters at random positions
    for _ in range(num_to_insert):
        pos = random.randint(0, len(chars))
        char = random.choice(string.ascii_lowercase)
        chars.insert(pos, char)

    return "".join(chars)


def inject_noise(text: str, noise_type: str, rate: float, seed: int = 42) -> str:
    """Apply noise injection to text.

    Args:
        text: Input text to corrupt
        noise_type: One of 'typos', 'ocr', 'deletions', 'insertions', 'mixed'
        rate: Corruption rate (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Corrupted text
    """
    if rate <= 0:
        return text

    if noise_type == "typos":
        return inject_typos(text, rate, seed)
    elif noise_type == "ocr":
        return inject_ocr_errors(text, rate, seed)
    elif noise_type == "deletions":
        return inject_deletions(text, rate, seed)
    elif noise_type == "insertions":
        return inject_insertions(text, rate, seed)
    elif noise_type == "mixed":
        # Apply a mix of all noise types at reduced rates
        result = text
        sub_rate = rate / 4
        result = inject_typos(result, sub_rate, seed)
        result = inject_ocr_errors(result, sub_rate, seed + 1)
        result = inject_deletions(result, sub_rate, seed + 2)
        result = inject_insertions(result, sub_rate, seed + 3)
        return result
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# ============================================================================
# Table Rendering Functions (Experiment B: Structured Data)
# ============================================================================


def table_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Convert table to markdown format.

    Example output:
    | Col1 | Col2 | Col3 |
    |------|------|------|
    | a    | b    | c    |
    """
    # Calculate column widths for alignment
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build markdown table
    lines = []
    # Header
    header_line = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |"
    )
    lines.append(header_line)
    # Separator
    sep_line = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines.append(sep_line)
    # Rows
    for row in rows:
        row_cells = []
        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else len(str(cell))
            row_cells.append(str(cell).ljust(width))
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def table_to_linearized(header: list[str], rows: list[list[str]]) -> str:
    """Convert table to linearized text format (row-by-row).

    Example output:
    Row 1: Col1=a, Col2=b, Col3=c
    Row 2: Col1=d, Col2=e, Col3=f
    """
    lines = []
    for i, row in enumerate(rows):
        cells = []
        for j, cell in enumerate(row):
            col_name = header[j] if j < len(header) else f"Col{j + 1}"
            cells.append(f"{col_name}={cell}")
        lines.append(f"Row {i + 1}: " + ", ".join(cells))
    return "\n".join(lines)


def render_table_to_image(
    header: list[str],
    rows: list[list[str]],
    output_path: str,
    max_width: int = 1400,
    padding: int = 20,
    cell_padding: int = 10,
) -> None:
    """Render table to an image with grid lines.

    Uses dark mode settings (same as text rendering) for consistency.
    """
    # Calculate column widths based on content
    col_widths = []
    for i, h in enumerate(header):
        max_cell_width = FONT.getbbox(h)[2]
        for row in rows:
            if i < len(row):
                cell_width = FONT.getbbox(str(row[i]))[2]
                max_cell_width = max(max_cell_width, cell_width)
        col_widths.append(max_cell_width + 2 * cell_padding)

    # Ensure table fits in max_width
    total_width = sum(col_widths) + 2 * padding
    if total_width > max_width:
        scale = (max_width - 2 * padding) / sum(col_widths)
        col_widths = [int(w * scale) for w in col_widths]
        total_width = sum(col_widths) + 2 * padding

    row_height = FONT_SIZE + 2 * cell_padding
    total_height = (len(rows) + 1) * row_height + 2 * padding  # +1 for header

    img = Image.new("RGB", (total_width, total_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw header
    x = padding
    y = padding
    for i, h in enumerate(header):
        # Draw cell background (slightly lighter for header)
        draw.rectangle([x, y, x + col_widths[i], y + row_height], outline="#555555")
        # Draw text centered in cell
        text_bbox = FONT.getbbox(h)
        text_x = x + (col_widths[i] - text_bbox[2]) // 2
        text_y = y + cell_padding
        draw.text((text_x, text_y), h, font=FONT, fill="#ffffff")  # White for header
        x += col_widths[i]

    # Draw rows
    for row_idx, row in enumerate(rows):
        y = padding + (row_idx + 1) * row_height
        x = padding
        for i, cell in enumerate(row):
            if i >= len(col_widths):
                break
            # Draw cell border
            draw.rectangle([x, y, x + col_widths[i], y + row_height], outline="#555555")
            # Draw text
            text_bbox = FONT.getbbox(str(cell))
            text_x = x + cell_padding
            text_y = y + cell_padding
            draw.text((text_x, text_y), str(cell), font=FONT, fill=FG_COLOR)
            x += col_widths[i]

    img.save(output_path)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip, handle common variations)."""
    s = answer.lower().strip()
    # Remove common punctuation
    s = s.rstrip(".")
    # Handle numeric variations
    s = s.replace(",", "")
    return s


def answers_match(predicted: str, gold_answers: list[str]) -> bool:
    """Check if predicted answer matches any gold answer."""
    pred_norm = normalize_answer(predicted)

    # Handle boolean answers specially
    boolean_true = {"true", "yes", "1", "correct", "right"}
    boolean_false = {"false", "no", "0", "incorrect", "wrong", "none"}

    for gold in gold_answers:
        gold_norm = normalize_answer(gold)

        # Exact match
        if gold_norm == pred_norm:
            return True

        # Boolean matching
        if gold_norm in boolean_true and pred_norm in boolean_true:
            return True
        if gold_norm in boolean_false and pred_norm in boolean_false:
            return True

        # Containment check
        if gold_norm in pred_norm or pred_norm in gold_norm:
            return True

    return False


def run_inference(
    prompt: str,
    image_path: str | Path,
    mode: str = "text",
    model: AutoModel | None = None,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[str, int, int]:
    """Run inference using the DeepSeek-OCR model.

    Args:
        prompt: The input prompt.
        image_path: Path to the image file.
        mode: Resolution mode key or 'text' for text-only (blank image).
        model: Pre-loaded model (optional).
        tokenizer: Pre-loaded tokenizer (optional).

    Returns:
        Tuple of (output_text, vision_tokens_count, output_tokens_count).
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model()

    if mode == "text":
        base_size = 512
        image_size = 512
        crop_mode = False
        vision_tokens = 0
        final_image_path = _ensure_blank_image()
    else:
        settings = MODE_SETTINGS[mode]
        base_size = settings.base_size
        image_size = settings.image_size
        crop_mode = settings.crop_mode
        final_image_path = str(image_path)

        # Calculate vision tokens
        try:
            with Image.open(final_image_path) as img:
                width, height = img.size
                vision_tokens = calculate_valid_vision_tokens(width, height, settings)
        except Exception as e:
            logger.error(f"Error calculating vision tokens for {final_image_path}: {e}")
            vision_tokens = 0

    if mode != "text":
        logger.info(
            f"Running inference (mode={mode}, vision_tokens={vision_tokens})..."
        )

    output = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=final_image_path,
        output_path=str(TMP_OUTPUT_PATH),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )

    output_tokens = tokenize_text(output, tokenizer)

    # Clear CUDA cache to prevent OOM in long-running experiments
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output, vision_tokens, output_tokens


def cmd_ocr(args: argparse.Namespace) -> None:
    """Run OCR evaluation on an image, optionally comparing to ground truth."""
    ground_truth = None
    if args.ground_truth:
        if Path(args.ground_truth).is_file():
            with open(args.ground_truth, "r", encoding="utf-8") as f:
                ground_truth = f.read()
        else:
            ground_truth = args.ground_truth

    if args.dry_run:
        settings = MODE_SETTINGS[args.mode]
        img = Image.open(args.image)
        width, height = img.size
        vision_tokens = calculate_valid_vision_tokens(width, height, settings)

        logger.info("DeepSeek-OCR Evaluation (Dry Run)")
        logger.info(f"[INPUT] Image: {args.image}, Dimensions: {width} x {height}")
        logger.info(f"[VISION TOKENS] Valid vision tokens: {vision_tokens}")

        if ground_truth:
            gt_tokens = tokenize_text(ground_truth)
            compression = gt_tokens / vision_tokens if vision_tokens > 0 else 0
            logger.info(f"[GROUND TRUTH] Text length: {len(ground_truth)} characters")
            logger.info(f"  Approx tokens: {gt_tokens}")
            logger.info(f"[COMPRESSION] Compression ratio: {compression:.2f}x")
        return

    logger.info("DeepSeek-OCR Evaluation")

    output, vision_tokens, output_tokens = run_inference(
        prompt=args.prompt,
        image_path=args.image,
        mode=args.mode,
    )
    compression = output_tokens / vision_tokens if vision_tokens > 0 else 0

    logger.info(f"[RESULTS] Vision tokens: {vision_tokens}")
    logger.info(
        f"  Output tokens: {output_tokens}, Compression ratio: {compression:.2f}x"
    )

    if ground_truth:
        metrics = calculate_edit_distance(output, ground_truth)
        logger.info(f"[ACCURACY] Edit distance: {metrics['edit_distance']} characters")
        logger.info(
            f"  Normalized ED: {metrics['normalized_ed']}, Precision: {metrics['precision']}%"
        )

    if args.show_output:
        logger.info("[OCR OUTPUT]")
        logger.info(output[:2000] + ("..." if len(output) > 2000 else ""))


def _ensure_blank_image() -> str:
    """Ensure a blank image exists for text-only inference and return its path."""
    if not TMP_BLANK_IMAGE.exists():
        Image.new("RGB", (32, 32), color="white").save(TMP_BLANK_IMAGE)
    return str(TMP_BLANK_IMAGE)


def save_experiment_results(
    results: dict, results_dir: Path, output_filename: str
) -> None:
    """Save experiment results to a JSON file."""
    output_path = results_dir / output_filename
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved to: {output_path}")


def setup_experiment_dirs(experiment_name: str) -> tuple[Path, Path]:
    """Sets up and returns data_dir and results_dir for an experiment."""
    root_dir = Path(__file__).parent
    data_dir = root_dir / ".cache" / experiment_name
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)
    return data_dir, results_dir


def cmd_quality(args: argparse.Namespace) -> None:
    """Run QuALITY long-document QA experiment comparing text vs vision accuracy."""
    data_dir, results_dir = setup_experiment_dirs("quality")

    logger.info("Loading QuALITY dataset...")
    try:
        ds = load_dataset("emozilla/quality", split="validation")
    except Exception as e:
        logger.error(f"Error loading QuALITY dataset: {e}")
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

    logger.info(f"Found {len(articles)} unique articles")
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

    logger.info(f"QUALITY EXPERIMENT: Mode={args.mode}, Articles={args.num_articles}")

    for article_hash, article_data in article_items:
        article = article_data["article"]
        questions = article_data["questions"][: args.questions_per_article]

        logger.info(f"Article: {article_hash} ({len(article.split())} words)")

        img_path = data_dir / f"{article_hash}.png"
        if not img_path.exists():
            render_text_to_image(article, str(img_path))

        # Calculate article token count once
        article_tokens = len(tokenizer.encode(article, add_special_tokens=False))
        exceeds_context = article_tokens > MODEL_CONTEXT_LIMIT

        logger.info(
            f"  Article tokens: {article_tokens}, Exceeds {MODEL_CONTEXT_LIMIT} limit: {exceeds_context}"
        )

        article_results = {
            "questions": [],
            "text_correct": 0,
            "vision_correct": 0,
            "article_tokens": article_tokens,
            "exceeds_context_limit": exceeds_context,
        }

        for qa in questions:
            question, options, expected = qa["question"], qa["options"], qa["answer"]
            options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))

            # Text condition
            text_prompt = f"<image>\n{article}\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer with just the option number (0, 1, 2, or 3):"
            text_output, _, _ = run_inference(
                text_prompt,
                "",  # No image for text mode
                mode="text",
                model=model,
                tokenizer=tokenizer,
            )
            text_tokens = (
                len(tokenizer.encode(article, add_special_tokens=False))
                + PROMPT_TOKEN_OVERHEAD
            )

            # Vision condition
            vision_prompt = f"<image>\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer with just the option number (0, 1, 2, or 3):"
            vision_output, _, _ = run_inference(
                vision_prompt,
                str(img_path),
                mode=args.mode,
                model=model,
                tokenizer=tokenizer,
            )
            vision_tokens = settings.tokens + PROMPT_TOKEN_OVERHEAD

            # Parse answers
            text_pred = next((int(c) for c in text_output.strip() if c in "0123"), -1)
            vision_pred = next(
                (int(c) for c in vision_output.strip() if c in "0123"), -1
            )

            text_correct = text_pred == expected
            vision_correct = vision_pred == expected

            logger.info(f"  Q: {question[:60]}...")
            logger.info(
                f"    Text: {text_pred} {'✓' if text_correct else '✗'}, Vision: {vision_pred} {'✓' if vision_correct else '✗'}"
            )

            # Store per-question results
            article_results["questions"].append(
                {
                    "question": question[:100],
                    "expected": expected,
                    "text_pred": text_pred,
                    "vision_pred": vision_pred,
                    "text_correct": text_correct,
                    "vision_correct": vision_correct,
                    "vision_beat_text": vision_correct and not text_correct,
                    "text_beat_vision": text_correct and not vision_correct,
                }
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

    # Analyze overflow hypothesis
    overflow_stats = {"vision_beat_text": 0, "text_beat_vision": 0, "total": 0}
    no_overflow_stats = {"vision_beat_text": 0, "text_beat_vision": 0, "total": 0}

    for article_hash, article_data in results["articles"].items():
        target = (
            overflow_stats
            if article_data["exceeds_context_limit"]
            else no_overflow_stats
        )
        for q in article_data["questions"]:
            target["total"] += 1
            if q["vision_beat_text"]:
                target["vision_beat_text"] += 1
            if q["text_beat_vision"]:
                target["text_beat_vision"] += 1

    results["summary"] = {
        "total_questions": n,
        "text_accuracy": text_acc,
        "vision_accuracy": vision_acc,
        "compression_ratio": round(compression, 2),
    }

    # Add overflow analysis to summary
    results["overflow_analysis"] = {
        "articles_exceeding_limit": sum(
            1 for a in results["articles"].values() if a["exceeds_context_limit"]
        ),
        "articles_within_limit": sum(
            1 for a in results["articles"].values() if not a["exceeds_context_limit"]
        ),
        "context_limit": MODEL_CONTEXT_LIMIT,
        "overflow_questions": overflow_stats,
        "no_overflow_questions": no_overflow_stats,
    }

    logger.info(
        f"RESULTS: Text={text_acc}%, Vision={vision_acc}%, Compression={compression:.1f}x"
    )

    # Log overflow analysis
    logger.info("OVERFLOW ANALYSIS:")
    logger.info(
        f"  Articles exceeding {MODEL_CONTEXT_LIMIT} tokens: {results['overflow_analysis']['articles_exceeding_limit']}"
    )
    logger.info(
        f"  Articles within limit: {results['overflow_analysis']['articles_within_limit']}"
    )
    if overflow_stats["total"] > 0:
        logger.info(
            f"  Overflow questions - Vision beat text: {overflow_stats['vision_beat_text']}/{overflow_stats['total']}, Text beat vision: {overflow_stats['text_beat_vision']}/{overflow_stats['total']}"
        )
    if no_overflow_stats["total"] > 0:
        logger.info(
            f"  No-overflow questions - Vision beat text: {no_overflow_stats['vision_beat_text']}/{no_overflow_stats['total']}, Text beat vision: {no_overflow_stats['text_beat_vision']}/{no_overflow_stats['total']}"
        )

    save_experiment_results(
        results, results_dir, f"quality_{args.mode}_{args.num_articles}articles.json"
    )


def truncate_text(
    text: str, tokenizer: AutoTokenizer, max_tokens: int, from_end: bool = False
) -> str:
    """Truncate text to max_tokens, either from beginning or end.

    Args:
        text: The text to truncate.
        tokenizer: The tokenizer to use for token counting.
        max_tokens: Maximum number of tokens to keep.
        from_end: If True, keep last max_tokens; if False, keep first max_tokens.

    Returns:
        Truncated text decoded back to string.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text

    if from_end:
        truncated_tokens = tokens[-max_tokens:]
    else:
        truncated_tokens = tokens[:max_tokens]

    return tokenizer.decode(truncated_tokens)


def cmd_truncation(args: argparse.Namespace) -> None:
    """Run truncation baseline experiment comparing vision vs truncated text on QuALITY.

    Tests four conditions:
    1. Full text (if fits in context)
    2. Truncated text - first N tokens (beginning of article)
    3. Truncated text - last N tokens (end of article)
    4. Vision (rendered full article, compressed)

    This addresses Lee et al.'s critique by testing whether vision beats truncation
    on QA tasks where full-document coverage matters.
    """
    data_dir, results_dir = setup_experiment_dirs("truncation")

    logger.info("Loading QuALITY dataset...")
    try:
        ds = load_dataset("emozilla/quality", split="validation")
    except Exception as e:
        logger.error(f"Error loading QuALITY dataset: {e}")
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

    logger.info(f"Found {len(articles)} unique articles")
    article_items = list(articles.items())[: args.num_articles]

    model, tokenizer = load_model()
    settings = MODE_SETTINGS[args.mode]

    # Token budget for truncation = vision tokens for this mode
    token_budget = settings.tokens
    logger.info(
        f"Token budget for truncation: {token_budget} (matching {args.mode} mode vision tokens)"
    )

    results = {
        "mode": args.mode,
        "token_budget": token_budget,
        "articles": {},
        "summary": {},
    }
    stats = {
        "full_text_correct": 0,
        "trunc_first_correct": 0,
        "trunc_last_correct": 0,
        "vision_correct": 0,
        "total": 0,
    }

    logger.info(
        f"TRUNCATION EXPERIMENT: Mode={args.mode}, Articles={args.num_articles}, Token Budget={token_budget}"
    )

    for article_hash, article_data in article_items:
        article = article_data["article"]
        questions = article_data["questions"][: args.questions_per_article]

        logger.info(f"\nArticle: {article_hash} ({len(article.split())} words)")

        # Render image for vision condition
        img_path = data_dir / f"{article_hash}.png"
        if not img_path.exists():
            render_text_to_image(article, str(img_path))

        # Calculate article token count
        article_tokens = len(tokenizer.encode(article, add_special_tokens=False))

        # Prepare truncated versions
        trunc_first = truncate_text(article, tokenizer, token_budget, from_end=False)
        trunc_last = truncate_text(article, tokenizer, token_budget, from_end=True)

        trunc_first_tokens = len(
            tokenizer.encode(trunc_first, add_special_tokens=False)
        )
        trunc_last_tokens = len(tokenizer.encode(trunc_last, add_special_tokens=False))

        logger.info(f"  Article tokens: {article_tokens}")
        logger.info(f"  Truncated (first): {trunc_first_tokens} tokens")
        logger.info(f"  Truncated (last): {trunc_last_tokens} tokens")
        logger.info(f"  Vision: {settings.tokens} tokens")

        article_results = {
            "questions": [],
            "article_tokens": article_tokens,
            "full_text_correct": 0,
            "trunc_first_correct": 0,
            "trunc_last_correct": 0,
            "vision_correct": 0,
        }

        for qa in questions:
            question, options, expected = qa["question"], qa["options"], qa["answer"]
            options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))

            question_suffix = f"\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\nAnswer with just the option number (0, 1, 2, or 3):"

            # Condition 1: Full text
            full_prompt = f"<image>\n{article}{question_suffix}"
            full_output, _, _ = run_inference(
                full_prompt, "", mode="text", model=model, tokenizer=tokenizer
            )
            full_pred = next((int(c) for c in full_output.strip() if c in "0123"), -1)
            full_correct = full_pred == expected

            # Condition 2: Truncated first N tokens
            trunc_first_prompt = f"<image>\n{trunc_first}{question_suffix}"
            trunc_first_output, _, _ = run_inference(
                trunc_first_prompt, "", mode="text", model=model, tokenizer=tokenizer
            )
            trunc_first_pred = next(
                (int(c) for c in trunc_first_output.strip() if c in "0123"), -1
            )
            trunc_first_correct = trunc_first_pred == expected

            # Condition 3: Truncated last N tokens
            trunc_last_prompt = f"<image>\n{trunc_last}{question_suffix}"
            trunc_last_output, _, _ = run_inference(
                trunc_last_prompt, "", mode="text", model=model, tokenizer=tokenizer
            )
            trunc_last_pred = next(
                (int(c) for c in trunc_last_output.strip() if c in "0123"), -1
            )
            trunc_last_correct = trunc_last_pred == expected

            # Condition 4: Vision (full article rendered)
            vision_prompt = f"<image>{question_suffix}"
            vision_output, _, _ = run_inference(
                vision_prompt,
                str(img_path),
                mode=args.mode,
                model=model,
                tokenizer=tokenizer,
            )
            vision_pred = next(
                (int(c) for c in vision_output.strip() if c in "0123"), -1
            )
            vision_correct = vision_pred == expected

            logger.info(f"  Q: {question[:50]}...")
            logger.info(
                f"    Full: {full_pred} {'✓' if full_correct else '✗'} | "
                f"First-{token_budget}: {trunc_first_pred} {'✓' if trunc_first_correct else '✗'} | "
                f"Last-{token_budget}: {trunc_last_pred} {'✓' if trunc_last_correct else '✗'} | "
                f"Vision: {vision_pred} {'✓' if vision_correct else '✗'}"
            )

            # Store results
            article_results["questions"].append(
                {
                    "question": question[:100],
                    "expected": expected,
                    "full_pred": full_pred,
                    "trunc_first_pred": trunc_first_pred,
                    "trunc_last_pred": trunc_last_pred,
                    "vision_pred": vision_pred,
                    "full_correct": full_correct,
                    "trunc_first_correct": trunc_first_correct,
                    "trunc_last_correct": trunc_last_correct,
                    "vision_correct": vision_correct,
                }
            )

            # Update counts
            if full_correct:
                article_results["full_text_correct"] += 1
                stats["full_text_correct"] += 1
            if trunc_first_correct:
                article_results["trunc_first_correct"] += 1
                stats["trunc_first_correct"] += 1
            if trunc_last_correct:
                article_results["trunc_last_correct"] += 1
                stats["trunc_last_correct"] += 1
            if vision_correct:
                article_results["vision_correct"] += 1
                stats["vision_correct"] += 1
            stats["total"] += 1

        results["articles"][article_hash] = article_results

    # Summary
    n = stats["total"]
    if n > 0:
        full_acc = round(stats["full_text_correct"] / n * 100, 1)
        trunc_first_acc = round(stats["trunc_first_correct"] / n * 100, 1)
        trunc_last_acc = round(stats["trunc_last_correct"] / n * 100, 1)
        vision_acc = round(stats["vision_correct"] / n * 100, 1)

        results["summary"] = {
            "total_questions": n,
            "token_budget": token_budget,
            "full_text_accuracy": full_acc,
            "trunc_first_accuracy": trunc_first_acc,
            "trunc_last_accuracy": trunc_last_acc,
            "vision_accuracy": vision_acc,
            "vision_beats_trunc_first": vision_acc > trunc_first_acc,
            "vision_beats_trunc_last": vision_acc > trunc_last_acc,
            "vision_beats_both_truncations": vision_acc
            > max(trunc_first_acc, trunc_last_acc),
        }

        logger.info("\n" + "=" * 60)
        logger.info("TRUNCATION EXPERIMENT RESULTS")
        logger.info("=" * 60)
        logger.info(f"Token budget: {token_budget} (matching {args.mode} mode)")
        logger.info(f"Total questions: {n}")
        logger.info("")
        logger.info(f"{'Condition':<20} | {'Accuracy':<10} | {'Correct':<10}")
        logger.info("-" * 45)
        logger.info(
            f"{'Full text':<20} | {full_acc:>8}% | {stats['full_text_correct']}/{n}"
        )
        logger.info(
            f"{'Trunc (first N)':<20} | {trunc_first_acc:>8}% | {stats['trunc_first_correct']}/{n}"
        )
        logger.info(
            f"{'Trunc (last N)':<20} | {trunc_last_acc:>8}% | {stats['trunc_last_correct']}/{n}"
        )
        logger.info(
            f"{'Vision':<20} | {vision_acc:>8}% | {stats['vision_correct']}/{n}"
        )
        logger.info("")
        logger.info(f"Vision beats truncation (first): {vision_acc > trunc_first_acc}")
        logger.info(f"Vision beats truncation (last): {vision_acc > trunc_last_acc}")

        save_experiment_results(
            results,
            results_dir,
            f"truncation_{args.mode}_{args.num_articles}articles.json",
        )


def cmd_noise(args: argparse.Namespace) -> None:
    """Run noise injection experiment (Experiment A) comparing text vs vision robustness.

    Tests whether vision encoders degrade more gracefully than text tokenizers
    when input text contains noise (typos, OCR errors, etc.).
    """
    data_dir, results_dir = setup_experiment_dirs("noise")

    logger.info("Loading QuALITY dataset...")
    try:
        ds = load_dataset("emozilla/quality", split="validation")
    except Exception as e:
        logger.error(f"Error loading QuALITY dataset: {e}")
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

    logger.info(f"Found {len(articles)} unique articles")
    article_items = list(articles.items())[: args.num_articles]

    model, tokenizer = load_model()

    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    results = {
        "mode": args.mode,
        "noise_type": args.noise_type,
        "noise_levels": noise_levels,
        "articles": {},
        "summary": {},
    }

    # Track accuracy at each noise level
    level_stats = {
        level: {"text_correct": 0, "vision_correct": 0, "total": 0}
        for level in noise_levels
    }

    logger.info(f"NOISE EXPERIMENT: Mode={args.mode}, Type={args.noise_type}")
    logger.info(f"Noise levels: {noise_levels}")
    logger.info(
        f"Articles: {args.num_articles}, Questions/article: {args.questions_per_article}"
    )

    for article_hash, article_data in article_items:
        article = article_data["article"]
        questions = article_data["questions"][: args.questions_per_article]

        logger.info(f"\nArticle: {article_hash} ({len(article.split())} words)")

        article_results = {
            "questions": [],
            "noise_levels": {},
        }

        for qa in questions:
            question, options, expected = qa["question"], qa["options"], qa["answer"]
            options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))

            question_results = {
                "question": question[:100],
                "expected": expected,
                "levels": {},
            }

            for level in noise_levels:
                # Apply noise to article text
                noisy_article = inject_noise(
                    article,
                    args.noise_type,
                    level,
                    seed=42 + int(level * 1000),  # Different seed per level
                )

                # Also apply noise to question if specified
                if args.noise_question:
                    noisy_question = inject_noise(
                        question, args.noise_type, level, seed=43
                    )
                    noisy_options = inject_noise(
                        options_text, args.noise_type, level, seed=44
                    )
                else:
                    noisy_question = question
                    noisy_options = options_text

                # Render noisy article to image
                img_path = data_dir / f"{article_hash}_noise{int(level * 100)}.png"
                if not img_path.exists():
                    render_text_to_image(noisy_article, str(img_path))

                # Text condition (noisy text directly)
                text_prompt = f"<image>\n{noisy_article}\n\nQuestion: {noisy_question}\n\nOptions:\n{noisy_options}\n\nAnswer with just the option number (0, 1, 2, or 3):"
                text_output, _, _ = run_inference(
                    text_prompt,
                    "",  # No image for text mode
                    mode="text",
                    model=model,
                    tokenizer=tokenizer,
                )

                # Vision condition (noisy text rendered to image)
                vision_prompt = f"<image>\n\nQuestion: {noisy_question}\n\nOptions:\n{noisy_options}\n\nAnswer with just the option number (0, 1, 2, or 3):"
                vision_output, _, _ = run_inference(
                    vision_prompt,
                    str(img_path),
                    mode=args.mode,
                    model=model,
                    tokenizer=tokenizer,
                )

                # Parse answers
                text_pred = next(
                    (int(c) for c in text_output.strip() if c in "0123"), -1
                )
                vision_pred = next(
                    (int(c) for c in vision_output.strip() if c in "0123"), -1
                )

                text_correct = text_pred == expected
                vision_correct = vision_pred == expected

                level_str = f"{int(level * 100)}%"
                logger.info(
                    f"  [{level_str:>3}] Q: {question[:40]}... | "
                    f"Text: {text_pred} {'✓' if text_correct else '✗'} | "
                    f"Vision: {vision_pred} {'✓' if vision_correct else '✗'}"
                )

                question_results["levels"][level] = {
                    "text_pred": text_pred,
                    "vision_pred": vision_pred,
                    "text_correct": text_correct,
                    "vision_correct": vision_correct,
                }

                level_stats[level]["total"] += 1
                if text_correct:
                    level_stats[level]["text_correct"] += 1
                if vision_correct:
                    level_stats[level]["vision_correct"] += 1

            article_results["questions"].append(question_results)

        results["articles"][article_hash] = article_results

    # Calculate summary statistics
    summary_by_level = {}
    for level in noise_levels:
        n = level_stats[level]["total"]
        if n > 0:
            text_acc = round(level_stats[level]["text_correct"] / n * 100, 1)
            vision_acc = round(level_stats[level]["vision_correct"] / n * 100, 1)
            summary_by_level[level] = {
                "total": n,
                "text_accuracy": text_acc,
                "vision_accuracy": vision_acc,
                "vision_advantage": round(vision_acc - text_acc, 1),
            }

    results["summary"] = {
        "by_level": summary_by_level,
        "noise_type": args.noise_type,
    }

    # Log results table
    logger.info("\n" + "=" * 70)
    logger.info("NOISE EXPERIMENT RESULTS")
    logger.info("=" * 70)
    logger.info(f"Noise type: {args.noise_type}")
    logger.info(f"Total questions per level: {level_stats[noise_levels[0]]['total']}")
    logger.info("")
    logger.info(
        f"{'Noise Level':<12} | {'Text Acc':<10} | {'Vision Acc':<10} | {'Δ (V-T)':<10}"
    )
    logger.info("-" * 50)

    for level in noise_levels:
        stats = summary_by_level.get(level, {})
        text_acc = stats.get("text_accuracy", 0)
        vision_acc = stats.get("vision_accuracy", 0)
        delta = stats.get("vision_advantage", 0)
        level_str = f"{int(level * 100)}%"
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        logger.info(
            f"{level_str:<12} | {text_acc:>8}% | {vision_acc:>8}% | {delta_str:>8}"
        )

    # Determine crossover point (if any)
    crossover = None
    for i, level in enumerate(noise_levels[1:], 1):
        prev_level = noise_levels[i - 1]
        prev_delta = summary_by_level.get(prev_level, {}).get("vision_advantage", 0)
        curr_delta = summary_by_level.get(level, {}).get("vision_advantage", 0)
        if prev_delta <= 0 and curr_delta > 0:
            crossover = level
            break

    if crossover:
        logger.info(
            f"\nCrossover point: Vision overtakes text at {int(crossover * 100)}% noise"
        )
    else:
        # Check if vision always wins or always loses
        all_deltas = [
            summary_by_level.get(lvl, {}).get("vision_advantage", 0)
            for lvl in noise_levels
        ]
        if all(d > 0 for d in all_deltas):
            logger.info("\nVision outperforms text at ALL noise levels")
        elif all(d <= 0 for d in all_deltas):
            logger.info(
                "\nText outperforms vision at ALL noise levels (no crossover found)"
            )

    save_experiment_results(
        results,
        results_dir,
        f"noise_{args.noise_type}_{args.mode}_{args.num_articles}articles.json",
    )


def cmd_augmented(args: argparse.Namespace) -> None:
    """Run augmented rendering experiment comparing plain vs highlighted text.

    Tests Hypothesis 3: Visual formatting can carry semantic signal that improves
    downstream task performance.

    Compares two conditions on QuALITY QA:
    1. Plain vision: Standard dark mode rendering
    2. Augmented vision: Semantic highlighting (entities=blue, numbers=green, quotes=purple)
    """
    data_dir, results_dir = setup_experiment_dirs("augmented")

    logger.info("Loading QuALITY dataset...")
    try:
        ds = load_dataset("emozilla/quality", split="validation")
    except Exception as e:
        logger.error(f"Error loading QuALITY dataset: {e}")
        return

    # Group questions by article (same as cmd_quality)
    articles_dict = {}
    for item in ds:
        article_hash = hashlib.md5(item["article"][:100].encode()).hexdigest()[:8]
        if article_hash not in articles_dict:
            articles_dict[article_hash] = {"article": item["article"], "questions": []}
        articles_dict[article_hash]["questions"].append(
            {
                "question": item["question"],
                "options": item["options"],
                "gold_label": item[
                    "answer"
                ],  # Note: QuALITY uses "answer" not "gold_label"
            }
        )

    # Filter for articles with enough questions
    articles = []
    for article_hash, article_data in articles_dict.items():
        if len(article_data["questions"]) >= args.questions_per_article:
            articles.append({"hash": article_hash, **article_data})
        if len(articles) >= args.num_articles:
            break

    model, tokenizer = load_model()

    results = {
        "mode": args.mode,
        "articles": [],
        "summary": {},
    }

    stats = {
        "plain_correct": 0,
        "augmented_correct": 0,
        "text_correct": 0,
        "total": 0,
    }

    logger.info(f"AUGMENTED RENDERING EXPERIMENT: Mode={args.mode}")
    logger.info(
        f"Articles: {len(articles)}, Questions/article: {args.questions_per_article}"
    )
    logger.info("=" * 70)

    for article_idx, item in enumerate(articles):
        article_text = item["article"]
        article_hash = item["hash"]

        logger.info(f"\nArticle {article_idx + 1}/{len(articles)}: {article_hash}")

        # Render plain and augmented versions
        plain_img_path = data_dir / f"article_{article_hash}_plain.png"
        augmented_img_path = data_dir / f"article_{article_hash}_augmented.png"

        if not plain_img_path.exists():
            render_text_to_image(article_text, str(plain_img_path))
        if not augmented_img_path.exists():
            render_augmented_text_to_image(article_text, str(augmented_img_path))

        # Process questions
        questions = item["questions"][: args.questions_per_article]

        for q_idx, q in enumerate(questions):
            question = q["question"]
            options = q["options"]
            correct = q["gold_label"]

            options_str = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
            prompt_suffix = f"\n\nQuestion: {question}\nOptions:\n{options_str}\n\nAnswer with just the number (0, 1, 2, or 3):"

            # 1. Plain vision
            plain_prompt = f"<image>{prompt_suffix}"
            plain_output, _, _ = run_inference(
                plain_prompt,
                str(plain_img_path),
                mode=args.mode,
                model=model,
                tokenizer=tokenizer,
            )
            plain_answer = next(
                (int(c) for c in plain_output.strip() if c in "0123"), -1
            )
            plain_correct = plain_answer == correct

            # 2. Augmented vision
            aug_prompt = f"<image>{prompt_suffix}"
            aug_output, _, _ = run_inference(
                aug_prompt,
                str(augmented_img_path),
                mode=args.mode,
                model=model,
                tokenizer=tokenizer,
            )
            aug_answer = next((int(c) for c in aug_output.strip() if c in "0123"), -1)
            aug_correct = aug_answer == correct

            # 3. Text-only baseline
            text_prompt = f"<image>\n{article_text}{prompt_suffix}"
            text_output, _, _ = run_inference(
                text_prompt, "", mode="text", model=model, tokenizer=tokenizer
            )
            text_answer = next((int(c) for c in text_output.strip() if c in "0123"), -1)
            text_correct = text_answer == correct

            # Log results
            p_mark = "✓" if plain_correct else "✗"
            a_mark = "✓" if aug_correct else "✗"
            t_mark = "✓" if text_correct else "✗"
            logger.info(
                f"  Q{q_idx + 1}: Plain={plain_answer}{p_mark} Aug={aug_answer}{a_mark} Text={text_answer}{t_mark} (correct={correct})"
            )

            stats["plain_correct"] += int(plain_correct)
            stats["augmented_correct"] += int(aug_correct)
            stats["text_correct"] += int(text_correct)
            stats["total"] += 1

        results["articles"].append(
            {
                "hash": article_hash,
                "questions": len(questions),
            }
        )

    # Summary
    n = stats["total"]
    if n > 0:
        plain_acc = stats["plain_correct"] / n * 100
        aug_acc = stats["augmented_correct"] / n * 100
        text_acc = stats["text_correct"] / n * 100

        results["summary"] = {
            "total_questions": n,
            "plain_accuracy": round(plain_acc, 1),
            "augmented_accuracy": round(aug_acc, 1),
            "text_accuracy": round(text_acc, 1),
            "plain_correct": stats["plain_correct"],
            "augmented_correct": stats["augmented_correct"],
            "text_correct": stats["text_correct"],
        }

        logger.info("\n" + "=" * 70)
        logger.info("AUGMENTED RENDERING EXPERIMENT RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total questions: {n}")
        logger.info("")
        logger.info(f"{'Condition':<20} | {'Accuracy':>10} | {'Correct':>10}")
        logger.info("-" * 50)
        logger.info(
            f"{'Plain Vision':<20} | {plain_acc:>9.1f}% | {stats['plain_correct']:>5}/{n}"
        )
        logger.info(
            f"{'Augmented Vision':<20} | {aug_acc:>9.1f}% | {stats['augmented_correct']:>5}/{n}"
        )
        logger.info(
            f"{'Text Only':<20} | {text_acc:>9.1f}% | {stats['text_correct']:>5}/{n}"
        )
        logger.info("")

        # Analysis
        if aug_acc > plain_acc:
            logger.info(
                f"✓ Augmented rendering improves accuracy by +{aug_acc - plain_acc:.1f}%"
            )
        elif aug_acc == plain_acc:
            logger.info("No difference between plain and augmented rendering")
        else:
            logger.info(
                f"Plain rendering better than augmented by +{plain_acc - aug_acc:.1f}%"
            )

    save_experiment_results(
        results, results_dir, f"augmented_{args.mode}_{args.num_articles}articles.json"
    )


def generate_cell_lookup_qa(
    header: list[str], rows: list[list[str]], seed: int = 42
) -> tuple[str, str]:
    """Generate a cell lookup question that can be answered from visible data.

    Returns: (question, answer) tuple
    """
    random.seed(seed)
    row_idx = random.randint(0, len(rows) - 1)
    col_idx = random.randint(0, len(header) - 1)

    question = f"What is the value in row {row_idx + 1}, column '{header[col_idx]}'?"
    answer = str(rows[row_idx][col_idx])
    return question, answer


def cmd_tables(args: argparse.Namespace) -> None:
    """Run table QA experiment comparing vision vs text table representations.

    Tests Hypothesis 2: Visual projection preserves 2D structure better than linearization.

    Uses synthetic cell-lookup questions that CAN be answered from visible data,
    testing spatial reasoning rather than aggregation.

    Compares three conditions:
    1. Vision: Table rendered as image
    2. Markdown: Table as markdown text
    3. Linearized: Table as row-by-row text (Row 1: Col1=val, Col2=val, ...)
    """
    import pandas as pd

    data_dir, results_dir = setup_experiment_dirs("tables")

    logger.info("Loading DataBench dataset for table sources...")
    try:
        qa_ds = load_dataset("cardiffnlp/databench", name="qa", split="train")
    except Exception as e:
        logger.error(f"Error loading DataBench dataset: {e}")
        return

    # Load diverse tables (we'll generate our own questions)
    tables = []
    seen_datasets = set()

    for item in qa_ds:
        ds_id = item["dataset"]
        if ds_id in seen_datasets:
            continue

        # Load the table sample
        try:
            df = pd.read_parquet(
                f"hf://datasets/cardiffnlp/databench/data/{ds_id}/sample.parquet"
            )
            # Accept any table with 3-20 rows, 3-8 cols (sample size is 20 rows)
            if 3 <= len(df) <= 20 and 3 <= len(df.columns) <= 8:
                # Convert all values to strings for rendering
                rows = [[str(cell) for cell in row] for row in df.values.tolist()]
                header = list(df.columns)

                # Generate cell lookup question
                question, answer = generate_cell_lookup_qa(
                    header, rows, seed=len(tables)
                )

                tables.append(
                    {
                        "id": ds_id,
                        "question": question,
                        "answer": answer,
                        "answer_type": "cell_lookup",
                        "header": header,
                        "rows": rows,
                    }
                )
                seen_datasets.add(ds_id)
        except Exception as e:
            logger.warning(f"Failed to load table {ds_id}: {e}")
            continue

        if len(tables) >= args.num_tables:
            break

    if len(tables) < args.num_tables:
        logger.warning(
            f"Only found {len(tables)} suitable tables (requested {args.num_tables})"
        )

    model, tokenizer = load_model()

    results = {
        "mode": args.mode,
        "tables": [],
        "summary": {},
    }

    stats = {
        "vision_correct": 0,
        "markdown_correct": 0,
        "linearized_correct": 0,
        "total": 0,
    }

    logger.info(f"TABLE EXPERIMENT: Mode={args.mode}, Tables={len(tables)}")
    logger.info("=" * 70)

    for i, item in enumerate(tables):
        table_id = item["id"]
        question = item["question"]
        gold_answer = item["answer"]
        gold_answers = [gold_answer]  # Wrap in list for answers_match
        header = item["header"]
        rows = item["rows"]

        logger.info(f"\nTable {i + 1}/{len(tables)}: {table_id}")
        logger.info(f"  Size: {len(rows)} rows × {len(header)} cols")
        logger.info(f"  Q: {question[:60]}...")
        logger.info(f"  Gold: {gold_answer}")

        # Create table hash for caching
        table_hash = hashlib.md5(
            f"{table_id}_{len(rows)}_{len(header)}".encode()
        ).hexdigest()[:8]

        # Render table to image
        img_path = data_dir / f"table_{table_hash}.png"
        if not img_path.exists():
            render_table_to_image(header, rows, str(img_path))

        # Prepare text representations
        markdown_table = table_to_markdown(header, rows)
        linearized_table = table_to_linearized(header, rows)

        # Common prompt - clearer instruction for table QA
        prompt_suffix = f"\n\nBased on the table above, answer the following question.\nQuestion: {question}\nAnswer:"

        # 1. Vision condition
        vision_prompt = f"<image>\nThis is a data table.{prompt_suffix}"
        vision_output, _, _ = run_inference(
            vision_prompt,
            str(img_path),
            mode=args.mode,
            model=model,
            tokenizer=tokenizer,
        )
        vision_correct = answers_match(vision_output.strip(), gold_answers)

        # 2. Markdown condition
        markdown_prompt = f"<image>\n{markdown_table}{prompt_suffix}"
        markdown_output, _, _ = run_inference(
            markdown_prompt, "", mode="text", model=model, tokenizer=tokenizer
        )
        markdown_correct = answers_match(markdown_output.strip(), gold_answers)

        # 3. Linearized condition
        linearized_prompt = f"<image>\n{linearized_table}{prompt_suffix}"
        linearized_output, _, _ = run_inference(
            linearized_prompt, "", mode="text", model=model, tokenizer=tokenizer
        )
        linearized_correct = answers_match(linearized_output.strip(), gold_answers)

        # Log results
        v_mark = "✓" if vision_correct else "✗"
        m_mark = "✓" if markdown_correct else "✗"
        l_mark = "✓" if linearized_correct else "✗"
        logger.info(f"  Vision: {vision_output[:30]}... {v_mark}")
        logger.info(f"  Markdown: {markdown_output[:30]}... {m_mark}")
        logger.info(f"  Linearized: {linearized_output[:30]}... {l_mark}")

        # Update stats
        stats["vision_correct"] += int(vision_correct)
        stats["markdown_correct"] += int(markdown_correct)
        stats["linearized_correct"] += int(linearized_correct)
        stats["total"] += 1

        results["tables"].append(
            {
                "id": table_id,
                "question": question,
                "gold_answer": gold_answer,
                "rows": len(rows),
                "cols": len(header),
                "vision_output": vision_output.strip(),
                "markdown_output": markdown_output.strip(),
                "linearized_output": linearized_output.strip(),
                "vision_correct": vision_correct,
                "markdown_correct": markdown_correct,
                "linearized_correct": linearized_correct,
            }
        )

    # Summary
    n = stats["total"]
    if n > 0:
        vision_acc = stats["vision_correct"] / n * 100
        markdown_acc = stats["markdown_correct"] / n * 100
        linearized_acc = stats["linearized_correct"] / n * 100

        results["summary"] = {
            "total_tables": n,
            "vision_accuracy": round(vision_acc, 1),
            "markdown_accuracy": round(markdown_acc, 1),
            "linearized_accuracy": round(linearized_acc, 1),
            "vision_correct": stats["vision_correct"],
            "markdown_correct": stats["markdown_correct"],
            "linearized_correct": stats["linearized_correct"],
        }

        logger.info("\n" + "=" * 70)
        logger.info("TABLE EXPERIMENT RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total tables: {n}")
        logger.info("")
        logger.info(f"{'Condition':<15} | {'Accuracy':>10} | {'Correct':>10}")
        logger.info("-" * 45)
        logger.info(
            f"{'Vision':<15} | {vision_acc:>9.1f}% | {stats['vision_correct']:>5}/{n}"
        )
        logger.info(
            f"{'Markdown':<15} | {markdown_acc:>9.1f}% | {stats['markdown_correct']:>5}/{n}"
        )
        logger.info(
            f"{'Linearized':<15} | {linearized_acc:>9.1f}% | {stats['linearized_correct']:>5}/{n}"
        )
        logger.info("")

        # Analysis
        if vision_acc > markdown_acc and vision_acc > linearized_acc:
            logger.info("✓ Vision outperforms both text representations")
        elif vision_acc > linearized_acc:
            logger.info("✓ Vision beats linearized text (structure preserved)")
        elif markdown_acc > linearized_acc:
            logger.info("Markdown beats linearized (some structure preserved in text)")
        else:
            logger.info("No clear winner between conditions")

    save_experiment_results(
        results, results_dir, f"tables_{args.mode}_{args.num_tables}tables.json"
    )


def cmd_finewiki(args: argparse.Namespace) -> None:
    """Run FineWiki language modeling experiment comparing text vs vision continuation."""
    data_dir, results_dir = setup_experiment_dirs("finewiki")

    logger.info("Loading FineWiki dataset...")
    try:
        ds = load_dataset(
            "HuggingFaceFW/finewiki", name="en", split="train", streaming=True
        )
    except Exception as e:
        logger.error(f"Error loading FineWiki dataset: {e}")
        return

    # Find articles with sufficient length
    articles = []
    min_words = args.context_words + args.continuation_words + 100
    logger.info(f"Finding {args.num_articles} articles with {min_words}+ words...")

    for item in ds:
        text = item["text"]
        words = text.split()
        if len(words) >= min_words:
            articles.append({"title": item.get("title", "Untitled"), "text": text})
            logger.info(f"  Found: {item.get('title', 'Untitled')[:50]}...")
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

    logger.info(f"FINEWIKI EXPERIMENT: Mode={args.mode}, Articles={args.num_articles}")

    for i, article in enumerate(articles):
        words = article["text"].split()
        context = " ".join(words[: args.context_words])
        continuation = " ".join(
            words[args.context_words : args.context_words + args.continuation_words]
        )

        logger.info(f"Article {i + 1}: {article['title'][:50]}...")

        img_path = data_dir / f"article_{i}.png"
        if not img_path.exists():
            render_text_to_image(context, str(img_path))

        # Text condition
        text_prompt = (
            f"<image>\n{context}\n\nContinue this text with the next sentence:"
        )
        text_output, _, _ = run_inference(
            text_prompt,
            "",  # No image for text mode
            mode="text",
            model=model,
            tokenizer=tokenizer,
        )
        text_tokens = (
            len(tokenizer.encode(context, add_special_tokens=False))
            + CONTINUATION_TOKEN_OVERHEAD
        )

        # Vision condition
        vision_prompt = "<image>\n\nContinue this text with the next sentence:"
        vision_output, _, _ = run_inference(
            vision_prompt,
            str(img_path),
            mode=args.mode,
            model=model,
            tokenizer=tokenizer,
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

        logger.info(
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

        logger.info(
            f"RESULTS: Text={text_avg:.3f}, Vision={vision_avg:.3f}, Compression={compression:.1f}x"
        )

        save_experiment_results(
            results,
            results_dir,
            f"finewiki_{args.mode}_{args.num_articles}articles.json",
        )


def cmd_omnidocbench(args: argparse.Namespace) -> None:
    """Run OmniDocBench evaluation (Benchmark used in DeepSeek-OCR paper)."""
    data_dir, results_dir = setup_experiment_dirs("omnidocbench")

    # 1. Ensure Annotation JSON exists
    json_path = data_dir / "OmniDocBench.json"
    if not json_path.exists():
        # Check if it was downloaded to root in previous steps
        root_json = Path("OmniDocBench.json")
        if root_json.exists():
            root_json.rename(json_path)
        else:
            logger.info("Downloading OmniDocBench.json...")
            # We use direct URL as it's not a standard HF file in the repo root usually,
            # but we found it via direct wget earlier.
            # Ideally we use hf_hub_download if it's in the repo.
            # The previous wget worked from https://huggingface.co/datasets/opendatalab/OmniDocBench/resolve/main/OmniDocBench.json
            try:
                hf_hub_download(
                    repo_id="opendatalab/OmniDocBench",
                    filename="OmniDocBench.json",
                    repo_type="dataset",
                    local_dir=str(data_dir),
                )
            except Exception as e:
                logger.error(f"Failed to download OmniDocBench.json: {e}")
                return

    # 2. Load Annotations
    logger.info("Loading OmniDocBench annotations...")
    with open(json_path, "r") as f:
        dataset = json.load(f)

    # 3. Filter/Select Items
    if args.num_articles > 0:
        dataset = dataset[: args.num_articles]

    logger.info(f"Evaluating on {len(dataset)} documents from OmniDocBench")

    model, tokenizer = load_model()

    results = {"mode": args.mode, "documents": [], "summary": {}}
    stats = {
        "edit_distance": 0,
        "normalized_ed": 0,
        "vision_tokens": 0,
        "output_tokens": 0,
        "total": 0,
    }

    logger.info(f"\nOMNIDOCBENCH EXPERIMENT: Mode={args.mode}, Docs={len(dataset)}")

    for i, item in enumerate(dataset):
        # Extract Ground Truth
        # Concatenate text blocks sorted by order
        blocks = item.get("layout_dets", [])
        blocks = [b for b in blocks if not b.get("ignore", False)]
        blocks.sort(key=lambda x: x.get("order") or 0)
        ground_truth = "\n".join(b.get("text", "") for b in blocks)

        # Get Image
        image_filename = item.get("page_info", {}).get("image_path")
        if not image_filename:
            logger.warning(f"Skipping item {i}: No image_path found")
            continue

        local_img_path = data_dir / "images" / image_filename
        if not local_img_path.exists():
            # Download image on demand
            try:
                hf_hub_download(
                    repo_id="opendatalab/OmniDocBench",
                    filename=f"images/{image_filename}",
                    repo_type="dataset",
                    local_dir=str(data_dir),
                )
            except Exception as e:
                logger.error(f"Failed to download image {image_filename}: {e}")
                continue

        logger.info(f"\nDocument {i + 1}: {image_filename}")

        # Run Inference
        output, vision_tokens, output_tokens = run_inference(
            prompt="<image>\nFree OCR.",  # Standard OCR prompt
            image_path=local_img_path,
            mode=args.mode,
            model=model,
            tokenizer=tokenizer,
        )

        # Metrics
        metrics = calculate_edit_distance(output, ground_truth)
        compression = output_tokens / vision_tokens if vision_tokens > 0 else 0

        logger.info(f"  Vision Tokens: {vision_tokens}")
        logger.info(f"  Compression: {compression:.1f}x")
        logger.info(f"  Precision: {metrics['precision']}%")

        stats["edit_distance"] += metrics["edit_distance"]
        stats["normalized_ed"] += metrics["normalized_ed"]
        stats["vision_tokens"] += vision_tokens
        stats["output_tokens"] += output_tokens
        stats["total"] += 1

        results["documents"].append(
            {
                "image": image_filename,
                "vision_tokens": vision_tokens,
                "output_tokens": output_tokens,
                "compression": round(compression, 2),
                "precision": metrics["precision"],
                "edit_distance": metrics["edit_distance"],
            }
        )

    # Summary
    n = stats["total"]
    if n > 0:
        avg_ned = stats["normalized_ed"] / n
        avg_precision = (1 - avg_ned) * 100
        avg_compression = (
            stats["output_tokens"] / stats["vision_tokens"]
            if stats["vision_tokens"] > 0
            else 0
        )

        results["summary"] = {
            "total_documents": n,
            "avg_precision": round(avg_precision, 2),
            "avg_normalized_ed": round(avg_ned, 4),
            "avg_compression": round(avg_compression, 2),
        }

        logger.info(
            f"\nRESULTS: Precision={avg_precision:.2f}%, Compression={avg_compression:.1f}x"
        )

        save_experiment_results(
            results,
            results_dir,
            f"omnidocbench_{args.mode}_{n}docs.json",
        )


def cmd_reproduce(args: argparse.Namespace) -> None:
    """Run a suite of experiments to reproduce paper results."""
    modes = ["tiny", "base", "large"]

    # Adjust parameters for fast run
    num_articles_quality = 1 if args.fast else 10
    num_articles_finewiki = 1 if args.fast else 20

    logger.info("=" * 60)
    logger.info("REPRODUCING DEEPSEEK-OCR PAPER RESULTS")
    logger.info("=" * 60)
    logger.info(f"Fast mode: {args.fast}")
    logger.info(f"Modes to test: {modes}")

    summary_results = {"quality": {}, "finewiki": {}, "omnidocbench": {}}

    # 0. OmniDocBench Experiments (The actual paper benchmark)
    logger.info("\n" + "=" * 60)
    logger.info("STARTING OMNIDOCBENCH EXPERIMENTS (OCR Benchmark)")
    logger.info("=" * 60)

    # Use a smaller number for reproduction if fast, but paper used full set.
    # The full set is 1355. Fast=1, Normal=10 (to save time, or 50?)
    # Let's match the other experiments
    num_docs_omni = 1 if args.fast else 10

    for mode in modes:
        if args.fast and mode == "large" and "omnidocbench" in summary_results:
            logger.info(
                f"Skipping OmniDocBench '{mode}' mode in fast run to avoid OOM."
            )
            continue
        logger.info(f"\nRunning OmniDocBench experiment for mode: {mode}")

        omni_args = argparse.Namespace(
            mode=mode, num_articles=num_docs_omni, command="omnidocbench"
        )

        cmd_omnidocbench(omni_args)

        data_dir, results_dir = setup_experiment_dirs("omnidocbench")
        result_file = results_dir / f"omnidocbench_{mode}_{num_docs_omni}docs.json"

        if result_file.exists():
            with open(result_file, "r") as f:
                data = json.load(f)
                summary_results["omnidocbench"][mode] = data["summary"]

    # 1. QuALITY Experiments
    logger.info("\n" + "=" * 60)
    logger.info("STARTING QUALITY EXPERIMENTS (Long-Document QA)")
    logger.info("=" * 60)

    for mode in modes:
        logger.info(f"\nRunning QuALITY experiment for mode: {mode}")

        # Create a namespace for the quality command
        quality_args = argparse.Namespace(
            mode=mode,
            num_articles=num_articles_quality,
            questions_per_article=5,
            command="quality",
        )

        # Run the experiment
        cmd_quality(quality_args)

        # Load the results to aggregate
        data_dir, results_dir = setup_experiment_dirs("quality")
        result_file = (
            results_dir / f"quality_{mode}_{num_articles_quality}articles.json"
        )

        if result_file.exists():
            with open(result_file, "r") as f:
                data = json.load(f)
                summary_results["quality"][mode] = data["summary"]

    # 2. FineWiki Experiments
    logger.info("\n" + "=" * 60)
    logger.info("STARTING FINEWIKI EXPERIMENTS (Language Modeling)")
    logger.info("=" * 60)

    for mode in modes:
        logger.info(f"\nRunning FineWiki experiment for mode: {mode}")

        # Create a namespace for the finewiki command
        finewiki_args = argparse.Namespace(
            mode=mode,
            num_articles=num_articles_finewiki,
            context_words=500,
            continuation_words=50,
            command="finewiki",
        )

        # Run the experiment
        cmd_finewiki(finewiki_args)

        # Load the results to aggregate
        data_dir, results_dir = setup_experiment_dirs("finewiki")
        result_file = (
            results_dir / f"finewiki_{mode}_{num_articles_finewiki}articles.json"
        )

        if result_file.exists():
            with open(result_file, "r") as f:
                data = json.load(f)
                summary_results["finewiki"][mode] = data["summary"]

        # 3. Print Summary Report

        logger.info("\n" + "=" * 60)

        logger.info("FINAL REPRODUCTION REPORT")

        logger.info("=" * 60)

        # OmniDocBench Table

        logger.info("\nOmniDocBench Results (OCR Accuracy):")

        logger.info(
            f"{'Mode':<10} | {'Vision Tokens':<15} | {'Precision':<10} | {'Compression':<12}"
        )

        logger.info("-" * 55)

        for mode in modes:
            res = summary_results["omnidocbench"].get(mode, {})

            if res:
                tokens = MODE_SETTINGS[mode].tokens

                prec = f"{res.get('avg_precision', 0)}%"

                comp = f"{res.get('avg_compression', 0)}x"

                logger.info(
                    f"{mode.capitalize():<10} | {tokens:<15} | {prec:<10} | {comp:<12}"
                )

        # QuALITY Table

        logger.info("\nQuALITY Results (Text vs Vision Accuracy):")

        logger.info(
            f"{'Mode':<10} | {'Vision Tokens':<15} | {'Text Acc':<10} | {'Vision Acc':<12} | {'Compression':<12}"
        )

        logger.info("-" * 70)

    for mode in modes:
        res = summary_results["quality"].get(mode, {})
        if res:
            tokens = MODE_SETTINGS[mode].tokens
            text_acc = f"{res.get('text_accuracy', 0)}%"
            vis_acc = f"{res.get('vision_accuracy', 0)}%"
            comp = f"{res.get('compression_ratio', 0)}x"
            logger.info(
                f"{mode.capitalize():<10} | {tokens:<15} | {text_acc:<10} | {vis_acc:<12} | {comp:<12}"
            )

    # FineWiki Table
    logger.info("\nFineWiki Results (Language Modeling Overlap):")
    logger.info(
        f"{'Mode':<10} | {'Vision Tokens':<15} | {'Text Avg':<10} | {'Vision Avg':<12} | {'Compression':<12}"
    )
    logger.info("-" * 70)

    for mode in modes:
        res = summary_results["finewiki"].get(mode, {})
        if res:
            tokens = MODE_SETTINGS[mode].tokens
            text_avg = f"{res.get('text_avg_overlap', 0):.3f}"
            vis_avg = f"{res.get('vision_avg_overlap', 0):.3f}"
            comp = f"{res.get('compression_ratio', 0)}x"
            logger.info(
                f"{mode.capitalize():<10} | {tokens:<15} | {text_avg:<10} | {vis_avg:<12} | {comp:<12}"
            )

    # Save summary
    _, results_dir = setup_experiment_dirs("reproduction")
    save_experiment_results(summary_results, results_dir, "reproduction_summary.json")


def main() -> None:
    """Main function to parse arguments and execute the specified command."""
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
        "--mode", type=str, default="base", choices=EXPERIMENT_MODES
    )
    quality_parser.add_argument("--num-articles", type=int, default=10)
    quality_parser.add_argument("--questions-per-article", type=int, default=5)

    # FineWiki command
    finewiki_parser = subparsers.add_parser(
        "finewiki", help="Run FineWiki language modeling experiment"
    )
    finewiki_parser.add_argument(
        "--mode", type=str, default="base", choices=EXPERIMENT_MODES
    )
    finewiki_parser.add_argument("--num-articles", type=int, default=20)
    finewiki_parser.add_argument("--context-words", type=int, default=500)
    finewiki_parser.add_argument("--continuation-words", type=int, default=50)

    # OmniDocBench command
    omni_parser = subparsers.add_parser(
        "omnidocbench", help="Run OmniDocBench evaluation"
    )
    omni_parser.add_argument(
        "--mode", type=str, default="base", choices=EXPERIMENT_MODES
    )
    omni_parser.add_argument(
        "--num-articles", type=int, default=10, help="Number of documents to evaluate"
    )

    # Truncation baseline command
    trunc_parser = subparsers.add_parser(
        "truncation", help="Run truncation baseline experiment (Experiment D)"
    )
    trunc_parser.add_argument(
        "--mode",
        type=str,
        default="large",
        choices=EXPERIMENT_MODES,
        help="Vision mode (determines token budget for truncation)",
    )
    trunc_parser.add_argument("--num-articles", type=int, default=5)
    trunc_parser.add_argument("--questions-per-article", type=int, default=5)

    # Noise injection experiment command (Experiment A)
    noise_parser = subparsers.add_parser(
        "noise", help="Run noise injection experiment (Experiment A)"
    )
    noise_parser.add_argument(
        "--mode",
        type=str,
        default="large",
        choices=EXPERIMENT_MODES,
        help="Vision mode to use",
    )
    noise_parser.add_argument(
        "--noise-type",
        type=str,
        default="typos",
        choices=["typos", "ocr", "deletions", "insertions", "mixed"],
        help="Type of noise to inject",
    )
    noise_parser.add_argument(
        "--noise-levels",
        type=str,
        default="0,0.02,0.05,0.10,0.15,0.20",
        help="Comma-separated noise rates (0.0 to 1.0)",
    )
    noise_parser.add_argument("--num-articles", type=int, default=3)
    noise_parser.add_argument("--questions-per-article", type=int, default=3)
    noise_parser.add_argument(
        "--noise-question",
        action="store_true",
        help="Also apply noise to question and options (default: article only)",
    )

    # Tables experiment command (Experiment B)
    tables_parser = subparsers.add_parser(
        "tables", help="Run WikiTableQuestions experiment (Experiment B)"
    )
    tables_parser.add_argument(
        "--mode",
        type=str,
        default="large",
        choices=EXPERIMENT_MODES,
        help="Vision mode for table images",
    )
    tables_parser.add_argument(
        "--num-tables", type=int, default=20, help="Number of tables to evaluate"
    )

    # Augmented rendering experiment command (Experiment C)
    augmented_parser = subparsers.add_parser(
        "augmented", help="Run augmented rendering experiment (Experiment C)"
    )
    augmented_parser.add_argument(
        "--mode",
        type=str,
        default="large",
        choices=EXPERIMENT_MODES,
        help="Vision mode to use",
    )
    augmented_parser.add_argument("--num-articles", type=int, default=3)
    augmented_parser.add_argument("--questions-per-article", type=int, default=3)

    # Reproduce command
    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Run a suite of experiments to reproduce paper results"
    )
    reproduce_parser.add_argument(
        "--fast",
        action="store_true",
        help="Run with fewer articles for quick verification",
    )

    args = parser.parse_args()

    if args.command == "ocr":
        cmd_ocr(args)
    elif args.command == "quality":
        cmd_quality(args)
    elif args.command == "finewiki":
        cmd_finewiki(args)
    elif args.command == "omnidocbench":
        cmd_omnidocbench(args)
    elif args.command == "truncation":
        cmd_truncation(args)
    elif args.command == "noise":
        cmd_noise(args)
    elif args.command == "tables":
        cmd_tables(args)
    elif args.command == "augmented":
        cmd_augmented(args)
    elif args.command == "reproduce":
        cmd_reproduce(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
