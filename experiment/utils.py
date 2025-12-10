"""Shared utilities for DeepSeek-OCR experiments."""

import platform
import sys

from PIL import Image, ImageDraw, ImageFont

# Global model cache
_model = None
_tokenizer = None

MODE_SETTINGS = {
    "tiny": {"base_size": 512, "image_size": 512, "vision_tokens": 64},
    "small": {"base_size": 640, "image_size": 640, "vision_tokens": 100},
    "base": {"base_size": 1024, "image_size": 1024, "vision_tokens": 256},
    "large": {"base_size": 1280, "image_size": 1280, "vision_tokens": 400},
}


def _get_mono_font_paths():
    """Get monospace font paths based on the current platform."""
    system = platform.system()
    if system == "Linux":
        return [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        ]
    elif system == "Darwin":  # macOS
        return [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Courier New.ttf",
        ]
    elif system == "Windows":
        return [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/lucon.ttf",
        ]
    return []


FONT_PATHS = _get_mono_font_paths()


def get_font(size: int = 14):
    """Load a monospace font of the specified size, with fallback to default.

    Args:
        size: Font size in points

    Returns:
        PIL ImageFont object
    """
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
        except IOError:
            continue

    print("Warning: No monospace font found, using default font", file=sys.stderr)
    return ImageFont.load_default()


def load_model():
    """Load the DeepSeek-OCR model (cached after first load).

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

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


def render_text_to_image(
    text: str,
    output_path: str,
    font_size: int = 12,
    max_width: int = 1200,
    padding: int = 30,
    line_spacing: int = 4,
    bg_color: str = "#1e1e1e",
    fg_color: str = "#d4d4d4",
) -> tuple:
    """Render text to image (dark mode for optimal OCR).

    Args:
        text: Text to render
        output_path: Path to save the image
        font_size: Font size in points
        max_width: Maximum image width
        padding: Padding around text
        line_spacing: Space between lines
        bg_color: Background color (hex)
        fg_color: Foreground/text color (hex)

    Returns:
        Tuple of (width, height, num_lines)
    """
    font = get_font(font_size)

    # Wrap text to fit width
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
