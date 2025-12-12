"""Image rendering and processing utilities.

This module handles text-to-image rendering and vision token calculations
for DeepSeek-OCR evaluation.
"""

import logging
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ..config import (
    ModeSettings,
    DEFAULT_FONT_SIZE,
    DEFAULT_BG_COLOR,
    DEFAULT_FG_COLOR,
)

logger = logging.getLogger(__name__)

# Font configuration
MONO_FONT_PATH = font_manager.findfont("monospace")
FONT_SIZE = DEFAULT_FONT_SIZE
BG_COLOR = DEFAULT_BG_COLOR
FG_COLOR = DEFAULT_FG_COLOR
FONT = ImageFont.truetype(MONO_FONT_PATH, FONT_SIZE)


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


def render_text_to_image_with_params(
    text: str,
    output_path: str,
    font_size: int = 12,
    font_type: str = "mono",
    blur_radius: float = 0.0,
    jpeg_quality: int | None = None,
    max_width: int = 1200,
    padding: int = 30,
    line_spacing: int = 4,
) -> None:
    """Render text to image with configurable parameters for ablation studies.

    Args:
        text: The input text to render.
        output_path: The path where the generated image will be saved.
        font_size: Font size in points (default 12).
        font_type: Font type - "mono", "serif", or "sans".
        blur_radius: Gaussian blur radius (0 = no blur).
        jpeg_quality: If set, save as JPEG with this quality (1-100). None = PNG.
        max_width: The maximum width of the output image.
        padding: The padding around the text content within the image.
        line_spacing: Additional spacing between lines of text.
    """
    import io

    # Select font based on type
    if font_type == "mono":
        font_path = MONO_FONT_PATH
    elif font_type == "serif":
        serif_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        ]
        font_path = next((f for f in serif_fonts if Path(f).exists()), MONO_FONT_PATH)
    elif font_type == "sans":
        sans_fonts = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        font_path = next((f for f in sans_fonts if Path(f).exists()), MONO_FONT_PATH)
    else:
        font_path = MONO_FONT_PATH

    font = ImageFont.truetype(font_path, font_size)

    # Wrap text to lines
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

    img = Image.new("RGB", (img_width, img_height), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=FG_COLOR)
        y += line_height

    # Apply blur if specified
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Save with appropriate format
    if jpeg_quality is not None:
        # Save as JPEG with quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        img = Image.open(buffer)
        img.save(output_path.replace(".png", ".jpg"), format="JPEG", quality=jpeg_quality)
    else:
        img.save(output_path)


def calculate_valid_vision_tokens(
    width: int, height: int, settings: ModeSettings
) -> int:
    """Calculate valid vision tokens based on image dimensions and mode settings.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        settings: Mode settings with base_size and crop_mode

    Returns:
        Number of vision tokens
    """
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
) -> int:
    """Calculate number of crops for high-resolution images.

    Args:
        width: Image width
        height: Image height
        tile_size: Size of each tile

    Returns:
        Number of crops
    """
    cols = max(1, (width + tile_size - 1) // tile_size)
    rows = max(1, (height + tile_size - 1) // tile_size)
    return cols * rows
