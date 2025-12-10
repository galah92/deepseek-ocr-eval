"""Render text documents to PNG images for the vision vs text experiment."""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse

import platform
import sys


def _get_platform_fonts():
    """Get font paths based on the current platform."""
    system = platform.system()

    if system == "Linux":
        return {
            "mono": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            ],
            "serif": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
            ],
            "sans": [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ],
        }
    elif system == "Darwin":  # macOS
        return {
            "mono": [
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Monaco.ttf",
                "/Library/Fonts/Courier New.ttf",
            ],
            "serif": [
                "/System/Library/Fonts/Times.ttc",
                "/Library/Fonts/Times New Roman.ttf",
                "/System/Library/Fonts/Georgia.ttf",
            ],
            "sans": [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/SFNSText.ttf",
            ],
        }
    elif system == "Windows":
        return {
            "mono": [
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/cour.ttf",
                "C:/Windows/Fonts/lucon.ttf",
            ],
            "serif": [
                "C:/Windows/Fonts/times.ttf",
                "C:/Windows/Fonts/georgia.ttf",
                "C:/Windows/Fonts/cambria.ttc",
            ],
            "sans": [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/calibri.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
            ],
        }
    else:
        # Fallback for unknown systems
        return {"mono": [], "serif": [], "sans": []}


FONT_CONFIGS = _get_platform_fonts()

SIZE_CONFIGS = {"small": 14, "medium": 20, "large": 28}

COLOR_CONFIGS = {
    "default": {"bg": "white", "fg": "black"},
    "dark": {"bg": "#1e1e1e", "fg": "#d4d4d4"},
    "sepia": {"bg": "#f4ecd8", "fg": "#5c4b37"},
    "blue": {"bg": "#e8f4fc", "fg": "#1a365d"},
    "low_contrast": {"bg": "#e0e0e0", "fg": "#606060"},
}

DOCUMENTS = {
    "employee_record": """Employee Record
---------------
Name: John Smith
Employee ID: EMP-2847
Department: Engineering
Position: Senior Developer
Start Date: March 15, 2023
Salary: $95,000
Manager: Sarah Chen
Office: Building A, Room 412""",
    "product_spec": """Product Specification
--------------------
Product: Wireless Headphones XR-500
SKU: WH-XR500-BLK
Price: $149.99
Battery Life: 30 hours
Weight: 250g
Bluetooth: 5.3
Colors: Black, White, Navy
Warranty: 2 years""",
    "invoice": """INVOICE #INV-2024-0892
Date: November 15, 2024
Due Date: December 15, 2024

Bill To:
  Acme Corporation
  123 Business Ave
  New York, NY 10001
  Contact: Robert Johnson
  Email: rjohnson@acme.com

Ship To:
  Acme Warehouse
  456 Industrial Blvd
  Newark, NJ 07102

Items:
  1. Widget Pro (WP-100)    x10   $50.00    $500.00
  2. Gadget Plus (GP-200)   x5    $75.00    $375.00
  3. Tool Kit (TK-300)      x2    $120.00   $240.00

Subtotal: $1,115.00
Tax (8%): $89.20
Shipping: $25.00
Total: $1,229.20

Payment Terms: Net 30""",
    "meeting_minutes": """Meeting Minutes
===============
Project: Phoenix Website Redesign
Date: October 28, 2024
Time: 2:00 PM - 3:30 PM
Location: Conference Room B

Attendees:
- Alice Wong (Project Manager)
- Bob Martinez (Lead Developer)
- Carol Davis (UX Designer)
- David Lee (Backend Engineer)
- Emma Wilson (QA Lead)

Absent: Frank Brown (Client Rep)

Agenda Items Discussed:

1. Sprint Review
   - Completed: User authentication, Dashboard UI
   - In Progress: Payment integration, Mobile responsive
   - Blocked: API rate limiting (waiting on vendor)

2. Timeline Update
   - Original deadline: December 1, 2024
   - Revised deadline: December 15, 2024
   - Reason: Scope increase from client

3. Budget Status
   - Allocated: $150,000
   - Spent: $98,500
   - Remaining: $51,500

Action Items:
- Bob: Fix checkout bug by Nov 1
- Carol: Finalize mobile mockups by Nov 3
- Emma: Create test plan for payment flow

Next Meeting: November 4, 2024 at 2:00 PM""",
}

# All rendering configurations: (name, font_type, font_size, color_scheme)
ALL_CONFIGS = [
    ("font_mono", "mono", 20, "default"),
    ("font_serif", "serif", 20, "default"),
    ("font_sans", "sans", 20, "default"),
    ("size_small", "mono", 14, "default"),
    ("size_medium", "mono", 20, "default"),
    ("size_large", "mono", 28, "default"),
    ("color_default", "mono", 20, "default"),
    ("color_dark", "mono", 20, "dark"),
    ("color_sepia", "mono", 20, "sepia"),
    ("color_blue", "mono", 20, "blue"),
    ("color_low_contrast", "mono", 20, "low_contrast"),
]

CONFIG_GROUPS = {
    "fonts": [c for c in ALL_CONFIGS if c[0].startswith("font_")],
    "sizes": [c for c in ALL_CONFIGS if c[0].startswith("size_")],
    "colors": [c for c in ALL_CONFIGS if c[0].startswith("color_")],
}


def get_font(font_type: str, font_size: int):
    """Load a font of the specified type and size, with fallback to default.

    Args:
        font_type: One of 'mono', 'serif', or 'sans'
        font_size: Font size in points

    Returns:
        PIL ImageFont object
    """
    font_paths = FONT_CONFIGS.get(font_type, FONT_CONFIGS.get("mono", []))
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            # Font file not found or cannot be read
            continue
        except IOError:
            # I/O error reading font file
            continue

    # Log warning when falling back to default font
    print(f"Warning: No {font_type} font found, using default font", file=sys.stderr)
    return ImageFont.load_default()


def render_text_to_image(
    text: str,
    output_path: str,
    font_size: int = 20,
    padding: int = 40,
    font_type: str = "mono",
    color_scheme: str = "default",
):
    font = get_font(font_type, font_size)
    colors = COLOR_CONFIGS.get(color_scheme, COLOR_CONFIGS["default"])
    lines = text.split("\n")

    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    max_width = 0
    line_height = font_size + 4
    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        max_width = max(max_width, bbox[2] - bbox[0])

    img_width = max_width + (padding * 2)
    img_height = (len(lines) * line_height) + (padding * 2)

    img = Image.new("RGB", (img_width, img_height), color=colors["bg"])
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=colors["fg"])
        y += line_height

    img.save(output_path)
    return img_width, img_height


def render_config(config_name: str, font_type: str, font_size: int, color_scheme: str):
    data_dir = Path(__file__).parent / "data" / config_name
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {config_name}: font={font_type}, size={font_size}, colors={color_scheme}")

    for name, text in DOCUMENTS.items():
        (data_dir / f"{name}.txt").write_text(text)
        render_text_to_image(
            text,
            str(data_dir / f"{name}.png"),
            font_size=font_size,
            font_type=font_type,
            color_scheme=color_scheme,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["all", "fonts", "sizes", "colors", "default"],
        default="default",
    )
    args = parser.parse_args()

    # Always create default data directory
    default_dir = Path(__file__).parent / "data"
    default_dir.mkdir(exist_ok=True)
    for name, text in DOCUMENTS.items():
        (default_dir / f"{name}.txt").write_text(text)
        render_text_to_image(text, str(default_dir / f"{name}.png"))

    if args.config == "default":
        configs = [("default", "mono", 20, "default")]
    elif args.config == "all":
        configs = ALL_CONFIGS
    else:
        configs = CONFIG_GROUPS.get(args.config, [])

    print(f"Rendering {len(configs)} configurations...")
    for config_name, font_type, font_size, color_scheme in configs:
        render_config(config_name, font_type, font_size, color_scheme)

    print("Done!")


if __name__ == "__main__":
    main()
