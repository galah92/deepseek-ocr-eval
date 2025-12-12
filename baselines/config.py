"""Configuration for compression baselines.

This module contains shared configuration settings for all compression
baselines, mirroring Lee et al.'s config.py structure.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModeSettings:
    """Settings for a vision resolution mode."""

    base_size: int
    image_size: int
    crop_mode: bool
    tokens: int | None  # None for dynamic modes


# Vision encoding modes and their settings
MODE_SETTINGS: dict[str, ModeSettings] = {
    "tiny": ModeSettings(512, 512, False, 64),
    "small": ModeSettings(640, 640, False, 100),
    "base": ModeSettings(1024, 1024, False, 256),
    "large": ModeSettings(1280, 1280, False, 400),
    "gundam": ModeSettings(1024, 640, True, None),
}

# Modes used in experiments (excludes gundam which has dynamic token count)
EXPERIMENT_MODES = ["tiny", "small", "base", "large"]

# Special token IDs for DeepSeek-OCR
# The <image> token ID used as placeholder for injected embeddings
IMAGE_TOKEN_ID = 128815

# Default rendering settings (dark mode for optimal OCR)
DEFAULT_FONT_SIZE = 12
DEFAULT_BG_COLOR = "#1e1e1e"
DEFAULT_FG_COLOR = "#d4d4d4"

# Gundam preset (used for vision bypass in mean pooling)
GUNDAM_PRESET = {
    "base_size": 512,
    "image_size": 512,
}

# Token overhead estimates for prompts
PROMPT_TOKEN_OVERHEAD = 50
CONTINUATION_TOKEN_OVERHEAD = 30

# Model context limit
MODEL_CONTEXT_LIMIT = 8192  # DeepSeek-OCR context window

# Temporary paths for experiments
TMP_OUTPUT_PATH = Path("/tmp/deepseek_ocr_output")
