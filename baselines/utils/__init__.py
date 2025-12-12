"""Utility functions for compression baselines.

This module provides helper functions for:
- Model loading and management
- Image rendering and processing
- Text generation and parsing
"""

from .model import load_model, get_device, count_parameters
from .image import (
    render_text_to_image,
    render_text_to_image_with_params,
    calculate_valid_vision_tokens,
    calculate_num_crops,
)
from .generation import (
    parse_mc_answer,
    clean_output,
    extract_number,
    format_mc_prompt,
)

__all__ = [
    # Model utilities
    "load_model",
    "get_device",
    "count_parameters",
    # Image utilities
    "render_text_to_image",
    "render_text_to_image_with_params",
    "calculate_valid_vision_tokens",
    "calculate_num_crops",
    # Generation utilities
    "parse_mc_answer",
    "clean_output",
    "extract_number",
    "format_mc_prompt",
]
