"""Utility functions for compression baselines.

This module provides helper functions for:
- Model loading and management
- Image rendering and processing
- Text generation and parsing
"""

from .generation import (
    clean_output,
    extract_number,
    format_mc_prompt,
    parse_mc_answer,
)
from .image import (
    calculate_num_crops,
    calculate_valid_vision_tokens,
    render_text_to_image,
    render_text_to_image_with_params,
)
from .model import count_parameters, get_device, load_model

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
