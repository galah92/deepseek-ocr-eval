"""Compression baselines for DeepSeek-OCR evaluation.

This module provides different compression methods for comparing
vision-based compression against text-based alternatives.

Available baselines:
- EmbeddingMeanPooler: Embedding-level mean pooling (Lee et al.)
- truncate_text: Simple truncation (first-N or last-N tokens)
- VisionEncoder: DeepSeek-OCR's vision encoding
- run_inference: Direct inference function
"""

from .config import (
    IMAGE_TOKEN_ID,
    EXPERIMENT_MODES,
    MODE_SETTINGS,
    ModeSettings,
    TMP_OUTPUT_PATH,
    MODEL_CONTEXT_LIMIT,
    PROMPT_TOKEN_OVERHEAD,
    DEFAULT_FONT_SIZE,
    DEFAULT_BG_COLOR,
    DEFAULT_FG_COLOR,
)
from .meanpool import EmbeddingMeanPooler, run_inference_mean_pool
from .truncation import truncate_text, truncate_text_first_n, truncate_text_last_n, count_tokens
from .vision import VisionEncoder, run_inference
from .utils import (
    load_model,
    get_device,
    render_text_to_image,
    render_text_to_image_with_params,
    calculate_valid_vision_tokens,
    parse_mc_answer,
    clean_output,
    format_mc_prompt,
)

__all__ = [
    # Config
    "IMAGE_TOKEN_ID",
    "EXPERIMENT_MODES",
    "MODE_SETTINGS",
    "ModeSettings",
    "TMP_OUTPUT_PATH",
    "MODEL_CONTEXT_LIMIT",
    "PROMPT_TOKEN_OVERHEAD",
    "DEFAULT_FONT_SIZE",
    "DEFAULT_BG_COLOR",
    "DEFAULT_FG_COLOR",
    # Baselines
    "EmbeddingMeanPooler",
    "run_inference_mean_pool",
    "truncate_text",
    "truncate_text_first_n",
    "truncate_text_last_n",
    "count_tokens",
    "VisionEncoder",
    "run_inference",
    # Utilities
    "load_model",
    "get_device",
    "render_text_to_image",
    "render_text_to_image_with_params",
    "calculate_valid_vision_tokens",
    "parse_mc_answer",
    "clean_output",
    "format_mc_prompt",
]
