"""Vision-based context compression via DeepSeek-OCR.

This module provides the core vision encoding functionality for
evaluating vision tokens as context compression.
"""

import logging
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .config import MODE_SETTINGS, TMP_OUTPUT_PATH
from .utils.image import calculate_valid_vision_tokens
from .utils.model import load_model

logger = logging.getLogger(__name__)

# Path for blank image (used in text-only baseline)
TMP_BLANK_IMAGE = Path("/tmp/blank_32x32.png")


def _ensure_blank_image() -> str:
    """Ensure a blank image exists for text-only baseline.

    Returns:
        Path to the blank image.
    """
    if not TMP_BLANK_IMAGE.exists():
        img = Image.new("RGB", (32, 32), color=(30, 30, 30))
        img.save(TMP_BLANK_IMAGE)
    return str(TMP_BLANK_IMAGE)


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

    output_tokens = len(tokenizer.encode(output, add_special_tokens=False))

    # Clear CUDA cache to prevent OOM in long-running experiments
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output, vision_tokens, output_tokens


class VisionEncoder:
    """Vision encoder wrapper for DeepSeek-OCR.

    Provides a class-based interface for vision encoding that matches
    the pattern used by other baselines (e.g., EmbeddingMeanPooler).
    """

    def __init__(
        self,
        model: AutoModel | None = None,
        tokenizer: AutoTokenizer | None = None,
        mode: str = "large",
    ):
        """Initialize vision encoder.

        Args:
            model: Pre-loaded model (optional).
            tokenizer: Pre-loaded tokenizer (optional).
            mode: Resolution mode (tiny, small, base, large, gundam).
        """
        if model is None or tokenizer is None:
            model, tokenizer = load_model()

        self.model = model
        self.tokenizer = tokenizer
        self.mode = mode

        settings = MODE_SETTINGS.get(mode)
        if settings is None:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: {list(MODE_SETTINGS.keys())}")
        self.settings = settings

    def encode_and_generate(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> tuple[str, int]:
        """Encode image and generate response.

        Args:
            image_path: Path to the rendered text image.
            prompt: The question/prompt to answer.
            max_new_tokens: Maximum tokens to generate (not used by infer API).

        Returns:
            Tuple of (generated_text, vision_tokens).
        """
        output, vision_tokens, _ = run_inference(
            prompt=prompt,
            image_path=image_path,
            mode=self.mode,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        return output, vision_tokens

    def get_vision_tokens(self, image_path: str | Path) -> int:
        """Calculate vision tokens for an image without running inference.

        Args:
            image_path: Path to the image.

        Returns:
            Number of vision tokens.
        """
        with Image.open(image_path) as img:
            width, height = img.size
            return calculate_valid_vision_tokens(width, height, self.settings)
