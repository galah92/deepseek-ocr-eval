"""Model loading and management utilities.

This module handles loading the DeepSeek-OCR model and provides
device management utilities.
"""

import logging

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# Global model cache
_model: AutoModel | None = None
_tokenizer: AutoTokenizer | None = None


def get_device() -> str:
    """Get the appropriate device (cuda or cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple[AutoModel, AutoTokenizer]:
    """Load the DeepSeek-OCR model (cached after first load).

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

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
    device = get_device()
    logger.info(f"Using device: {device}")
    _model = _model.eval().to(device)
    return _model, _tokenizer


def count_parameters(model: AutoModel) -> dict:
    """Count model parameters (total, trainable, frozen).

    Args:
        model: The model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": (trainable / total * 100) if total > 0 else 0,
    }


def freeze_vision_encoder(model: AutoModel) -> None:
    """Freeze the vision encoder parameters.

    Args:
        model: The DeepSeek-OCR model
    """
    if hasattr(model, "vision_encoder"):
        for param in model.vision_encoder.parameters():
            param.requires_grad = False


def unfreeze_decoder(model: AutoModel) -> None:
    """Unfreeze the language decoder parameters.

    Args:
        model: The DeepSeek-OCR model
    """
    if hasattr(model, "model"):
        for param in model.model.parameters():
            param.requires_grad = True
