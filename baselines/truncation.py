"""Truncation baseline for context compression.

This module implements simple truncation baselines that keep either
the first N or last N tokens from the context.
"""

import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def truncate_text(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    from_end: bool = False,
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

    truncated_tokens = tokens[-max_tokens:] if from_end else tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


def truncate_text_first_n(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> str:
    """Keep only the first N tokens of text.

    This simulates what happens when context is too long and gets truncated
    from the end, losing information about the latter parts of the document.

    Args:
        text: The text to truncate.
        tokenizer: The tokenizer to use.
        max_tokens: Number of tokens to keep.

    Returns:
        Truncated text.
    """
    return truncate_text(text, tokenizer, max_tokens, from_end=False)


def truncate_text_last_n(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> str:
    """Keep only the last N tokens of text.

    This simulates recency-biased truncation which Lee et al. found
    performs well for language modeling tasks.

    Args:
        text: The text to truncate.
        tokenizer: The tokenizer to use.
        max_tokens: Number of tokens to keep.

    Returns:
        Truncated text.
    """
    return truncate_text(text, tokenizer, max_tokens, from_end=True)


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Count the number of tokens in text.

    Args:
        text: The text to count tokens for.
        tokenizer: The tokenizer to use.

    Returns:
        Number of tokens.
    """
    return len(tokenizer.encode(text, add_special_tokens=False))
