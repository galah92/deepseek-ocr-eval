"""Text generation utilities.

This module provides helper functions for text generation and
output parsing for DeepSeek-OCR evaluation.
"""

import logging
import re

logger = logging.getLogger(__name__)


def parse_mc_answer(output: str) -> int:
    """Parse multiple-choice answer (0-3) from model output.

    Looks for the first digit 0-3 in the output string.

    Args:
        output: Raw model output string.

    Returns:
        Integer 0-3 if found, -1 if no valid answer detected.
    """
    return next((int(c) for c in output.strip() if c in "0123"), -1)


def clean_output(output: str) -> str:
    """Clean model output by removing special tokens and whitespace.

    Args:
        output: Raw model output string.

    Returns:
        Cleaned output string.
    """
    # Remove common special tokens
    output = re.sub(r"<[^>]+>", "", output)
    # Clean up whitespace
    output = " ".join(output.split())
    return output.strip()


def extract_number(output: str) -> int | None:
    """Extract the first number from model output.

    Args:
        output: Raw model output string.

    Returns:
        First integer found, or None if no number detected.
    """
    match = re.search(r"\d+", output)
    return int(match.group()) if match else None


def format_mc_prompt(
    context: str,
    question: str,
    options: list[str],
    instruction: str = "Answer with just the option number (0, 1, 2, or 3):",
) -> str:
    """Format a multiple-choice question prompt.

    Args:
        context: The context/article text.
        question: The question to ask.
        options: List of answer options.
        instruction: Instruction for the model.

    Returns:
        Formatted prompt string.
    """
    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    return f"{context}\n\nQuestion: {question}\n\nOptions:\n{options_text}\n\n{instruction}"
