"""QuALITY long-document experiment: Vision tokens vs text tokens for context compression."""

import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from .utils import MODE_SETTINGS, load_model, render_text_to_image

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Overhead estimate: ~100 tokens for question formatting, options list, and prompt template
# Breakdown: "Question: " (~3) + question (~20-40) + "Options:" (~2) + 4 options (~40) + "Answer..." (~5)
PROMPT_OVERHEAD_TOKENS = 100


def run_text_condition(context: str, question: str, options: list) -> dict:
    """Run QA with text context."""
    model, tokenizer = load_model()

    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = f"""<image>
{context}

Question: {question}

Options:
{options_text}

Answer with just the option number (0, 1, 2, or 3):"""

    # Use minimal blank image
    blank_path = "/tmp/blank_32x32.png"
    Image.new("RGB", (32, 32), color="white").save(blank_path)

    context_tokens = len(tokenizer.encode(context, add_special_tokens=False))

    output = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=blank_path,
        output_path="/tmp/quality_text_output",
        base_size=512,
        image_size=512,
        crop_mode=False,
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )

    return {
        "answer": output.strip(),
        "context_tokens": context_tokens,
        "total_tokens": context_tokens + PROMPT_OVERHEAD_TOKENS,
    }


def run_vision_condition(
    image_path: str, question: str, options: list, mode: str = "small"
) -> dict:
    """Run QA with vision context (rendered text image)."""
    model, tokenizer = load_model()
    settings = MODE_SETTINGS.get(mode, MODE_SETTINGS["small"])

    options_text = "\n".join(f"{i}. {opt}" for i, opt in enumerate(options))
    prompt = f"""<image>

Question: {question}

Options:
{options_text}

Answer with just the option number (0, 1, 2, or 3):"""

    output = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path="/tmp/quality_vision_output",
        base_size=settings["base_size"],
        image_size=settings["image_size"],
        crop_mode=False,
        save_results=False,
        test_compress=True,
        eval_mode=True,
    )

    return {
        "answer": output.strip(),
        "vision_tokens": settings["vision_tokens"],
        "total_tokens": settings["vision_tokens"] + PROMPT_OVERHEAD_TOKENS,
    }


def parse_answer(response: str) -> int:
    """Extract answer number from response."""
    response = response.strip()
    for char in response:
        if char in "0123":
            return int(char)
    return -1


def run_quality_experiment(
    mode: str = "small", num_articles: int = 10, questions_per_article: int = 5
):
    """Run experiment on QuALITY dataset."""
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "quality_data"
    data_dir.mkdir(exist_ok=True)
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading QuALITY dataset...")
    try:
        ds = load_dataset("emozilla/quality", split="validation")
    except Exception as e:
        print(f"Error loading QuALITY dataset: {e}", file=sys.stderr)
        print(
            "Please check your internet connection and HuggingFace access.",
            file=sys.stderr,
        )
        print(
            "Dataset: https://huggingface.co/datasets/emozilla/quality", file=sys.stderr
        )
        return None

    # Group questions by article
    articles = {}
    for item in ds:
        article_hash = hashlib.md5(item["article"][:100].encode()).hexdigest()[:8]
        if article_hash not in articles:
            articles[article_hash] = {"article": item["article"], "questions": []}
        articles[article_hash]["questions"].append(
            {
                "question": item["question"],
                "options": item["options"],
                "answer": item["answer"],
                "hard": item["hard"],
            }
        )

    print(f"Found {len(articles)} unique articles")

    article_items = list(articles.items())[:num_articles]

    results = {"mode": mode, "articles": {}, "summary": {}}
    stats = {
        "text_correct": 0,
        "vision_correct": 0,
        "text_tokens": 0,
        "vision_tokens": 0,
        "total": 0,
    }

    print(f"\n{'=' * 70}")
    print("QUALITY EXPERIMENT: VISION vs TEXT TOKENS")
    print(
        f"Mode: {mode}, Articles: {num_articles}, Questions/article: {questions_per_article}"
    )
    print("=" * 70)

    for article_hash, article_data in article_items:
        article = article_data["article"]
        questions = article_data["questions"][:questions_per_article]

        word_count = len(article.split())
        print(f"\n{'─' * 70}")
        print(f"Article: {article_hash} ({word_count} words)")
        print(f"{'─' * 70}")

        # Render article to image (dark mode, small font for density)
        img_path = data_dir / f"{article_hash}.png"
        if not img_path.exists():
            w, h, lines = render_text_to_image(article, str(img_path))
            print(f"  Rendered: {w}x{h}px, {lines} lines")

        article_results = {"questions": [], "text_correct": 0, "vision_correct": 0}

        for qa in questions:
            question = qa["question"]
            options = qa["options"]
            expected = qa["answer"]

            print(f"\n  Q: {question[:70]}...")
            print(f"  Expected: {expected} ({options[expected][:40]}...)")

            text_result = run_text_condition(article, question, options)
            vision_result = run_vision_condition(str(img_path), question, options, mode)

            text_pred = parse_answer(text_result["answer"])
            vision_pred = parse_answer(vision_result["answer"])

            text_correct = text_pred == expected
            vision_correct = vision_pred == expected

            print(
                f"  Text [{text_result['total_tokens']} tok]: {text_pred} {'✓' if text_correct else '✗'}"
            )
            print(
                f"  Vision [{vision_result['total_tokens']} tok]: {vision_pred} {'✓' if vision_correct else '✗'}"
            )

            article_results["questions"].append(
                {
                    "question": question,
                    "expected": expected,
                    "text": {
                        "pred": text_pred,
                        "correct": text_correct,
                        "tokens": text_result["total_tokens"],
                    },
                    "vision": {
                        "pred": vision_pred,
                        "correct": vision_correct,
                        "tokens": vision_result["total_tokens"],
                    },
                }
            )

            if text_correct:
                article_results["text_correct"] += 1
                stats["text_correct"] += 1
            if vision_correct:
                article_results["vision_correct"] += 1
                stats["vision_correct"] += 1

            stats["text_tokens"] += text_result["total_tokens"]
            stats["vision_tokens"] += vision_result["total_tokens"]
            stats["total"] += 1

        n = len(questions)
        article_results["text_accuracy"] = (
            round(article_results["text_correct"] / n * 100, 1) if n > 0 else 0
        )
        article_results["vision_accuracy"] = (
            round(article_results["vision_correct"] / n * 100, 1) if n > 0 else 0
        )
        results["articles"][article_hash] = article_results
        print(
            f"\n  Article Summary: Text {article_results['text_correct']}/{n}, Vision {article_results['vision_correct']}/{n}"
        )

    # Summary
    n = stats["total"]
    text_acc = round(stats["text_correct"] / n * 100, 1) if n > 0 else 0
    vision_acc = round(stats["vision_correct"] / n * 100, 1) if n > 0 else 0
    compression = (
        stats["text_tokens"] / stats["vision_tokens"]
        if stats["vision_tokens"] > 0
        else 0
    )

    results["summary"] = {
        "total_questions": n,
        "text_correct": stats["text_correct"],
        "vision_correct": stats["vision_correct"],
        "text_accuracy": text_acc,
        "vision_accuracy": vision_acc,
        "text_tokens": stats["text_tokens"],
        "vision_tokens": stats["vision_tokens"],
        "compression_ratio": round(compression, 2),
    }

    output_path = results_dir / f"quality_experiment_{mode}_{num_articles}articles.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nQuestions: {n}")
    print("\nAccuracy:")
    print(f"  Text condition:   {text_acc}% ({stats['text_correct']}/{n})")
    print(f"  Vision condition: {vision_acc}% ({stats['vision_correct']}/{n})")
    print("\nToken Efficiency:")
    print(f"  Text tokens:   {stats['text_tokens']}")
    print(f"  Vision tokens: {stats['vision_tokens']}")
    print(f"  Compression:   {compression:.1f}x")
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="base", choices=["tiny", "small", "base", "large"]
    )
    parser.add_argument("--num-articles", type=int, default=10)
    parser.add_argument("--questions-per-article", type=int, default=5)
    args = parser.parse_args()
    run_quality_experiment(args.mode, args.num_articles, args.questions_per_article)
