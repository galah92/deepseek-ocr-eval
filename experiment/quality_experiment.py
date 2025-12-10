"""QuALITY long-document experiment: Vision tokens vs text tokens for context compression."""

import json
import hashlib
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

_model = None
_tokenizer = None

MODE_SETTINGS = {
    'tiny': {'base_size': 512, 'image_size': 512, 'vision_tokens': 64},
    'small': {'base_size': 640, 'image_size': 640, 'vision_tokens': 100},
    'base': {'base_size': 1024, 'image_size': 1024, 'vision_tokens': 256},
    'large': {'base_size': 1280, 'image_size': 1280, 'vision_tokens': 400},
}

import platform

def _get_mono_font_paths():
    """Get monospace font paths based on the current platform."""
    system = platform.system()
    if system == 'Linux':
        return [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        ]
    elif system == 'Darwin':  # macOS
        return [
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Courier New.ttf",
        ]
    elif system == 'Windows':
        return [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "C:/Windows/Fonts/lucon.ttf",
        ]
    return []

FONT_PATHS = _get_mono_font_paths()


def get_font(size: int = 14):
    """Load a monospace font of the specified size, with fallback to default.

    Args:
        size: Font size in points

    Returns:
        PIL ImageFont object
    """
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            # Font file not found or cannot be read
            continue
        except IOError:
            # I/O error reading font file
            continue

    print(f"Warning: No monospace font found, using default font", file=sys.stderr)
    return ImageFont.load_default()


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = 'deepseek-ai/DeepSeek-OCR'
    print(f"Loading model from {model_name}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        model_name, attn_implementation='eager',
        trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16
    )
    _model = _model.eval().cuda()
    # Enable torch compile for faster inference
    # _model = torch.compile(_model, mode="reduce-overhead")
    return _model, _tokenizer


def render_long_text_to_image(text: str, output_path: str, font_size: int = 12,
                               max_width: int = 1200, max_height: int = 1600,
                               padding: int = 30, line_spacing: int = 4,
                               bg_color: str = '#1e1e1e', fg_color: str = '#d4d4d4') -> tuple:
    """Render long text to a single tall image. Returns (width, height, num_lines)."""
    font = get_font(font_size)

    # Wrap text to fit width
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
        words = paragraph.split()
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = font.getbbox(test_line)
            if bbox[2] > max_width - 2 * padding:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(' '.join(current_line))

    line_height = font_size + line_spacing
    img_height = len(lines) * line_height + 2 * padding
    img_width = max_width

    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, font=font, fill=fg_color)
        y += line_height

    img.save(output_path)
    return img_width, img_height, len(lines)


def run_text_condition(context: str, question: str, options: list) -> dict:
    """Run QA with text context."""
    model, tokenizer = load_model()

    # Format multiple choice question
    options_text = '\n'.join(f'{i}. {opt}' for i, opt in enumerate(options))
    prompt = f"""<image>
{context}

Question: {question}

Options:
{options_text}

Answer with just the option number (0, 1, 2, or 3):"""

    # Use minimal blank image
    blank_path = "/tmp/blank_32x32.png"
    Image.new('RGB', (32, 32), color='white').save(blank_path)

    context_tokens = len(tokenizer.encode(context, add_special_tokens=False))

    output = model.infer(
        tokenizer, prompt=prompt, image_file=blank_path,
        output_path='/tmp/quality_text_output', base_size=512, image_size=512,
        crop_mode=False, save_results=False, test_compress=True, eval_mode=True
    )

    return {
        'answer': output.strip(),
        'context_tokens': context_tokens,
        'total_tokens': context_tokens + 100,  # question + options overhead
    }


def run_vision_condition(image_path: str, question: str, options: list, mode: str = 'small') -> dict:
    """Run QA with vision context (rendered text image)."""
    model, tokenizer = load_model()
    settings = MODE_SETTINGS.get(mode, MODE_SETTINGS['small'])

    options_text = '\n'.join(f'{i}. {opt}' for i, opt in enumerate(options))
    prompt = f"""<image>

Question: {question}

Options:
{options_text}

Answer with just the option number (0, 1, 2, or 3):"""

    output = model.infer(
        tokenizer, prompt=prompt, image_file=image_path,
        output_path='/tmp/quality_vision_output',
        base_size=settings['base_size'], image_size=settings['image_size'],
        crop_mode=False, save_results=False, test_compress=True, eval_mode=True
    )

    return {
        'answer': output.strip(),
        'vision_tokens': settings['vision_tokens'],
        'total_tokens': settings['vision_tokens'] + 100,
    }


def parse_answer(response: str) -> int:
    """Extract answer number from response."""
    response = response.strip()
    # Try to find a digit 0-3
    for char in response:
        if char in '0123':
            return int(char)
    return -1


def run_quality_experiment(mode: str = 'small', num_articles: int = 10, questions_per_article: int = 5):
    """Run experiment on QuALITY dataset."""
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "quality_data"
    data_dir.mkdir(exist_ok=True)
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading QuALITY dataset...")
    ds = load_dataset('emozilla/quality', split='validation')

    # Group questions by article
    articles = {}
    for item in ds:
        article_hash = hashlib.md5(item['article'][:100].encode()).hexdigest()[:8]
        if article_hash not in articles:
            articles[article_hash] = {
                'article': item['article'],
                'questions': []
            }
        articles[article_hash]['questions'].append({
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer'],
            'hard': item['hard']
        })

    print(f"Found {len(articles)} unique articles")

    # Select articles
    article_items = list(articles.items())[:num_articles]

    results = {'mode': mode, 'articles': {}, 'summary': {}}
    stats = {
        'text_correct': 0, 'vision_correct': 0,
        'text_tokens': 0, 'vision_tokens': 0, 'total': 0
    }

    print(f"\n{'='*70}")
    print(f"QUALITY EXPERIMENT: VISION vs TEXT TOKENS")
    print(f"Mode: {mode}, Articles: {num_articles}, Questions/article: {questions_per_article}")
    print('='*70)

    for article_hash, article_data in article_items:
        article = article_data['article']
        questions = article_data['questions'][:questions_per_article]

        word_count = len(article.split())
        print(f"\n{'─'*70}")
        print(f"Article: {article_hash} ({word_count} words)")
        print(f"{'─'*70}")

        # Render article to image (dark mode, small font for density)
        img_path = data_dir / f"{article_hash}.png"
        if not img_path.exists():
            w, h, lines = render_long_text_to_image(article, str(img_path))
            print(f"  Rendered: {w}x{h}px, {lines} lines")

        article_results = {'questions': [], 'text_correct': 0, 'vision_correct': 0}

        for qa in questions:
            question = qa['question']
            options = qa['options']
            expected = qa['answer']

            print(f"\n  Q: {question[:70]}...")
            print(f"  Expected: {expected} ({options[expected][:40]}...)")

            # Run both conditions
            text_result = run_text_condition(article, question, options)
            vision_result = run_vision_condition(str(img_path), question, options, mode)

            text_pred = parse_answer(text_result['answer'])
            vision_pred = parse_answer(vision_result['answer'])

            text_correct = text_pred == expected
            vision_correct = vision_pred == expected

            print(f"  Text [{text_result['total_tokens']} tok]: {text_pred} {'✓' if text_correct else '✗'}")
            print(f"  Vision [{vision_result['total_tokens']} tok]: {vision_pred} {'✓' if vision_correct else '✗'}")

            article_results['questions'].append({
                'question': question,
                'expected': expected,
                'text': {'pred': text_pred, 'correct': text_correct, 'tokens': text_result['total_tokens']},
                'vision': {'pred': vision_pred, 'correct': vision_correct, 'tokens': vision_result['total_tokens']},
            })

            if text_correct:
                article_results['text_correct'] += 1
                stats['text_correct'] += 1
            if vision_correct:
                article_results['vision_correct'] += 1
                stats['vision_correct'] += 1

            stats['text_tokens'] += text_result['total_tokens']
            stats['vision_tokens'] += vision_result['total_tokens']
            stats['total'] += 1

        n = len(questions)
        article_results['text_accuracy'] = round(article_results['text_correct'] / n * 100, 1)
        article_results['vision_accuracy'] = round(article_results['vision_correct'] / n * 100, 1)
        results['articles'][article_hash] = article_results
        print(f"\n  Article Summary: Text {article_results['text_correct']}/{n}, Vision {article_results['vision_correct']}/{n}")

    # Summary
    n = stats['total']
    text_acc = round(stats['text_correct'] / n * 100, 1) if n > 0 else 0
    vision_acc = round(stats['vision_correct'] / n * 100, 1) if n > 0 else 0
    compression = stats['text_tokens'] / stats['vision_tokens'] if stats['vision_tokens'] > 0 else 0

    results['summary'] = {
        'total_questions': n,
        'text_correct': stats['text_correct'],
        'vision_correct': stats['vision_correct'],
        'text_accuracy': text_acc,
        'vision_accuracy': vision_acc,
        'text_tokens': stats['text_tokens'],
        'vision_tokens': stats['vision_tokens'],
        'compression_ratio': round(compression, 2),
    }

    output_path = results_dir / f"quality_experiment_{mode}_{num_articles}articles.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print('='*70)
    print(f"\nQuestions: {n}")
    print(f"\nAccuracy:")
    print(f"  Text condition:   {text_acc}% ({stats['text_correct']}/{n})")
    print(f"  Vision condition: {vision_acc}% ({stats['vision_correct']}/{n})")
    print(f"\nToken Efficiency:")
    print(f"  Text tokens:   {stats['text_tokens']}")
    print(f"  Vision tokens: {stats['vision_tokens']}")
    print(f"  Compression:   {compression:.1f}x")
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='base', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--num-articles', type=int, default=10)
    parser.add_argument('--questions-per-article', type=int, default=5)
    args = parser.parse_args()
    run_quality_experiment(args.mode, args.num_articles, args.questions_per_article)
