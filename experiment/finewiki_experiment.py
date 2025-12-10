"""FineWiki language modeling experiment: Vision tokens vs text tokens for context.

This experiment matches the methodology from Lee et al. (2024) "Optical Context Compression
Is Just (Bad) Autoencoding" which uses FineWiki for perplexity evaluation.

Methodology:
1. Load Wikipedia articles from FineWiki dataset
2. Split each article into context + continuation
3. Text condition: Use raw text as context, predict continuation
4. Vision condition: Render context as image, use vision tokens, predict continuation
5. Measure prediction accuracy / perplexity proxy
"""

import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
import random

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
    return _model, _tokenizer


def render_text_to_image(text: str, output_path: str, font_size: int = 12,
                         max_width: int = 1200, padding: int = 30, line_spacing: int = 4,
                         bg_color: str = '#1e1e1e', fg_color: str = '#d4d4d4') -> tuple:
    """Render text to image (dark mode for optimal OCR). Returns (width, height, num_lines)."""
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


def run_text_condition(context: str, continuation: str) -> dict:
    """Run language modeling with text context - predict next words."""
    model, tokenizer = load_model()

    # Ask model to complete the text
    prompt = f"""<image>
{context}

Continue this text with the next sentence:"""

    # Use minimal blank image
    blank_path = "/tmp/blank_32x32.png"
    Image.new('RGB', (32, 32), color='white').save(blank_path)

    context_tokens = len(tokenizer.encode(context, add_special_tokens=False))

    output = model.infer(
        tokenizer, prompt=prompt, image_file=blank_path,
        output_path='/tmp/finewiki_text_output', base_size=512, image_size=512,
        crop_mode=False, save_results=False, test_compress=True, eval_mode=True
    )

    return {
        'prediction': output.strip(),
        'context_tokens': context_tokens,
        'total_tokens': context_tokens + 50,
    }


def run_vision_condition(image_path: str, continuation: str, mode: str = 'small') -> dict:
    """Run language modeling with vision context - predict next words."""
    model, tokenizer = load_model()
    settings = MODE_SETTINGS.get(mode, MODE_SETTINGS['small'])

    prompt = """<image>

Continue this text with the next sentence:"""

    output = model.infer(
        tokenizer, prompt=prompt, image_file=image_path,
        output_path='/tmp/finewiki_vision_output',
        base_size=settings['base_size'], image_size=settings['image_size'],
        crop_mode=False, save_results=False, test_compress=True, eval_mode=True
    )

    return {
        'prediction': output.strip(),
        'vision_tokens': settings['vision_tokens'],
        'total_tokens': settings['vision_tokens'] + 50,
    }


def compute_overlap_score(prediction: str, target: str) -> float:
    """Compute word overlap score between prediction and target."""
    pred_words = set(prediction.lower().split())
    target_words = set(target.lower().split())

    if not target_words:
        return 0.0

    overlap = len(pred_words & target_words)
    return overlap / len(target_words)


def compute_first_word_match(prediction: str, target: str) -> bool:
    """Check if the first word of prediction matches target."""
    pred_words = prediction.strip().split()
    target_words = target.strip().split()

    if not pred_words or not target_words:
        return False

    return pred_words[0].lower().strip('.,!?;:') == target_words[0].lower().strip('.,!?;:')


def run_finewiki_experiment(mode: str = 'small', num_articles: int = 20,
                            context_words: int = 500, continuation_words: int = 50):
    """Run experiment on FineWiki dataset."""
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "finewiki_data"
    data_dir.mkdir(exist_ok=True)
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading FineWiki dataset (English)...")
    ds = load_dataset('HuggingFaceFW/finewiki', name='en', split='train', streaming=True)

    # Get articles with sufficient length
    articles = []
    min_words = context_words + continuation_words + 100

    print(f"Finding {num_articles} articles with {min_words}+ words...")
    for item in ds:
        text = item['text']
        words = text.split()
        if len(words) >= min_words:
            articles.append({
                'title': item.get('title', 'Untitled'),
                'text': text,
                'word_count': len(words)
            })
            print(f"  Found: {item.get('title', 'Untitled')[:50]}... ({len(words)} words)")
        if len(articles) >= num_articles:
            break

    print(f"\nCollected {len(articles)} articles")

    results = {'mode': mode, 'articles': [], 'summary': {}}
    stats = {
        'text_overlap_sum': 0, 'vision_overlap_sum': 0,
        'text_first_word': 0, 'vision_first_word': 0,
        'text_tokens': 0, 'vision_tokens': 0, 'total': 0
    }

    print(f"\n{'='*70}")
    print(f"FINEWIKI EXPERIMENT: VISION vs TEXT TOKENS")
    print(f"Mode: {mode}, Articles: {num_articles}")
    print(f"Context: {context_words} words, Continuation: {continuation_words} words")
    print('='*70)

    for i, article in enumerate(articles):
        words = article['text'].split()

        # Split into context and continuation
        context = ' '.join(words[:context_words])
        continuation = ' '.join(words[context_words:context_words + continuation_words])

        print(f"\n{'─'*70}")
        print(f"Article {i+1}: {article['title'][:50]}...")
        print(f"{'─'*70}")
        print(f"  Context: {context[:100]}...")
        print(f"  Target continuation: {continuation[:100]}...")

        # Render context to image
        img_path = data_dir / f"article_{i}.png"
        if not img_path.exists():
            w, h, lines = render_text_to_image(context, str(img_path))
            print(f"  Rendered: {w}x{h}px, {lines} lines")

        # Run both conditions
        text_result = run_text_condition(context, continuation)
        vision_result = run_vision_condition(str(img_path), continuation, mode)

        # Compute scores
        text_overlap = compute_overlap_score(text_result['prediction'], continuation)
        vision_overlap = compute_overlap_score(vision_result['prediction'], continuation)
        text_first = compute_first_word_match(text_result['prediction'], continuation)
        vision_first = compute_first_word_match(vision_result['prediction'], continuation)

        print(f"\n  Text prediction: {text_result['prediction'][:80]}...")
        print(f"  Vision prediction: {vision_result['prediction'][:80]}...")
        print(f"\n  Text [{text_result['total_tokens']} tok]: overlap={text_overlap:.2f}, first_word={'Y' if text_first else 'N'}")
        print(f"  Vision [{vision_result['total_tokens']} tok]: overlap={vision_overlap:.2f}, first_word={'Y' if vision_first else 'N'}")

        article_result = {
            'title': article['title'],
            'context_preview': context[:200],
            'continuation': continuation,
            'text': {
                'prediction': text_result['prediction'],
                'overlap': text_overlap,
                'first_word_match': text_first,
                'tokens': text_result['total_tokens']
            },
            'vision': {
                'prediction': vision_result['prediction'],
                'overlap': vision_overlap,
                'first_word_match': vision_first,
                'tokens': vision_result['total_tokens']
            }
        }
        results['articles'].append(article_result)

        stats['text_overlap_sum'] += text_overlap
        stats['vision_overlap_sum'] += vision_overlap
        stats['text_first_word'] += int(text_first)
        stats['vision_first_word'] += int(vision_first)
        stats['text_tokens'] += text_result['total_tokens']
        stats['vision_tokens'] += vision_result['total_tokens']
        stats['total'] += 1

    # Summary
    n = stats['total']
    if n > 0:
        text_avg_overlap = stats['text_overlap_sum'] / n
        vision_avg_overlap = stats['vision_overlap_sum'] / n
        text_first_rate = stats['text_first_word'] / n
        vision_first_rate = stats['vision_first_word'] / n
        compression = stats['text_tokens'] / stats['vision_tokens'] if stats['vision_tokens'] > 0 else 0

        results['summary'] = {
            'total_articles': n,
            'text_avg_overlap': round(text_avg_overlap, 3),
            'vision_avg_overlap': round(vision_avg_overlap, 3),
            'text_first_word_rate': round(text_first_rate, 3),
            'vision_first_word_rate': round(vision_first_rate, 3),
            'text_tokens': stats['text_tokens'],
            'vision_tokens': stats['vision_tokens'],
            'compression_ratio': round(compression, 2),
        }

        output_path = results_dir / f"finewiki_experiment_{mode}_{num_articles}articles.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY")
        print('='*70)
        print(f"\nArticles: {n}")
        print(f"\nOverlap Score (higher = better):")
        print(f"  Text condition:   {text_avg_overlap:.3f}")
        print(f"  Vision condition: {vision_avg_overlap:.3f}")
        print(f"\nFirst Word Match Rate:")
        print(f"  Text condition:   {text_first_rate:.1%} ({stats['text_first_word']}/{n})")
        print(f"  Vision condition: {vision_first_rate:.1%} ({stats['vision_first_word']}/{n})")
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
    parser.add_argument('--num-articles', type=int, default=20)
    parser.add_argument('--context-words', type=int, default=500)
    parser.add_argument('--continuation-words', type=int, default=50)
    args = parser.parse_args()
    run_finewiki_experiment(args.mode, args.num_articles, args.context_words, args.continuation_words)
