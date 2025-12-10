"""Vision tokens vs text tokens experiment for context compression."""

import json
import re
import time
from pathlib import Path
import torch

_model = None
_tokenizer = None

MODE_SETTINGS = {
    'tiny': {'base_size': 512, 'image_size': 512, 'vision_tokens': 64},
    'small': {'base_size': 640, 'image_size': 640, 'vision_tokens': 100},
    'base': {'base_size': 1024, 'image_size': 1024, 'vision_tokens': 256},
    'large': {'base_size': 1280, 'image_size': 1280, 'vision_tokens': 400},
}


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModel, AutoTokenizer

    model_name = 'deepseek-ai/DeepSeek-OCR'
    print(f"Loading model from {model_name}...")

    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        model_name, attn_implementation='eager',
        trust_remote_code=True, use_safetensors=True
    )
    _model = _model.eval().cuda().to(torch.bfloat16)
    return _model, _tokenizer


def run_text_condition(context: str, question: str) -> dict:
    from PIL import Image

    model, tokenizer = load_model()

    blank_path = "/tmp/blank_1x1.png"
    Image.new('RGB', (32, 32), color='white').save(blank_path)

    prompt = f"<image>\n{context}\n\nQuestion: {question}\nAnswer:"
    context_tokens = len(tokenizer.encode(context, add_special_tokens=False))
    question_tokens = len(tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False))

    output = model.infer(
        tokenizer, prompt=prompt, image_file=blank_path,
        output_path='/tmp/text_output', base_size=512, image_size=512,
        crop_mode=False, save_results=False, test_compress=True, eval_mode=True
    )

    return {
        'answer': output.strip(),
        'context_tokens': context_tokens,
        'question_tokens': question_tokens,
        'total_tokens': context_tokens + question_tokens + 64,
    }


def run_vision_condition(image_path: str, question: str, mode: str = 'small') -> dict:
    model, tokenizer = load_model()
    settings = MODE_SETTINGS.get(mode, MODE_SETTINGS['small'])

    prompt = f"<image>\n\nQuestion: {question}\nAnswer:"
    question_tokens = len(tokenizer.encode(f"Question: {question}\nAnswer:", add_special_tokens=False))

    output = model.infer(
        tokenizer, prompt=prompt, image_file=image_path,
        output_path='/tmp/vision_output',
        base_size=settings['base_size'], image_size=settings['image_size'],
        crop_mode=False, save_results=False, test_compress=True, eval_mode=True
    )

    return {
        'answer': output.strip(),
        'vision_tokens': settings['vision_tokens'],
        'question_tokens': question_tokens,
        'total_tokens': settings['vision_tokens'] + question_tokens,
    }


def check_answer(predicted: str, expected: str) -> bool:
    pred = predicted.lower().strip()
    exp = expected.lower().strip()

    if not pred:
        return False
    if pred == exp or exp in pred:
        return True
    if pred in exp and len(pred) >= 2:
        return True

    pred_clean = re.sub(r'[^\w\s]', '', pred)
    exp_clean = re.sub(r'[^\w\s]', '', exp)
    return pred_clean == exp_clean or exp_clean in pred_clean or (pred_clean in exp_clean and len(pred_clean) >= 2)


def run_experiment(mode: str = 'small', data_config: str = 'default'):
    experiment_dir = Path(__file__).parent
    data_dir = experiment_dir / "data" if data_config == 'default' else experiment_dir / "data" / data_config

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return None

    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    with open(experiment_dir / "questions.json") as f:
        questions_data = json.load(f)

    results = {'mode': mode, 'data_config': data_config, 'documents': {}, 'summary': {}}
    stats = {'text_correct': 0, 'vision_correct': 0, 'text_tokens': 0, 'vision_tokens': 0, 'total': 0}

    print(f"\n{'='*70}")
    print(f"VISION TOKENS VS TEXT TOKENS FOR CONTEXT")
    print(f"Mode: {mode}, Data config: {data_config}")
    print('='*70)

    for doc_name, doc_data in questions_data.items():
        txt_path = data_dir / f"{doc_name}.txt"
        img_path = data_dir / f"{doc_name}.png"

        if not txt_path.exists() or not img_path.exists():
            continue

        context = txt_path.read_text()
        print(f"\n{'─'*70}\nDocument: {doc_name} ({len(context)} chars)\n{'─'*70}")

        doc_results = {'questions': [], 'text_correct': 0, 'vision_correct': 0}

        for qa in doc_data['questions']:
            question, expected = qa['q'], qa['a']
            print(f"\n  Q: {question}\n  Expected: {expected}")

            text_result = run_text_condition(context, question)
            vision_result = run_vision_condition(str(img_path), question, mode)

            text_correct = check_answer(text_result['answer'], expected)
            vision_correct = check_answer(vision_result['answer'], expected)

            print(f"  Text [{text_result['total_tokens']} tok]: {text_result['answer'][:60]} {'✓' if text_correct else '✗'}")
            print(f"  Vision [{vision_result['total_tokens']} tok]: {vision_result['answer'][:60]} {'✓' if vision_correct else '✗'}")

            doc_results['questions'].append({
                'question': question, 'expected': expected,
                'text': {'answer': text_result['answer'], 'correct': text_correct, 'total_tokens': text_result['total_tokens']},
                'vision': {'answer': vision_result['answer'], 'correct': vision_correct, 'total_tokens': vision_result['total_tokens']},
            })

            if text_correct:
                doc_results['text_correct'] += 1
                stats['text_correct'] += 1
            if vision_correct:
                doc_results['vision_correct'] += 1
                stats['vision_correct'] += 1

            stats['text_tokens'] += text_result['total_tokens']
            stats['vision_tokens'] += vision_result['total_tokens']
            stats['total'] += 1

        n = len(doc_data['questions'])
        doc_results['text_accuracy'] = round(doc_results['text_correct'] / n * 100, 1)
        doc_results['vision_accuracy'] = round(doc_results['vision_correct'] / n * 100, 1)
        results['documents'][doc_name] = doc_results
        print(f"\n  Summary: Text {doc_results['text_correct']}/{n}, Vision {doc_results['vision_correct']}/{n}")

    n = stats['total']
    text_acc = round(stats['text_correct'] / n * 100, 1) if n > 0 else 0
    vision_acc = round(stats['vision_correct'] / n * 100, 1) if n > 0 else 0
    compression = stats['text_tokens'] / stats['vision_tokens'] if stats['vision_tokens'] > 0 else 0

    results['summary'] = {
        'total_questions': n,
        'text_correct': stats['text_correct'], 'vision_correct': stats['vision_correct'],
        'text_accuracy': text_acc, 'vision_accuracy': vision_acc,
        'text_tokens': stats['text_tokens'], 'vision_tokens': stats['vision_tokens'],
        'compression_ratio': round(compression, 2),
        'tokens_saved': stats['text_tokens'] - stats['vision_tokens'],
    }

    output_name = f"context_experiment_{mode}_{data_config}.json" if data_config != 'default' else f"context_experiment_{mode}.json"
    output_path = results_dir / output_name
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}\nRESULTS SUMMARY\n{'='*70}")
    print(f"\nQuestions: {n}")
    print(f"\nAccuracy:\n  Text condition:   {text_acc}% ({stats['text_correct']}/{n})")
    print(f"  Vision condition: {vision_acc}% ({stats['vision_correct']}/{n})")
    print(f"\nToken Efficiency:\n  Text tokens:   {stats['text_tokens']}")
    print(f"  Vision tokens: {stats['vision_tokens']}\n  Compression:   {compression:.2f}x")
    print(f"  Tokens saved:  {stats['text_tokens'] - stats['vision_tokens']}")

    print(f"\nConclusion:")
    if vision_acc >= text_acc - 5 and compression > 1:
        print(f"  ✓ HYPOTHESIS SUPPORTED\n    Vision achieves {compression:.1f}x compression with comparable accuracy")
    elif vision_acc >= text_acc - 10:
        print(f"  ~ HYPOTHESIS PARTIALLY SUPPORTED\n    Vision: {vision_acc}%, Text: {text_acc}%")
    else:
        print(f"  ✗ HYPOTHESIS NOT SUPPORTED\n    Vision accuracy ({vision_acc}%) significantly lower than text ({text_acc}%)")

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='small', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--data-config', default='default')
    args = parser.parse_args()
    run_experiment(args.mode, args.data_config)
