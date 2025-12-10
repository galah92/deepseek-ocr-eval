"""
Run experiments across all text-to-image configurations.
Compares fonts, sizes, and color schemes.
"""

import json
from pathlib import Path
from run_experiment import run_experiment

# Configurations to test
CONFIGS = {
    'fonts': ['font_mono', 'font_serif', 'font_sans'],
    'sizes': ['size_small', 'size_medium', 'size_large'],
    'colors': ['color_default', 'color_dark', 'color_sepia', 'color_blue', 'color_low_contrast'],
}


def run_all_experiments(mode: str = 'small', config_type: str = 'all'):
    """Run experiments for all configurations of a given type."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    if config_type == 'all':
        configs_to_run = []
        for configs in CONFIGS.values():
            configs_to_run.extend(configs)
        # Remove duplicates (font_mono = size_medium = color_default)
        configs_to_run = list(set(configs_to_run))
    else:
        configs_to_run = CONFIGS.get(config_type, [])

    all_results = {}

    print("=" * 70)
    print(f"RUNNING EXPERIMENTS ACROSS {len(configs_to_run)} CONFIGURATIONS")
    print(f"Mode: {mode}")
    print("=" * 70)

    for config in configs_to_run:
        print(f"\n>>> Running experiment with config: {config}")
        result = run_experiment(mode=mode, data_config=config)
        if result:
            all_results[config] = result['summary']

    # Generate comparison report
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    # Sort by vision accuracy
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['vision_accuracy'],
        reverse=True
    )

    print(f"\n{'Config':<25} {'Text Acc':>10} {'Vision Acc':>12} {'Compression':>12}")
    print("-" * 60)

    for config, summary in sorted_results:
        print(f"{config:<25} {summary['text_accuracy']:>9.1f}% {summary['vision_accuracy']:>11.1f}% {summary['compression_ratio']:>11.2f}x")

    # Save comparison
    comparison = {
        'mode': mode,
        'configs_tested': configs_to_run,
        'results': all_results,
        'ranking': [{'config': c, **s} for c, s in sorted_results]
    }

    output_path = results_dir / f"comparison_{config_type}_{mode}.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison saved to: {output_path}")
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='small', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--config-type', default='all',
                        choices=['all', 'fonts', 'sizes', 'colors'],
                        help='Which configuration type to test')
    args = parser.parse_args()
    run_all_experiments(args.mode, args.config_type)
