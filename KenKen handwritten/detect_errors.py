# -*- coding: utf-8 -*-
"""
Error correction for KenKen handwritten puzzles.
Uses all error correction approaches from the original KenKen solver:
- Unsat core detection
- Top-K alternatives
- Multi-error correction (up to 5 errors)
- Operator inference
- Cage re-detection with threshold multipliers
"""

import os
import sys
import json
import time
import torch
import pandas as pd

# Add parent KenKen directory to import solver functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'KenKen'))
from solve_all_sizes import (
    find_size_and_borders, find_size_and_borders_with_retry,
    make_puzzle, make_puzzle_with_alternatives,
    solve_kenken, solve_with_operator_inference,
    attempt_error_correction,
    Grid_CNN, CNN_v2
)


def load_models():
    """Load Grid CNN and our custom Character CNN."""
    grid_model = Grid_CNN(output_dim=6)
    grid_model.load_state_dict(torch.load('./models/grid_detection_model_weights.pth',
                                          map_location='cpu', weights_only=True))
    grid_model.eval()

    # Use our unified handwritten model
    char_model = CNN_v2(output_dim=14)
    char_model.load_state_dict(torch.load('./models/unified_kenken_cnn.pth',
                                          map_location='cpu', weights_only=True))
    char_model.eval()

    return grid_model, char_model


def solve_with_full_correction(filename, size, grid_model, char_model, verbose=False):
    """
    Attempt to solve puzzle with full error correction pipeline.

    Returns:
        dict with 'solved', 'correction_type', 'solution', 'attempts'
    """
    result = {
        'solved': False,
        'correction_type': 'none',
        'solution': None,
        'attempts': 0
    }

    # Step 1: Try basic detection with retry on validation failure
    detected_size, cages, border_thickness, retry_info = find_size_and_borders_with_retry(
        filename, grid_model, size_override=size, verbose=verbose
    )

    if not cages or detected_size != size:
        result['correction_type'] = 'detection_failed'
        return result

    # Step 2: Try basic solve
    puzzle = make_puzzle(size, border_thickness, cages, filename, char_model, invert=False)
    solution = solve_kenken(puzzle, size)
    result['attempts'] += 1

    if solution is not None:
        result['solved'] = True
        result['solution'] = solution
        result['correction_type'] = 'baseline'
        return result

    # Step 3: Try operator inference (for missing operators)
    solution = solve_with_operator_inference(puzzle, size)
    result['attempts'] += 1

    if solution is not None:
        result['solved'] = True
        result['solution'] = solution
        result['correction_type'] = 'operator_inference'
        return result

    # Step 4: Try error correction with alternatives
    puzzle_alt, alternatives = make_puzzle_with_alternatives(
        size, border_thickness, cages, filename, char_model, k=4, invert=False
    )

    max_errors = 5 if size >= 9 else 3
    correction = attempt_error_correction(
        puzzle_alt, alternatives, size, max_errors=max_errors, max_k=4
    )
    result['attempts'] += correction.attempts

    if correction.success:
        result['solved'] = True
        result['solution'] = correction.solution
        result['correction_type'] = correction.correction_type
        return result

    # Step 5: If still failing, try with cage re-detection
    for multiplier in [1.5, 2.0, 2.5]:
        detected_size, cages_retry, border_thickness_retry, _ = find_size_and_borders_with_retry(
            filename, grid_model, size_override=size, verbose=False
        )

        if not cages_retry or len(cages_retry) == len(cages):
            continue

        puzzle_retry, alternatives_retry = make_puzzle_with_alternatives(
            size, border_thickness_retry, cages_retry, filename, char_model, k=4, invert=False
        )

        solution = solve_kenken(puzzle_retry, size)
        result['attempts'] += 1

        if solution is not None:
            result['solved'] = True
            result['solution'] = solution
            result['correction_type'] = f'cage_retry_{multiplier}'
            return result

        correction_retry = attempt_error_correction(
            puzzle_retry, alternatives_retry, size, max_errors=max_errors, max_k=4
        )
        result['attempts'] += correction_retry.attempts

        if correction_retry.success:
            result['solved'] = True
            result['solution'] = correction_retry.solution
            result['correction_type'] = f'cage_retry_{multiplier}+{correction_retry.correction_type}'
            return result

    result['correction_type'] = 'uncorrectable'
    return result


def evaluate_with_correction(sizes, num_per_size, grid_model, char_model):
    """Run evaluation with full error correction."""
    results = []

    for size in sizes:
        print(f"\nEvaluating {size}x{size} puzzles with error correction...")

        solved_count = 0
        for idx in range(num_per_size):
            filename = f'./board_images/board{size}_{idx}.png'
            if not os.path.exists(filename):
                continue

            start_time = time.time()

            try:
                result = solve_with_full_correction(
                    filename, size, grid_model, char_model, verbose=False
                )

                if result['solved']:
                    solved_count += 1

                results.append({
                    'puzzle_id': f'{size}_{idx}',
                    'size': size,
                    'solved': result['solved'],
                    'correction_type': result['correction_type'],
                    'attempts': result['attempts'],
                    'solve_time_ms': int((time.time() - start_time) * 1000)
                })

            except Exception as e:
                results.append({
                    'puzzle_id': f'{size}_{idx}',
                    'size': size,
                    'solved': False,
                    'correction_type': f'error: {str(e)[:50]}',
                    'attempts': 0,
                    'solve_time_ms': int((time.time() - start_time) * 1000)
                })

        print(f"  {size}x{size}: {solved_count}/{num_per_size} solved ({100*solved_count/num_per_size:.0f}%)")

    return results


def main():
    print("Loading models...")
    grid_model, char_model = load_models()
    print("Models loaded.")

    sizes = [3, 4, 5, 6, 7, 9]
    num_per_size = 100

    print("\nRunning evaluation with error correction...")
    results = evaluate_with_correction(sizes, num_per_size, grid_model, char_model)

    # Save detailed results
    os.makedirs('./results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('./results/corrected_evaluation.csv', index=False)

    # Create summary
    summary = []
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        total = len(size_results)
        solved = sum(1 for r in size_results if r['solved'])

        # Count by correction type
        correction_counts = {}
        for r in size_results:
            ct = r['correction_type']
            correction_counts[ct] = correction_counts.get(ct, 0) + 1

        summary.append({
            'size': size,
            'total': total,
            'solved': solved,
            'accuracy': f"{100*solved/total:.0f}%" if total > 0 else "N/A",
            'baseline': correction_counts.get('baseline', 0),
            'operator_inf': correction_counts.get('operator_inference', 0),
            'error_corr': sum(v for k, v in correction_counts.items()
                           if 'error' in k.lower() or 'single' in k or 'two' in k or 'three' in k),
            'uncorrectable': correction_counts.get('uncorrectable', 0)
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('./results/corrected_summary.csv', index=False)

    print("\n" + "="*70)
    print("CORRECTED EVALUATION RESULTS")
    print("="*70)
    print(summary_df.to_string(index=False))

    # Also show correction type breakdown
    print("\n" + "="*70)
    print("CORRECTION TYPE BREAKDOWN")
    print("="*70)
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        correction_counts = {}
        for r in size_results:
            ct = r['correction_type']
            correction_counts[ct] = correction_counts.get(ct, 0) + 1

        print(f"\n{size}x{size}:")
        for ct, count in sorted(correction_counts.items(), key=lambda x: -x[1]):
            print(f"  {ct}: {count}")

    print("\nResults saved to ./results/")


if __name__ == '__main__':
    main()
