# -*- coding: utf-8 -*-
"""
Error correction evaluation for KenKen handwritten puzzles with 300px cells.

Uses the solver from KenKen-300px with the handwritten CNN model.
"""

import os
import sys
import time
import torch
import pandas as pd
from collections import Counter

# Add KenKen-300px directory to import solver functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'KenKen-300px'))
from solve_all_sizes import (
    find_size_and_borders, find_size_and_borders_with_retry,
    make_puzzle, make_puzzle_with_alternatives,
    solve_kenken, solve_with_operator_inference,
    attempt_error_correction, attempt_constraint_based_correction,
    Grid_CNN, CNN_v2, CELL_SIZE
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
    solution, inferences = solve_with_operator_inference(puzzle, size)
    result['attempts'] += 1

    if solution is not None:
        result['solved'] = True
        result['solution'] = solution
        result['correction_type'] = 'operator_inference'
        return result

    # Step 4: Try constraint-based error correction with higher k
    puzzle_alt, alternatives = make_puzzle_with_alternatives(
        size, border_thickness, cages, filename, char_model, k=8, invert=False
    )

    # Use constraint-based correction which prioritizes impossible targets
    correction = attempt_constraint_based_correction(
        puzzle_alt, alternatives, size, max_errors=None, max_k=8
    )
    result['attempts'] += correction.correction_attempts

    if correction.success:
        result['solved'] = True
        result['solution'] = correction.solution
        result['correction_type'] = f'constraint_{correction.correction_type}'
        return result

    # Step 5: If still failing, try with cage re-detection
    for multiplier in [1.5, 2.0, 2.5]:
        detected_size, cages_retry, border_thickness_retry, _ = find_size_and_borders_with_retry(
            filename, grid_model, size_override=size, verbose=False
        )

        if not cages_retry or len(cages_retry) == len(cages):
            continue

        puzzle_retry, alternatives_retry = make_puzzle_with_alternatives(
            size, border_thickness_retry, cages_retry, filename, char_model, k=8, invert=False
        )

        solution = solve_kenken(puzzle_retry, size)
        result['attempts'] += 1

        if solution is not None:
            result['solved'] = True
            result['solution'] = solution
            result['correction_type'] = f'cage_retry_{multiplier}'
            return result

        correction_retry = attempt_constraint_based_correction(
            puzzle_retry, alternatives_retry, size, max_errors=None, max_k=8
        )
        result['attempts'] += correction_retry.correction_attempts

        if correction_retry.success:
            result['solved'] = True
            result['solution'] = correction_retry.solution
            result['correction_type'] = f'cage_retry_{multiplier}+constraint_{correction_retry.correction_type}'
            return result

    result['correction_type'] = 'uncorrectable'
    return result


def evaluate_with_correction(sizes, num_per_size, grid_model, char_model):
    """Run evaluation with full error correction."""
    results = []

    for size in sizes:
        board_size = size * CELL_SIZE
        print(f"\nEvaluating {size}x{size} puzzles ({board_size}x{board_size}px) with error correction...")

        baseline_count = 0
        corrected_count = 0
        correction_types = Counter()

        for i in range(num_per_size):
            filename = f'./board_images/board{size}_{i}.png'

            if not os.path.exists(filename):
                continue

            start_time = time.time()
            result = solve_with_full_correction(filename, size, grid_model, char_model)
            solve_time = (time.time() - start_time) * 1000

            correction_types[result['correction_type']] += 1

            if result['solved']:
                corrected_count += 1
                if result['correction_type'] == 'baseline':
                    baseline_count += 1

            results.append({
                'size': size,
                'puzzle_id': i,
                'baseline_solved': result['correction_type'] == 'baseline',
                'corrected_solved': result['solved'],
                'correction_type': result['correction_type'],
                'attempts': result['attempts'],
                'solve_time_ms': solve_time
            })

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{num_per_size}] Baseline: {baseline_count}, Corrected: {corrected_count}")

        baseline_acc = baseline_count / num_per_size * 100
        corrected_acc = corrected_count / num_per_size * 100
        print(f"  {size}x{size}: Baseline {baseline_acc:.0f}% -> Corrected {corrected_acc:.0f}%")
        print(f"    Breakdown: {dict(correction_types)}")

    return results


def main():
    print("=" * 70)
    print("KenKen Handwritten Evaluation - 300px Cells")
    print("=" * 70)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load models
    print("\nLoading models...")
    grid_model, char_model = load_models()
    print("Models loaded.")

    # Run evaluation
    sizes = [3, 4, 5, 6, 7, 9]
    num_per_size = 100

    results = evaluate_with_correction(sizes, num_per_size, grid_model, char_model)

    # Save results
    os.makedirs('./results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('./results/corrected_evaluation.csv', index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_data = []
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        if not size_results:
            continue

        baseline = sum(1 for r in size_results if r['baseline_solved'])
        corrected = sum(1 for r in size_results if r['corrected_solved'])
        total = len(size_results)

        board_size = size * CELL_SIZE
        print(f"{size}x{size} ({board_size}x{board_size}px): Baseline {baseline}/{total} ({baseline/total*100:.0f}%) -> "
              f"Corrected {corrected}/{total} ({corrected/total*100:.0f}%)")

        summary_data.append({
            'size': size,
            'board_size': f'{board_size}x{board_size}',
            'baseline_solved': baseline,
            'corrected_solved': corrected,
            'total': total,
            'baseline_accuracy': baseline / total * 100,
            'corrected_accuracy': corrected / total * 100
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('./results/corrected_summary.csv', index=False)

    print(f"\nResults saved to ./results/")


if __name__ == '__main__':
    main()
