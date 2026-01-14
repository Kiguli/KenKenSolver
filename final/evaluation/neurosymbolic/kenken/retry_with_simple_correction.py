# -*- coding: utf-8 -*-
"""
Retry uncorrectable puzzles using attempt_error_correction.

This script re-runs puzzles that failed with attempt_constraint_based_correction
using the simpler attempt_error_correction approach which doesn't filter alternatives.

Usage:
    python retry_with_simple_correction.py
"""

import os
import sys
import csv
import time
import torch
from pathlib import Path
from itertools import combinations, product
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

# Add paths
script_dir = Path(__file__).parent
base_dir = script_dir.parent.parent.parent.parent

# Add archive solver to path (has all dependencies)
archive_solver_dir = base_dir / 'archive' / 'KenKen-handwritten-v2' / 'solver'
sys.path.insert(0, str(archive_solver_dir))

# Import from archive solve_all_sizes.py
from solve_all_sizes import (
    solve_kenken, make_puzzle_with_alternatives,
    find_size_and_borders_with_retry, detect_errors_via_unsat_core,
    generate_cage_alternatives, CorrectionResult
)

# Try to import ImprovedCNN from archive models (for class definition)
archive_models_dir = base_dir / 'archive' / 'KenKen-handwritten-v2' / 'models'
sys.path.insert(0, str(archive_models_dir))

# Model weights are in final/models/handwritten_v2
final_models_dir = base_dir / 'final' / 'models' / 'handwritten_v2'

try:
    from improved_cnn import ImprovedCNN
    HAS_IMPROVED_CNN = True
except ImportError:
    HAS_IMPROVED_CNN = False
    print("Warning: ImprovedCNN not available")


def attempt_error_correction(puzzle, alternatives, size, max_errors=3, max_k=4, max_suspects=15):
    """
    Attempt to correct OCR errors using top-K predictions.

    This is the simpler approach that doesn't filter alternatives by constraints.
    It just tries all alternatives from the top-K predictions.
    """
    max_attempts_per_level = [50, 100, 200, 400, 600, 800, 1000, 1500]

    suspects, num_probes = detect_errors_via_unsat_core(puzzle, size)
    total_solve_calls = num_probes

    if not suspects:
        solution = solve_kenken(puzzle, size)
        total_solve_calls += 1
        if solution:
            return CorrectionResult(
                success=True,
                correction_type="none",
                num_errors_corrected=0,
                solution=solution,
                correction_attempts=0
            )

    suspects = suspects[:max_suspects]
    correction_attempts = 0

    suspect_alternatives = []
    for cage_idx in suspects:
        alts = generate_cage_alternatives(alternatives[cage_idx], max_k)
        if alts:
            suspect_alternatives.append((cage_idx, alts))

    error_names = ["single", "two", "three", "four", "five", "six", "seven", "eight"]

    for num_errors in range(1, max_errors + 1):
        if num_errors > len(suspect_alternatives):
            break

        max_attempts = max_attempts_per_level[num_errors - 1] if num_errors <= len(max_attempts_per_level) else 2000
        error_name = error_names[num_errors - 1] if num_errors <= len(error_names) else f"{num_errors}"

        attempts_this_level = 0
        cage_combos = list(combinations(range(len(suspect_alternatives)), num_errors))

        for combo_indices in cage_combos:
            if attempts_this_level >= max_attempts:
                break

            combo_data = [suspect_alternatives[i] for i in combo_indices]
            cage_indices = [d[0] for d in combo_data]
            alt_lists = [d[1] for d in combo_data]

            max_alts_per_cage = max(1, int(max_attempts ** (1.0 / num_errors)))
            trimmed_alt_lists = [alts[:max_alts_per_cage] for alts in alt_lists]

            for alt_combo in product(*trimmed_alt_lists):
                if attempts_this_level >= max_attempts:
                    break

                test_puzzle = puzzle.copy()
                for cage_idx, (new_cage, conf, preds) in zip(cage_indices, alt_combo):
                    test_puzzle[cage_idx] = new_cage

                solution = solve_kenken(test_puzzle, size)
                total_solve_calls += 1
                correction_attempts += 1
                attempts_this_level += 1

                if solution is not None:
                    return CorrectionResult(
                        success=True,
                        correction_type=error_name,
                        num_errors_corrected=num_errors,
                        solution=solution,
                        correction_attempts=correction_attempts
                    )

    return CorrectionResult(
        success=False,
        correction_type="still_uncorrectable",
        num_errors_corrected=0,
        solution=None,
        correction_attempts=correction_attempts
    )


def load_model():
    """Load the character recognition model."""
    # Use KenKen improved CNN from final/models/handwritten_v2
    improved_model_path = final_models_dir / 'kenken_improved_cnn.pth'

    if improved_model_path.exists() and HAS_IMPROVED_CNN:
        print(f"Loading ImprovedCNN from {improved_model_path}")
        model = ImprovedCNN(output_dim=14)
        model.load_state_dict(torch.load(improved_model_path, map_location='cpu', weights_only=False))
    else:
        raise FileNotFoundError(f"Model not found at {improved_model_path}")

    model.eval()
    return model


def load_results_csv(csv_path):
    """Load the results CSV and return rows."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_results_csv(csv_path, rows, fieldnames=None):
    """Save rows to CSV."""
    if not fieldnames:
        fieldnames = rows[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 70)
    print("Retry Uncorrectable Puzzles with Simple Error Correction")
    print("=" * 70)
    print()

    # Paths
    base_dir = script_dir.parent.parent.parent.parent
    results_csv = base_dir / 'final' / 'results' / 'neurosymbolic' / 'kenken_handwritten_v2.csv'
    board_dir = base_dir / 'final' / 'benchmarks' / 'KenKen' / 'Handwritten'

    if not results_csv.exists():
        print(f"Error: Results CSV not found at {results_csv}")
        return

    # Load model
    model = load_model()

    # Load existing results
    rows = load_results_csv(results_csv)
    print(f"Loaded {len(rows)} results from CSV")

    # Find uncorrectable puzzles
    uncorrectable = [r for r in rows if r['correction_type'] == 'still_uncorrectable']
    print(f"Found {len(uncorrectable)} uncorrectable puzzles to retry")
    print()

    # Dynamic max_errors based on size
    max_errors_by_size = {3: 3, 4: 4, 5: 5, 6: 5, 7: 6, 9: 6}

    # Track results
    newly_solved = []
    still_unsolved = []
    results_by_size = defaultdict(lambda: {'retried': 0, 'solved': 0, 'methods': []})

    # Process each uncorrectable puzzle
    for i, row in enumerate(uncorrectable):
        size = int(row['size'])
        puzzle_id = int(row['puzzle_id'])

        board_path = board_dir / f'{size}x{size}' / f'board{size}_{puzzle_id}.png'
        if not board_path.exists():
            print(f"Warning: Board image not found: {board_path}")
            still_unsolved.append(row)
            continue

        print(f"[{i+1}/{len(uncorrectable)}] Retrying {size}x{size} puzzle {puzzle_id}...", end=' ')
        sys.stdout.flush()

        results_by_size[size]['retried'] += 1

        start_time = time.time()

        try:
            # Get size and borders
            detected_size, cages, border_thickness, retry_info = find_size_and_borders_with_retry(
                str(board_path), None, size, verbose=False
            )

            if not cages:
                print("no cages detected")
                still_unsolved.append(row)
                continue

            # Extract puzzle with alternatives
            puzzle, alternatives = make_puzzle_with_alternatives(
                size, border_thickness, cages, str(board_path), model, k=8, invert=False
            )

            # Try simple error correction with higher max_errors
            max_errors = max_errors_by_size.get(size, 5)
            correction = attempt_error_correction(
                puzzle, alternatives, size, max_errors=max_errors, max_k=8, max_suspects=20
            )

            elapsed = (time.time() - start_time) * 1000

            if correction.success:
                print(f"SOLVED! ({correction.correction_type}, {elapsed:.0f}ms)")

                # Update the row
                row['corrected_solved'] = 'True'
                row['correction_type'] = f"simple_{correction.correction_type}"
                row['solve_time_ms'] = str(float(row['solve_time_ms']) + elapsed)

                newly_solved.append({
                    'size': size,
                    'puzzle_id': puzzle_id,
                    'correction_type': correction.correction_type,
                    'additional_time_ms': elapsed
                })

                results_by_size[size]['solved'] += 1
                results_by_size[size]['methods'].append(correction.correction_type)
            else:
                print(f"still unsolvable ({elapsed:.0f}ms)")
                still_unsolved.append(row)

        except Exception as e:
            print(f"error: {e}")
            import traceback
            traceback.print_exc()
            still_unsolved.append(row)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Summary by size
    print("By Size:")
    for size in sorted(results_by_size.keys()):
        stats = results_by_size[size]
        methods = ', '.join(set(stats['methods'])) if stats['methods'] else 'none'
        print(f"  {size}x{size}: {stats['solved']}/{stats['retried']} newly solved ({methods})")

    print()
    print(f"Total newly solved: {len(newly_solved)} / {len(uncorrectable)}")
    print()

    # Updated accuracy table
    if newly_solved:
        print("Detailed Results:")
        for result in newly_solved[:20]:  # Show first 20
            print(f"  Puzzle {result['size']}_{result['puzzle_id']}: {result['correction_type']}")
        if len(newly_solved) > 20:
            print(f"  ... and {len(newly_solved) - 20} more")

    # Calculate new totals
    print()
    print("Updated Accuracy by Size:")
    print("| Size | Previous | New | Change |")
    print("|------|----------|-----|--------|")

    size_totals = defaultdict(lambda: {'total': 0, 'solved': 0})
    for row in rows:
        size = int(row['size'])
        size_totals[size]['total'] += 1
        if row['corrected_solved'] == 'True':
            size_totals[size]['solved'] += 1

    for size in sorted(size_totals.keys()):
        total = size_totals[size]['total']
        new_solved = size_totals[size]['solved']
        prev_solved = new_solved - results_by_size[size]['solved']
        new_pct = new_solved / total * 100
        prev_pct = prev_solved / total * 100
        change = new_pct - prev_pct
        print(f"| {size}×{size}  | {prev_pct:.0f}%      | {new_pct:.0f}%  | +{change:.0f}%    |")

    # Save updated results
    save_results_csv(results_csv, rows)
    print()
    print(f"Results saved to {results_csv}")


if __name__ == '__main__':
    main()
