# -*- coding: utf-8 -*-
"""
Confidence-Based Error Correction for Unsolved KenKen Puzzles.

This script retries unsolved (uncorrectable) puzzles using a pure
confidence-based approach:
1. Identify characters with lowest OCR confidence across all cages
2. Swap 1, 2, or 3 lowest-confidence characters for alternatives
3. Apply two-digit constraints (first digit=1, no 7/8/9 as second digit)

Usage:
    python retry_unsolved.py
"""

import os
import sys
import csv
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations, product
from collections import defaultdict

# Add parent directories to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent / 'models'))  # For improved_cnn

# Import from solve_all_sizes.py
from solve_all_sizes import (
    solve_kenken, make_puzzle_with_alternatives, update_puzzle_from_predictions,
    find_size_and_borders_with_retry, CNN_v2
)

# Try to import ImprovedCNN
try:
    from improved_cnn import ImprovedCNN
    HAS_IMPROVED_CNN = True
except ImportError:
    HAS_IMPROVED_CNN = False
    print("Note: ImprovedCNN not available, will use CNN_v2")


# =============================================================================
# Confidence-Based Error Correction
# =============================================================================

def get_lowest_confidence_chars(alternatives, num_chars=6):
    """
    Find the N characters with lowest confidence across all cages.

    Returns list of (cage_idx, char_idx, confidence, top_k_alternatives)
    sorted by confidence ascending (lowest first).
    """
    all_chars = []
    for cage_idx, alt_data in enumerate(alternatives):
        for char_idx, top_k in enumerate(alt_data['top_k']):
            if top_k and len(top_k) > 1:  # Must have alternatives
                conf = top_k[0][1]  # Confidence of top prediction
                all_chars.append((cage_idx, char_idx, conf, top_k))

    # Sort by confidence ascending (lowest first)
    all_chars.sort(key=lambda x: x[2])
    return all_chars[:num_chars]


def is_valid_two_digit_prediction(preds, size):
    """
    Validate two-digit numbers for KenKen targets.
    - First digit must be 1 (targets range from 10-max based on size)
    - Second digit cannot be 7, 8, or 9 (invalid for most KenKen sizes)

    Returns True if valid or not a two-digit target.
    """
    if len(preds) < 2:
        return True  # Single digit targets are always valid

    # For two-digit targets (excluding operator at end)
    # Operators are: 10=+, 11=÷, 12=×, 13=-
    has_operator = preds[-1] >= 10
    digit_preds = preds[:-1] if has_operator else preds

    if len(digit_preds) == 2:
        first_digit = digit_preds[0]
        second_digit = digit_preds[1]

        # First digit must be 1 for valid KenKen targets (10-19 range typical)
        if first_digit != 1:
            return False

        # Second digit cannot be 7, 8, or 9 for smaller puzzles
        # For 9x9, targets can go up to ~72 for multiplication, but two-digit
        # handwritten typically maxes at 16-18 for addition cages
        if size <= 7 and second_digit in [7, 8, 9]:
            return False

    return True


def apply_char_swaps(alternatives, chars_to_swap, alt_choices):
    """
    Apply character swaps and rebuild the puzzle.

    Args:
        alternatives: Full alternatives data for all cages
        chars_to_swap: List of (cage_idx, char_idx, conf, top_k) tuples
        alt_choices: List of alternative indices to use (1=2nd best, 2=3rd best, etc.)

    Returns:
        Modified puzzle representation
    """
    # Build a map of swaps: (cage_idx, char_idx) -> new_prediction
    swaps = {}
    for (cage_idx, char_idx, _, top_k), alt_idx in zip(chars_to_swap, alt_choices):
        if alt_idx < len(top_k):
            new_pred = top_k[alt_idx][0]
            swaps[(cage_idx, char_idx)] = new_pred

    # Rebuild puzzle with swaps applied
    modified_puzzle = []
    for cage_idx, alt_data in enumerate(alternatives):
        # Start with original predictions
        preds = list(alt_data['predictions'])

        # Apply any swaps for this cage
        for char_idx in range(len(preds)):
            if (cage_idx, char_idx) in swaps:
                preds[char_idx] = swaps[(cage_idx, char_idx)]

        # Rebuild cage from predictions
        cage = alt_data['cage']
        new_cage = update_puzzle_from_predictions(cage, preds)
        modified_puzzle.append(new_cage)

    return modified_puzzle


def try_confidence_based_corrections(puzzle, alternatives, size, max_swaps=3, verbose=False):
    """
    Try swapping 1, 2, or 3 lowest-confidence characters.
    For each character, try 2nd, 3rd, and 4th alternatives.

    Returns:
        (solution, correction_type, swap_details) or (None, "still_uncorrectable", None)
    """
    # Get lowest confidence characters (get extra to have options)
    lowest_conf_chars = get_lowest_confidence_chars(alternatives, max_swaps * 3)

    if verbose:
        print(f"  Lowest confidence chars: {[(c[0], c[1], f'{c[2]:.3f}') for c in lowest_conf_chars[:6]]}")

    if not lowest_conf_chars:
        return None, "still_uncorrectable", None

    total_attempts = 0

    for num_swaps in range(1, min(max_swaps, len(lowest_conf_chars)) + 1):
        # Try all combinations of num_swaps characters from lowest confidence
        for chars_to_swap in combinations(lowest_conf_chars[:max_swaps * 2], num_swaps):
            # For each character, try alternatives 1, 2, 3 (indices 1, 2, 3 in top_k)
            alt_indices_options = []
            for c in chars_to_swap:
                max_alts = min(4, len(c[3]))  # Up to 4 alternatives (including original)
                alt_indices_options.append(range(1, max_alts))  # Skip 0 (original)

            if not all(alt_indices_options):
                continue

            for alt_choices in product(*alt_indices_options):
                # Create modified puzzle
                test_puzzle = apply_char_swaps(alternatives, chars_to_swap, alt_choices)

                # Validate two-digit constraints for each cage
                valid = True
                for cage_idx, alt_data in enumerate(alternatives):
                    preds = list(alt_data['predictions'])
                    for char_idx in range(len(preds)):
                        if any(c[0] == cage_idx and c[1] == char_idx for c in chars_to_swap):
                            # This char was swapped
                            swap_info = [(c, a) for c, a in zip(chars_to_swap, alt_choices)
                                        if c[0] == cage_idx and c[1] == char_idx]
                            if swap_info:
                                c, alt_idx = swap_info[0]
                                if alt_idx < len(c[3]):
                                    preds[char_idx] = c[3][alt_idx][0]

                    if not is_valid_two_digit_prediction(preds, size):
                        valid = False
                        break

                if not valid:
                    continue

                # Try to solve
                total_attempts += 1
                solution = solve_kenken(test_puzzle, size)

                if solution is not None:
                    # Build swap details
                    swap_details = []
                    for (cage_idx, char_idx, conf, top_k), alt_idx in zip(chars_to_swap, alt_choices):
                        orig = top_k[0][0]
                        new = top_k[alt_idx][0]
                        swap_details.append({
                            'cage': cage_idx,
                            'char': char_idx,
                            'original': orig,
                            'new': new,
                            'orig_conf': conf,
                            'new_conf': top_k[alt_idx][1]
                        })

                    return solution, f"confidence_swap_{num_swaps}", swap_details

    if verbose:
        print(f"  Tried {total_attempts} combinations, no solution found")

    return None, "still_uncorrectable", None


# =============================================================================
# Main Script
# =============================================================================

def load_model():
    """Load the character recognition model."""
    # Try improved CNN first, fall back to original
    # Models are in parent directory (KenKen-handwritten-v2/models/)
    improved_model_path = script_dir.parent / 'models' / 'improved_cnn_weights.pth'
    original_model_path = script_dir / 'models' / 'character_recognition_v2_model_weights.pth'

    if improved_model_path.exists() and HAS_IMPROVED_CNN:
        print(f"Loading ImprovedCNN from {improved_model_path}")
        model = ImprovedCNN(output_dim=14)
        model.load_state_dict(torch.load(improved_model_path, map_location='cpu'))
    elif original_model_path.exists():
        print(f"Loading CNN_v2 from {original_model_path}")
        model = CNN_v2(output_dim=14)
        model.load_state_dict(torch.load(original_model_path, map_location='cpu'))
    else:
        raise FileNotFoundError("No model weights found!")

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
    print("Confidence-Based Error Correction for Unsolved KenKen Puzzles")
    print("=" * 70)
    print()

    # Change to script directory
    os.chdir(script_dir)

    # Paths
    results_csv = script_dir / 'results' / 'unified_solver_evaluation.csv'
    board_dir = script_dir.parent / 'board_images'

    if not results_csv.exists():
        print(f"Error: Results CSV not found at {results_csv}")
        return

    # Load model
    model = load_model()

    # Load existing results
    rows = load_results_csv(results_csv)
    print(f"Loaded {len(rows)} results from CSV")

    # Find uncorrectable puzzles
    uncorrectable = [r for r in rows if r['correction_type'] == 'uncorrectable']
    print(f"Found {len(uncorrectable)} uncorrectable puzzles to retry")
    print()

    # Track results
    newly_solved = []
    still_unsolved = []
    results_by_size = defaultdict(lambda: {'retried': 0, 'solved': 0, 'methods': []})

    # Process each uncorrectable puzzle
    for i, row in enumerate(uncorrectable):
        size = int(row['size'])
        puzzle_id = int(row['puzzle_id'])

        board_path = board_dir / f'board{size}_{puzzle_id}.png'
        if not board_path.exists():
            print(f"Warning: Board image not found: {board_path}")
            still_unsolved.append(row)
            continue

        print(f"[{i+1}/{len(uncorrectable)}] Retrying {size}x{size} puzzle {puzzle_id}...", end=' ')

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
                size, border_thickness, cages, str(board_path), model, k=4, invert=False
            )

            # Try confidence-based correction
            solution, correction_type, swap_details = try_confidence_based_corrections(
                puzzle, alternatives, size, max_swaps=3, verbose=False
            )

            elapsed = (time.time() - start_time) * 1000

            if solution is not None:
                print(f"SOLVED! ({correction_type})")

                # Update the row
                row['corrected_solved'] = 'True'
                row['correction_type'] = correction_type
                row['solve_time_ms'] = str(float(row['solve_time_ms']) + elapsed)

                newly_solved.append({
                    'size': size,
                    'puzzle_id': puzzle_id,
                    'correction_type': correction_type,
                    'swap_details': swap_details,
                    'additional_time_ms': elapsed
                })

                results_by_size[size]['solved'] += 1
                results_by_size[size]['methods'].append(correction_type)
            else:
                print("still unsolvable")
                row['correction_type'] = 'still_uncorrectable'
                still_unsolved.append(row)

        except Exception as e:
            print(f"error: {e}")
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

    # Detailed results
    if newly_solved:
        print("Detailed Results:")
        for result in newly_solved:
            details = result['swap_details']
            if details:
                swaps_str = ', '.join([f"cage{d['cage']}[{d['char']}]: {d['original']}->{d['new']}"
                                       for d in details])
            else:
                swaps_str = 'N/A'
            print(f"  Puzzle {result['size']}_{result['puzzle_id']}: {result['correction_type']} ({swaps_str})")

    # Updated accuracy table
    print()
    print("Updated Accuracy:")
    print("| Size | Previous | New | Improvement |")
    print("|------|----------|-----|-------------|")

    size_totals = defaultdict(lambda: {'total': 0, 'prev_solved': 0, 'new_solved': 0})
    for row in rows:
        size = int(row['size'])
        size_totals[size]['total'] += 1
        if row['corrected_solved'] == 'True':
            size_totals[size]['new_solved'] += 1
        # Count original solved (before our changes) from base_solved
        if row.get('base_solved') == 'True' or (row['correction_type'] not in ['uncorrectable', 'still_uncorrectable']
                                                  and 'confidence_swap' not in row['correction_type']):
            size_totals[size]['prev_solved'] += 1

    # Recalculate prev_solved from original uncorrectable count
    original_uncorrectable = {3: 0, 4: 9, 5: 56, 6: 56, 7: 85, 9: 85}

    for size in sorted(size_totals.keys()):
        total = size_totals[size]['total']
        prev_solved = total - original_uncorrectable.get(size, 0)
        new_solved = size_totals[size]['new_solved']
        prev_pct = prev_solved / total * 100
        new_pct = new_solved / total * 100
        improvement = new_pct - prev_pct
        print(f"| {size}×{size}  | {prev_pct:.0f}%      | {new_pct:.0f}%  | +{improvement:.0f}%         |")

    # Save updated results
    save_results_csv(results_csv, rows)
    print()
    print(f"Results saved to {results_csv}")

    # Also save to final location
    final_csv = script_dir.parent.parent.parent / 'final' / 'results' / 'neurosymbolic' / 'kenken_handwritten_v2.csv'
    if final_csv.parent.exists():
        save_results_csv(final_csv, rows)
        print(f"Results copied to {final_csv}")


if __name__ == '__main__':
    main()
