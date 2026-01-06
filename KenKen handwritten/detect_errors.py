"""
Enhanced Error Detection and Correction for Handwritten KenKen.

Uses Z3's unsatisfiable core to identify problematic cages, then explores
alternative character predictions using Top-K + multi-error correction.

Based on the approach in "90-10 split detect handwritten digit errors/predict_digits.py"
but adapted for KenKen's cage-based structure (multiple characters per cage).

Reports both "base accuracy" and "corrected accuracy" for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from z3 import Int, Solver, And, Or, Distinct, Sum, sat, unsat, Bool
from torchvision import transforms
import json
import time
import sys
import os
from itertools import combinations, product
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Import from evaluate.py
from evaluate import (
    Grid_CNN, CNN_v2, transform,
    find_size_and_borders, segment_cell, get_predictions,
    BOARD_SIZE, IMG_SIZE, SCALE_FACTOR
)


# =============================================================================
# Top-K Predictions
# =============================================================================

def get_predictions_with_top_k(characters, model, k=4):
    """
    Get predictions with top-K alternatives using unified model.

    For each character in a cage, returns:
    - Best prediction
    - Top-K alternatives with confidence scores
    """
    predictions = []
    top_k_predictions = []

    with torch.no_grad():
        for c in characters:
            im = torch.tensor(c, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = model(im)
            probs = F.softmax(output, dim=1).squeeze()

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            pred = sorted_indices[0].item()
            top_k = [(sorted_indices[j].item(), sorted_probs[j].item())
                     for j in range(min(k, len(sorted_probs)))]

            predictions.append(pred)
            top_k_predictions.append(top_k)

    return predictions, top_k_predictions


def update_puzzle_from_predictions(cage, predictions):
    """Update puzzle cage from character predictions."""
    puzzle = {"cells": cage, "op": "", "target": 0}

    if len(predictions) == 0:
        return puzzle

    if len(predictions) == 1:
        puzzle["target"] = predictions[0]
    else:
        target = 0
        for i in range(len(predictions) - 1):
            power = len(predictions) - 2 - i
            target += predictions[i] * (10 ** power)

        if predictions[-1] == 10:
            op = "add"
        elif predictions[-1] == 11:
            op = "div"
        elif predictions[-1] == 12:
            op = "mul"
        elif predictions[-1] == 13:
            op = "sub"
        else:
            target = target * 10 + predictions[-1]
            op = ""

        puzzle["target"] = target
        puzzle["op"] = op

    return puzzle


def make_puzzle_with_alternatives(size, border_thickness, cages, filename, model, k=4):
    """
    Extract puzzle from image with alternative predictions stored.

    Returns both the puzzle and the top-K alternatives for each cage.
    """
    img = Image.open(filename).convert('L')
    grid = np.array(img)
    puzzle = []
    alternatives = []

    for cage in cages:
        characters = segment_cell(grid, size, border_thickness + 5, cage[0][0], cage[0][1])
        predictions, top_k = get_predictions_with_top_k(characters, model, k)

        cage_puzzle = update_puzzle_from_predictions(cage, predictions)
        puzzle.append(cage_puzzle)
        alternatives.append({
            'cage': cage,
            'characters': characters,
            'predictions': predictions,
            'top_k': top_k
        })

    return puzzle, alternatives


# =============================================================================
# Z3 Solver with Constraint Tracking
# =============================================================================

def solve_kenken(puzzle, size):
    """Solve KenKen puzzle (fast, no tracking)."""
    known_values = {}
    for block in puzzle:
        if block["op"] == "" and len(block["cells"]) == 1:
            i, j = block["cells"][0]
            known_values[(i, j)] = block["target"]

    X = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i, j) in known_values:
                X[i][j] = known_values[(i, j)]
            else:
                X[i][j] = Int(f"x_{i+1}_{j+1}")

    s = Solver()
    s.set("timeout", 30000)

    # Range constraints
    for i in range(size):
        for j in range(size):
            if (i, j) not in known_values:
                s.add(And(1 <= X[i][j], X[i][j] <= size))

    # Row/column distinctness
    for i in range(size):
        s.add(Distinct(X[i]))
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    # Cage constraints
    for block in puzzle:
        op = block["op"]
        target = block["target"]
        block_cells = block["cells"]

        vars_in_block = []
        for i, j in block_cells:
            if (i, j) in known_values:
                vars_in_block.append(known_values[(i, j)])
            else:
                vars_in_block.append(X[i][j])

        if op == "":
            if len(block_cells) == 1:
                i, j = block_cells[0]
                if (i, j) not in known_values:
                    s.add(X[i][j] == target)
        elif op == "add":
            s.add(Sum(vars_in_block) == target)
        elif op == "mul":
            if len(vars_in_block) == 1:
                s.add(vars_in_block[0] == target)
            else:
                prod = vars_in_block[0]
                for v in vars_in_block[1:]:
                    prod = prod * v
                s.add(prod == target)
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            s.add(Or(a - b == target, b - a == target))
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            int_target = int(target)
            s.add(Or(a == b * int_target, b == a * int_target))

    if s.check() == sat:
        m = s.model()
        solution = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i, j) in known_values:
                    row.append(known_values[(i, j)])
                else:
                    row.append(m.evaluate(X[i][j]).as_long())
            solution.append(row)
        return solution
    return None


def solve_kenken_with_tracking(puzzle, size):
    """
    Solve KenKen puzzle with constraint tracking for unsat core extraction.

    Returns (solution, suspect_cage_indices) where:
    - solution is the solved grid or None
    - suspect_cage_indices is list of cage indices in unsat core
    """
    known_values = {}
    for block in puzzle:
        if block["op"] == "" and len(block["cells"]) == 1:
            i, j = block["cells"][0]
            known_values[(i, j)] = block["target"]

    X = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i, j) in known_values:
                X[i][j] = known_values[(i, j)]
            else:
                X[i][j] = Int(f"x_{i+1}_{j+1}")

    s = Solver()
    s.set("timeout", 30000)

    # Core constraints (unlikely to be wrong)
    for i in range(size):
        for j in range(size):
            if (i, j) not in known_values:
                s.add(And(1 <= X[i][j], X[i][j] <= size))

    for i in range(size):
        s.add(Distinct(X[i]))
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    # Track cage constraints with assert_and_track
    cage_trackers = {}

    for idx, block in enumerate(puzzle):
        op = block["op"]
        target = block["target"]
        block_cells = block["cells"]

        vars_in_block = []
        for i, j in block_cells:
            if (i, j) in known_values:
                vars_in_block.append(known_values[(i, j)])
            else:
                vars_in_block.append(X[i][j])

        constraint = None
        if op == "":
            if len(block_cells) == 1:
                i, j = block_cells[0]
                if (i, j) not in known_values:
                    constraint = X[i][j] == target
        elif op == "add":
            constraint = Sum(vars_in_block) == target
        elif op == "mul":
            if len(vars_in_block) == 1:
                constraint = vars_in_block[0] == target
            else:
                prod = vars_in_block[0]
                for v in vars_in_block[1:]:
                    prod = prod * v
                constraint = prod == target
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraint = Or(a - b == target, b - a == target)
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            int_target = int(target)
            constraint = Or(a == b * int_target, b == a * int_target)

        if constraint is not None:
            tracker = Bool(f"cage_{idx}")
            cage_trackers[tracker] = idx
            s.assert_and_track(constraint, tracker)

    if s.check() == sat:
        m = s.model()
        solution = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i, j) in known_values:
                    row.append(known_values[(i, j)])
                else:
                    row.append(m.evaluate(X[i][j]).as_long())
            solution.append(row)
        return solution, []

    # Extract unsat core
    core = s.unsat_core()
    suspect_indices = [cage_trackers[t] for t in core if t in cage_trackers]
    return None, suspect_indices


# =============================================================================
# Error Detection via Unsat Core + Probing
# =============================================================================

@dataclass
class DetectionResult:
    """Result of error detection."""
    suspects: List[int]  # Cage indices
    num_probes: int
    detection_time: float


def detect_errors_via_unsat_core(puzzle, size, max_probes=30):
    """
    Detect which cages are likely errors using unsat core + probing.

    Strategy:
    1. Get initial unsat core (cages involved in conflicts)
    2. For each suspect, try removing it and check if puzzle becomes solvable
    3. Rank by how often each cage appears in conflicts
    """
    start_time = time.time()

    # Initial solve with tracking
    solution, initial_suspects = solve_kenken_with_tracking(puzzle, size)

    if solution is not None:
        return DetectionResult(suspects=[], num_probes=1, detection_time=time.time() - start_time)

    if not initial_suspects:
        return DetectionResult(suspects=[], num_probes=1, detection_time=time.time() - start_time)

    suspect_counts = Counter()
    num_probes = 1

    # Probe by removing each suspect cage
    probes_to_do = initial_suspects[:max_probes]

    for cage_idx in probes_to_do:
        # Create puzzle without this cage
        test_puzzle = [puzzle[i] for i in range(len(puzzle)) if i != cage_idx]

        sol, new_suspects = solve_kenken_with_tracking(test_puzzle, size)
        num_probes += 1

        if sol is not None:
            # Removing this cage made it solvable - highly suspect
            suspect_counts[cage_idx] += 10
        else:
            # Still unsolvable, but track which cages appear in new unsat core
            for s in new_suspects:
                # Adjust index since we removed a cage
                adjusted_s = s if s < cage_idx else s + 1
                suspect_counts[adjusted_s] += 1
            if cage_idx in initial_suspects:
                suspect_counts[cage_idx] += 1

    # Sort by suspicion score
    sorted_suspects = sorted(suspect_counts.keys(), key=lambda p: -suspect_counts[p])

    if not sorted_suspects:
        sorted_suspects = initial_suspects

    return DetectionResult(
        suspects=sorted_suspects,
        num_probes=num_probes,
        detection_time=time.time() - start_time
    )


# =============================================================================
# Multi-Error Correction with Top-K Alternatives
# =============================================================================

@dataclass
class CorrectionResult:
    """Result of correction attempt."""
    success: bool
    correction_type: str  # "none", "single", "two", "three", etc.
    num_errors_corrected: int
    cages_corrected: List[int]
    solution: Optional[List[List[int]]]
    correction_attempts: int
    total_solve_calls: int


def generate_cage_alternatives(alternatives_data, max_k=4):
    """
    Generate all alternative interpretations of a cage.

    For a cage with N characters, each with K alternatives,
    returns list of (new_predictions, confidence) tuples.
    """
    top_k_list = alternatives_data['top_k']
    original_preds = alternatives_data['predictions']
    cage = alternatives_data['cage']

    if not top_k_list:
        return []

    # Build list of alternatives for each character position
    char_alternatives = []
    for char_idx, char_top_k in enumerate(top_k_list):
        # Include alternatives beyond the first (original) prediction
        alts = [(pred, conf) for pred, conf in char_top_k[:max_k]]
        char_alternatives.append(alts)

    # Generate all combinations (skip first which is original)
    all_combos = []
    for combo in product(*char_alternatives):
        preds = [p for p, _ in combo]
        conf = sum(c for _, c in combo) / len(combo)  # Average confidence

        # Skip if this is the original prediction
        if preds == original_preds:
            continue

        # Create new cage puzzle
        new_cage = update_puzzle_from_predictions(cage, preds)
        all_combos.append((new_cage, conf, preds))

    # Sort by confidence (highest first)
    all_combos.sort(key=lambda x: -x[1])

    return all_combos


def attempt_correction_with_top_k(
    puzzle: List[dict],
    alternatives: List[dict],
    size: int,
    max_errors: int = 5,
    max_k: int = 4,
    max_suspects: int = 15,
    max_attempts_per_level: Optional[List[int]] = None
) -> Tuple[DetectionResult, CorrectionResult]:
    """
    Use unsat core detection + top-K predictions for multi-error correction.

    For each suspect cage, tries alternative predictions (2nd, 3rd, 4th best).
    Can correct multiple cages simultaneously.
    """
    if max_attempts_per_level is None:
        # Reduced from [100, 500, 1000, 2000, 5000] for faster evaluation
        max_attempts_per_level = [50, 100, 200, 400, 800]

    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    # Step 1: Detect suspects
    detection = detect_errors_via_unsat_core(puzzle, size)
    total_solve_calls = detection.num_probes

    # Check if already solved
    if not detection.suspects:
        solution = solve_kenken(puzzle, size)
        total_solve_calls += 1
        if solution:
            return detection, CorrectionResult(
                success=True,
                correction_type="none",
                num_errors_corrected=0,
                cages_corrected=[],
                solution=solution,
                correction_attempts=0,
                total_solve_calls=total_solve_calls
            )

    suspects = detection.suspects[:max_suspects]
    correction_attempts = 0

    # Pre-compute alternatives for each suspect cage
    suspect_alternatives = []
    for cage_idx in suspects:
        alts = generate_cage_alternatives(alternatives[cage_idx], max_k)
        if alts:
            suspect_alternatives.append((cage_idx, alts))

    error_names = ["single", "two", "three", "four", "five"]

    # Try correcting 1, 2, 3, ... up to max_errors cages
    for num_errors in range(1, max_errors + 1):
        if num_errors > len(suspect_alternatives):
            break

        max_attempts = max_attempts_per_level[num_errors - 1]
        error_name = error_names[num_errors - 1] if num_errors <= len(error_names) else f"{num_errors}"

        attempts_this_level = 0

        # Generate combinations of suspect cages
        cage_combos = list(combinations(range(len(suspect_alternatives)), num_errors))

        for combo_indices in cage_combos:
            if attempts_this_level >= max_attempts:
                break

            # Get the cages and their alternatives
            combo_data = [suspect_alternatives[i] for i in combo_indices]
            cage_indices = [d[0] for d in combo_data]
            alt_lists = [d[1] for d in combo_data]

            # Limit alternatives per cage to keep combinations manageable
            max_alts_per_cage = max(1, int(max_attempts ** (1.0 / num_errors)))
            trimmed_alt_lists = [alts[:max_alts_per_cage] for alts in alt_lists]

            # Try all combinations of alternatives
            for alt_combo in product(*trimmed_alt_lists):
                if attempts_this_level >= max_attempts:
                    break

                # Build modified puzzle
                test_puzzle = puzzle.copy()
                for cage_idx, (new_cage, conf, preds) in zip(cage_indices, alt_combo):
                    test_puzzle[cage_idx] = new_cage

                solution = solve_kenken(test_puzzle, size)
                total_solve_calls += 1
                correction_attempts += 1
                attempts_this_level += 1

                if solution is not None:
                    return detection, CorrectionResult(
                        success=True,
                        correction_type=error_name,
                        num_errors_corrected=num_errors,
                        cages_corrected=list(cage_indices),
                        solution=solution,
                        correction_attempts=correction_attempts,
                        total_solve_calls=total_solve_calls
                    )

    # Could not correct
    return detection, CorrectionResult(
        success=False,
        correction_type="uncorrectable",
        num_errors_corrected=0,
        cages_corrected=[],
        solution=None,
        correction_attempts=correction_attempts,
        total_solve_calls=total_solve_calls
    )


def solve_with_error_correction(size, cages, border_thickness, filename, model, max_k=4, max_errors=5):
    """
    Attempt to solve a puzzle, correcting OCR errors if needed.

    Returns:
        solution: The solution grid (or None)
        corrected: Whether error correction was applied
        correction_type: Type of correction applied
        stats: Dictionary with timing and attempt counts
    """
    start_time = time.time()

    # Get initial puzzle with alternatives
    puzzle, alternatives = make_puzzle_with_alternatives(
        size, border_thickness, cages, filename, model, k=max_k
    )

    # Try initial solution
    solution = solve_kenken(puzzle, size)
    if solution is not None:
        return solution, False, "none", {
            'solve_time_ms': (time.time() - start_time) * 1000,
            'correction_attempts': 0,
            'total_solve_calls': 1
        }

    # Apply error correction
    detection, correction = attempt_correction_with_top_k(
        puzzle, alternatives, size,
        max_errors=max_errors,
        max_k=max_k
    )

    return correction.solution, correction.success, correction.correction_type, {
        'solve_time_ms': (time.time() - start_time) * 1000,
        'correction_attempts': correction.correction_attempts,
        'total_solve_calls': correction.total_solve_calls,
        'num_suspects': len(detection.suspects),
        'detection_time_ms': detection.detection_time * 1000
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    print("=" * 70)
    print("Handwritten KenKen Solver - Enhanced Error Correction")
    print("Using Top-K alternatives + Multi-error correction")
    print("=" * 70)
    print()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load models
    print("Loading models...")

    grid_detection = Grid_CNN(output_dim=5)
    state_dict = torch.load('./models/grid_detection_model_weights.pth', weights_only=False)
    grid_detection.load_state_dict(state_dict)
    grid_detection.eval()
    print("  Grid detection model loaded")

    char_model = CNN_v2(output_dim=14)
    model_path = './models/unified_kenken_cnn.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Unified model not found at {model_path}. "
            "Run train_unified_cnn.py first."
        )
    state_dict = torch.load(model_path, weights_only=False)
    char_model.load_state_dict(state_dict)
    char_model.eval()
    print("  Unified character model loaded")

    print()

    # Load puzzle data
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    results = []
    base_accuracy = {}
    corrected_accuracy = {}
    avg_time = {}
    correction_breakdown = {}

    # Only evaluate sizes 3-7 (grid CNN doesn't support 9x9)
    sizes_to_eval = [3, 4, 5, 6, 7]

    for size in sizes_to_eval:
        size_str = str(size)
        if size_str not in puzzles_ds:
            continue

        puzzles = puzzles_ds[size_str]
        print(f"Evaluating {size}x{size} puzzles ({len(puzzles)} total)...")
        base_solved = 0
        corrected_solved = 0
        total_time = 0
        correction_types = Counter()

        for i in range(len(puzzles)):
            filename = f"./board_images/board{size}_{i}.png"

            if not os.path.exists(filename):
                continue

            try:
                detected_size, cages, border_thickness = find_size_and_borders(filename, grid_detection)
                solution, corrected, correction_type, stats = solve_with_error_correction(
                    detected_size, cages, border_thickness, filename, char_model,
                    max_k=4, max_errors=5
                )

                total_time += stats['solve_time_ms']
                correction_types[correction_type] += 1

                if solution is not None:
                    if not corrected:
                        base_solved += 1
                        corrected_solved += 1
                    else:
                        corrected_solved += 1

                results.append({
                    'puzzle_id': i,
                    'size': size,
                    'base_solved': solution is not None and not corrected,
                    'corrected_solved': solution is not None,
                    'correction_type': correction_type,
                    'solve_time_ms': stats['solve_time_ms'],
                    'correction_attempts': stats['correction_attempts'],
                    'total_solve_calls': stats['total_solve_calls']
                })

            except Exception as e:
                results.append({
                    'puzzle_id': i,
                    'size': size,
                    'base_solved': False,
                    'corrected_solved': False,
                    'correction_type': 'error',
                    'error': str(e),
                    'solve_time_ms': 0
                })

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(puzzles)}] Base: {base_solved}, Corrected: {corrected_solved}")
                sys.stdout.flush()

        base_accuracy[size] = base_solved / len(puzzles) * 100
        corrected_accuracy[size] = corrected_solved / len(puzzles) * 100
        avg_time[size] = total_time / len(puzzles)
        correction_breakdown[size] = dict(correction_types)

        print(f"  {size}x{size}: Base {base_accuracy[size]:.1f}% -> Corrected {corrected_accuracy[size]:.1f}%")
        print(f"    Breakdown: {dict(correction_types)}")
        print()

    # Save results
    os.makedirs('./results', exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv('./results/error_correction_evaluation.csv', index=False)

    # Summary
    summary = []
    for size in sizes_to_eval:
        if size in base_accuracy:
            summary.append({
                'size': size,
                'base_accuracy': base_accuracy[size],
                'corrected_accuracy': corrected_accuracy[size],
                'improvement': corrected_accuracy[size] - base_accuracy[size],
                'avg_time_ms': avg_time[size]
            })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('./results/error_correction_summary.csv', index=False)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Size':<8} {'Base':<12} {'Corrected':<12} {'Improvement':<12}")
    print("-" * 44)
    for size in sizes_to_eval:
        if size in base_accuracy:
            imp = corrected_accuracy[size] - base_accuracy[size]
            print(f"{size}x{size:<5} {base_accuracy[size]:.1f}%{'':<6} {corrected_accuracy[size]:.1f}%{'':<6} +{imp:.1f}%")
    print()


if __name__ == '__main__':
    main()
