"""
Sudoku Solver V2 - Using KenKen V2 ImprovedCNN

Uses the ImprovedCNN trained for KenKen (which includes digits 0-9) to solve
handwritten Sudoku puzzles. The CNN has 14 output classes, but we only use
the first 10 (digits 0-9) for Sudoku.

Pipeline: Image -> ImprovedCNN (digit recognition) -> Z3 (constraint solving) -> Solution

Supports:
- 4x4 Sudoku (2x2 boxes)
- 9x9 Sudoku (3x3 boxes)
"""

import json
import os
import sys
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from z3 import Int, Solver, And, Distinct, sat, unsat, Bool
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import combinations

# Add parent directory for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.improved_cnn import ImprovedCNN


# =============================================================================
# Constants
# =============================================================================

BOARD_PIXELS = 900
NUM_DIGITS = 10  # Classes 0-9

# Path to KenKen V2 model weights (has 14 classes: 0-9 + operators)
KENKEN_MODEL_CLASSES = 14


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path):
    """Load the ImprovedCNN model trained for KenKen."""
    model = ImprovedCNN(output_dim=KENKEN_MODEL_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# =============================================================================
# Cell Extraction and Recognition
# =============================================================================

def get_cell_size(size):
    """Get cell size in pixels for a given puzzle size."""
    return BOARD_PIXELS // size


def get_box_size(size):
    """Get box size for a given puzzle size."""
    return 2 if size == 4 else 3


def extract_cell(board_img, row, col, size, target_size=28):
    """Extract and preprocess a single cell from board image."""
    cell_size = get_cell_size(size)
    margin = 8  # Avoid grid lines

    y_start = row * cell_size + margin
    y_end = (row + 1) * cell_size - margin
    x_start = col * cell_size + margin
    x_end = (col + 1) * cell_size - margin

    cell = board_img[y_start:y_end, x_start:x_end]

    cell_pil = Image.fromarray((cell * 255).astype(np.uint8))
    cell_pil = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(cell_pil) / 255.0


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


def recognize_with_alternatives(cell_img, model):
    """
    Recognize digit using ImprovedCNN, returning probability distribution.

    Only uses classes 0-9 (masks out operator classes 10-13).
    """
    # Invert so ink is high value (CNN expects white digit on black background)
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)

        # Only use digit classes (0-9), mask out operators (10-13)
        digit_logits = logits[:, :NUM_DIGITS]
        probs = F.softmax(digit_logits, dim=1).squeeze()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Build top-k list
    top_k = [(sorted_indices[i].item(), sorted_probs[i].item()) for i in range(min(4, len(sorted_indices)))]

    return {
        'prediction': sorted_indices[0].item(),
        'confidence': sorted_probs[0].item(),
        'second_best': sorted_indices[1].item() if len(sorted_indices) > 1 else 0,
        'second_conf': sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0,
        'all_probs': probs.numpy(),
        'top_k': top_k
    }


def extract_puzzle_with_alternatives(image_path, model, size):
    """Extract puzzle values from board image with alternatives."""
    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * size for _ in range(size)]
    alternatives = {}

    for row in range(size):
        for col in range(size):
            cell = extract_cell(board_img, row, col, size)

            if is_cell_empty(cell):
                puzzle[row][col] = 0
            else:
                recog = recognize_with_alternatives(cell, model)
                puzzle[row][col] = recog['prediction']
                alternatives[(row, col)] = recog

    return puzzle, alternatives


# =============================================================================
# Z3 Solvers
# =============================================================================

def solve_sudoku_with_tracking(size, given_cells):
    """
    Solve Sudoku with constraint tracking for unsat core extraction.
    """
    box_size = get_box_size(size)
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    clue_trackers = {}

    # Cell range constraints
    for i in range(size):
        for j in range(size):
            s.add(X[i][j] >= 1, X[i][j] <= size)

    # Given cell constraints WITH TRACKING
    for (i, j), val in given_cells.items():
        tracker = Bool(f"clue_{i}_{j}")
        clue_trackers[tracker] = (i, j)
        s.assert_and_track(X[i][j] == val, tracker)

    # Row uniqueness
    for i in range(size):
        s.add(Distinct([X[i][j] for j in range(size)]))

    # Column uniqueness
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    # Box uniqueness
    for box_row in range(size // box_size):
        for box_col in range(size // box_size):
            cells = []
            for i in range(box_size):
                for j in range(box_size):
                    cells.append(X[box_row * box_size + i][box_col * box_size + j])
            s.add(Distinct(cells))

    if s.check() == sat:
        m = s.model()
        solution = [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
        return solution, []
    else:
        core = s.unsat_core()
        suspects = [clue_trackers[t] for t in core if t in clue_trackers]
        return None, suspects


def solve_sudoku(size, given_cells):
    """Solve Sudoku (regular, no tracking)."""
    box_size = get_box_size(size)
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    for i in range(size):
        for j in range(size):
            s.add(And(X[i][j] >= 1, X[i][j] <= size))

    for (i, j), val in given_cells.items():
        s.add(X[i][j] == val)

    for i in range(size):
        s.add(Distinct([X[i][j] for j in range(size)]))

    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    for box_row in range(size // box_size):
        for box_col in range(size // box_size):
            cells = []
            for i in range(box_size):
                for j in range(box_size):
                    cells.append(X[box_row * box_size + i][box_col * box_size + j])
            s.add(Distinct(cells))

    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
    return None


# =============================================================================
# Error Detection and Correction
# =============================================================================

@dataclass
class DetectionResult:
    suspects: List[Tuple[int, int]]
    num_probes: int
    detection_time: float


@dataclass
class CorrectionResult:
    success: bool
    correction_type: str
    num_errors_corrected: int
    positions_corrected: List[Tuple[int, int]]
    original_values: List[int]
    corrected_values: List[int]
    solution: Optional[List[List[int]]]
    correction_attempts: int
    total_solve_calls: int


def detect_errors_via_unsat_core(puzzle, size, max_probes=50):
    """Detect errors using unsat core analysis."""
    start_time = time.time()

    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    solution, initial_suspects = solve_sudoku_with_tracking(size, given_cells)

    if solution is not None:
        return DetectionResult(suspects=[], num_probes=1,
                               detection_time=time.time() - start_time)

    if not initial_suspects:
        return DetectionResult(suspects=[], num_probes=1,
                               detection_time=time.time() - start_time)

    suspect_counts = Counter()
    num_probes = 1

    for pos in initial_suspects[:max_probes]:
        modified = given_cells.copy()
        del modified[pos]

        sol, new_suspects = solve_sudoku_with_tracking(size, modified)
        num_probes += 1

        if sol is not None:
            suspect_counts[pos] += 10
        else:
            for s in new_suspects:
                suspect_counts[s] += 1
            if pos in initial_suspects:
                suspect_counts[pos] += 1

    sorted_suspects = sorted(suspect_counts.keys(), key=lambda p: -suspect_counts[p])

    if not sorted_suspects:
        sorted_suspects = initial_suspects

    return DetectionResult(
        suspects=sorted_suspects,
        num_probes=num_probes,
        detection_time=time.time() - start_time
    )


def attempt_correction_with_detection(
    puzzle, alternatives, size,
    max_errors=3, max_suspects_per_level=20, max_attempts_per_level=None
):
    """Use unsat core detection to guide correction attempts."""
    if max_attempts_per_level is None:
        max_attempts_per_level = [20, 100, 300]

    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    detection = detect_errors_via_unsat_core(puzzle, size)
    total_solve_calls = detection.num_probes

    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    if not detection.suspects:
        solution = solve_sudoku(size, given_cells)
        total_solve_calls += 1
        if solution:
            return detection, CorrectionResult(
                success=True, correction_type="none", num_errors_corrected=0,
                positions_corrected=[], original_values=[], corrected_values=[],
                solution=solution, correction_attempts=0,
                total_solve_calls=total_solve_calls
            )

    suspects = detection.suspects
    correction_attempts = 0

    # Filter suspects to those with valid second-best alternatives
    valid_suspects = []
    for pos in suspects[:max_suspects_per_level]:
        if pos in alternatives:
            second_best = alternatives[pos]['second_best']
            if second_best != 0:  # 0 means no alternative
                valid_suspects.append(pos)

    error_names = ["single", "two", "three"]

    for num_errors in range(1, max_errors + 1):
        if num_errors > len(valid_suspects):
            break

        max_attempts = max_attempts_per_level[num_errors - 1]
        error_name = error_names[num_errors - 1] if num_errors <= len(error_names) else f"{num_errors}"

        combos = list(combinations(valid_suspects, num_errors))
        combos = combos[:max_attempts]

        for positions in combos:
            modified = given_cells.copy()
            corrected_values = []
            original_values = []

            for pos in positions:
                original_values.append(given_cells[pos])
                second_best = alternatives[pos]['second_best']
                modified[pos] = second_best
                corrected_values.append(second_best)

            solution = solve_sudoku(size, modified)
            total_solve_calls += 1
            correction_attempts += 1

            if solution is not None:
                return detection, CorrectionResult(
                    success=True, correction_type=error_name,
                    num_errors_corrected=num_errors,
                    positions_corrected=list(positions),
                    original_values=original_values,
                    corrected_values=corrected_values,
                    solution=solution, correction_attempts=correction_attempts,
                    total_solve_calls=total_solve_calls
                )

    return detection, CorrectionResult(
        success=False, correction_type="uncorrectable", num_errors_corrected=0,
        positions_corrected=[], original_values=[], corrected_values=[],
        solution=None, correction_attempts=correction_attempts,
        total_solve_calls=total_solve_calls
    )


# =============================================================================
# Evaluation Utilities
# =============================================================================

def compare_puzzles(extracted, expected, size):
    """Compare extracted puzzle to expected and return error positions."""
    errors = []
    for i in range(size):
        for j in range(size):
            if extracted[i][j] != expected[i][j]:
                errors.append((i, j))
    return errors


def solutions_match(computed, expected, size):
    """Check if computed solution matches expected."""
    if computed is None:
        return False
    for i in range(size):
        for j in range(size):
            if computed[i][j] != expected[i][j]:
                return False
    return True


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_puzzles(model, puzzles, image_dir, size):
    """Evaluate puzzles and return results."""
    results = []

    for idx in range(len(puzzles)):
        image_path = os.path.join(image_dir, f"board{size}_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        # Extract puzzle
        start_time = time.time()
        extracted_puzzle, alternatives = extract_puzzle_with_alternatives(image_path, model, size)
        extract_time = time.time() - start_time

        # Count extraction accuracy
        actual_errors = compare_puzzles(extracted_puzzle, expected_puzzle, size)
        num_actual_errors = len(actual_errors)

        # Run detection and correction
        detection, correction = attempt_correction_with_detection(
            extracted_puzzle, alternatives, size
        )
        total_time = time.time() - start_time

        # Verify solution against ground truth
        final_matches_truth = False
        if correction.solution:
            final_matches_truth = solutions_match(correction.solution, expected_solution, size)

        result = {
            'size': size,
            'puzzle_idx': idx,
            'num_actual_errors': num_actual_errors,
            'correction_type': correction.correction_type,
            'final_success': correction.success and final_matches_truth,
            'total_time': total_time,
            'correction_attempts': correction.correction_attempts,
            'total_solve_calls': correction.total_solve_calls
        }

        results.append(result)

        if (idx + 1) % 25 == 0:
            success_count = sum(1 for r in results if r['final_success'])
            print(f"  [{idx+1}/{len(puzzles)}] Success: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    return results


def main():
    print("=" * 70)
    print("Sudoku Solver V2 - Using KenKen V2 ImprovedCNN")
    print("=" * 70)

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Model path
    model_path = os.path.join(parent_dir, "models/improved_cnn_weights.pth")

    # Use board images from 90-10 split handwritten
    image_dir = os.path.join(os.path.dirname(parent_dir),
                             "90-10 split handwritten/Handwritten_Sudoku/board_images")

    # Use puzzles from 90-10 split handwritten
    puzzles_path = os.path.join(os.path.dirname(parent_dir),
                                "90-10 split handwritten/Handwritten_Sudoku/puzzles/puzzles_dict.json")

    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Check paths
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    if not os.path.exists(image_dir):
        print(f"Error: Board images not found at {image_dir}")
        return

    if not os.path.exists(puzzles_path):
        print(f"Error: Puzzles not found at {puzzles_path}")
        return

    # Load model
    print(f"\nLoading ImprovedCNN model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load puzzles
    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    all_results = []

    # Evaluate each size
    for size_str in ['4', '9']:
        size = int(size_str)
        puzzles = puzzles_ds.get(size_str, [])

        if not puzzles:
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating {len(puzzles)} {size}x{size} puzzles...")
        print("=" * 70)

        results = evaluate_puzzles(model, puzzles, image_dir, size)
        all_results.extend(results)

        # Summary
        success_count = sum(1 for r in results if r['final_success'])
        correction_counts = Counter(r['correction_type'] for r in results)
        avg_time = sum(r['total_time'] for r in results) / len(results)

        print(f"\n{size}x{size} Summary:")
        print(f"  Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        print(f"  Breakdown: {dict(correction_counts)}")
        print(f"  Avg time: {avg_time:.3f}s")

    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print("=" * 70)

    for size in [4, 9]:
        size_results = [r for r in all_results if r['size'] == size]
        if size_results:
            success_count = sum(1 for r in size_results if r['final_success'])
            print(f"\n{size}x{size}: {success_count}/{len(size_results)} ({success_count/len(size_results)*100:.1f}%)")

    # Save results
    if all_results:
        csv_path = os.path.join(results_dir, "sudoku_v2_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
