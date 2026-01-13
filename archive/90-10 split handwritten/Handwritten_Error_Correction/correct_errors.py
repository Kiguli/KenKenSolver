"""
Error Detection and Correction for Handwritten Puzzles.

When CNN misclassifies digits causing Z3 to fail (UNSAT), this script
identifies and corrects errors by substituting low-confidence predictions
with their second-best alternatives.

Supports:
- Single-digit error correction: O(n) complexity
- Two-digit error correction: O(n²) complexity, used when single fails

Algorithm:
1. Extract puzzle from image with CNN confidence scores
2. Try to solve - if successful, done
3. If UNSAT, attempt single-error correction:
   - Sort clues by confidence (lowest first)
   - For each clue, substitute with second-best prediction and try solving
4. If single-error fails, attempt two-error correction:
   - Consider pairs of low-confidence clues
   - Substitute both with their second-best predictions
5. If solvable, we found the correction

Puzzle types supported:
- Sudoku 4x4 and 9x9
- HexaSudoku 16x16
"""

import json
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from z3 import Int, Solver, And, Distinct, sat
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# CNN Model
# =============================================================================

class CNN_v2(nn.Module):
    """CNN for digit/character recognition (same architecture as training)."""

    def __init__(self, output_dim):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(model_path, num_classes):
    """Load trained CNN model."""
    model = CNN_v2(output_dim=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# =============================================================================
# Cell Extraction and Recognition
# =============================================================================

def extract_cell(board_img, row, col, cell_size, target_size=28):
    """Extract and preprocess a single cell from board image."""
    margin = 8  # Avoid grid lines
    y_start = row * cell_size + margin
    y_end = (row + 1) * cell_size - margin
    x_start = col * cell_size + margin
    x_end = (col + 1) * cell_size - margin

    cell = board_img[y_start:y_end, x_start:x_end]

    # Resize to target size
    cell_pil = Image.fromarray((cell * 255).astype(np.uint8))
    cell_pil = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(cell_pil) / 255.0


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


def recognize_with_alternatives(cell_img, model):
    """
    Recognize character in cell using CNN, returning full probability distribution.

    Returns:
        dict with keys:
        - prediction: top predicted class
        - confidence: probability of top prediction
        - second_best: second most likely class
        - second_conf: probability of second prediction
        - all_probs: full probability distribution
    """
    # Invert so ink is high value (MNIST convention for training)
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

        # Sort by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    return {
        'prediction': sorted_indices[0].item(),
        'confidence': sorted_probs[0].item(),
        'second_best': sorted_indices[1].item(),
        'second_conf': sorted_probs[1].item(),
        'all_probs': probs.numpy()
    }


def extract_puzzle_with_alternatives(image_path, model, size, board_pixels):
    """
    Extract puzzle values from board image with confidence alternatives.

    Returns:
        puzzle: size x size grid of values (0 = empty)
        alternatives: dict of {(row, col): recognition_info} for non-empty cells
    """
    cell_size = board_pixels // size

    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * size for _ in range(size)]
    alternatives = {}

    for row in range(size):
        for col in range(size):
            cell = extract_cell(board_img, row, col, cell_size)

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

def solve_sudoku(size, given_cells):
    """
    Solve Sudoku using Z3 constraint solver.

    Args:
        size: Grid size (4 or 9)
        given_cells: dict of {(row, col): value} for known cells

    Returns:
        size x size solution grid, or None if unsatisfiable
    """
    box_size = 2 if size == 4 else 3
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    # Cell range constraints
    for i in range(size):
        for j in range(size):
            s.add(And(X[i][j] >= 1, X[i][j] <= size))

    # Given cell constraints
    for (i, j), val in given_cells.items():
        s.add(X[i][j] == val)

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

    # Solve
    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
    return None


def solve_hexasudoku(given_cells):
    """
    Solve 16x16 HexaSudoku using Z3 constraint solver.

    Args:
        given_cells: dict of {(row, col): value} for known cells

    Returns:
        16x16 solution grid, or None if unsatisfiable
    """
    size = 16
    box_size = 4
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    # Cell range constraints
    for i in range(size):
        for j in range(size):
            s.add(And(X[i][j] >= 1, X[i][j] <= size))

    # Given cell constraints
    for (i, j), val in given_cells.items():
        s.add(X[i][j] == val)

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

    # Solve
    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
    return None


# =============================================================================
# Error Correction
# =============================================================================

@dataclass
class CorrectionResult:
    """Result of a single-digit correction attempt."""
    position: Tuple[int, int]
    original_value: int
    corrected_value: int
    original_conf: float
    second_conf: float
    solution: List[List[int]]
    attempts: int


@dataclass
class TwoDigitCorrectionResult:
    """Result of a two-digit correction attempt."""
    position1: Tuple[int, int]
    position2: Tuple[int, int]
    original_value1: int
    original_value2: int
    corrected_value1: int
    corrected_value2: int
    original_conf1: float
    original_conf2: float
    second_conf1: float
    second_conf2: float
    solution: List[List[int]]
    attempts: int


def attempt_single_error_correction(
    puzzle: List[List[int]],
    alternatives: Dict[Tuple[int, int], Dict],
    size: int,
    solver_fn,
    max_attempts: int = None
) -> Tuple[Optional[CorrectionResult], int, List[float]]:
    """
    Attempt to correct a single misclassified digit.

    Tries substituting each clue with its second-best CNN prediction,
    processing clues in order of lowest confidence first.

    Args:
        puzzle: Extracted puzzle grid
        alternatives: Dict of recognition info for each clue position
        size: Grid size
        solver_fn: Function to solve puzzle (takes given_cells dict)
        max_attempts: Maximum substitutions to try (None = all)

    Returns:
        (CorrectionResult or None, total_attempts, attempt_times)
    """
    # Build given_cells dict
    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    # Sort positions by confidence (lowest first = most likely errors)
    sorted_positions = sorted(
        alternatives.keys(),
        key=lambda pos: alternatives[pos]['confidence']
    )

    if max_attempts:
        sorted_positions = sorted_positions[:max_attempts]

    attempt_times = []

    for attempt_num, pos in enumerate(sorted_positions, 1):
        # Get second-best prediction
        second_best = alternatives[pos]['second_best']

        # Skip if second_best is 0 (empty) - doesn't make sense
        if second_best == 0:
            continue

        # Create modified puzzle with substitution
        modified_cells = given_cells.copy()
        modified_cells[pos] = second_best

        # Try to solve with this substitution
        start_time = time.time()
        solution = solver_fn(modified_cells)
        attempt_times.append(time.time() - start_time)

        if solution is not None:
            return CorrectionResult(
                position=pos,
                original_value=given_cells[pos],
                corrected_value=second_best,
                original_conf=alternatives[pos]['confidence'],
                second_conf=alternatives[pos]['second_conf'],
                solution=solution,
                attempts=attempt_num
            ), attempt_num, attempt_times

    return None, len(sorted_positions), attempt_times


def attempt_two_error_correction(
    puzzle: List[List[int]],
    alternatives: Dict[Tuple[int, int], Dict],
    size: int,
    solver_fn,
    max_candidates: int = 30,
    max_pairs: int = 500
) -> Tuple[Optional[TwoDigitCorrectionResult], int, List[float]]:
    """
    Attempt to correct two misclassified digits simultaneously.

    Tries substituting pairs of clues with their second-best CNN predictions,
    processing clue pairs in order of combined lowest confidence.

    Args:
        puzzle: Extracted puzzle grid
        alternatives: Dict of recognition info for each clue position
        size: Grid size
        solver_fn: Function to solve puzzle (takes given_cells dict)
        max_candidates: Max number of lowest-confidence clues to consider
        max_pairs: Maximum number of pairs to try

    Returns:
        (TwoDigitCorrectionResult or None, total_attempts, attempt_times)
    """
    from itertools import combinations

    # Build given_cells dict
    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    # Sort positions by confidence (lowest first = most likely errors)
    sorted_positions = sorted(
        alternatives.keys(),
        key=lambda pos: alternatives[pos]['confidence']
    )

    # Limit to top N lowest-confidence candidates
    candidates = sorted_positions[:min(max_candidates, len(sorted_positions))]

    # Filter out positions where second_best is 0 (empty)
    valid_candidates = [pos for pos in candidates if alternatives[pos]['second_best'] != 0]

    if len(valid_candidates) < 2:
        return None, 0, []

    # Generate pairs sorted by combined confidence (sum of both confidences)
    pairs = list(combinations(valid_candidates, 2))
    pairs.sort(key=lambda p: alternatives[p[0]]['confidence'] + alternatives[p[1]]['confidence'])

    # Limit number of pairs
    pairs = pairs[:max_pairs]

    attempt_times = []

    for attempt_num, (pos1, pos2) in enumerate(pairs, 1):
        second_best1 = alternatives[pos1]['second_best']
        second_best2 = alternatives[pos2]['second_best']

        # Create modified puzzle with both substitutions
        modified_cells = given_cells.copy()
        modified_cells[pos1] = second_best1
        modified_cells[pos2] = second_best2

        # Try to solve with this substitution
        start_time = time.time()
        solution = solver_fn(modified_cells)
        attempt_times.append(time.time() - start_time)

        if solution is not None:
            return TwoDigitCorrectionResult(
                position1=pos1,
                position2=pos2,
                original_value1=given_cells[pos1],
                original_value2=given_cells[pos2],
                corrected_value1=second_best1,
                corrected_value2=second_best2,
                original_conf1=alternatives[pos1]['confidence'],
                original_conf2=alternatives[pos2]['confidence'],
                second_conf1=alternatives[pos1]['second_conf'],
                second_conf2=alternatives[pos2]['second_conf'],
                solution=solution,
                attempts=attempt_num
            ), attempt_num, attempt_times

    return None, len(pairs), attempt_times


# =============================================================================
# Evaluation Utilities
# =============================================================================

def compare_puzzles(extracted, expected, size):
    """Compare extracted puzzle to expected and return error info."""
    errors = []
    for i in range(size):
        for j in range(size):
            if extracted[i][j] != expected[i][j]:
                errors.append((i, j, extracted[i][j], expected[i][j]))
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


def value_to_char(val, is_hex=False):
    """Convert value to display character."""
    if val == 0:
        return "."
    elif val <= 9:
        return str(val)
    elif is_hex:
        return chr(ord('A') + val - 10)
    return str(val)


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_puzzle_type(
    puzzle_type: str,
    puzzles: List[Dict],
    model,
    image_dir: str,
    size: int,
    board_pixels: int,
    solver_fn,
    max_correction_attempts: int = None,
    max_two_error_candidates: int = 30,
    max_two_error_pairs: int = 500
) -> List[Dict]:
    """
    Evaluate puzzles with error correction (single and two-digit).

    Returns list of result dicts for each puzzle.
    """
    results = []
    is_hex = size == 16

    for idx in range(len(puzzles)):
        if size == 16:
            image_path = os.path.join(image_dir, f"board16_{idx}.png")
        else:
            image_path = os.path.join(image_dir, f"board{size}_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        # Step 1: Extract puzzle with alternatives
        start_extract = time.time()
        extracted_puzzle, alternatives = extract_puzzle_with_alternatives(
            image_path, model, size, board_pixels
        )
        extract_time = time.time() - start_extract

        # Count clues
        total_clues = sum(1 for i in range(size) for j in range(size)
                         if extracted_puzzle[i][j] != 0)

        # Compare extraction to ground truth
        extraction_errors = compare_puzzles(extracted_puzzle, expected_puzzle, size)
        num_misclassified = len(extraction_errors)
        extraction_accuracy = 1.0 - (num_misclassified / (size * size))

        # Step 2: Try to solve extracted puzzle
        given_cells = {(i, j): extracted_puzzle[i][j]
                       for i in range(size) for j in range(size)
                       if extracted_puzzle[i][j] != 0}

        start_solve = time.time()
        solution = solver_fn(given_cells)
        first_solve_time = time.time() - start_solve

        # Determine original status
        if solution is not None and solutions_match(solution, expected_solution, size):
            original_status = "solved"
        elif solution is None:
            original_status = "unsat"
        else:
            original_status = "wrong_solution"

        # Initialize result
        result = {
            'puzzle_type': puzzle_type,
            'puzzle_idx': idx,
            'total_clues': total_clues,
            'original_status': original_status,
            'original_extraction_accuracy': extraction_accuracy,
            'num_misclassified_original': num_misclassified,
            # Single-error correction fields
            'single_correction_attempted': False,
            'single_correction_found': False,
            'single_attempts': 0,
            'error_row': None,
            'error_col': None,
            'original_value': None,
            'corrected_value': None,
            'original_confidence': None,
            'second_best_confidence': None,
            # Two-error correction fields
            'two_error_correction_attempted': False,
            'two_error_correction_found': False,
            'two_error_attempts': 0,
            'error2_row': None,
            'error2_col': None,
            'original_value2': None,
            'corrected_value2': None,
            'original_confidence2': None,
            'second_best_confidence2': None,
            # Timing
            'time_first_solve_attempt': first_solve_time,
            'time_per_correction_attempt': 0,
            'time_total_all_attempts': first_solve_time,
            # Final status
            'final_status': original_status if original_status == "solved" else "uncorrectable",
            'final_matches_ground_truth': original_status == "solved",
            'correction_type': 'none' if original_status == "solved" else None,
            'ground_truth_error_positions': str([(e[0], e[1]) for e in extraction_errors])
        }

        # Step 3: If not solved correctly, attempt single-error correction
        all_attempt_times = []
        if original_status != "solved":
            result['single_correction_attempted'] = True

            correction, attempts, attempt_times = attempt_single_error_correction(
                extracted_puzzle,
                alternatives,
                size,
                solver_fn,
                max_attempts=max_correction_attempts
            )

            result['single_attempts'] = attempts
            all_attempt_times.extend(attempt_times)

            if correction:
                result['single_correction_found'] = True
                result['error_row'] = correction.position[0]
                result['error_col'] = correction.position[1]
                result['original_value'] = correction.original_value
                result['corrected_value'] = correction.corrected_value
                result['original_confidence'] = correction.original_conf
                result['second_best_confidence'] = correction.second_conf

                # Verify against ground truth
                matches_truth = solutions_match(correction.solution, expected_solution, size)
                result['final_matches_ground_truth'] = matches_truth
                result['final_status'] = "corrected_single" if matches_truth else "wrong_correction"
                result['correction_type'] = 'single'

        # Step 4: If single-error correction failed, try two-error correction
        if original_status != "solved" and not result['single_correction_found']:
            result['two_error_correction_attempted'] = True

            two_correction, two_attempts, two_attempt_times = attempt_two_error_correction(
                extracted_puzzle,
                alternatives,
                size,
                solver_fn,
                max_candidates=max_two_error_candidates,
                max_pairs=max_two_error_pairs
            )

            result['two_error_attempts'] = two_attempts
            all_attempt_times.extend(two_attempt_times)

            if two_correction:
                result['two_error_correction_found'] = True
                # First error position
                result['error_row'] = two_correction.position1[0]
                result['error_col'] = two_correction.position1[1]
                result['original_value'] = two_correction.original_value1
                result['corrected_value'] = two_correction.corrected_value1
                result['original_confidence'] = two_correction.original_conf1
                result['second_best_confidence'] = two_correction.second_conf1
                # Second error position
                result['error2_row'] = two_correction.position2[0]
                result['error2_col'] = two_correction.position2[1]
                result['original_value2'] = two_correction.original_value2
                result['corrected_value2'] = two_correction.corrected_value2
                result['original_confidence2'] = two_correction.original_conf2
                result['second_best_confidence2'] = two_correction.second_conf2

                # Verify against ground truth
                matches_truth = solutions_match(two_correction.solution, expected_solution, size)
                result['final_matches_ground_truth'] = matches_truth
                result['final_status'] = "corrected_two" if matches_truth else "wrong_correction"
                result['correction_type'] = 'two'
            else:
                result['final_status'] = "uncorrectable"

        # Update timing
        if all_attempt_times:
            result['time_per_correction_attempt'] = sum(all_attempt_times) / len(all_attempt_times)
            result['time_total_all_attempts'] = first_solve_time + sum(all_attempt_times)

        results.append(result)

        # Progress
        if (idx + 1) % 25 == 0:
            correct_count = sum(1 for r in results if r['final_matches_ground_truth'])
            print(f"  {puzzle_type}: {idx+1}/{len(puzzles)} - "
                  f"Final accuracy: {correct_count/len(results)*100:.1f}%")

    return results


def main():
    print("=" * 70)
    print("Handwritten Error Detection and Correction")
    print("=" * 70)

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    sudoku_model_path = os.path.join(parent_dir, "Handwritten_Sudoku/models/handwritten_sudoku_cnn.pth")
    hex_model_path = os.path.join(parent_dir, "Handwritten_HexaSudoku/models/handwritten_hex_cnn.pth")

    sudoku_image_dir = os.path.join(parent_dir, "Handwritten_Sudoku/board_images")
    hex_image_dir = os.path.join(parent_dir, "Handwritten_HexaSudoku/board_images")

    puzzles_dir = os.path.join(base_dir, "puzzles")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # ==========================================================================
    # Evaluate Sudoku (4x4 and 9x9)
    # ==========================================================================
    if os.path.exists(sudoku_model_path):
        print(f"\n{'='*70}")
        print("Evaluating Sudoku (4x4 and 9x9)")
        print("=" * 70)

        model = load_model(sudoku_model_path, num_classes=10)
        print(f"Loaded Sudoku model from {sudoku_model_path}")

        with open(os.path.join(puzzles_dir, "puzzles_dict.json"), "r") as f:
            puzzles_ds = json.load(f)

        for size_str in ['4', '9']:
            size = int(size_str)
            puzzles = puzzles_ds.get(size_str, [])

            if not puzzles:
                continue

            print(f"\nEvaluating {len(puzzles)} {size}x{size} puzzles...")

            results = evaluate_puzzle_type(
                puzzle_type=f"sudoku_{size}x{size}",
                puzzles=puzzles,
                model=model,
                image_dir=sudoku_image_dir,
                size=size,
                board_pixels=900,
                solver_fn=lambda gc, s=size: solve_sudoku(s, gc),
                max_correction_attempts=None  # Try all for Sudoku
            )

            all_results.extend(results)

            # Summary for this size
            original_correct = sum(1 for r in results if r['original_status'] == "solved")
            final_correct = sum(1 for r in results if r['final_matches_ground_truth'])
            single_corrected = sum(1 for r in results if r['single_correction_found'])
            two_corrected = sum(1 for r in results if r['two_error_correction_found'])

            print(f"\n{size}x{size} Summary:")
            print(f"  Original accuracy: {original_correct}/{len(results)} ({original_correct/len(results)*100:.1f}%)")
            print(f"  Single-error corrections: {single_corrected}")
            print(f"  Two-error corrections: {two_corrected}")
            print(f"  Final accuracy: {final_correct}/{len(results)} ({final_correct/len(results)*100:.1f}%)")
    else:
        print(f"\nSkipping Sudoku - model not found at {sudoku_model_path}")

    # ==========================================================================
    # Evaluate HexaSudoku (16x16)
    # ==========================================================================
    if os.path.exists(hex_model_path):
        print(f"\n{'='*70}")
        print("Evaluating HexaSudoku (16x16)")
        print("=" * 70)

        model = load_model(hex_model_path, num_classes=17)
        print(f"Loaded HexaSudoku model from {hex_model_path}")

        with open(os.path.join(puzzles_dir, "unique_puzzles_dict.json"), "r") as f:
            puzzles_ds = json.load(f)

        puzzles = puzzles_ds.get('16', [])

        if puzzles:
            print(f"\nEvaluating {len(puzzles)} 16x16 puzzles...")

            results = evaluate_puzzle_type(
                puzzle_type="hexasudoku_16x16",
                puzzles=puzzles,
                model=model,
                image_dir=hex_image_dir,
                size=16,
                board_pixels=1600,
                solver_fn=solve_hexasudoku,
                max_correction_attempts=50  # Limit for efficiency
            )

            all_results.extend(results)

            # Summary
            original_correct = sum(1 for r in results if r['original_status'] == "solved")
            final_correct = sum(1 for r in results if r['final_matches_ground_truth'])
            single_corrected = sum(1 for r in results if r['single_correction_found'])
            two_corrected = sum(1 for r in results if r['two_error_correction_found'])

            print(f"\n16x16 Summary:")
            print(f"  Original accuracy: {original_correct}/{len(results)} ({original_correct/len(results)*100:.1f}%)")
            print(f"  Single-error corrections: {single_corrected}")
            print(f"  Two-error corrections: {two_corrected}")
            print(f"  Final accuracy: {final_correct}/{len(results)} ({final_correct/len(results)*100:.1f}%)")
    else:
        print(f"\nSkipping HexaSudoku - model not found at {hex_model_path}")

    # ==========================================================================
    # Save Results
    # ==========================================================================
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print("=" * 70)

    # Group by puzzle type
    by_type = {}
    for r in all_results:
        pt = r['puzzle_type']
        if pt not in by_type:
            by_type[pt] = []
        by_type[pt].append(r)

    summary_lines = []

    for pt in sorted(by_type.keys()):
        results = by_type[pt]
        n = len(results)

        original_correct = sum(1 for r in results if r['original_status'] == "solved")
        final_correct = sum(1 for r in results if r['final_matches_ground_truth'])
        single_corrections = sum(1 for r in results if r['single_correction_found'])
        two_corrections = sum(1 for r in results if r['two_error_correction_found'])
        total_corrections = single_corrections + two_corrections
        wrong_corrections = sum(1 for r in results if r['final_status'] == "wrong_correction")

        # Calculate average attempts for successful corrections
        avg_single_attempts = 0
        avg_two_attempts = 0
        if single_corrections > 0:
            avg_single_attempts = sum(r['single_attempts'] for r in results if r['single_correction_found']) / single_corrections
        if two_corrections > 0:
            avg_two_attempts = sum(r['two_error_attempts'] for r in results if r['two_error_correction_found']) / two_corrections

        avg_time = sum(r['time_total_all_attempts'] for r in results) / n

        print(f"\n{pt}:")
        print(f"  Original: {original_correct}/{n} ({original_correct/n*100:.1f}%)")
        print(f"  Single-error corrections: {single_corrections}")
        print(f"  Two-error corrections: {two_corrections}")
        print(f"  Total corrections: {total_corrections}")
        if wrong_corrections > 0:
            print(f"  Wrong corrections: {wrong_corrections}")
        print(f"  Final: {final_correct}/{n} ({final_correct/n*100:.1f}%)")
        print(f"  Improvement: +{final_correct - original_correct} puzzles")
        if single_corrections > 0:
            print(f"  Avg single-error attempts: {avg_single_attempts:.1f}")
        if two_corrections > 0:
            print(f"  Avg two-error attempts: {avg_two_attempts:.1f}")
        print(f"  Avg total time: {avg_time:.3f}s")

        summary_lines.append(f"{pt}: {original_correct/n*100:.1f}% -> {final_correct/n*100:.1f}% "
                            f"(+{final_correct - original_correct}, single:{single_corrections}, two:{two_corrections})")

    # Save CSV
    if all_results:
        csv_path = os.path.join(results_dir, "correction_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nDetailed results saved to {csv_path}")

        # Save summary
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Error Correction Summary\n")
            f.write("=" * 50 + "\n\n")
            for line in summary_lines:
                f.write(line + "\n")
        print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
