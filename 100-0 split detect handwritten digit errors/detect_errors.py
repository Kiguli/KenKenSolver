"""
Constraint-Based Error Detection for Handwritten Puzzles (100-0 Split).

Uses Z3's unsatisfiable core (unsat core) to identify which clues are causing
constraint violations, without checking ground truth. This approach identifies
errors through logical analysis rather than CNN confidence scores.

100-0 Split Configuration:
- CNN was trained on ALL available data (MNIST train+test, EMNIST train+test)
- Board images use the SAME training data (not held-out test data)
- This tests CNN performance when recognizing digits it has memorized
- Expected result: Near 100% recognition accuracy

Algorithm:
1. Solve puzzle with constraint tracking (each clue gets a Boolean tracker)
2. If UNSAT, extract unsat core to find which clues are in conflict
3. Probe each suspect by removing it to refine the suspect list
4. Try substitutions only on detected suspects (not all clues)

Supports:
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
from z3 import Int, Solver, And, Distinct, sat, unsat, Bool, Implies
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from itertools import combinations


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
    margin = 8
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
    """Recognize character in cell using CNN, returning probability distribution."""
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    return {
        'prediction': sorted_indices[0].item(),
        'confidence': sorted_probs[0].item(),
        'second_best': sorted_indices[1].item(),
        'second_conf': sorted_probs[1].item(),
        'all_probs': probs.numpy()
    }


def extract_puzzle_with_alternatives(image_path, model, size, board_pixels):
    """Extract puzzle values from board image with alternatives."""
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
# Z3 Solvers with Constraint Tracking
# =============================================================================

def solve_sudoku_with_tracking(size, given_cells):
    """
    Solve Sudoku with constraint tracking for unsat core extraction.

    Returns:
        (solution, suspects) where suspects is list of (row, col) positions
        that are involved in the unsatisfiable core.
    """
    box_size = 2 if size == 4 else 3
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    # Track which clues are in the problem
    clue_trackers = {}  # {Bool: (row, col)}

    # Cell range constraints (structural - not tracked)
    for i in range(size):
        for j in range(size):
            s.add(X[i][j] >= 1, X[i][j] <= size)

    # Given cell constraints WITH TRACKING
    for (i, j), val in given_cells.items():
        tracker = Bool(f"clue_{i}_{j}")
        clue_trackers[tracker] = (i, j)
        # Use assert_and_track for proper unsat core support
        s.assert_and_track(X[i][j] == val, tracker)

    # Row uniqueness (structural)
    for i in range(size):
        s.add(Distinct([X[i][j] for j in range(size)]))

    # Column uniqueness (structural)
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    # Box uniqueness (structural)
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
        solution = [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
        return solution, []
    else:
        # Get unsat core
        core = s.unsat_core()
        suspects = []
        for tracker in core:
            if tracker in clue_trackers:
                suspects.append(clue_trackers[tracker])
        return None, suspects


def solve_hexasudoku_with_tracking(given_cells):
    """
    Solve 16x16 HexaSudoku with constraint tracking.

    Returns:
        (solution, suspects) where suspects is list of (row, col) positions
        that are involved in the unsatisfiable core.
    """
    size = 16
    box_size = 4
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
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)], []
    else:
        core = s.unsat_core()
        suspects = [clue_trackers[t] for t in core if t in clue_trackers]
        return None, suspects


# Regular solvers (without tracking, for correction attempts)
def solve_sudoku(size, given_cells):
    """Solve Sudoku (regular, no tracking)."""
    box_size = 2 if size == 4 else 3
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


def solve_hexasudoku(given_cells):
    """Solve HexaSudoku (regular, no tracking)."""
    size = 16
    box_size = 4
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
# Error Detection via Unsat Core
# =============================================================================

@dataclass
class DetectionResult:
    """Result of error detection."""
    suspects: List[Tuple[int, int]]
    num_probes: int
    detection_time: float


def detect_errors_via_unsat_core(
    puzzle: List[List[int]],
    size: int,
    tracking_solver_fn,
    max_probes: int = 50
) -> DetectionResult:
    """
    Detect which clues are likely errors using unsat core analysis.

    Algorithm:
    1. Initial solve with tracking -> get suspect list from unsat core
    2. For each suspect, try removing it and re-solving
       - If solvable: this clue alone causes the conflict (high weight)
       - If still UNSAT: check which clues remain in conflict
    3. Rank suspects by conflict participation count

    Returns:
        DetectionResult with suspects sorted by conflict participation
    """
    start_time = time.time()

    # Build given_cells dict
    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    # Initial solve with tracking
    solution, initial_suspects = tracking_solver_fn(given_cells)

    if solution is not None:
        # No errors - puzzle solves directly
        return DetectionResult(
            suspects=[],
            num_probes=1,
            detection_time=time.time() - start_time
        )

    if not initial_suspects:
        # UNSAT but no suspects in core (shouldn't happen often)
        return DetectionResult(
            suspects=[],
            num_probes=1,
            detection_time=time.time() - start_time
        )

    # Refine suspects by probing
    suspect_counts = Counter()
    num_probes = 1  # Count initial solve

    # Probe each initial suspect
    probes_to_do = initial_suspects[:max_probes]

    for pos in probes_to_do:
        # Remove this clue and try solving
        modified = given_cells.copy()
        del modified[pos]

        sol, new_suspects = tracking_solver_fn(modified)
        num_probes += 1

        if sol is not None:
            # Removing this clue alone makes it solvable!
            # This is a strong indicator this clue is wrong
            suspect_counts[pos] += 10
        else:
            # Still UNSAT - see which clues remain in conflict
            for s in new_suspects:
                suspect_counts[s] += 1
            # Also weight the removed position if it was in original core
            if pos in initial_suspects:
                suspect_counts[pos] += 1

    # Sort suspects by conflict participation count (highest first)
    sorted_suspects = sorted(
        suspect_counts.keys(),
        key=lambda p: -suspect_counts[p]
    )

    # If we have no suspects from probing, use initial suspects
    if not sorted_suspects:
        sorted_suspects = initial_suspects

    return DetectionResult(
        suspects=sorted_suspects,
        num_probes=num_probes,
        detection_time=time.time() - start_time
    )


# =============================================================================
# Error Correction Using Detected Suspects
# =============================================================================

@dataclass
class CorrectionResult:
    """Result of correction attempt."""
    success: bool
    correction_type: str  # "none", "single", "two", "three", "four", "uncorrectable"
    num_errors_corrected: int
    positions_corrected: List[Tuple[int, int]]
    original_values: List[int]
    corrected_values: List[int]
    solution: Optional[List[List[int]]]
    correction_attempts: int
    total_solve_calls: int


def attempt_correction_with_detection(
    puzzle: List[List[int]],
    alternatives: Dict[Tuple[int, int], Dict],
    size: int,
    tracking_solver_fn,
    regular_solver_fn,
    max_errors: int = 4,
    max_suspects_per_level: int = 20,
    max_attempts_per_level: Optional[List[int]] = None
) -> Tuple[DetectionResult, CorrectionResult]:
    """
    Use unsat core detection to guide correction attempts.

    Supports correcting 1, 2, 3, 4 errors by trying combinations of suspects.
    The unsat core approach makes this feasible by limiting the search space
    to only clues that participate in constraint conflicts.

    Args:
        puzzle: Extracted puzzle values
        alternatives: CNN probability info for each clue position
        size: Puzzle size (4, 9, or 16)
        tracking_solver_fn: Solver that returns unsat core suspects
        regular_solver_fn: Fast solver for correction attempts
        max_errors: Maximum number of errors to try correcting (default 4)
        max_suspects_per_level: Max suspects to consider at each error level
        max_attempts_per_level: Max correction attempts per error level
            Default: [20, 100, 300, 500] for 1-4 errors

    Returns:
        (DetectionResult, CorrectionResult)
    """
    if max_attempts_per_level is None:
        # Default limits for up to 4 errors
        max_attempts_per_level = [20, 100, 300, 500]

    # Extend if needed
    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    # Step 1: Detect suspects via unsat core
    detection = detect_errors_via_unsat_core(
        puzzle, size, tracking_solver_fn
    )

    total_solve_calls = detection.num_probes

    # Build given_cells
    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    # Check if already solved
    if not detection.suspects:
        # Try solving to check if it works
        solution = regular_solver_fn(given_cells)
        total_solve_calls += 1
        if solution:
            return detection, CorrectionResult(
                success=True,
                correction_type="none",
                num_errors_corrected=0,
                positions_corrected=[],
                original_values=[],
                corrected_values=[],
                solution=solution,
                correction_attempts=0,
                total_solve_calls=total_solve_calls
            )

    suspects = detection.suspects
    correction_attempts = 0

    # Filter suspects to only those with valid alternatives
    valid_suspects = []
    for pos in suspects[:max_suspects_per_level]:
        if pos in alternatives:
            second_best = alternatives[pos]['second_best']
            if second_best != 0:
                valid_suspects.append(pos)

    # Error level names
    error_names = ["single", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

    # Try correcting 1, 2, 3, ... up to max_errors
    for num_errors in range(1, max_errors + 1):
        if num_errors > len(valid_suspects):
            # Not enough suspects to try this many errors
            break

        max_attempts = max_attempts_per_level[num_errors - 1]
        error_name = error_names[num_errors - 1] if num_errors <= len(error_names) else f"{num_errors}"

        # Generate combinations of suspects
        combos = list(combinations(valid_suspects, num_errors))

        # Limit number of attempts
        combos = combos[:max_attempts]

        for positions in combos:
            # Build modified puzzle with second-best predictions
            modified = given_cells.copy()
            corrected_values = []
            original_values = []

            for pos in positions:
                original_values.append(given_cells[pos])
                second_best = alternatives[pos]['second_best']
                modified[pos] = second_best
                corrected_values.append(second_best)

            solution = regular_solver_fn(modified)
            total_solve_calls += 1
            correction_attempts += 1

            if solution is not None:
                return detection, CorrectionResult(
                    success=True,
                    correction_type=error_name,
                    num_errors_corrected=num_errors,
                    positions_corrected=list(positions),
                    original_values=original_values,
                    corrected_values=corrected_values,
                    solution=solution,
                    correction_attempts=correction_attempts,
                    total_solve_calls=total_solve_calls
                )

    # Could not correct
    return detection, CorrectionResult(
        success=False,
        correction_type="uncorrectable",
        num_errors_corrected=0,
        positions_corrected=[],
        original_values=[],
        corrected_values=[],
        solution=None,
        correction_attempts=correction_attempts,
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

def evaluate_puzzle_type(
    puzzle_type: str,
    puzzles: List[Dict],
    model,
    image_dir: str,
    size: int,
    board_pixels: int,
    tracking_solver_fn,
    regular_solver_fn
) -> List[Dict]:
    """Evaluate puzzles with unsat core-based error detection."""
    results = []

    for idx in range(len(puzzles)):
        if size == 16:
            image_path = os.path.join(image_dir, f"board16_{idx}.png")
        else:
            image_path = os.path.join(image_dir, f"board{size}_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        # Extract puzzle
        extracted_puzzle, alternatives = extract_puzzle_with_alternatives(
            image_path, model, size, board_pixels
        )

        # Find actual errors (for validation only - not used in algorithm)
        actual_errors = compare_puzzles(extracted_puzzle, expected_puzzle, size)
        num_actual_errors = len(actual_errors)

        # Run detection and correction
        start_time = time.time()
        detection, correction = attempt_correction_with_detection(
            extracted_puzzle,
            alternatives,
            size,
            tracking_solver_fn,
            regular_solver_fn
        )
        total_time = time.time() - start_time

        # Check if detection found the actual errors
        suspects_set = set(detection.suspects)
        actual_errors_set = set(actual_errors)
        suspects_include_actual = actual_errors_set.issubset(suspects_set) if actual_errors_set else True
        num_actual_in_suspects = len(suspects_set & actual_errors_set)

        # Verify solution against ground truth
        final_matches_truth = False
        if correction.solution:
            final_matches_truth = solutions_match(correction.solution, expected_solution, size)

        result = {
            'puzzle_type': puzzle_type,
            'puzzle_idx': idx,
            'num_clues': sum(1 for i in range(size) for j in range(size) if extracted_puzzle[i][j] != 0),
            # Actual errors (from ground truth - for validation only)
            'num_actual_errors': num_actual_errors,
            'actual_error_positions': str(actual_errors),
            # Detection results
            'num_suspects_detected': len(detection.suspects),
            'suspect_positions': str(detection.suspects[:10]),  # First 10
            'detection_probes': detection.num_probes,
            'detection_time': detection.detection_time,
            # How well detection worked
            'suspects_include_all_actual': suspects_include_actual,
            'num_actual_in_suspects': num_actual_in_suspects,
            # Correction results
            'correction_type': correction.correction_type,
            'correction_attempts': correction.correction_attempts,
            'total_solve_calls': correction.total_solve_calls,
            'positions_corrected': str(correction.positions_corrected),
            # Final outcome
            'final_success': correction.success and final_matches_truth,
            'final_matches_ground_truth': final_matches_truth,
            'total_time': total_time
        }

        results.append(result)

        # Progress
        if (idx + 1) % 25 == 0:
            success_count = sum(1 for r in results if r['final_success'])
            print(f"  {puzzle_type}: {idx+1}/{len(puzzles)} - "
                  f"Success: {success_count/len(results)*100:.1f}%")

    return results


def main():
    print("=" * 70)
    print("Constraint-Based Error Detection (Unsat Core Analysis)")
    print("100-0 Split (Training Data Recognition)")
    print("=" * 70)

    # Paths - use models from this folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Use 100-0 trained model
    hex_model_path = os.path.join(base_dir, "models/handwritten_hex_cnn.pth")

    # Use local board_images (generated from training data)
    hex_image_dir = os.path.join(base_dir, "board_images")

    puzzles_dir = os.path.join(base_dir, "puzzles")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # ==========================================================================
    # Evaluate HexaSudoku (16x16) - main focus
    # ==========================================================================
    if os.path.exists(hex_model_path):
        print(f"\n{'='*70}")
        print("Evaluating HexaSudoku (16x16)")
        print("=" * 70)

        model = load_model(hex_model_path, num_classes=17)
        print(f"Loaded HexaSudoku model from {hex_model_path}")

        puzzles_path = os.path.join(puzzles_dir, "unique_puzzles_dict.json")
        if not os.path.exists(puzzles_path):
            # Try parent puzzles directory
            puzzles_path = os.path.join(parent_dir, "90-10 split detect handwritten digit errors/puzzles/unique_puzzles_dict.json")

        if os.path.exists(puzzles_path):
            with open(puzzles_path, "r") as f:
                puzzles_ds = json.load(f)

            puzzles = puzzles_ds.get('16', [])

            if puzzles:
                print(f"\nEvaluating {len(puzzles)} 16x16 puzzles...")
                print(f"Using board images from: {hex_image_dir}")

                results = evaluate_puzzle_type(
                    puzzle_type="hexasudoku_16x16",
                    puzzles=puzzles,
                    model=model,
                    image_dir=hex_image_dir,
                    size=16,
                    board_pixels=1600,
                    tracking_solver_fn=solve_hexasudoku_with_tracking,
                    regular_solver_fn=solve_hexasudoku
                )

                all_results.extend(results)

                # Summary
                success_count = sum(1 for r in results if r['final_success'])
                detection_found_actual = sum(1 for r in results if r['suspects_include_all_actual'])
                correction_counts = Counter(r['correction_type'] for r in results)
                avg_suspects = sum(r['num_suspects_detected'] for r in results) / len(results)
                avg_solve_calls = sum(r['total_solve_calls'] for r in results) / len(results)

                print(f"\n16x16 Summary:")
                print(f"  Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
                print(f"  Detection found actual errors: {detection_found_actual}/{len(results)}")
                print(f"  Corrections by error count: {dict(correction_counts)}")
                print(f"  Avg suspects detected: {avg_suspects:.1f}")
                print(f"  Avg solve calls: {avg_solve_calls:.1f}")
        else:
            print(f"\nSkipping HexaSudoku - puzzles not found at {puzzles_path}")
    else:
        print(f"\nSkipping HexaSudoku - model not found at {hex_model_path}")
        print("Run train_cnn.py first to train the model.")

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

        success_count = sum(1 for r in results if r['final_success'])

        # Count by correction type
        correction_counts = Counter(r['correction_type'] for r in results)
        no_errors = correction_counts.get('none', 0)
        single_corrected = correction_counts.get('single', 0)
        two_corrected = correction_counts.get('two', 0)
        three_corrected = correction_counts.get('three', 0)
        four_corrected = correction_counts.get('four', 0)
        uncorrectable = correction_counts.get('uncorrectable', 0)

        detection_accuracy = sum(1 for r in results if r['suspects_include_all_actual']) / n
        avg_suspects = sum(r['num_suspects_detected'] for r in results) / n
        avg_solve_calls = sum(r['total_solve_calls'] for r in results) / n
        avg_time = sum(r['total_time'] for r in results) / n

        print(f"\n{pt}:")
        print(f"  Final success: {success_count}/{n} ({success_count/n*100:.1f}%)")
        print(f"  Breakdown:")
        print(f"    - No errors (direct solve): {no_errors}")
        print(f"    - 1-error corrections: {single_corrected}")
        print(f"    - 2-error corrections: {two_corrected}")
        print(f"    - 3-error corrections: {three_corrected}")
        print(f"    - 4-error corrections: {four_corrected}")
        print(f"    - Uncorrectable: {uncorrectable}")
        print(f"  Detection found actual errors: {detection_accuracy*100:.1f}%")
        print(f"  Avg suspects per puzzle: {avg_suspects:.1f}")
        print(f"  Avg solve calls: {avg_solve_calls:.1f}")
        print(f"  Avg time: {avg_time:.3f}s")

        summary_lines.append(f"{pt}: {success_count/n*100:.1f}% success, "
                            f"avg {avg_solve_calls:.1f} solve calls")

    # Save CSV
    if all_results:
        csv_path = os.path.join(results_dir, "detection_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nDetailed results saved to {csv_path}")

        # Save summary
        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Constraint-Based Error Detection Summary (100-0 Split)\n")
            f.write("(Using Z3 Unsat Core Analysis)\n")
            f.write("=" * 50 + "\n\n")
            for line in summary_lines:
                f.write(line + "\n")
        print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
