"""
HexaSudoku Numeric Solver V2 - Using KenKen V2 ImprovedCNN

Solves 16x16 HexaSudoku puzzles where values 10-16 are rendered as two-digit
numbers (10, 11, 12, 13, 14, 15, 16) instead of hex letters (A-G).

Key improvements over V1:
- Uses ImprovedCNN (99.9% digit accuracy vs ~95% for CNN_v2)
- Should eliminate 1<->7 confusion which caused 64% of V1 errors in tens position

Domain constraints:
- Values 1-9: Single digit recognition
- Values 10-16: Two-digit recognition (tens=1, ones=0-6)

Pipeline: Image -> ImprovedCNN (digit recognition) -> Z3 (constraint solving) -> Solution
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

BOARD_PIXELS = 1600
SIZE = 16
BOX_SIZE = 4
NUM_DIGITS = 10  # Classes 0-9

# Path to KenKen V2 model weights (has 14 classes: 0-9 + operators)
KENKEN_MODEL_CLASSES = 14

# Two-digit cell detection
INK_THRESHOLD = 0.02


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

def extract_cell(board_img, row, col, cell_size, margin=8):
    """Extract a cell from board image."""
    y_start = row * cell_size + margin
    y_end = (row + 1) * cell_size - margin
    x_start = col * cell_size + margin
    x_end = (col + 1) * cell_size - margin

    return board_img[y_start:y_end, x_start:x_end]


def get_ink_density(cell_img):
    """Calculate ink density (amount of dark pixels)."""
    return 1.0 - np.mean(cell_img)


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


def classify_cell_type(cell_img):
    """
    Determine if cell has 0, 1, or 2 digits based on ink distribution.

    Returns: 'empty', 'single', or 'double'
    """
    if is_cell_empty(cell_img):
        return 'empty'

    h, w = cell_img.shape
    total_ink = get_ink_density(cell_img)

    if total_ink < INK_THRESHOLD:
        return 'empty'

    # Check center region ink density
    center = cell_img[:, int(w*0.25):int(w*0.75)]
    center_ink = get_ink_density(center)

    center_ratio = center_ink / total_ink if total_ink > 0 else 0

    # Single digits have high center ink density (centered digit)
    # Double digits have lower center ink density (two digits spread out)
    if center_ratio > 1.7:
        return 'single'
    else:
        return 'double'


def extract_left_digit(cell_img, target_size=28):
    """Extract left digit region from a two-digit cell."""
    h, w = cell_img.shape

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = 0
    x2 = w // 2

    digit_region = cell_img[y1:y2, x1:x2]

    pil_img = Image.fromarray((digit_region * 255).astype(np.uint8))
    resized = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(resized) / 255.0


def extract_right_digit(cell_img, target_size=28):
    """Extract right digit region from a two-digit cell."""
    h, w = cell_img.shape

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = w // 2
    x2 = w

    digit_region = cell_img[y1:y2, x1:x2]

    pil_img = Image.fromarray((digit_region * 255).astype(np.uint8))
    resized = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(resized) / 255.0


def extract_single_digit(cell_img, target_size=28):
    """Extract and resize a single digit cell."""
    pil_img = Image.fromarray((cell_img * 255).astype(np.uint8))
    resized = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def recognize_digit(digit_img, model):
    """
    Recognize a single digit using the ImprovedCNN.

    Only uses classes 0-9 (masks out operator classes 10-13).
    """
    # Invert for CNN (expects white digit on black background)
    digit_inverted = 1.0 - digit_img

    with torch.no_grad():
        tensor = torch.tensor(digit_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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


def recognize_cell(cell_img, model):
    """
    Recognize the value in a cell (handles both single and double digits).

    Domain constraints for HexaSudoku (values 1-16):
    - Single digits: 1-9
    - Double digits: 10-16 (tens=1, ones=0-6)

    Returns:
        dict with value, cell_type, confidence, alternatives
    """
    cell_type = classify_cell_type(cell_img)

    if cell_type == 'empty':
        return {
            'value': 0,
            'cell_type': 'empty',
            'confidence': 1.0,
            'alternatives': None
        }

    if cell_type == 'single':
        digit_img = extract_single_digit(cell_img)
        recog = recognize_digit(digit_img, model)

        # For single digit, value is 1-9
        # Class 0 could be misrecognition, treat as error (should be 1-9)
        value = recog['prediction']
        if value == 0:
            # 0 shouldn't appear in puzzle (values are 1-16)
            # Use second best if available
            if recog['second_best'] > 0:
                value = recog['second_best']
            else:
                value = 1  # Fallback

        return {
            'value': value,
            'cell_type': 'single',
            'confidence': recog['confidence'],
            'alternatives': {
                'single': recog
            }
        }

    else:  # double
        left_img = extract_left_digit(cell_img)
        right_img = extract_right_digit(cell_img)

        left_recog = recognize_digit(left_img, model)
        right_recog = recognize_digit(right_img, model)

        # Get digits
        tens_digit = left_recog['prediction']
        ones_digit = right_recog['prediction']

        # DOMAIN CONSTRAINT 1: Tens digit must be 1 for two-digit cells (values 10-16)
        # This eliminates 1->7 confusion which accounts for 64% of V1 double-digit errors
        tens_digit = 1  # Force to 1 (the only valid option for HexaSudoku 10-16)

        # DOMAIN CONSTRAINT 2: Ones digit must be 0-6 (values 10-16 only)
        if ones_digit > 6:
            # Find best valid ones digit from top-K predictions
            found_valid = False
            for digit, _ in right_recog.get('top_k', []):
                if digit <= 6:
                    ones_digit = digit
                    found_valid = True
                    break
            if not found_valid:
                ones_digit = 6  # Fallback: clamp to max valid

        value = tens_digit * 10 + ones_digit  # Always in range [10, 16]

        # Combined confidence
        combined_conf = left_recog['confidence'] * right_recog['confidence']

        return {
            'value': value,
            'cell_type': 'double',
            'confidence': combined_conf,
            'alternatives': {
                'tens': left_recog,
                'ones': right_recog
            }
        }


def extract_puzzle_with_alternatives(image_path, model):
    """Extract puzzle values from board image with alternatives."""
    cell_size = BOARD_PIXELS // SIZE

    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * SIZE for _ in range(SIZE)]
    alternatives = {}

    for row in range(SIZE):
        for col in range(SIZE):
            cell = extract_cell(board_img, row, col, cell_size)
            recog = recognize_cell(cell, model)

            puzzle[row][col] = recog['value']

            if recog['cell_type'] != 'empty':
                alternatives[(row, col)] = recog

    return puzzle, alternatives


# =============================================================================
# Z3 Solvers
# =============================================================================

def solve_hexasudoku_with_tracking(given_cells):
    """Solve 16x16 HexaSudoku with constraint tracking."""
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    clue_trackers = {}

    for i in range(SIZE):
        for j in range(SIZE):
            s.add(X[i][j] >= 1, X[i][j] <= SIZE)

    for (i, j), val in given_cells.items():
        tracker = Bool(f"clue_{i}_{j}")
        clue_trackers[tracker] = (i, j)
        s.assert_and_track(X[i][j] == val, tracker)

    for i in range(SIZE):
        s.add(Distinct([X[i][j] for j in range(SIZE)]))

    for j in range(SIZE):
        s.add(Distinct([X[i][j] for i in range(SIZE)]))

    for box_row in range(SIZE // BOX_SIZE):
        for box_col in range(SIZE // BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(SIZE)] for i in range(SIZE)], []
    else:
        core = s.unsat_core()
        suspects = [clue_trackers[t] for t in core if t in clue_trackers]
        return None, suspects


def solve_hexasudoku(given_cells):
    """Solve HexaSudoku (regular, no tracking)."""
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    for i in range(SIZE):
        for j in range(SIZE):
            s.add(And(X[i][j] >= 1, X[i][j] <= SIZE))

    for (i, j), val in given_cells.items():
        s.add(X[i][j] == val)

    for i in range(SIZE):
        s.add(Distinct([X[i][j] for j in range(SIZE)]))

    for j in range(SIZE):
        s.add(Distinct([X[i][j] for i in range(SIZE)]))

    for box_row in range(SIZE // BOX_SIZE):
        for box_col in range(SIZE // BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(SIZE)] for i in range(SIZE)]
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


def detect_errors_via_unsat_core(puzzle, max_probes=50):
    """Detect errors using unsat core analysis."""
    start_time = time.time()

    given_cells = {}
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    solution, initial_suspects = solve_hexasudoku_with_tracking(given_cells)

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

        sol, new_suspects = solve_hexasudoku_with_tracking(modified)
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


def get_alternative_values(recog, max_alternatives=3):
    """
    Get alternative values for error correction.

    For single digits: use second, third best predictions (1-9 only)
    For double digits: only try alternatives for the ones digit (tens is always 1)

    DOMAIN CONSTRAINT: For double-digit cells, only values {10-16} are valid.
    """
    alternatives = []

    if recog['cell_type'] == 'single':
        single = recog['alternatives']['single']
        probs = single['all_probs']
        sorted_idx = np.argsort(probs)[::-1]
        # Only valid single digits are 1-9
        for idx in sorted_idx[1:max_alternatives + 1]:
            if 1 <= idx <= 9:
                alternatives.append(int(idx))

    elif recog['cell_type'] == 'double':
        ones = recog['alternatives']['ones']
        ones_probs = ones['all_probs']

        # DOMAIN CONSTRAINT: Only consider ones digits 0-6 (valid for values 10-16)
        ones_sorted = [i for i in np.argsort(ones_probs)[::-1] if i <= 6][:max_alternatives + 1]

        current_ones = recog['value'] % 10

        # Generate alternatives by changing the ones digit (tens is always 1)
        for o in ones_sorted:
            if o != current_ones:
                new_val = 10 + o  # Tens is always 1
                if new_val not in alternatives:
                    alternatives.append(new_val)

    return alternatives[:max_alternatives]


def attempt_correction_with_detection(
    puzzle, alternatives,
    max_errors=4, max_suspects_per_level=20, max_attempts_per_level=None
):
    """Use unsat core detection to guide correction attempts."""
    if max_attempts_per_level is None:
        max_attempts_per_level = [20, 100, 300, 500]

    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    detection = detect_errors_via_unsat_core(puzzle)
    total_solve_calls = detection.num_probes

    given_cells = {}
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    if not detection.suspects:
        solution = solve_hexasudoku(given_cells)
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

    # Build alternatives for each suspect
    suspect_alternatives = []
    for pos in suspects[:max_suspects_per_level]:
        if pos in alternatives:
            alt_values = get_alternative_values(alternatives[pos])
            if alt_values:
                suspect_alternatives.append((pos, alt_values))

    error_names = ["single", "two", "three", "four"]

    for num_errors in range(1, max_errors + 1):
        if num_errors > len(suspect_alternatives):
            break

        max_attempts = max_attempts_per_level[num_errors - 1]
        error_name = error_names[num_errors - 1] if num_errors <= len(error_names) else f"{num_errors}"

        combos = list(combinations(range(len(suspect_alternatives)), num_errors))
        combos = combos[:max_attempts]

        for combo_indices in combos:
            combo_suspects = [suspect_alternatives[i] for i in combo_indices]
            positions = [s[0] for s in combo_suspects]
            alt_lists = [s[1] for s in combo_suspects]

            # Try first alternative for each position
            modified = given_cells.copy()
            original_values = []
            corrected_values = []

            for pos, alts in zip(positions, alt_lists):
                original_values.append(given_cells[pos])
                modified[pos] = alts[0]
                corrected_values.append(alts[0])

            solution = solve_hexasudoku(modified)
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

def compare_puzzles(extracted, expected):
    """Compare extracted puzzle to expected and return error positions."""
    errors = []
    for i in range(SIZE):
        for j in range(SIZE):
            if extracted[i][j] != expected[i][j]:
                errors.append((i, j))
    return errors


def solutions_match(computed, expected):
    """Check if computed solution matches expected."""
    if computed is None:
        return False
    for i in range(SIZE):
        for j in range(SIZE):
            if computed[i][j] != expected[i][j]:
                return False
    return True


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_puzzles(model, puzzles, image_dir):
    """Evaluate puzzles and return results."""
    results = []

    for idx in range(len(puzzles)):
        image_path = os.path.join(image_dir, f"board16_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        # Extract puzzle
        start_time = time.time()
        extracted_puzzle, alternatives = extract_puzzle_with_alternatives(image_path, model)
        extract_time = time.time() - start_time

        # Count extraction accuracy
        actual_errors = compare_puzzles(extracted_puzzle, expected_puzzle)
        num_actual_errors = len(actual_errors)

        # Run detection and correction
        detection, correction = attempt_correction_with_detection(
            extracted_puzzle, alternatives
        )
        total_time = time.time() - start_time

        # Verify solution against ground truth
        final_matches_truth = False
        if correction.solution:
            final_matches_truth = solutions_match(correction.solution, expected_solution)

        result = {
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
    print("HexaSudoku Numeric Solver V2 - Using KenKen V2 ImprovedCNN")
    print("=" * 70)

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    # Model path
    model_path = os.path.join(parent_dir, "models/improved_cnn_weights.pth")

    # Use board images from 90-10 split handwritten digits only
    image_dir = os.path.join(os.path.dirname(parent_dir),
                             "90-10 split handwritten digits only/board_images")

    # Use puzzles from 90-10 split handwritten digits only
    puzzles_path = os.path.join(os.path.dirname(parent_dir),
                                "90-10 split handwritten digits only/puzzles/unique_puzzles_dict.json")

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

    puzzles = puzzles_ds.get('16', [])

    if not puzzles:
        print("Error: No puzzles found")
        return

    print(f"\n{'='*70}")
    print(f"Evaluating {len(puzzles)} 16x16 puzzles...")
    print(f"Using board images from: {image_dir}")
    print("=" * 70)

    results = evaluate_puzzles(model, puzzles, image_dir)

    # Summary
    success_count = sum(1 for r in results if r['final_success'])
    correction_counts = Counter(r['correction_type'] for r in results)
    avg_time = sum(r['total_time'] for r in results) / len(results)
    avg_solve_calls = sum(r['total_solve_calls'] for r in results) / len(results)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\nHexaSudoku 16x16 (Numeric):")
    print(f"  Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print(f"  Breakdown: {dict(correction_counts)}")
    print(f"  Avg time: {avg_time:.3f}s")
    print(f"  Avg solve calls: {avg_solve_calls:.1f}")

    # Compare to V1
    v1_result = 58  # V1 achieved 58% with domain constraints
    print(f"\nComparison to V1:")
    print(f"  V1: {v1_result}%")
    print(f"  V2: {success_count/len(results)*100:.1f}%")
    print(f"  Change: {success_count/len(results)*100 - v1_result:+.1f}%")

    # Save results
    if results:
        csv_path = os.path.join(results_dir, "hexasudoku_numeric_v2_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
