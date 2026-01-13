"""
Constraint-Based Error Detection for Digits-Only HexaSudoku.

Key difference from letter-based version:
- Values 1-9: Single digit recognition
- Values 10-16: Two-digit recognition (extract and recognize each digit separately)

Uses Z3's unsat core to identify clues causing constraint violations.
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
from z3 import Int, Solver, And, Distinct, sat, Bool
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import combinations


# =============================================================================
# Constants for Two-Digit Detection
# =============================================================================

# Cell layout constants (must match generate_images.py)
CELL_SIZE = 100
TWO_DIGIT_SIZE = 40
TWO_DIGIT_LEFT_X = 10
TWO_DIGIT_RIGHT_X = 50
TWO_DIGIT_Y = 30

# Thresholds for digit detection
INK_THRESHOLD = 0.02  # Minimum ink density to consider a region as having content
TWO_DIGIT_RATIO_THRESHOLD = 0.3  # Both halves must have at least this ratio of max ink


# =============================================================================
# CNN Model
# =============================================================================

class CNN_v2(nn.Module):
    """CNN for digit recognition (0-9 + empty)."""

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
# Cell Extraction and Two-Digit Recognition
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
    # Ink is dark (low values), so we invert for density
    return 1.0 - np.mean(cell_img)


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


def classify_cell_type(cell_img):
    """
    Determine if cell has 0, 1, or 2 digits based on ink distribution.

    Key insight: Single digits (70x70) are centered with high ink concentration.
    Double digits (40x40 each) spread ink across the cell with lower center density.

    Returns:
        'empty', 'single', or 'double'
    """
    if is_cell_empty(cell_img):
        return 'empty'

    h, w = cell_img.shape
    total_ink = get_ink_density(cell_img)

    if total_ink < INK_THRESHOLD:
        return 'empty'

    # Check center region ink density (middle 50% of cell)
    center = cell_img[:, int(w*0.25):int(w*0.75)]
    center_ink = get_ink_density(center)

    # Single digits have high center ink density (centered 70x70 digit)
    # Double digits have lower center ink density (two 40x40 digits spread out)
    # The center_ink/total_ink ratio is a good discriminator
    center_ratio = center_ink / total_ink if total_ink > 0 else 0

    # Based on observation:
    # - Single digits: center_ink ~0.13-0.17, total ~0.06-0.09, ratio ~2.0
    # - Double digits: center_ink ~0.04-0.07, total ~0.03-0.05, ratio ~1.2-1.5
    # Threshold at 1.7 separates them well
    if center_ratio > 1.7:
        return 'single'
    else:
        return 'double'


def extract_left_digit(cell_img, target_size=28):
    """Extract left digit region from a two-digit cell."""
    h, w = cell_img.shape

    # Use the known layout
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = 0
    x2 = w // 2

    digit_region = cell_img[y1:y2, x1:x2]

    # Resize to 28x28 for CNN
    pil_img = Image.fromarray((digit_region * 255).astype(np.uint8))
    resized = pil_img.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(resized) / 255.0


def extract_right_digit(cell_img, target_size=28):
    """Extract right digit region from a two-digit cell."""
    h, w = cell_img.shape

    # Use the known layout
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = w // 2
    x2 = w

    digit_region = cell_img[y1:y2, x1:x2]

    # Resize to 28x28 for CNN
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
    Recognize a single digit using the CNN.

    Returns:
        dict with prediction, confidence, second_best, top_k, etc.
    """
    # Invert for CNN (CNN expects white digit on black background)
    digit_inverted = 1.0 - digit_img

    with torch.no_grad():
        tensor = torch.tensor(digit_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Build top-k list (excluding empty class 10)
    top_k = []
    for i in range(len(sorted_indices)):
        digit = sorted_indices[i].item()
        conf = sorted_probs[i].item()
        if digit != 10 and len(top_k) < 4:
            top_k.append((digit, conf))

    return {
        'prediction': sorted_indices[0].item(),
        'confidence': sorted_probs[0].item(),
        'second_best': sorted_indices[1].item(),
        'second_conf': sorted_probs[1].item(),
        'all_probs': probs.numpy(),
        'top_k': top_k
    }


def recognize_cell(cell_img, model):
    """
    Recognize the value in a cell (handles both single and double digits).

    Returns:
        dict with:
        - value: recognized value (1-16)
        - cell_type: 'empty', 'single', or 'double'
        - confidence: overall confidence
        - alternatives: info for error correction
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

        # For single digit, value is the digit itself (1-9)
        # Class 10 is empty, so we filter that
        if recog['prediction'] == 10:  # Empty class
            return {
                'value': 0,
                'cell_type': 'empty',
                'confidence': recog['confidence'],
                'alternatives': recog
            }

        return {
            'value': recog['prediction'],
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

        # Get digits (filter out empty class 10)
        tens_digit = left_recog['prediction']
        ones_digit = right_recog['prediction']

        # DOMAIN CONSTRAINT 1: Tens digit must be 1 for two-digit cells (values 10-16)
        # This eliminates 1→7 confusion which accounts for 64% of double-digit errors
        tens_digit = 1  # Force to 1 (the only valid option for HexaSudoku 10-16)

        # DOMAIN CONSTRAINT 2: Ones digit must be 0-6 (values 10-16 only, no 17-19)
        if ones_digit == 10:  # Empty class
            ones_digit = 0
        elif ones_digit > 6:
            # Find best valid ones digit from top-K predictions
            found_valid = False
            for digit, _ in right_recog.get('top_k', []):
                if digit <= 6 and digit != 10:
                    ones_digit = digit
                    found_valid = True
                    break
            if not found_valid:
                ones_digit = 6  # Fallback: clamp to max valid

        value = tens_digit * 10 + ones_digit  # Always in range [10, 16]

        # Combined confidence is product of individual confidences
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
            recog = recognize_cell(cell, model)

            puzzle[row][col] = recog['value']

            if recog['cell_type'] != 'empty':
                alternatives[(row, col)] = recog

    return puzzle, alternatives


# =============================================================================
# Z3 Solver with Constraint Tracking
# =============================================================================

def solve_hexasudoku_with_tracking(given_cells):
    """Solve 16x16 HexaSudoku with constraint tracking."""
    size = 16
    box_size = 4
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    clue_trackers = {}

    for i in range(size):
        for j in range(size):
            s.add(X[i][j] >= 1, X[i][j] <= size)

    for (i, j), val in given_cells.items():
        tracker = Bool(f"clue_{i}_{j}")
        clue_trackers[tracker] = (i, j)
        s.assert_and_track(X[i][j] == val, tracker)

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
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)], []
    else:
        core = s.unsat_core()
        suspects = [clue_trackers[t] for t in core if t in clue_trackers]
        return None, suspects


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


def detect_errors_via_unsat_core(puzzle, size, tracking_solver_fn, max_probes=50):
    """Detect errors using unsat core analysis."""
    start_time = time.time()

    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    solution, initial_suspects = tracking_solver_fn(given_cells)

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

        sol, new_suspects = tracking_solver_fn(modified)
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

    For single digits: use second, third best predictions
    For double digits: only try alternatives for the ones digit (tens is always 1)

    DOMAIN CONSTRAINT: For double-digit cells, only values {10-16} are valid.
    - Tens digit is always 1
    - Ones digit must be in {0, 1, 2, 3, 4, 5, 6}
    """
    alternatives = []

    if recog['cell_type'] == 'single':
        single = recog['alternatives']['single']
        # Get top alternatives (excluding empty class 10)
        probs = single['all_probs']
        sorted_idx = np.argsort(probs)[::-1]
        for idx in sorted_idx[1:max_alternatives + 1]:
            if idx != 10:  # Skip empty class
                alternatives.append(int(idx))

    elif recog['cell_type'] == 'double':
        ones = recog['alternatives']['ones']
        ones_probs = ones['all_probs']

        # DOMAIN CONSTRAINT: Only consider ones digits 0-6 (valid for values 10-16)
        # Filter to valid ones digits and sort by probability
        ones_sorted = [i for i in np.argsort(ones_probs)[::-1]
                       if i != 10 and i <= 6][:max_alternatives + 1]

        current_ones = recog['value'] % 10

        # Generate alternatives by changing the ones digit (tens is always 1)
        for o in ones_sorted:
            if o != current_ones:
                new_val = 10 + o  # Tens is always 1
                if new_val not in alternatives:
                    alternatives.append(new_val)

        # Note: We don't try changing the tens digit since it must be 1

    return alternatives[:max_alternatives]


def attempt_correction_with_detection(
    puzzle, alternatives, size, tracking_solver_fn, regular_solver_fn,
    max_errors=4, max_suspects_per_level=20, max_attempts_per_level=None
):
    """Use unsat core detection to guide correction attempts."""
    if max_attempts_per_level is None:
        max_attempts_per_level = [20, 100, 300, 500]

    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    detection = detect_errors_via_unsat_core(puzzle, size, tracking_solver_fn)
    total_solve_calls = detection.num_probes

    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    if not detection.suspects:
        solution = regular_solver_fn(given_cells)
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

            solution = regular_solver_fn(modified)
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
# Evaluation
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


def evaluate_puzzle_type(puzzle_type, puzzles, model, image_dir, size, board_pixels,
                         tracking_solver_fn, regular_solver_fn):
    """Evaluate puzzles with unsat core-based error detection."""
    results = []

    for idx in range(len(puzzles)):
        image_path = os.path.join(image_dir, f"board16_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        extracted_puzzle, alternatives = extract_puzzle_with_alternatives(
            image_path, model, size, board_pixels
        )

        actual_errors = compare_puzzles(extracted_puzzle, expected_puzzle, size)
        num_actual_errors = len(actual_errors)

        start_time = time.time()
        detection, correction = attempt_correction_with_detection(
            extracted_puzzle, alternatives, size,
            tracking_solver_fn, regular_solver_fn
        )
        total_time = time.time() - start_time

        suspects_set = set(detection.suspects)
        actual_errors_set = set(actual_errors)
        suspects_include_actual = actual_errors_set.issubset(suspects_set) if actual_errors_set else True
        num_actual_in_suspects = len(suspects_set & actual_errors_set)

        final_matches_truth = False
        if correction.solution:
            final_matches_truth = solutions_match(correction.solution, expected_solution, size)

        result = {
            'puzzle_type': puzzle_type,
            'puzzle_idx': idx,
            'num_clues': sum(1 for i in range(size) for j in range(size) if extracted_puzzle[i][j] != 0),
            'num_actual_errors': num_actual_errors,
            'actual_error_positions': str(actual_errors),
            'num_suspects_detected': len(detection.suspects),
            'suspect_positions': str(detection.suspects[:10]),
            'detection_probes': detection.num_probes,
            'detection_time': detection.detection_time,
            'suspects_include_all_actual': suspects_include_actual,
            'num_actual_in_suspects': num_actual_in_suspects,
            'correction_type': correction.correction_type,
            'correction_attempts': correction.correction_attempts,
            'total_solve_calls': correction.total_solve_calls,
            'positions_corrected': str(correction.positions_corrected),
            'final_success': correction.success and final_matches_truth,
            'final_matches_ground_truth': final_matches_truth,
            'total_time': total_time
        }

        results.append(result)

        if (idx + 1) % 25 == 0:
            success_count = sum(1 for r in results if r['final_success'])
            print(f"  {puzzle_type}: {idx+1}/{len(puzzles)} - "
                  f"Success: {success_count/len(results)*100:.1f}%")

    return results


def main():
    print("=" * 70)
    print("Constraint-Based Error Detection (Digits Only)")
    print("Two-digit recognition for values 10-16")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "models/handwritten_digit_cnn.pth")
    image_dir = os.path.join(base_dir, "board_images")
    puzzles_dir = os.path.join(base_dir, "puzzles")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    if os.path.exists(model_path):
        print(f"\n{'='*70}")
        print("Evaluating HexaSudoku (16x16) - Digits Only")
        print("=" * 70)

        model = load_model(model_path, num_classes=11)
        print(f"Loaded digit model from {model_path}")

        puzzles_path = os.path.join(puzzles_dir, "unique_puzzles_dict.json")

        if os.path.exists(puzzles_path):
            with open(puzzles_path, "r") as f:
                puzzles_ds = json.load(f)

            puzzles = puzzles_ds.get('16', [])

            if puzzles:
                print(f"\nEvaluating {len(puzzles)} 16x16 puzzles...")
                print(f"Using board images from: {image_dir}")

                results = evaluate_puzzle_type(
                    puzzle_type="hexasudoku_16x16_digits",
                    puzzles=puzzles, model=model,
                    image_dir=image_dir, size=16, board_pixels=1600,
                    tracking_solver_fn=solve_hexasudoku_with_tracking,
                    regular_solver_fn=solve_hexasudoku
                )

                all_results.extend(results)

                success_count = sum(1 for r in results if r['final_success'])
                correction_counts = Counter(r['correction_type'] for r in results)
                avg_suspects = sum(r['num_suspects_detected'] for r in results) / len(results)
                avg_solve_calls = sum(r['total_solve_calls'] for r in results) / len(results)

                print(f"\n16x16 Summary:")
                print(f"  Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
                print(f"  Corrections by type: {dict(correction_counts)}")
                print(f"  Avg suspects detected: {avg_suspects:.1f}")
                print(f"  Avg solve calls: {avg_solve_calls:.1f}")
        else:
            print(f"\nSkipping - puzzles not found at {puzzles_path}")
    else:
        print(f"\nSkipping - model not found at {model_path}")
        print("Run train_cnn.py first to train the model.")

    # Save Results
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print("=" * 70)

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
        correction_counts = Counter(r['correction_type'] for r in results)

        avg_solve_calls = sum(r['total_solve_calls'] for r in results) / n
        avg_time = sum(r['total_time'] for r in results) / n

        print(f"\n{pt}:")
        print(f"  Final success: {success_count}/{n} ({success_count/n*100:.1f}%)")
        print(f"  Breakdown: {dict(correction_counts)}")
        print(f"  Avg solve calls: {avg_solve_calls:.1f}")
        print(f"  Avg time: {avg_time:.3f}s")

        summary_lines.append(f"{pt}: {success_count/n*100:.1f}% success")

    if all_results:
        csv_path = os.path.join(results_dir, "detection_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nDetailed results saved to {csv_path}")

        summary_path = os.path.join(results_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Constraint-Based Error Detection Summary (Digits Only)\n")
            f.write("=" * 50 + "\n\n")
            for line in summary_lines:
                f.write(line + "\n")
        print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
