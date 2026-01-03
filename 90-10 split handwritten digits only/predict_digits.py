"""
Extended Error Correction using Top-K CNN Predictions (Digits Only).

This extends detect_errors.py to try not just the second-best CNN prediction,
but also the 3rd and 4th best predictions for each suspect cell.

Key difference from letter-based version:
- Values 10-16 are rendered as two-digit numbers (e.g., "16" = "1" + "6")
- Each digit position in a two-digit cell gets independent Top-K alternatives
- Error correction considers combinations of alternatives for both digit positions

Supports:
- HexaSudoku 16x16 (the main use case for this experiment)
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
from itertools import combinations, product


# =============================================================================
# Constants
# =============================================================================

SIZE = 16
BOX_SIZE = 4
BOARD_PIXELS = 1600
CELL_SIZE = BOARD_PIXELS // SIZE  # 100px

# Two-digit extraction parameters (matching generate_images.py)
TWO_DIGIT_SIZE = 40
TWO_DIGIT_LEFT_X = 10
TWO_DIGIT_RIGHT_X = 50
TWO_DIGIT_Y = 30

# Cell classification parameters
CELL_MARGIN = 8
SINGLE_INK_THRESHOLD = 0.01  # Minimum ink for a non-empty cell
TWO_DIGIT_RATIO_THRESHOLD = 0.15  # If both halves have >15% of total ink, it's double
TWO_DIGIT_MIN_INK = 0.005  # Minimum ink in each half for double digit


# =============================================================================
# CNN Model (same as detect_errors.py)
# =============================================================================

class CNN_v2(nn.Module):
    """CNN for digit recognition."""

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
# Cell Extraction and Classification
# =============================================================================

def extract_cell(board_img, row, col):
    """Extract a single cell from board image."""
    y_start = row * CELL_SIZE + CELL_MARGIN
    y_end = (row + 1) * CELL_SIZE - CELL_MARGIN
    x_start = col * CELL_SIZE + CELL_MARGIN
    x_end = (col + 1) * CELL_SIZE - CELL_MARGIN

    return board_img[y_start:y_end, x_start:x_end]


def get_ink_density(cell_img):
    """Calculate ink density (dark pixels) in cell. Returns fraction of ink."""
    return np.mean(1.0 - cell_img)


def classify_cell_type(cell_img):
    """
    Classify cell as 'empty', 'single', or 'double' digit.

    Key insight: Single digits (70x70) are centered with high ink concentration.
    Double digits (40x40 each) spread ink across the cell with lower center density.
    """
    total_ink = get_ink_density(cell_img)

    if total_ink < SINGLE_INK_THRESHOLD:
        return 'empty'

    h, w = cell_img.shape

    # Check center region ink density (middle 50% of cell)
    center = cell_img[:, int(w*0.25):int(w*0.75)]
    center_ink = get_ink_density(center)

    # Single digits have high center ink density (centered 70x70 digit)
    # Double digits have lower center ink density (two 40x40 digits spread out)
    center_ratio = center_ink / total_ink if total_ink > 0 else 0

    # Threshold at 1.7 separates single vs double digit cells
    if center_ratio > 1.7:
        return 'single'
    else:
        return 'double'


def extract_single_digit(cell_img, target_size=28):
    """Extract single digit from cell, centered."""
    cell_pil = Image.fromarray((cell_img * 255).astype(np.uint8))
    resized = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def extract_left_digit(cell_img, target_size=28):
    """Extract left digit from two-digit cell."""
    h, w = cell_img.shape

    # Use relative positions for robustness with different cell sizes after margin removal
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = 0
    x2 = w // 2

    left_region = cell_img[y1:y2, x1:x2]

    cell_pil = Image.fromarray((left_region * 255).astype(np.uint8))
    resized = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def extract_right_digit(cell_img, target_size=28):
    """Extract right digit from two-digit cell."""
    h, w = cell_img.shape

    # Use relative positions for robustness with different cell sizes after margin removal
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = w // 2
    x2 = w

    right_region = cell_img[y1:y2, x1:x2]

    cell_pil = Image.fromarray((right_region * 255).astype(np.uint8))
    resized = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


# =============================================================================
# Recognition with Top-K
# =============================================================================

def recognize_digit_with_top_k(digit_img, model, k=4):
    """
    Recognize a single digit image using CNN, returning top-K predictions.

    Returns dict with:
        - prediction: top prediction (0-9)
        - confidence: confidence of top prediction
        - top_k: list of (digit, confidence) for top K predictions
    """
    # Invert: MNIST expects ink=HIGH, our cells have ink=LOW (dark on white)
    digit_inverted = 1.0 - digit_img

    with torch.no_grad():
        tensor = torch.tensor(digit_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()

        # Sort by probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Build top-k list, excluding empty class (10) for digit recognition
    top_k = []
    for i in range(len(sorted_indices)):
        digit = sorted_indices[i].item()
        conf = sorted_probs[i].item()
        if digit != 10 and len(top_k) < k:  # Exclude empty class
            top_k.append((digit, conf))

    if not top_k:
        top_k = [(0, 0.0)]  # Fallback

    return {
        'prediction': top_k[0][0],
        'confidence': top_k[0][1],
        'top_k': top_k
    }


def recognize_cell_with_top_k(cell_img, model, k=4):
    """
    Recognize cell value with Top-K alternatives.

    Returns dict with:
        - value: recognized value (1-16)
        - cell_type: 'empty', 'single', or 'double'
        - alternatives: list of alternative values with confidences
        - digit_details: detailed info for each digit position (for two-digit cells)
    """
    cell_type = classify_cell_type(cell_img)

    if cell_type == 'empty':
        return {
            'value': 0,
            'cell_type': 'empty',
            'alternatives': [],
            'digit_details': None
        }

    elif cell_type == 'single':
        digit_img = extract_single_digit(cell_img)
        recog = recognize_digit_with_top_k(digit_img, model, k)

        # Single digit values: 1-9
        value = recog['prediction']
        if value == 0:
            value = 10  # If CNN predicts 0, it might be "10" misclassified

        # Build alternatives from top-k
        alternatives = []
        for digit, conf in recog['top_k']:
            alt_value = digit if digit != 0 else 10
            if alt_value != value:
                alternatives.append({'value': alt_value, 'confidence': conf})

        return {
            'value': value,
            'cell_type': 'single',
            'alternatives': alternatives[:k-1],
            'digit_details': {'single': recog}
        }

    else:  # double
        left_img = extract_left_digit(cell_img)
        right_img = extract_right_digit(cell_img)

        left_recog = recognize_digit_with_top_k(left_img, model, k)
        right_recog = recognize_digit_with_top_k(right_img, model, k)

        # DOMAIN CONSTRAINT 1: Tens digit must be 1 for two-digit cells (values 10-16)
        # This eliminates 1→7 confusion which accounts for 64% of double-digit errors
        tens_digit = 1  # Force to 1 (the only valid option for HexaSudoku 10-16)

        # DOMAIN CONSTRAINT 2: Ones digit must be 0-6 (values 10-16 only, no 17-19)
        ones_digit = right_recog['prediction']
        if ones_digit > 6 or ones_digit == 10:  # Invalid or empty class
            # Find best valid ones digit from top-K predictions
            found_valid = False
            for digit, _ in right_recog['top_k']:
                if digit <= 6:
                    ones_digit = digit
                    found_valid = True
                    break
            if not found_valid:
                ones_digit = 0  # Fallback

        value = tens_digit * 10 + ones_digit  # Always in range [10, 16]

        # Build alternatives by trying other valid ones digits (tens is always 1)
        alternatives = []

        # DOMAIN CONSTRAINT: Only try valid ones digits (0-6)
        for ones, ones_conf in right_recog['top_k']:
            if ones <= 6 and ones != ones_digit:  # Valid and different from current
                alt_val = 10 + ones  # Tens is always 1
                avg_conf = ones_conf
                alternatives.append({'value': alt_val, 'confidence': avg_conf,
                                   'type': 'ones_alt', 'tens': 1, 'ones': ones})

        # Note: We don't try changing the tens digit since it must be 1

        # Remove duplicates and sort by confidence
        seen = set([value])
        unique_alts = []
        for alt in sorted(alternatives, key=lambda x: -x['confidence']):
            if alt['value'] not in seen:
                seen.add(alt['value'])
                unique_alts.append(alt)

        return {
            'value': value,
            'cell_type': 'double',
            'alternatives': unique_alts[:k*2],  # More alternatives for double digits
            'digit_details': {
                'left': left_recog,
                'right': right_recog
            }
        }


def extract_puzzle_with_top_k(image_path, model, k=4):
    """
    Extract puzzle values from board image with top-K alternatives.

    Returns:
        puzzle: 16x16 list of recognized values
        cell_info: dict mapping (row, col) to recognition details
    """
    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * SIZE for _ in range(SIZE)]
    cell_info = {}

    for row in range(SIZE):
        for col in range(SIZE):
            cell = extract_cell(board_img, row, col)
            recog = recognize_cell_with_top_k(cell, model, k)

            puzzle[row][col] = recog['value']
            if recog['value'] != 0:
                cell_info[(row, col)] = recog

    return puzzle, cell_info


# =============================================================================
# Z3 Solver with Constraint Tracking
# =============================================================================

def solve_hexasudoku_with_tracking(given_cells):
    """Solve 16x16 HexaSudoku with constraint tracking for unsat core."""
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    clue_trackers = {}

    # Cell range constraints
    for i in range(SIZE):
        for j in range(SIZE):
            s.add(X[i][j] >= 1, X[i][j] <= SIZE)

    # Given cell constraints with tracking
    for (i, j), val in given_cells.items():
        tracker = Bool(f"clue_{i}_{j}")
        clue_trackers[tracker] = (i, j)
        s.assert_and_track(X[i][j] == val, tracker)

    # Row constraints
    for i in range(SIZE):
        s.add(Distinct([X[i][j] for j in range(SIZE)]))

    # Column constraints
    for j in range(SIZE):
        s.add(Distinct([X[i][j] for i in range(SIZE)]))

    # Box constraints (4x4)
    for box_row in range(SIZE // BOX_SIZE):
        for box_col in range(SIZE // BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

    if s.check() == sat:
        m = s.model()
        solution = [[m.evaluate(X[i][j]).as_long() for j in range(SIZE)] for i in range(SIZE)]
        return solution, []
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
# Error Detection via Unsat Core
# =============================================================================

@dataclass
class DetectionResult:
    """Result of error detection."""
    suspects: List[Tuple[int, int]]
    num_probes: int
    detection_time: float


def detect_errors_via_unsat_core(puzzle, max_probes=50):
    """Detect which clues are likely errors using unsat core analysis."""
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

    probes_to_do = initial_suspects[:max_probes]

    for pos in probes_to_do:
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


# =============================================================================
# Extended Correction with Top-K
# =============================================================================

@dataclass
class CorrectionResult:
    """Result of correction attempt."""
    success: bool
    correction_type: str
    num_errors_corrected: int
    positions_corrected: List[Tuple[int, int]]
    original_values: List[int]
    corrected_values: List[int]
    solution: Optional[List[List[int]]]
    correction_attempts: int
    total_solve_calls: int


def attempt_correction_with_top_k(
    puzzle: List[List[int]],
    cell_info: Dict[Tuple[int, int], Dict],
    max_errors: int = 5,
    max_suspects_per_level: int = 15,
    max_attempts_per_level: Optional[List[int]] = None
) -> Tuple[DetectionResult, CorrectionResult]:
    """
    Use unsat core detection + top-K predictions for correction.

    For two-digit cells, alternatives include combinations of different
    tens and ones digit predictions.
    """
    if max_attempts_per_level is None:
        max_attempts_per_level = [100, 400, 1000, 2000, 4000]

    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    # Step 1: Detect suspects via unsat core
    detection = detect_errors_via_unsat_core(puzzle)
    total_solve_calls = detection.num_probes

    # Build given_cells
    given_cells = {}
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    # Check if already solved
    if not detection.suspects:
        solution = solve_hexasudoku(given_cells)
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

    # Build list of (position, [alternative_values]) for each suspect
    suspect_alternatives = []
    for pos in suspects[:max_suspects_per_level]:
        if pos in cell_info:
            info = cell_info[pos]
            # Get alternative values
            alt_values = [alt['value'] for alt in info.get('alternatives', [])]
            # Filter to valid range and remove duplicates
            alt_values = [v for v in alt_values if 1 <= v <= 16 and v != info['value']]
            if alt_values:
                suspect_alternatives.append((pos, alt_values))

    error_names = ["single", "two", "three", "four", "five"]

    # Try correcting 1, 2, 3, ... up to max_errors
    for num_errors in range(1, max_errors + 1):
        if num_errors > len(suspect_alternatives):
            break

        max_attempts = max_attempts_per_level[num_errors - 1]
        error_name = error_names[num_errors - 1] if num_errors <= len(error_names) else f"{num_errors}"

        attempts_this_level = 0

        # Generate combinations of suspect positions
        position_combos = list(combinations(range(len(suspect_alternatives)), num_errors))

        for combo_indices in position_combos:
            if attempts_this_level >= max_attempts:
                break

            # Get the suspects and their alternatives for this combination
            combo_suspects = [suspect_alternatives[i] for i in combo_indices]
            positions = [s[0] for s in combo_suspects]
            alt_lists = [s[1] for s in combo_suspects]

            # Try all combinations of alternatives
            for value_combo in product(*alt_lists):
                if attempts_this_level >= max_attempts:
                    break

                # Build modified puzzle
                modified = given_cells.copy()
                original_values = []
                corrected_values = []

                for pos, val in zip(positions, value_combo):
                    original_values.append(given_cells[pos])
                    modified[pos] = val
                    corrected_values.append(val)

                solution = solve_hexasudoku(modified)
                total_solve_calls += 1
                correction_attempts += 1
                attempts_this_level += 1

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

def main():
    print("=" * 70)
    print("Extended Error Correction (Top-K CNN Predictions) - Digits Only")
    print("Tries 2nd, 3rd, and 4th best predictions for each suspect")
    print("Values 10-16 rendered as two-digit numbers")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "models/handwritten_digit_cnn.pth")
    image_dir = os.path.join(base_dir, "board_images")
    puzzles_path = os.path.join(base_dir, "puzzles/unique_puzzles_dict.json")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train_cnn.py first.")
        return

    print(f"\nLoading model from {model_path}")
    model = load_model(model_path, num_classes=11)  # 0-9 + empty

    # Load puzzles
    if not os.path.exists(puzzles_path):
        print(f"Error: Puzzles not found at {puzzles_path}")
        return

    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"Loaded {len(puzzles)} puzzles")

    # Check if images exist
    if not os.path.exists(image_dir):
        print(f"Error: Board images not found at {image_dir}")
        print("Run generate_images.py first.")
        return

    # Evaluate
    print(f"\n{'='*70}")
    print("Evaluating HexaSudoku 16x16 (Digits Only)")
    print("=" * 70)

    results = []
    cell_type_stats = {'single': 0, 'double': 0, 'empty': 0}

    for idx in range(len(puzzles)):
        image_path = os.path.join(image_dir, f"board16_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        # Extract puzzle with top-K alternatives
        extracted_puzzle, cell_info = extract_puzzle_with_top_k(image_path, model, k=4)

        # Count cell types
        for pos, info in cell_info.items():
            cell_type_stats[info['cell_type']] += 1

        # Find actual errors
        actual_errors = compare_puzzles(extracted_puzzle, expected_puzzle)
        num_actual_errors = len(actual_errors)

        # Run detection and correction
        start_time = time.time()
        detection, correction = attempt_correction_with_top_k(
            extracted_puzzle,
            cell_info,
            max_errors=5
        )
        total_time = time.time() - start_time

        # Check detection accuracy
        suspects_set = set(detection.suspects)
        actual_errors_set = set(actual_errors)
        suspects_include_actual = actual_errors_set.issubset(suspects_set) if actual_errors_set else True
        num_actual_in_suspects = len(suspects_set & actual_errors_set)

        # Verify solution
        final_matches_truth = False
        if correction.solution:
            final_matches_truth = solutions_match(correction.solution, expected_solution)

        result = {
            'puzzle_idx': idx,
            'num_clues': sum(1 for i in range(SIZE) for j in range(SIZE) if extracted_puzzle[i][j] != 0),
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
            'corrected_values': str(correction.corrected_values),
            'final_success': correction.success and final_matches_truth,
            'final_matches_ground_truth': final_matches_truth,
            'total_time': total_time
        }

        results.append(result)

        if (idx + 1) % 25 == 0:
            success_count = sum(1 for r in results if r['final_success'])
            print(f"  Progress: {idx+1}/{len(puzzles)} - "
                  f"Success: {success_count/len(results)*100:.1f}%")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    n = len(results)
    success_count = sum(1 for r in results if r['final_success'])

    correction_counts = Counter(r['correction_type'] for r in results)
    no_errors = correction_counts.get('none', 0)
    single = correction_counts.get('single', 0)
    two = correction_counts.get('two', 0)
    three = correction_counts.get('three', 0)
    four = correction_counts.get('four', 0)
    five = correction_counts.get('five', 0)
    uncorrectable = correction_counts.get('uncorrectable', 0)

    detection_accuracy = sum(1 for r in results if r['suspects_include_all_actual']) / n if n > 0 else 0
    avg_suspects = sum(r['num_suspects_detected'] for r in results) / n if n > 0 else 0
    avg_solve_calls = sum(r['total_solve_calls'] for r in results) / n if n > 0 else 0
    avg_time = sum(r['total_time'] for r in results) / n if n > 0 else 0

    print(f"\nHexaSudoku 16x16 (Digits Only):")
    print(f"  Final success: {success_count}/{n} ({success_count/n*100:.1f}%)" if n > 0 else "  No results")
    print(f"\n  Cell type distribution:")
    print(f"    - Single digit cells: {cell_type_stats['single']}")
    print(f"    - Double digit cells: {cell_type_stats['double']}")
    print(f"\n  Correction breakdown:")
    print(f"    - Direct solve (no errors): {no_errors}")
    print(f"    - 1-error corrections: {single}")
    print(f"    - 2-error corrections: {two}")
    print(f"    - 3-error corrections: {three}")
    print(f"    - 4-error corrections: {four}")
    print(f"    - 5-error corrections: {five}")
    print(f"    - Uncorrectable: {uncorrectable}")
    print(f"\n  Performance:")
    print(f"    - Detection found all actual errors: {detection_accuracy*100:.1f}%")
    print(f"    - Avg suspects detected: {avg_suspects:.1f}")
    print(f"    - Avg solve calls: {avg_solve_calls:.1f}")
    print(f"    - Avg time: {avg_time:.3f}s")

    # Save results
    if results:
        csv_path = os.path.join(results_dir, "prediction_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDetailed results saved to {csv_path}")

        summary_path = os.path.join(results_dir, "prediction_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("Extended Error Correction (Top-K CNN Predictions) - Digits Only\n")
            f.write("Tries 2nd, 3rd, and 4th best predictions for each suspect\n")
            f.write("Values 10-16 rendered as two-digit numbers\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Final success: {success_count}/{n} ({success_count/n*100:.1f}%)\n" if n > 0 else "No results\n")
            f.write(f"Avg solve calls: {avg_solve_calls:.1f}\n")
        print(f"Summary saved to {summary_path}")

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
