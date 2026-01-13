# -*- coding: utf-8 -*-
import sys
sys.stdout = sys.stderr  # Force unbuffered output
"""
Unified Solver for all Sudoku/HexaSudoku puzzle types.

Uses the unified ImprovedCNN (17 classes: 0-9 digits + A-G letters)
to recognize characters and Z3 SMT solver for constraint solving.

Supported puzzle types:
- Sudoku 4x4: digits 1-4 (900x900 boards, 225px cells)
- Sudoku 9x9: digits 1-9 (900x900 boards, 100px cells)
- HexaSudoku A-G: values 1-16 as 1-9,A-G (1600x1600 boards, 100px cells)
- HexaSudoku Numeric: values 1-16 as 1-9,10-16 (1600x1600 boards, 100px cells)

Usage:
    python solve_all_sizes.py
"""

import sys
import json
import os
import csv
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from z3 import Int, Solver, And, Distinct, sat, Bool
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from itertools import combinations

# Add parent directories to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir / 'models'))

from improved_cnn import ImprovedCNN


# =============================================================================
# Constants
# =============================================================================

NUM_CLASSES = 17  # 0-9 + A-G

# Board configurations by puzzle type
BOARD_CONFIGS = {
    'sudoku_4x4': {'size': 4, 'board_pixels': 900, 'box_size': 2},
    'sudoku_9x9': {'size': 9, 'board_pixels': 900, 'box_size': 3},
    'hexasudoku_ag': {'size': 16, 'board_pixels': 1600, 'box_size': 4},
    'hexasudoku_numeric': {'size': 16, 'board_pixels': 1600, 'box_size': 4},
}

# Two-digit layout for HexaSudoku Numeric (matching generate_images.py)
TWO_DIGIT_RATIO_THRESHOLD = 0.3
INK_THRESHOLD = 0.02


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path):
    """Load trained unified CNN model."""
    model = ImprovedCNN(output_dim=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# =============================================================================
# Cell Extraction (matching training preprocessing)
# =============================================================================

import cv2 as cv

IMG_SIZE = 28


def extract_cell(board_img, row, col, cell_size, margin=8):
    """Extract a cell from board image."""
    y_start = row * cell_size + margin
    y_end = (row + 1) * cell_size - margin
    x_start = col * cell_size + margin
    x_end = (col + 1) * cell_size - margin
    return board_img[y_start:y_end, x_start:x_end]


def get_contours(threshold_image):
    """Find contours in thresholded image."""
    kernel = np.ones((2, 2), np.uint8)
    img_dilate = cv.dilate(threshold_image, kernel, iterations=1)
    contours, _ = cv.findContours(img_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def extract_character_image(cell_img, contour):
    """Extract and normalize a single character from contour (matches training)."""
    x, y, w, h = cv.boundingRect(contour)

    # Filter too small contours
    if w < 5 or h < 5:
        return None

    char_img = cell_img[y:y+h, x:x+w]

    if char_img.size == 0:
        return None

    # Resize preserving aspect ratio
    char_pil = Image.fromarray((char_img * 255).astype(np.uint8))
    aspect = w / h

    if aspect > 1:
        new_w = IMG_SIZE - 4  # Leave margin
        new_h = max(1, int((IMG_SIZE - 4) / aspect))
    else:
        new_h = IMG_SIZE - 4
        new_w = max(1, int((IMG_SIZE - 4) * aspect))

    resized = char_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return np.array(canvas).astype(np.float32) / 255.0


def extract_single_character(cell_img):
    """
    Extract a single character from cell using contour detection.
    Matches the training preprocessing pipeline.
    """
    # Convert to uint8 for OpenCV
    img_uint8 = (cell_img * 255).astype(np.uint8)
    _, thresh = cv.threshold(img_uint8, 127, 255, cv.THRESH_BINARY_INV)

    # Check if cell has content
    if np.sum(thresh) < 100:  # Mostly empty
        return None

    # Find contours
    contours = get_contours(thresh)

    if not contours:
        return None

    # Find largest contour by area
    largest_contour = max(contours, key=cv.contourArea)

    # Extract character
    char_img = extract_character_image(cell_img, largest_contour)

    return char_img


def resize_to_28x28(img):
    """Resize image to 28x28 for CNN input (fallback)."""
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    resized = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def get_ink_density(cell_img):
    """Calculate ink density (amount of dark pixels)."""
    return 1.0 - np.mean(cell_img)


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


# =============================================================================
# Recognition Functions
# =============================================================================

def recognize_with_topk(img, model, k=5):
    """
    Recognize character with top-k predictions.

    Returns dict with prediction, confidence, top_k list.
    """
    # NO inversion - model trained on white bg, black ink (same as board images)
    with torch.no_grad():
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    top_k = [(sorted_indices[i].item(), sorted_probs[i].item()) for i in range(min(k, len(sorted_indices)))]

    return {
        'prediction': sorted_indices[0].item(),
        'confidence': sorted_probs[0].item(),
        'top_k': top_k,
        'all_probs': probs.numpy()
    }


# =============================================================================
# Sudoku Recognition (4x4 and 9x9)
# =============================================================================

def recognize_sudoku_cell(cell_img, model, max_digit):
    """
    Recognize a Sudoku cell (single digit 1-max_digit).

    Args:
        cell_img: Cell image (0-1 float, white bg)
        model: CNN model
        max_digit: Maximum digit value (4 for 4x4, 9 for 9x9)

    Returns:
        dict with value, confidence, alternatives
    """
    # Use contour-based extraction (matching training)
    char_img = extract_single_character(cell_img)

    if char_img is None:
        return {'value': 0, 'confidence': 1.0, 'alternatives': None}

    recog = recognize_with_topk(char_img, model)

    # Filter to valid digits (1 to max_digit)
    valid_top_k = [(d, c) for d, c in recog['top_k'] if 1 <= d <= max_digit]

    if not valid_top_k:
        # Fallback: find best valid digit
        probs = recog['all_probs']
        best_digit = np.argmax(probs[1:max_digit+1]) + 1
        return {
            'value': best_digit,
            'confidence': probs[best_digit],
            'alternatives': {'single': recog}
        }

    return {
        'value': valid_top_k[0][0],
        'confidence': valid_top_k[0][1],
        'alternatives': {'single': recog, 'valid_top_k': valid_top_k}
    }


def extract_sudoku_puzzle(image_path, model, size, board_pixels):
    """Extract Sudoku puzzle from image."""
    cell_size = board_pixels // size

    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * size for _ in range(size)]
    alternatives = {}

    for row in range(size):
        for col in range(size):
            cell = extract_cell(board_img, row, col, cell_size)
            recog = recognize_sudoku_cell(cell, model, max_digit=size)

            puzzle[row][col] = recog['value']
            if recog['value'] != 0:
                alternatives[(row, col)] = recog

    return puzzle, alternatives


# =============================================================================
# HexaSudoku A-G Recognition (16x16 with letters)
# =============================================================================

def recognize_hexasudoku_ag_cell(cell_img, model):
    """
    Recognize a HexaSudoku A-G cell.

    Values 1-9: digits 1-9 (CNN classes 1-9)
    Values 10-16: letters A-G (CNN classes 10-16)

    Returns:
        dict with value (1-16), confidence, alternatives
    """
    # Use contour-based extraction (matching training)
    char_img = extract_single_character(cell_img)

    if char_img is None:
        return {'value': 0, 'confidence': 1.0, 'alternatives': None}

    recog = recognize_with_topk(char_img, model)

    # Filter to valid values (1-16: classes 1-9 and 10-16)
    valid_top_k = [(d, c) for d, c in recog['top_k'] if 1 <= d <= 16]

    if not valid_top_k:
        # Fallback
        probs = recog['all_probs']
        best_val = np.argmax(probs[1:17]) + 1
        return {
            'value': best_val,
            'confidence': probs[best_val],
            'alternatives': {'single': recog}
        }

    return {
        'value': valid_top_k[0][0],
        'confidence': valid_top_k[0][1],
        'alternatives': {'single': recog, 'valid_top_k': valid_top_k}
    }


def extract_hexasudoku_ag_puzzle(image_path, model):
    """Extract HexaSudoku A-G puzzle from image."""
    size = 16
    board_pixels = 1600
    cell_size = board_pixels // size

    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * size for _ in range(size)]
    alternatives = {}

    for row in range(size):
        for col in range(size):
            cell = extract_cell(board_img, row, col, cell_size)
            recog = recognize_hexasudoku_ag_cell(cell, model)

            puzzle[row][col] = recog['value']
            if recog['value'] != 0:
                alternatives[(row, col)] = recog

    return puzzle, alternatives


# =============================================================================
# HexaSudoku Numeric Recognition (16x16 with two-digit numbers)
# =============================================================================

# Ink density threshold for empty cell detection
INK_THRESHOLD_NUMERIC = 0.02


def classify_cell_type_numeric(cell_img):
    """
    Determine if cell has 0, 1, or 2 digits based on ink distribution.

    Uses ink density ratio method (proven more reliable than contour counting):
    - Single digits: high ink concentration in center (centered digit)
    - Double digits: ink spread across cell (two digits side by side)

    Returns: 'empty', 'single', or 'double'
    """
    if is_cell_empty(cell_img):
        return 'empty'

    h, w = cell_img.shape
    total_ink = get_ink_density(cell_img)

    if total_ink < INK_THRESHOLD_NUMERIC:
        return 'empty'

    # Check center region ink density (middle 50% of cell width)
    center = cell_img[:, int(w * 0.25):int(w * 0.75)]
    center_ink = get_ink_density(center)
    center_ratio = center_ink / total_ink if total_ink > 0 else 0

    # Single digits have high center ink density (~2.0 ratio)
    # Double digits have lower center ink density (~1.2-1.5 ratio)
    # Threshold 1.7 discriminates well between the two
    return 'single' if center_ratio > 1.7 else 'double'


def extract_left_digit_fixed(cell_img):
    """
    Extract left digit region from a two-digit cell using fixed spatial split.

    The left half contains the tens digit (always "1" for values 10-16).
    """
    h, w = cell_img.shape
    # Crop to middle 70% vertically, left 50% horizontally
    y1, y2 = int(h * 0.15), int(h * 0.85)
    digit_region = cell_img[y1:y2, 0:w // 2]

    # Resize to 28x28 for CNN
    pil_img = Image.fromarray((digit_region * 255).astype(np.uint8))
    resized = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def extract_right_digit_fixed(cell_img):
    """
    Extract right digit region from a two-digit cell using fixed spatial split.

    The right half contains the ones digit (0-6 for values 10-16).
    """
    h, w = cell_img.shape
    # Crop to middle 70% vertically, right 50% horizontally
    y1, y2 = int(h * 0.15), int(h * 0.85)
    digit_region = cell_img[y1:y2, w // 2:w]

    # Resize to 28x28 for CNN
    pil_img = Image.fromarray((digit_region * 255).astype(np.uint8))
    resized = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def extract_single_digit_numeric(cell_img):
    """
    Extract and resize a single digit cell for numeric HexaSudoku.

    Uses contour-based extraction for single digits (matches training pipeline).
    """
    # Try contour-based extraction first (matches training)
    char_img = extract_single_character(cell_img)
    if char_img is not None:
        return char_img

    # Fallback: simple resize
    pil_img = Image.fromarray((cell_img * 255).astype(np.uint8))
    resized = pil_img.resize((28, 28), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def recognize_hexasudoku_numeric_cell(cell_img, model):
    """
    Recognize a HexaSudoku numeric cell.

    Values 1-9: single digit (classes 1-9)
    Values 10-16: two digits "1X" where X is 0-6 (tens=1, ones=0-6)

    DOMAIN CONSTRAINTS:
    - Single-digit cells: values 1-9 only (not 0, not 10-16)
    - Two-digit cells: tens digit must be 1 (forced, not recognized)
    - Two-digit cells: ones digit must be 0-6 (7, 8, 9 never appear)
    """
    # Use ink density method for cell type classification (more reliable)
    cell_type = classify_cell_type_numeric(cell_img)

    if cell_type == 'empty':
        return {'value': 0, 'cell_type': 'empty', 'confidence': 1.0, 'alternatives': None}

    if cell_type == 'single':
        # Single digit extraction and recognition
        digit_img = extract_single_digit_numeric(cell_img)
        recog = recognize_with_topk(digit_img, model)

        # DOMAIN CONSTRAINT: Single digit must be 1-9 (not 0, not letters A-G)
        valid_top_k = [(d, c) for d, c in recog['top_k'] if 1 <= d <= 9]

        if not valid_top_k:
            # Fallback: find best valid digit in range 1-9
            probs = recog['all_probs']
            best_digit = np.argmax(probs[1:10]) + 1
            return {
                'value': best_digit,
                'cell_type': 'single',
                'confidence': float(probs[best_digit]),
                'alternatives': {'single': recog}
            }

        return {
            'value': valid_top_k[0][0],
            'cell_type': 'single',
            'confidence': valid_top_k[0][1],
            'alternatives': {'single': recog, 'valid_top_k': valid_top_k}
        }

    else:  # cell_type == 'double'
        # Two-digit extraction using fixed spatial split
        left_img = extract_left_digit_fixed(cell_img)
        right_img = extract_right_digit_fixed(cell_img)

        left_recog = recognize_with_topk(left_img, model)
        right_recog = recognize_with_topk(right_img, model)

        # DOMAIN CONSTRAINT: Tens digit MUST be 1 for values 10-16
        # We force this rather than recognize it (eliminates 1<->7 confusion)
        tens_digit = 1

        # DOMAIN CONSTRAINT: Ones digit must be 0-6 (7, 8, 9 never appear in two-digit)
        ones_pred = right_recog['prediction']
        if ones_pred > 6:
            # Find best valid ones digit from top-k predictions
            valid_ones = [(d, c) for d, c in right_recog['top_k'] if 0 <= d <= 6]
            ones_digit = valid_ones[0][0] if valid_ones else 0
        else:
            ones_digit = ones_pred

        value = tens_digit * 10 + ones_digit  # Always in [10, 16]
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


def extract_hexasudoku_numeric_puzzle(image_path, model):
    """Extract HexaSudoku numeric puzzle from image."""
    size = 16
    board_pixels = 1600
    cell_size = board_pixels // size

    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * size for _ in range(size)]
    alternatives = {}

    for row in range(size):
        for col in range(size):
            cell = extract_cell(board_img, row, col, cell_size)
            recog = recognize_hexasudoku_numeric_cell(cell, model)

            puzzle[row][col] = recog['value']
            if recog['value'] != 0:
                alternatives[(row, col)] = recog

    return puzzle, alternatives


# =============================================================================
# Z3 Solvers
# =============================================================================

def solve_sudoku(size, box_size, given_cells):
    """Solve Sudoku using Z3."""
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    # Cell range
    for i in range(size):
        for j in range(size):
            s.add(And(X[i][j] >= 1, X[i][j] <= size))

    # Given cells
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

    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
    return None


def solve_with_tracking(size, box_size, given_cells):
    """Solve with constraint tracking for error detection."""
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


# =============================================================================
# Error Correction
# =============================================================================

@dataclass
class CorrectionResult:
    success: bool
    correction_type: str
    num_errors_corrected: int
    positions_corrected: List[Tuple[int, int]]
    solution: Optional[List[List[int]]]


def get_alternative_values(recog, puzzle_type, max_alternatives=3):
    """Get alternative values for error correction."""
    alternatives = []

    if puzzle_type in ['sudoku_4x4', 'sudoku_9x9']:
        if 'single' in recog.get('alternatives', {}):
            single = recog['alternatives']['single']
            valid_top_k = recog['alternatives'].get('valid_top_k', [])
            current = recog['value']
            for d, _ in valid_top_k:
                if d != current and len(alternatives) < max_alternatives:
                    alternatives.append(d)

    elif puzzle_type == 'hexasudoku_ag':
        if 'single' in recog.get('alternatives', {}):
            valid_top_k = recog['alternatives'].get('valid_top_k', [])
            current = recog['value']
            for d, _ in valid_top_k:
                if d != current and len(alternatives) < max_alternatives:
                    alternatives.append(d)

    elif puzzle_type == 'hexasudoku_numeric':
        cell_type = recog.get('cell_type', 'single')

        if cell_type == 'single':
            if 'single' in recog.get('alternatives', {}):
                single = recog['alternatives']['single']
                probs = single['all_probs']
                current = recog['value']
                sorted_idx = np.argsort(probs)[::-1]
                for idx in sorted_idx:
                    if 1 <= idx <= 9 and idx != current and len(alternatives) < max_alternatives:
                        alternatives.append(int(idx))

        elif cell_type == 'double':
            if 'ones' in recog.get('alternatives', {}):
                ones = recog['alternatives']['ones']
                ones_probs = ones['all_probs']
                current_ones = recog['value'] % 10

                # Only consider valid ones digits 0-6
                ones_sorted = [(i, ones_probs[i]) for i in range(7) if i != current_ones]
                ones_sorted.sort(key=lambda x: -x[1])

                for o, _ in ones_sorted[:max_alternatives]:
                    new_val = 10 + o
                    if new_val not in alternatives:
                        alternatives.append(new_val)

    return alternatives[:max_alternatives]


def attempt_error_correction(puzzle, alternatives, size, box_size, puzzle_type, max_errors=3):
    """Attempt to correct errors using Z3 unsat core guidance."""
    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]

    # Try direct solve first
    solution = solve_sudoku(size, box_size, given_cells)
    if solution is not None:
        return CorrectionResult(success=True, correction_type="none",
                               num_errors_corrected=0, positions_corrected=[],
                               solution=solution)

    # Get unsat core suspects
    _, suspects = solve_with_tracking(size, box_size, given_cells)

    if not suspects:
        return CorrectionResult(success=False, correction_type="unsolvable",
                               num_errors_corrected=0, positions_corrected=[],
                               solution=None)

    # Build alternatives for suspects
    suspect_alternatives = []
    for pos in suspects:
        if pos in alternatives:
            alt_values = get_alternative_values(alternatives[pos], puzzle_type)
            if alt_values:
                suspect_alternatives.append((pos, alt_values))

    # Try single error corrections
    for pos, alts in suspect_alternatives:
        for alt_val in alts:
            modified = given_cells.copy()
            modified[pos] = alt_val
            solution = solve_sudoku(size, box_size, modified)
            if solution is not None:
                return CorrectionResult(success=True, correction_type="single",
                                       num_errors_corrected=1, positions_corrected=[pos],
                                       solution=solution)

    # Try double error corrections
    if max_errors >= 2 and len(suspect_alternatives) >= 2:
        for i, (pos1, alts1) in enumerate(suspect_alternatives):
            for pos2, alts2 in suspect_alternatives[i+1:]:
                for alt1 in alts1[:2]:
                    for alt2 in alts2[:2]:
                        modified = given_cells.copy()
                        modified[pos1] = alt1
                        modified[pos2] = alt2
                        solution = solve_sudoku(size, box_size, modified)
                        if solution is not None:
                            return CorrectionResult(success=True, correction_type="double",
                                                   num_errors_corrected=2,
                                                   positions_corrected=[pos1, pos2],
                                                   solution=solution)

    return CorrectionResult(success=False, correction_type="uncorrectable",
                           num_errors_corrected=0, positions_corrected=[],
                           solution=None)


# =============================================================================
# Unified Evaluation
# =============================================================================

def evaluate_puzzle(puzzle_type, puzzles, model, image_dir):
    """Evaluate a set of puzzles."""
    config = BOARD_CONFIGS[puzzle_type]
    size = config['size']
    board_pixels = config['board_pixels']
    box_size = config['box_size']

    results = []

    for idx in range(len(puzzles)):
        if puzzle_type.startswith('sudoku'):
            image_path = os.path.join(image_dir, f"board{size}_{idx}.png")
        else:
            image_path = os.path.join(image_dir, f"board16_{idx}.png")

        if not os.path.exists(image_path):
            continue

        expected_puzzle = puzzles[idx]['puzzle']
        expected_solution = puzzles[idx]['solution']

        # Extract puzzle
        start_time = time.time()

        if puzzle_type == 'sudoku_4x4':
            extracted, alternatives = extract_sudoku_puzzle(image_path, model, 4, 900)
        elif puzzle_type == 'sudoku_9x9':
            extracted, alternatives = extract_sudoku_puzzle(image_path, model, 9, 900)
        elif puzzle_type == 'hexasudoku_ag':
            extracted, alternatives = extract_hexasudoku_ag_puzzle(image_path, model)
        elif puzzle_type == 'hexasudoku_numeric':
            extracted, alternatives = extract_hexasudoku_numeric_puzzle(image_path, model)

        extract_time = time.time() - start_time

        # Calculate extraction accuracy
        extraction_matches = sum(1 for i in range(size) for j in range(size)
                                 if extracted[i][j] == expected_puzzle[i][j])
        extraction_accuracy = extraction_matches / (size * size)

        # Solve with error correction
        start_time = time.time()
        correction = attempt_error_correction(extracted, alternatives, size, box_size, puzzle_type)
        solve_time = time.time() - start_time

        # Check solution
        final_match = False
        if correction.solution:
            final_match = all(correction.solution[i][j] == expected_solution[i][j]
                             for i in range(size) for j in range(size))

        results.append({
            'puzzle_type': puzzle_type,
            'puzzle_idx': idx,
            'extraction_accuracy': extraction_accuracy,
            'correction_type': correction.correction_type,
            'num_errors_corrected': correction.num_errors_corrected,
            'final_success': final_match,
            'extract_time': extract_time,
            'solve_time': solve_time
        })

        if (idx + 1) % 25 == 0:
            success_count = sum(1 for r in results if r['final_success'])
            print(f"  {puzzle_type}: {idx+1}/{len(puzzles)} - "
                  f"Success: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Unified Sudoku/HexaSudoku Solver")
    print("Using ImprovedCNN (17 classes)")
    print("=" * 70)

    # Paths
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'unified_cnn_weights.pth'
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Data directories
    data_dirs = {
        'sudoku': Path("/Users/ben/Library/Mobile Documents/com~apple~CloudDocs/Active Work/NeuroSymbolic/KenKenSolver/archive/90-10 split handwritten/Handwritten_Sudoku"),
        'hexasudoku_ag': Path("/Users/ben/Library/Mobile Documents/com~apple~CloudDocs/Active Work/NeuroSymbolic/KenKenSolver/archive/90-10 split handwritten/Handwritten_HexaSudoku"),
        'hexasudoku_numeric': Path("/Users/ben/Library/Mobile Documents/com~apple~CloudDocs/Active Work/NeuroSymbolic/KenKenSolver/archive/90-10 split handwritten digits only"),
    }

    # Check model
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Run training/train_unified_cnn.py first.")
        return

    print(f"\nLoading model from {model_path}...")
    model = load_model(str(model_path))
    print("Model loaded successfully.")

    all_results = []

    # Evaluate Sudoku 4x4 and 9x9
    sudoku_dir = data_dirs['sudoku']
    if sudoku_dir.exists():
        puzzles_path = sudoku_dir / 'puzzles' / 'puzzles_dict.json'
        if puzzles_path.exists():
            with open(puzzles_path) as f:
                puzzles_ds = json.load(f)

            image_dir = str(sudoku_dir / 'board_images')

            for size_str in ['4', '9']:
                puzzles = puzzles_ds.get(size_str, [])
                if puzzles:
                    puzzle_type = f'sudoku_{size_str}x{size_str}'
                    print(f"\n{'='*60}")
                    print(f"Evaluating {puzzle_type} ({len(puzzles)} puzzles)")
                    print("=" * 60)

                    results = evaluate_puzzle(puzzle_type, puzzles, model, image_dir)
                    all_results.extend(results)

                    success_count = sum(1 for r in results if r['final_success'])
                    print(f"\n{puzzle_type}: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Evaluate HexaSudoku A-G
    hexag_dir = data_dirs['hexasudoku_ag']
    if hexag_dir.exists():
        puzzles_path = hexag_dir / 'puzzles' / 'unique_puzzles_dict.json'
        if puzzles_path.exists():
            with open(puzzles_path) as f:
                puzzles_ds = json.load(f)

            puzzles = puzzles_ds.get('16', [])
            if puzzles:
                print(f"\n{'='*60}")
                print(f"Evaluating hexasudoku_ag ({len(puzzles)} puzzles)")
                print("=" * 60)

                image_dir = str(hexag_dir / 'board_images')
                results = evaluate_puzzle('hexasudoku_ag', puzzles, model, image_dir)
                all_results.extend(results)

                success_count = sum(1 for r in results if r['final_success'])
                print(f"\nhexasudoku_ag: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Evaluate HexaSudoku Numeric
    hexnum_dir = data_dirs['hexasudoku_numeric']
    if hexnum_dir.exists():
        puzzles_path = hexnum_dir / 'puzzles' / 'unique_puzzles_dict.json'
        if puzzles_path.exists():
            with open(puzzles_path) as f:
                puzzles_ds = json.load(f)

            puzzles = puzzles_ds.get('16', [])
            if puzzles:
                print(f"\n{'='*60}")
                print(f"Evaluating hexasudoku_numeric ({len(puzzles)} puzzles)")
                print("=" * 60)

                image_dir = str(hexnum_dir / 'board_images')
                results = evaluate_puzzle('hexasudoku_numeric', puzzles, model, image_dir)
                all_results.extend(results)

                success_count = sum(1 for r in results if r['final_success'])
                print(f"\nhexasudoku_numeric: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # Summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print("=" * 70)

    by_type = {}
    for r in all_results:
        pt = r['puzzle_type']
        if pt not in by_type:
            by_type[pt] = []
        by_type[pt].append(r)

    for pt in sorted(by_type.keys()):
        results = by_type[pt]
        n = len(results)
        success_count = sum(1 for r in results if r['final_success'])
        correction_counts = Counter(r['correction_type'] for r in results)
        avg_extraction = sum(r['extraction_accuracy'] for r in results) / n

        print(f"\n{pt}:")
        print(f"  Success: {success_count}/{n} ({success_count/n*100:.1f}%)")
        print(f"  Avg extraction accuracy: {avg_extraction*100:.1f}%")
        print(f"  Corrections: {dict(correction_counts)}")

    # Save results
    if all_results:
        csv_path = results_dir / 'unified_evaluation.csv'
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
