"""
Error Detection and Correction for 9x9 KenKen puzzles.

Uses Z3's unsatisfiable core to identify problematic cages, then explores
alternative character predictions using Top-K + multi-error correction.

Usage:
    python detect_errors_9x9.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from z3 import Int, Solver, And, Or, Distinct, Sum, sat, unsat, Bool, Then, Tactic
import time
import sys
import os
from itertools import combinations, product
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

BOARD_SIZE = 900
IMG_SIZE = 28
SCALE_FACTOR = 2


class CNN_v2(nn.Module):
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


# =============================================================================
# Image Processing Functions (from evaluate_9x9_only.py)
# =============================================================================

def find_h_borders(h_lines, size, epsilon, delta):
    cell_size = ((BOARD_SIZE * SCALE_FACTOR) // size)
    vertical_window = (cell_size - delta, cell_size + delta)
    h_borders = np.zeros((size - 1, size))
    horizontal_window = (epsilon, cell_size - epsilon)

    for i in range(size - 1):
        window = h_lines[(h_lines['y1'] >= vertical_window[0]) & (h_lines['y1'] <= vertical_window[1])]
        if len(window) == 0:
            vertical_window = (vertical_window[0] + cell_size, vertical_window[1] + cell_size)
            continue
        max_val = (window[['y1', 'y2']].min(axis=1)).max()
        min_val = (window[['y1', 'y2']].max(axis=1)).min()

        if max_val - min_val > int(11 * SCALE_FACTOR):
            for j in range(size):
                y_vals = window[(np.maximum(window['x1'], window['x2']) >= horizontal_window[0]) &
                               (np.minimum(window['x1'], window['x2']) <= horizontal_window[1])]['y1'].values
                if max_val in y_vals or min_val in y_vals:
                    h_borders[i][j] = 1
                horizontal_window = (horizontal_window[0] + cell_size, horizontal_window[1] + cell_size)
            horizontal_window = (epsilon, cell_size - epsilon)
        vertical_window = (vertical_window[0] + cell_size, vertical_window[1] + cell_size)

    return h_borders


def find_v_borders(v_lines, size, epsilon, delta):
    cell_size = ((BOARD_SIZE * SCALE_FACTOR) // size)
    horizontal_window = (cell_size - delta, cell_size + delta)
    v_borders = np.zeros((size, size - 1))
    vertical_window = (epsilon, cell_size - epsilon)

    for i in range(size - 1):
        window = v_lines[(v_lines['x1'] >= horizontal_window[0]) & (v_lines['x1'] <= horizontal_window[1])]
        if len(window) == 0:
            horizontal_window = (horizontal_window[0] + cell_size, horizontal_window[1] + cell_size)
            continue
        max_val = (window[['x1', 'x2']].min(axis=1)).max()
        min_val = (window[['x1', 'x2']].max(axis=1)).min()

        if max_val - min_val > 11:
            for j in range(size):
                x_vals = window[(np.maximum(window['y1'], window['y2']) >= vertical_window[0]) &
                               (np.minimum(window['y1'], window['y2']) <= vertical_window[1])]['x1'].values
                if max_val in x_vals or min_val in x_vals:
                    v_borders[j][i] = 1
                vertical_window = (vertical_window[0] + cell_size, vertical_window[1] + cell_size)
            vertical_window = (epsilon, cell_size - epsilon)
        horizontal_window = (horizontal_window[0] + cell_size, horizontal_window[1] + cell_size)

    return v_borders


def make_cage_from_borders(start_cell, visited, h_borders, v_borders):
    cage = [start_cell]
    neighbors = [start_cell]
    visited[start_cell[0]][start_cell[1]] = 1

    while neighbors:
        for neighbor in neighbors:
            start_cell = neighbor
            neighbors = []
            if start_cell[1] > 0 and visited[start_cell[0]][start_cell[1] - 1] == 0 and v_borders[start_cell[0]][start_cell[1] - 1] == 0:
                cage.append([start_cell[0], start_cell[1] - 1])
                visited[start_cell[0]][start_cell[1] - 1] = 1
                neighbors.append([start_cell[0], start_cell[1] - 1])

            if start_cell[0] > 0 and visited[start_cell[0] - 1][start_cell[1]] == 0 and h_borders[start_cell[0] - 1][start_cell[1]] == 0:
                cage.append([start_cell[0] - 1, start_cell[1]])
                visited[start_cell[0] - 1][start_cell[1]] = 1
                neighbors.append([start_cell[0] - 1, start_cell[1]])

            if start_cell[1] < len(v_borders) - 1 and visited[start_cell[0]][start_cell[1] + 1] == 0 and v_borders[start_cell[0]][start_cell[1]] == 0:
                cage.append([start_cell[0], start_cell[1] + 1])
                visited[start_cell[0]][start_cell[1] + 1] = 1
                neighbors.append([start_cell[0], start_cell[1] + 1])

            if start_cell[0] < len(v_borders) - 1 and visited[start_cell[0] + 1][start_cell[1]] == 0 and h_borders[start_cell[0]][start_cell[1]] == 0:
                cage.append([start_cell[0] + 1, start_cell[1]])
                visited[start_cell[0] + 1][start_cell[1]] = 1
                neighbors.append([start_cell[0] + 1, start_cell[1]])
    return cage


def construct_cages(h_borders, v_borders):
    size = len(v_borders)
    cages = []
    visited = np.zeros((size, size))
    for row in range(size):
        for col in range(size):
            start_cell = [row, col]
            if visited[row][col] == 0:
                cages.append(make_cage_from_borders(start_cell, visited, h_borders, v_borders))
    return cages


def get_border_thickness(lines):
    v_lines = lines[lines['x1'] == lines['x2']]
    if len(v_lines) == 0:
        return 32
    return min(v_lines['x1'])


def find_size_and_borders_with_size(filename, size):
    src = cv.imread(filename)
    resized = cv.resize(src, (BOARD_SIZE * SCALE_FACTOR, BOARD_SIZE * SCALE_FACTOR))
    filtered = cv.pyrMeanShiftFiltering(resized, sp=5, sr=40)
    dst = cv.Canny(filtered, 50, 200, None, 3)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 360, 75, None, 50, 15)
    if linesP is None:
        return size, [], 16
    linesP = np.squeeze(linesP, axis=1)
    if linesP.ndim == 1:
        linesP = linesP.reshape(1, -1)
    lines_df = pd.DataFrame(linesP, columns=['x1', 'y1', 'x2', 'y2'])
    h_lines = lines_df[abs(lines_df['y1'] - lines_df['y2']) < 2]
    v_lines = lines_df[abs(lines_df['x1'] - lines_df['x2']) < 2]
    border_thickness = get_border_thickness(lines_df)

    cages = construct_cages(
        find_h_borders(h_lines, size, border_thickness, border_thickness),
        find_v_borders(v_lines, size, border_thickness, border_thickness)
    )
    return size, cages, border_thickness // SCALE_FACTOR


def get_contours(img, size=9):
    """
    Get character contours from cell image.

    For 9x9 puzzles, use smaller dilation kernel to avoid merging close characters.
    """
    img = (img * 255).astype(np.uint8)
    _, inp = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

    e_kernel = np.ones((1, 1), np.uint8)
    inp = cv.erode(inp, e_kernel, iterations=1)

    # Use smaller dilation for 9x9 to prevent character merging
    if size >= 9:
        d_kernel = np.ones((2, 2), np.uint8)
    else:
        d_kernel = np.ones((3, 3), np.uint8)
    inp = cv.dilate(inp, d_kernel, iterations=1)

    ctrs, hierarchy = cv.findContours(inp.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ctrs = sorted(ctrs, key=lambda cnt: cv.boundingRect(cnt)[0])

    if len(ctrs) == 0:
        return []

    boxes = [cv.boundingRect(ctrs[0])]
    count = 1
    while count < len(ctrs):
        x, y, w, h = boxes[-1]
        x2, y2, w2, h2 = cv.boundingRect(ctrs[count])
        # Only merge if significantly overlapping (not just touching)
        overlap_threshold = 3 if size >= 9 else 0
        if x2 < (x + w - overlap_threshold):
            h = abs(y - y2) + h2 if y < y2 else abs(y - y2) + h
            w = max(w, w2)
            y = min(y, y2)
            boxes[-1] = (x, y, w, h)
        else:
            boxes.append((x2, y2, w2, h2))
        count += 1

    return boxes


def get_character(img, box):
    x, y, w, h = box
    cropped = img[y:y + h, x:x + w]
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8)).convert('L')

    aspect = w / h
    if aspect > 1:
        new_w = IMG_SIZE
        new_h = int(IMG_SIZE / aspect)
    else:
        new_h = IMG_SIZE
        new_w = int(IMG_SIZE * aspect)

    resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))

    result = np.array(canvas).astype(np.float32) / 255.0
    # Invert so character pixels are 1.0 (high) and background is 0.0
    # This matches the training data convention
    result = 1.0 - result
    return result


def segment_cell(grid, size, border_thickness, row, col):
    cell_size = len(grid) // size

    # Adaptive height factor: larger boards need more vertical coverage
    if size <= 6:
        height_factor = 0.5
    elif size == 7:
        height_factor = 0.6
    else:  # 9x9+
        height_factor = 0.7

    vertical_end = int(row * cell_size + cell_size * height_factor)

    cell = grid[row * cell_size + border_thickness: vertical_end,
                col * cell_size + border_thickness: (col + 1) * cell_size - border_thickness]
    cell = (cell / 255.0).astype('float64')
    contours = get_contours(cell, size)  # Pass size for adaptive contour detection
    characters = []
    for box in contours:
        characters.append(get_character(cell, box))
    return characters


# =============================================================================
# Top-K Predictions
# =============================================================================

def get_predictions_with_top_k(characters, model, k=4):
    """Get predictions with top-K alternatives."""
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
    """Extract puzzle from image with alternative predictions stored."""
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

def get_valid_values_for_mul(target, size, num_cells):
    """Get valid cell values for multiplication cage (domain tightening)."""
    valid = set()
    for v in range(1, size + 1):
        if target % v == 0:  # v must be a factor of target
            valid.add(v)
    return valid


def get_valid_values_for_div(target, size):
    """Get valid cell values for division cage (domain tightening)."""
    valid = set()
    int_target = int(target)
    for v in range(1, size + 1):
        # Either v * target <= size, or v is divisible by target
        if v * int_target <= size or (v % int_target == 0 and v // int_target >= 1):
            valid.add(v)
    return valid


def solve_kenken(puzzle, size):
    """Solve KenKen puzzle with optimized Z3 tactics."""
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

    # Use optimized solver with tactics: simplify -> propagate-values -> solve-eqs -> smt
    tactic = Then(
        Tactic('simplify'),
        Tactic('propagate-values'),
        Tactic('solve-eqs'),
        Tactic('smt')
    )
    s = tactic.solver()
    s.set("timeout", 30000)

    # Base domain constraints
    for i in range(size):
        for j in range(size):
            if (i, j) not in known_values:
                s.add(And(1 <= X[i][j], X[i][j] <= size))

    # Latin square constraints
    for i in range(size):
        s.add(Distinct(X[i]))
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    # Cage constraints with domain tightening
    for block in puzzle:
        op = block["op"]
        target = block["target"]
        block_cells = block["cells"]

        vars_in_block = []
        unknown_cells = []
        for i, j in block_cells:
            if (i, j) in known_values:
                vars_in_block.append(known_values[(i, j)])
            else:
                vars_in_block.append(X[i][j])
                unknown_cells.append((i, j))

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
                # Domain tightening: cells must be factors of target
                valid_values = get_valid_values_for_mul(target, size, len(block_cells))
                for i, j in unknown_cells:
                    s.add(Or([X[i][j] == v for v in valid_values]))
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            s.add(Or(a - b == target, b - a == target))
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            int_target = int(target)
            s.add(Or(a == b * int_target, b == a * int_target))
            # Domain tightening for division
            valid_values = get_valid_values_for_div(target, size)
            for i, j in unknown_cells:
                s.add(Or([X[i][j] == v for v in valid_values]))

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
    """Solve KenKen puzzle with constraint tracking for unsat core extraction."""
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

    # Note: Cannot use tactics with assert_and_track, use regular Solver
    s = Solver()
    s.set("timeout", 30000)

    # Base domain constraints
    for i in range(size):
        for j in range(size):
            if (i, j) not in known_values:
                s.add(And(1 <= X[i][j], X[i][j] <= size))

    # Latin square constraints
    for i in range(size):
        s.add(Distinct(X[i]))
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    cage_trackers = {}

    for idx, block in enumerate(puzzle):
        op = block["op"]
        target = block["target"]
        block_cells = block["cells"]

        vars_in_block = []
        unknown_cells = []
        for i, j in block_cells:
            if (i, j) in known_values:
                vars_in_block.append(known_values[(i, j)])
            else:
                vars_in_block.append(X[i][j])
                unknown_cells.append((i, j))

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
                # Domain tightening for multiplication
                valid_values = get_valid_values_for_mul(target, size, len(block_cells))
                for i, j in unknown_cells:
                    s.add(Or([X[i][j] == v for v in valid_values]))
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraint = Or(a - b == target, b - a == target)
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            int_target = int(target)
            constraint = Or(a == b * int_target, b == a * int_target)
            # Domain tightening for division
            valid_values = get_valid_values_for_div(target, size)
            for i, j in unknown_cells:
                s.add(Or([X[i][j] == v for v in valid_values]))

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

    core = s.unsat_core()
    suspect_indices = [cage_trackers[t] for t in core if t in cage_trackers]
    return None, suspect_indices


# =============================================================================
# Error Detection and Correction
# =============================================================================

@dataclass
class DetectionResult:
    suspects: List[int]
    num_probes: int
    detection_time: float


@dataclass
class CorrectionResult:
    success: bool
    correction_type: str
    num_errors_corrected: int
    cages_corrected: List[int]
    solution: Optional[List[List[int]]]
    correction_attempts: int
    total_solve_calls: int


def detect_errors_via_unsat_core(puzzle, size, max_probes=30):
    """Detect which cages are likely errors using unsat core + probing."""
    start_time = time.time()

    solution, initial_suspects = solve_kenken_with_tracking(puzzle, size)

    if solution is not None:
        return DetectionResult(suspects=[], num_probes=1, detection_time=time.time() - start_time)

    if not initial_suspects:
        return DetectionResult(suspects=[], num_probes=1, detection_time=time.time() - start_time)

    suspect_counts = Counter()
    num_probes = 1

    probes_to_do = initial_suspects[:max_probes]

    for cage_idx in probes_to_do:
        test_puzzle = [puzzle[i] for i in range(len(puzzle)) if i != cage_idx]

        sol, new_suspects = solve_kenken_with_tracking(test_puzzle, size)
        num_probes += 1

        if sol is not None:
            suspect_counts[cage_idx] += 10
        else:
            for s in new_suspects:
                adjusted_s = s if s < cage_idx else s + 1
                suspect_counts[adjusted_s] += 1
            if cage_idx in initial_suspects:
                suspect_counts[cage_idx] += 1

    sorted_suspects = sorted(suspect_counts.keys(), key=lambda p: -suspect_counts[p])

    if not sorted_suspects:
        sorted_suspects = initial_suspects

    return DetectionResult(
        suspects=sorted_suspects,
        num_probes=num_probes,
        detection_time=time.time() - start_time
    )


def generate_cage_alternatives(alternatives_data, max_k=4):
    """Generate all alternative interpretations of a cage."""
    top_k_list = alternatives_data['top_k']
    original_preds = alternatives_data['predictions']
    cage = alternatives_data['cage']

    if not top_k_list:
        return []

    char_alternatives = []
    for char_idx, char_top_k in enumerate(top_k_list):
        alts = [(pred, conf) for pred, conf in char_top_k[:max_k]]
        char_alternatives.append(alts)

    all_combos = []
    for combo in product(*char_alternatives):
        preds = [p for p, _ in combo]
        conf = sum(c for _, c in combo) / len(combo)

        if preds == original_preds:
            continue

        new_cage = update_puzzle_from_predictions(cage, preds)
        all_combos.append((new_cage, conf, preds))

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
    """Use unsat core detection + top-K predictions for multi-error correction."""
    if max_attempts_per_level is None:
        max_attempts_per_level = [50, 100, 200, 400, 800]

    while len(max_attempts_per_level) < max_errors:
        max_attempts_per_level.append(max_attempts_per_level[-1] * 2)

    detection = detect_errors_via_unsat_core(puzzle, size)
    total_solve_calls = detection.num_probes

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

    suspect_alternatives = []
    for cage_idx in suspects:
        alts = generate_cage_alternatives(alternatives[cage_idx], max_k)
        if alts:
            suspect_alternatives.append((cage_idx, alts))

    error_names = ["single", "two", "three", "four", "five"]

    for num_errors in range(1, max_errors + 1):
        if num_errors > len(suspect_alternatives):
            break

        max_attempts = max_attempts_per_level[num_errors - 1]
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
                    return detection, CorrectionResult(
                        success=True,
                        correction_type=error_name,
                        num_errors_corrected=num_errors,
                        cages_corrected=list(cage_indices),
                        solution=solution,
                        correction_attempts=correction_attempts,
                        total_solve_calls=total_solve_calls
                    )

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
    """Attempt to solve a puzzle, correcting OCR errors if needed."""
    start_time = time.time()

    puzzle, alternatives = make_puzzle_with_alternatives(
        size, border_thickness, cages, filename, model, k=max_k
    )

    solution = solve_kenken(puzzle, size)
    if solution is not None:
        return solution, False, "none", {
            'solve_time_ms': (time.time() - start_time) * 1000,
            'correction_attempts': 0,
            'total_solve_calls': 1
        }

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
    print("9x9 KenKen Solver - Error Correction Evaluation")
    print("Using Top-K alternatives + Multi-error correction")
    print("=" * 70)
    print()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading character recognition model...")
    model = CNN_v2(output_dim=14)
    state_dict = torch.load('./models/character_recognition_v2_model_weights.pth', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")
    print()

    results = []
    size = 9
    base_solved = 0
    corrected_solved = 0
    total_time = 0
    correction_types = Counter()

    print(f"Evaluating 9x9 puzzles (100 total)...")

    for i in range(100):
        filename = f"./board_images/board{size}_{i}.png"

        if not os.path.exists(filename):
            continue

        try:
            _, cages, border_thickness = find_size_and_borders_with_size(filename, size)
            solution, corrected, correction_type, stats = solve_with_error_correction(
                size, cages, border_thickness, filename, model,
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

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/100] Base: {base_solved}, Corrected: {corrected_solved}")
            sys.stdout.flush()

    base_accuracy = base_solved / 100 * 100
    corrected_accuracy = corrected_solved / 100 * 100
    avg_time = total_time / 100

    print()
    print(f"  9x9: Base {base_accuracy:.1f}% -> Corrected {corrected_accuracy:.1f}%")
    print(f"    Breakdown: {dict(correction_types)}")
    print()

    # Save results
    os.makedirs('./results', exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv('./results/9x9_error_correction_evaluation.csv', index=False)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"9x9 KenKen:")
    print(f"  Base accuracy: {base_accuracy:.1f}%")
    print(f"  Corrected accuracy: {corrected_accuracy:.1f}%")
    print(f"  Improvement: +{corrected_accuracy - base_accuracy:.1f}%")
    print(f"  Average time: {avg_time:.0f}ms")
    print()
    print(f"Results saved to ./results/9x9_error_correction_evaluation.csv")
    print()


if __name__ == '__main__':
    main()
