#!/usr/bin/env python3
"""
Run the NeuroSymbolic KenKen solver evaluation on 9x9 puzzles only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from z3 import *
import time
import sys
import os

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


def get_contours(img):
    img = (img * 255).astype(np.uint8)
    _, inp = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

    e_kernel = np.ones((1, 1), np.uint8)
    inp = cv.erode(inp, e_kernel, iterations=1)
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
        if x2 < (x + w):
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
    return result


def segment_cell(grid, size, border_thickness, row, col):
    cell_size = len(grid) // size
    cell = grid[row * cell_size + border_thickness: row * cell_size + cell_size // 2,
                col * cell_size + border_thickness: (col + 1) * cell_size - border_thickness]
    cell = (cell / 255.0).astype('float64')
    contours = get_contours(cell)
    characters = []
    for box in contours:
        characters.append(get_character(cell, box))
    return characters


def get_predictions(characters, model):
    predictions = []
    with torch.no_grad():
        for c in characters:
            im = torch.tensor(c, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = model(im)
            predictions.append(torch.argmax(output, dim=1).item())
    return predictions


def update_puzzle_from_predictions(puzzle, predictions):
    if len(predictions) == 0:
        puzzle["target"] = 0
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
            op = ""
        puzzle["target"] = target
        puzzle["op"] = op
    return puzzle


def make_puzzle_from_image(size, border_thickness, cages, filename, model):
    img = Image.open(filename).convert('L')
    grid = np.array(img)
    puzzle = []
    for cage in cages:
        puzzle.append({"cells": cage, "op": "", "target": 0})
        characters = segment_cell(grid, size, border_thickness + 5, cage[0][0], cage[0][1])
        predictions = get_predictions(characters, model)
        puzzle[-1] = update_puzzle_from_predictions(puzzle[-1], predictions)
    return puzzle


def parse_block_constraints_optimized(puzzle, cells, size, known_values):
    """
    Optimized constraint generation with:
    1. Pre-filled singletons (skip constraint, use known_values)
    2. Integer-only division (avoid Real arithmetic)
    3. Domain tightening from cage arithmetic
    """
    constraints = []

    for block in puzzle:
        op = block["op"]
        target = block["target"]
        block_cells = block["cells"]

        # Get variables, substituting known values
        vars_in_block = []
        for i, j in block_cells:
            if (i, j) in known_values:
                vars_in_block.append(known_values[(i, j)])
            else:
                vars_in_block.append(cells[i][j])

        if op == "":
            # Singleton - already handled via known_values, but add constraint for safety
            if len(block_cells) == 1:
                i, j = block_cells[0]
                if (i, j) not in known_values:
                    constraints.append(cells[i][j] == target)
        elif op == "add":
            constraints.append(Sum(vars_in_block) == target)
            # Domain tightening: each cell <= target - (n-1) where n is cage size
            n = len(block_cells)
            if n > 1:
                max_val = min(size, target - (n - 1))
                for i, j in block_cells:
                    if (i, j) not in known_values and max_val < size:
                        constraints.append(cells[i][j] <= max_val)
        elif op == "mul":
            # Build product
            if len(vars_in_block) == 1:
                constraints.append(vars_in_block[0] == target)
            else:
                product = vars_in_block[0]
                for v in vars_in_block[1:]:
                    product = product * v
                constraints.append(product == target)
            # Domain tightening: each cell must divide target
            for i, j in block_cells:
                if (i, j) not in known_values:
                    valid_divisors = [d for d in range(1, size + 1) if target % d == 0]
                    if len(valid_divisors) < size:
                        constraints.append(Or([cells[i][j] == d for d in valid_divisors]))
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a - b == target, b - a == target))
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            # Integer-only division: avoid Real arithmetic
            # a/b == target means a == b * target
            # b/a == target means b == a * target
            int_target = int(target)
            constraints.append(Or(a == b * int_target, b == a * int_target))
        else:
            raise ValueError(f"Unsupported operation or malformed block: {block}")

    return constraints


def evaluate_puzzle(puzzle, size):
    """
    Optimized KenKen solver with:
    1. Pre-filled singletons
    2. Integer-only constraints
    3. Solver tactics for better propagation
    4. Timeout to avoid infinite solving
    """
    # Step 1: Extract known values from singletons
    known_values = {}
    for block in puzzle:
        if block["op"] == "" and len(block["cells"]) == 1:
            i, j = block["cells"][0]
            known_values[(i, j)] = block["target"]

    # Step 2: Create variables only for unknown cells
    X = [[None for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if (i, j) in known_values:
                X[i][j] = known_values[(i, j)]  # Use integer directly
            else:
                X[i][j] = Int(f"x_{i+1}_{j+1}")

    # Step 3: Build constraints
    constraints = []

    # Cell range constraints (only for unknowns)
    for i in range(size):
        for j in range(size):
            if (i, j) not in known_values:
                constraints.append(And(1 <= X[i][j], X[i][j] <= size))

    # Row distinctness
    for i in range(size):
        constraints.append(Distinct(X[i]))

    # Column distinctness
    for j in range(size):
        constraints.append(Distinct([X[i][j] for i in range(size)]))

    # Cage constraints (optimized)
    constraints.extend(parse_block_constraints_optimized(puzzle, X, size, known_values))

    # Step 4: Create solver with tactics
    # Use tactics that help with constraint propagation
    try:
        tactic = Then('simplify', 'propagate-values', 'solve-eqs', 'smt')
        s = tactic.solver()
    except:
        # Fallback to regular solver if tactics fail
        s = Solver()

    # Set timeout (60 seconds)
    s.set("timeout", 60000)

    s.add(constraints)

    if s.check() == sat:
        m = s.model()
        solution = []
        for i in range(size):
            row = []
            for j in range(size):
                if (i, j) in known_values:
                    row.append(known_values[(i, j)])
                else:
                    row.append(m.evaluate(X[i][j]))
            solution.append(row)
        return solution
    else:
        return None


def run_evaluation(model):
    results = []
    csv_path = './results/9x9_evaluation.csv'

    print("=" * 70)
    print("9x9 KENKEN EVALUATION")
    print("=" * 70)
    print()
    sys.stdout.flush()

    size = 9
    print(f"[Size {size}x{size}] Evaluating 100 puzzles...")
    sys.stdout.flush()

    size_start = time.time()
    solved_count = 0

    for i in range(100):
        filename = f"./board_images/board{size}_{i}.png"

        start_time = time.time()
        size_detected = size
        size_correct = True
        solved = False
        error_type = "none"
        error_message = ""

        try:
            size_detected, cages, border = find_size_and_borders_with_size(filename, size)
            puzzle = make_puzzle_from_image(size_detected, border, cages, filename, model)
            solution = evaluate_puzzle(puzzle, size_detected)

            if solution is None:
                error_type = "z3_unsolvable"
                error_message = "Z3 could not satisfy constraints"
            else:
                solved = True

        except Exception as e:
            error_type = "exception"
            error_message = str(e)

        end_time = time.time()
        time_ms = (end_time - start_time) * 1000

        if solved:
            solved_count += 1

        results.append({
            'size': size,
            'puzzle_index': i,
            'filename': f"./board_images/board{size}_{i}.png",
            'size_detected': size_detected,
            'size_correct': size_correct,
            'solved': solved,
            'solve_time_ms': time_ms,
            'error_type': error_type,
            'error_message': error_message
        })

        # Save incrementally after each puzzle
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

        print(f"  [{i+1}/100] {'SOLVED' if solved else 'FAILED'} in {time_ms/1000:.1f}s (total solved: {solved_count})")
        sys.stdout.flush()

    size_end = time.time()
    size_time = size_end - size_start
    avg_time = size_time / 100 * 1000

    print()
    print(f"  Done: {solved_count}/100 solved ({solved_count}%) - Avg time: {avg_time:.0f}ms")
    print()
    sys.stdout.flush()

    print(f"Results saved to {csv_path}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Size 9: {solved_count}/100 solved ({solved_count}%)")
    print(f"Average solve time: {avg_time:.0f}ms")
    print()
    print("Error breakdown:")
    error_counts = df[df['error_type'] != 'none']['error_type'].value_counts()
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count} failures")
    if len(error_counts) == 0:
        print("  No errors!")
    print()

    return df


if __name__ == "__main__":
    print("Loading character recognition model...")
    character_model = CNN_v2(output_dim=14)
    state_dict = torch.load('./models/character_recognition_v2_model_weights.pth', weights_only=False)
    character_model.load_state_dict(state_dict)
    character_model.eval()
    print("Model loaded.")
    print()

    results_df = run_evaluation(character_model)
    print("Done!")
