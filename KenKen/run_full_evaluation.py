#!/usr/bin/env python3
"""
Full evaluation of optimized KenKen solver on 3x3-7x7 puzzles.
Uses the same optimizations as the 9x9 solver:
1. Pre-filled singletons
2. Integer-only division
3. Domain tightening
4. Solver tactics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from z3 import *
from torchvision import transforms
import json
import time
import sys
import os

# Constants
BOARD_SIZE = 900
IMG_SIZE = 28
SCALE_FACTOR = 2

# ============================================================
# CNN MODELS
# ============================================================

class Grid_CNN(nn.Module):
    def __init__(self, output_dim):
        super(Grid_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(262144, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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


# ============================================================
# IMAGE PROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])


def get_size(filename, grid_model):
    im = Image.open(filename).convert("RGBA")
    im = transform(im).unsqueeze(0)
    output = grid_model(im)
    prediction = torch.argmax(output, dim=1).item()
    return prediction + 3


def find_h_borders(h_lines, size, epsilon, delta):
    cell_size = ((BOARD_SIZE * SCALE_FACTOR) // size)
    vertical_window = (cell_size - delta, cell_size + delta)
    h_borders = np.zeros((size - 1, size))
    horizontal_window = (epsilon, cell_size - epsilon)

    for i in range(size - 1):
        window = h_lines[(h_lines['y1'] >= vertical_window[0]) & (h_lines['y1'] <= vertical_window[1])]
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


def make_cage(start_cell, visited, h_borders, v_borders):
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
                cages.append(make_cage(start_cell, visited, h_borders, v_borders))
    return cages


def get_border_thickness(lines):
    v_lines = lines[lines['x1'] == lines['x2']]
    return min(v_lines['x1'])


def find_size_and_borders(filename, grid_model):
    src = cv.imread(filename)
    resized = cv.resize(src, (BOARD_SIZE * SCALE_FACTOR, BOARD_SIZE * SCALE_FACTOR))
    filtered = cv.pyrMeanShiftFiltering(resized, sp=5, sr=40)
    dst = cv.Canny(filtered, 50, 200, None, 3)
    linesP = cv.HoughLinesP(dst, 1, np.pi / 360, 75, None, 50, 15)
    linesP = np.squeeze(linesP, axis=1)
    lines_df = pd.DataFrame(linesP, columns=['x1', 'y1', 'x2', 'y2'])
    h_lines = lines_df[abs(lines_df['y1'] - lines_df['y2']) < 2]
    v_lines = lines_df[abs(lines_df['x1'] - lines_df['x2']) < 2]
    border_thickness = get_border_thickness(lines_df)

    size = get_size(filename, grid_model)
    cages = construct_cages(
        find_h_borders(h_lines, size, border_thickness, border_thickness),
        find_v_borders(v_lines, size, border_thickness, border_thickness)
    )
    return size, cages, border_thickness // SCALE_FACTOR


# ============================================================
# CHARACTER RECOGNITION
# ============================================================

def get_contours(img):
    img = (img * 255).astype(np.uint8)
    _, inp = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    e_kernel = np.ones((1, 1), np.uint8)
    inp = cv.erode(inp, e_kernel, iterations=1)
    d_kernel = np.ones((3, 3), np.uint8)
    inp = cv.dilate(inp, d_kernel, iterations=1)
    ctrs, _ = cv.findContours(inp.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ctrs = sorted(ctrs, key=lambda cnt: cv.boundingRect(cnt)[0])

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
    return np.array(canvas).astype(np.float32) / 255.0


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


def get_predictions(characters, char_model):
    predictions = []
    with torch.no_grad():
        for c in characters:
            im = torch.tensor(c, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = char_model(im)
            predictions.append(torch.argmax(output, dim=1).item())
    return predictions


def update_puzzle(puzzle, predictions):
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


def make_puzzle(size, border_thickness, cages, filename, char_model):
    img = Image.open(filename).convert('L')
    grid = np.array(img)
    puzzle = []
    for cage in cages:
        puzzle.append({"cells": cage, "op": "", "target": 0})
        characters = segment_cell(grid, size, border_thickness + 5, cage[0][0], cage[0][1])
        predictions = get_predictions(characters, char_model)
        puzzle[-1] = update_puzzle(puzzle[-1], predictions)
    return puzzle


# ============================================================
# OPTIMIZED Z3 SOLVER
# ============================================================

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
            # Domain tightening: each cell <= target - (n-1)
            n = len(block_cells)
            if n > 1:
                max_val = min(size, target - (n - 1))
                for i, j in block_cells:
                    if (i, j) not in known_values and max_val < size:
                        constraints.append(cells[i][j] <= max_val)
        elif op == "mul":
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
    try:
        tactic = Then('simplify', 'propagate-values', 'solve-eqs', 'smt')
        s = tactic.solver()
    except:
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


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 70)
    print("KenKen Solver Full Evaluation (3x3 to 7x7) - Optimized Z3")
    print("=" * 70)
    print()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load models
    print("Loading models...")
    character_model = CNN_v2(output_dim=14)
    state_dict = torch.load('./models/character_recognition_v2_model_weights.pth', weights_only=False)
    character_model.load_state_dict(state_dict)
    character_model.eval()

    grid_detection = Grid_CNN(output_dim=5)
    state_dict = torch.load('./models/grid_detection_model_weights.pth', weights_only=False)
    grid_detection.load_state_dict(state_dict)
    grid_detection.eval()
    print("  Models loaded successfully")
    print()

    # Load puzzle data
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    # Results tracking
    results = []
    accuracy = {}
    avg_time = {}

    # Run evaluation for sizes 3-7
    for size in range(3, 8):
        print(f"Evaluating {size}x{size} puzzles...")
        size_start = time.time()
        solved_count = 0
        puzzle_count = len(puzzles_ds[str(size)])

        for i in range(puzzle_count):
            filepath = f"./board_images/board{size}_{i}.png"
            start = time.time()

            try:
                s, cages, b_t = find_size_and_borders(filepath, grid_detection)
                puzzle = make_puzzle(s, b_t, cages, filepath, character_model)
                solution = evaluate_puzzle(puzzle, s)

                if solution:
                    solved_count += 1
                    solved = True
                else:
                    solved = False
            except Exception as e:
                solved = False

            end = time.time()
            time_ms = (end - start) * 1000

            results.append({
                'size': size,
                'puzzle_index': i,
                'solved': solved,
                'solve_time_ms': time_ms
            })

            # Progress indicator every 10 puzzles
            if (i + 1) % 10 == 0 or i == puzzle_count - 1:
                print(f"  [{i+1}/{puzzle_count}] solved: {solved_count}")
                sys.stdout.flush()

        size_end = time.time()
        size_time = size_end - size_start

        accuracy[size] = solved_count / puzzle_count * 100
        avg_time[size] = size_time / puzzle_count * 1000

        print(f"  Done: {solved_count}/{puzzle_count} ({accuracy[size]:.1f}%) - Avg time: {avg_time[size]:.0f}ms")
        print()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('./results/optimized_evaluation.csv', index=False)

    # Save summary
    summary_df = pd.DataFrame({
        'size': list(accuracy.keys()),
        'accuracy': [accuracy[s] for s in accuracy.keys()],
        'avg_time_ms': [avg_time[s] for s in avg_time.keys()]
    })
    summary_df.to_csv('./results/optimized_summary.csv', index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Size':<6} {'Accuracy':<12} {'Avg Time':<12}")
    print("-" * 30)
    for size in range(3, 8):
        print(f"{size}x{size:<4} {accuracy[size]:.1f}%{'':<6} {avg_time[size]:.0f}ms")
    print()
    print(f"Results saved to ./results/optimized_evaluation.csv")
    print(f"Summary saved to ./results/optimized_summary.csv")


if __name__ == "__main__":
    main()
