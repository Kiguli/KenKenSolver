#!/usr/bin/env python3
"""
Run the NeuroSymbolic KenKen solver on all puzzles (sizes 3-9) and generate detailed results.
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
import time
import sys
import os

# Constants
BOARD_SIZE = 900
IMG_SIZE = 28
SCALE_FACTOR = 2

# ============================================================
# MODEL DEFINITIONS
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
# LOAD MODELS
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

# Load character recognition model
character_model = CNN_v2(output_dim=14)
state_dict = torch.load('./models/character_recognition_v2_model_weights.pth', weights_only=False)
character_model.load_state_dict(state_dict)
character_model.eval()

# Load grid detection model
grid_detection = Grid_CNN(output_dim=5)
state_dict = torch.load('./models/grid_detection_model_weights.pth', weights_only=False)
grid_detection.load_state_dict(state_dict)
grid_detection.eval()


# ============================================================
# SIZE DETECTION
# ============================================================

def get_size(filename):
    """Detect board size using Grid_CNN (only works for sizes 3-7)."""
    im = Image.open(filename).convert("RGBA")
    im = transform(im).unsqueeze(0)
    output = grid_detection(im)
    prediction = torch.argmax(output, dim=1).item()
    return prediction + 3


# ============================================================
# BORDER DETECTION (OpenCV)
# ============================================================

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


def find_size_and_borders(filename):
    """Original function - uses Grid_CNN to detect size."""
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

    size = get_size(filename)
    cages = construct_cages(
        find_h_borders(h_lines, size, border_thickness, border_thickness),
        find_v_borders(v_lines, size, border_thickness, border_thickness)
    )
    return size, cages, border_thickness // SCALE_FACTOR


def find_size_and_borders_with_size(filename, size):
    """Modified function - uses provided size instead of Grid_CNN."""
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

    cages = construct_cages(
        find_h_borders(h_lines, size, border_thickness, border_thickness),
        find_v_borders(v_lines, size, border_thickness, border_thickness)
    )
    return size, cages, border_thickness // SCALE_FACTOR


# ============================================================
# IMAGE SEGMENTATION
# ============================================================

def get_contours(img):
    img = (img * 255).astype(np.uint8)
    _, inp = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

    e_kernel = np.ones((1, 1), np.uint8)
    inp = cv.erode(inp, e_kernel, iterations=1)
    d_kernel = np.ones((3, 3), np.uint8)
    inp = cv.dilate(inp, d_kernel, iterations=1)

    ctrs, hierarchy = cv.findContours(inp.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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


# ============================================================
# CHARACTER CLASSIFICATION
# ============================================================

def get_predictions(characters):
    predictions = []
    with torch.no_grad():
        for c in characters:
            im = torch.tensor(c, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = character_model(im)
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


def make_puzzle(size, border_thickness, cages, filename):
    img = Image.open(filename).convert('L')
    grid = np.array(img)
    puzzle = []
    for cage in cages:
        puzzle.append({"cells": cage, "op": "", "target": 0})
        characters = segment_cell(grid, size, border_thickness + 5, cage[0][0], cage[0][1])
        predictions = get_predictions(characters)
        puzzle[-1] = update_puzzle(puzzle[-1], predictions)
    return puzzle


# ============================================================
# Z3 SOLVER
# ============================================================

def parse_block_constraints(puzzle, cells):
    constraints = []
    for block in puzzle:
        op = block["op"]
        target = block["target"]
        vars_in_block = [cells[i][j] for i, j in block["cells"]]
        if op == "":
            constraints.append(vars_in_block[0] == target)
        elif op == "add":
            constraints.append(Sum(vars_in_block) == target)
        elif op == "mul":
            product = vars_in_block[0]
            for v in vars_in_block[1:]:
                product *= v
            constraints.append(product == target)
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a - b == target, b - a == target))
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a / b == target, b / a == target))
        else:
            raise ValueError(f"Unsupported operation or malformed block: {block}")
    return constraints


def evaluate_puzzle(puzzle, size):
    X = [[Int("x_%s_%s" % (i + 1, j + 1)) for j in range(size)] for i in range(size)]
    cells_c = [And(1 <= X[i][j], X[i][j] <= size) for i in range(size) for j in range(size)]
    rows_c = [Distinct(X[i]) for i in range(size)]
    cols_c = [Distinct([X[i][j] for i in range(size)]) for j in range(size)]
    constraints = cells_c + rows_c + cols_c + parse_block_constraints(puzzle, X)
    instance = [[0] * size] * size
    instance = [If(instance[i][j] == 0, True, X[i][j] == instance[i][j]) for i in range(size) for j in range(size)]
    s = Solver()
    problem = constraints + instance
    s.add(problem)
    if s.check() == sat:
        m = s.model()
        solution = [[m.evaluate(X[i][j]) for j in range(size)] for i in range(size)]
        return solution
    else:
        return None


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():
    results = []
    csv_path = './results/detailed_evaluation.csv'

    print("=" * 70)
    print("KENKEN NEUROSYMBOLIC SOLVER - FULL EVALUATION")
    print("=" * 70)
    print()
    sys.stdout.flush()

    for size in range(3, 10):
        print(f"[Size {size}x{size}] Evaluating 100 puzzles...")
        sys.stdout.flush()

        size_start = time.time()
        solved_count = 0
        size_correct_count = 0

        for i in range(100):
            filename = f"./board_images/board{size}_{i}.png"

            start_time = time.time()
            size_detected = None
            size_correct = False
            solved = False
            error_type = "none"
            error_message = ""

            try:
                if size <= 7:
                    # Use Grid_CNN for size detection
                    size_detected, cages, border = find_size_and_borders(filename)
                    size_correct = (size_detected == size)
                    if not size_correct:
                        error_type = "size_detection"
                        error_message = f"Expected {size}, detected {size_detected}"
                else:
                    # Bypass Grid_CNN for sizes 8-9
                    size_detected, cages, border = find_size_and_borders_with_size(filename, size)
                    size_correct = True  # N/A - we provided the size

                puzzle = make_puzzle(size_detected, border, cages, filename)
                solution = evaluate_puzzle(puzzle, size_detected)

                if solution is None:
                    if error_type == "none":
                        error_type = "z3_unsolvable"
                        error_message = "Z3 could not satisfy constraints"
                else:
                    solved = True

            except Exception as e:
                if error_type == "none":
                    error_type = "exception"
                    error_message = str(e)

            end_time = time.time()
            time_ms = (end_time - start_time) * 1000

            if solved:
                solved_count += 1
            if size_correct:
                size_correct_count += 1

            results.append({
                'size': size,
                'puzzle_index': i,
                'filename': filename,
                'size_detected': size_detected,
                'size_correct': size_correct,
                'solved': solved,
                'solve_time_ms': time_ms,
                'error_type': error_type,
                'error_message': error_message
            })

            # Save CSV after every puzzle for monitoring
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)

            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/100...")
                sys.stdout.flush()

        size_end = time.time()
        size_time = size_end - size_start
        avg_time = size_time / 100 * 1000

        size_detection_str = f"{size_correct_count}/100" if size <= 7 else "N/A (bypassed)"
        print(f"  Done: {solved_count}/100 solved ({solved_count}%) - Size detection: {size_detection_str} - Avg time: {avg_time:.0f}ms")
        print()
        sys.stdout.flush()

    print(f"Results saved to {csv_path}")
    print()

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("=== RESULTS BY SIZE ===")
    for size in range(3, 10):
        size_df = df[df['size'] == size]
        solved = size_df['solved'].sum()
        total = len(size_df)
        avg_time = size_df['solve_time_ms'].mean()

        if size <= 7:
            size_correct = size_df['size_correct'].sum()
            print(f"Size {size}: {solved}/{total} solved ({solved/total*100:.1f}%) - avg {avg_time:.0f}ms")
            print(f"  - Size detection: {size_correct}/{total} correct")
        else:
            print(f"Size {size}: {solved}/{total} solved ({solved/total*100:.1f}%) - avg {avg_time:.0f}ms")
            print(f"  - Size detection: N/A (bypassed)")

    print()
    print("=== ERROR BREAKDOWN ===")
    error_counts = df[df['error_type'] != 'none']['error_type'].value_counts()
    for error_type, count in error_counts.items():
        print(f"{error_type}: {count} failures")

    if len(error_counts) == 0:
        print("No errors!")

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    run_evaluation()
