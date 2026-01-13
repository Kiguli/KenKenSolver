#!/usr/bin/env python3
"""
Generate 100 9x9 KenKen puzzles from Sudoku solutions, create board images,
and run the NeuroSymbolic solver evaluation pipeline.
"""

import json
import numpy as np
import random
import math
import os
import sys
import time
from PIL import Image
import pandas as pd
from z3 import *

# For evaluation pipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from torchvision import transforms

# ============================================================
# CONSTANTS
# ============================================================

BOARD_SIZE = 900
IMG_SIZE = 28
SCALE_FACTOR = 2
MAX_CAGE_SIZE = 4
MAX_SINGLETONS = 2
GRID_SIZE = 9

KENKEN_DIR = os.path.dirname(os.path.abspath(__file__))
SUDOKU_DIR = os.path.join(os.path.dirname(KENKEN_DIR), 'Sudoku')

# ============================================================
# Z3 PUZZLE VALIDATION
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
    """Validate puzzle with Z3 solver."""
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
# KENKEN PUZZLE GENERATION FROM SOLUTION GRID
# ============================================================

def get_target(cage, grid):
    """Calculate target value for a cage based on operation."""
    if cage["op"] == "sub":
        return abs(grid[cage["cells"][0][0]][cage["cells"][0][1]] -
                   grid[cage["cells"][1][0]][cage["cells"][1][1]])
    elif cage["op"] == "div":
        a = grid[cage["cells"][0][0]][cage["cells"][0][1]]
        b = grid[cage["cells"][1][0]][cage["cells"][1][1]]
        return max(a / b, b / a)
    elif cage["op"] == "mul":
        s = 1
        for cell in cage["cells"]:
            s *= grid[cell[0]][cell[1]]
        return s
    else:  # add or empty
        s = 0
        for cell in cage["cells"]:
            s += grid[cell[0]][cell[1]]
        return s


def get_unassigned_neighbors(grid_size, assigned, cell):
    """Get neighboring cells that haven't been assigned to a cage yet."""
    row, col = cell
    neighbors = []
    if row > 0:
        neighbors.append([row-1, col])
    if row < grid_size-1:
        neighbors.append([row+1, col])
    if col > 0:
        neighbors.append([row, col-1])
    if col < grid_size-1:
        neighbors.append([row, col+1])
    unassigned_neighbors = [neighbor for neighbor in neighbors if neighbor not in assigned]
    return unassigned_neighbors


def choose_unassigned(grid_size, assigned):
    """Choose a random unassigned cell."""
    if grid_size**2 == len(assigned):
        return None
    while True:
        row = np.random.randint(0, grid_size)
        col = np.random.randint(0, grid_size)
        if [row, col] not in assigned:
            return [row, col]


def generate_ops(cage, grid):
    """Generate operation for a cage."""
    cage_size = len(cage["cells"])
    options = []

    if cage_size == 1:
        return ""

    if cage_size == 2:
        a = grid[cage["cells"][0][0]][cage["cells"][0][1]]
        b = grid[cage["cells"][1][0]][cage["cells"][1][1]]
        if a % b == 0 or b % a == 0:
            options += ["div", "div"]
        options += ["sub", "mul", "add"]
    else:
        options = ["mul", "add"]

    return np.random.choice(options)


def make_cage(puzzle, grid, grid_size, assigned, cage_size, index):
    """Create a single cage of the given size."""
    to_assign = []
    start_cell = choose_unassigned(grid_size, assigned)

    puzzle.append({"cells": [], "op": "", "target": 0})

    num_attempts = 10

    if not start_cell:
        return

    if cage_size == 1:
        puzzle[index]["cells"].append(start_cell)
        assigned.append(start_cell)
        puzzle[index]["target"] = grid[start_cell[0]][start_cell[1]]
        return

    puzzle[index]["cells"].append(start_cell)
    to_assign.append(start_cell)

    while len(puzzle[index]["cells"]) < cage_size:
        neighbors = []
        for cell in puzzle[index]["cells"]:
            neighbors.extend(get_unassigned_neighbors(grid_size, assigned + to_assign, [cell[0], cell[1]]))

        if not neighbors:
            if num_attempts == 0:
                puzzle[index]["cells"] = []
                return

            num_attempts -= 1
            start_cell = choose_unassigned(grid_size, assigned)
            puzzle[index]["cells"] = [start_cell]
            to_assign = [start_cell]
            continue

        neighbor = neighbors[np.random.randint(0, len(neighbors))]
        puzzle[index]["cells"].append(neighbor)
        to_assign.append(neighbor)

    assigned += to_assign

    puzzle[index]["op"] = generate_ops(puzzle[index], grid)
    puzzle[index]["target"] = get_target(puzzle[index], grid)


def generate_puzzle_from_solution_simple(grid, grid_size, max_cage_size, max_singletons):
    """
    Generate a KenKen puzzle from a given solution grid using a simpler,
    more reliable algorithm that guarantees all cells are covered.
    """
    # Track which cells are assigned
    assigned = [[False] * grid_size for _ in range(grid_size)]
    puzzle = []
    singleton_count = 0

    def get_neighbors(row, col):
        """Get unassigned adjacent neighbors."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and not assigned[nr][nc]:
                neighbors.append((nr, nc))
        return neighbors

    def grow_cage(start_row, start_col, target_size):
        """Grow a cage from a starting cell to target size."""
        cage_cells = [[start_row, start_col]]
        assigned[start_row][start_col] = True

        while len(cage_cells) < target_size:
            # Collect all unassigned neighbors of the current cage
            all_neighbors = []
            for r, c in cage_cells:
                for nr, nc in get_neighbors(r, c):
                    if [nr, nc] not in cage_cells and (nr, nc) not in all_neighbors:
                        all_neighbors.append((nr, nc))

            if not all_neighbors:
                break  # Can't grow further

            # Choose a random neighbor
            nr, nc = random.choice(all_neighbors)
            cage_cells.append([nr, nc])
            assigned[nr][nc] = True

        return cage_cells

    # Iterate through all cells in row-major order
    for row in range(grid_size):
        for col in range(grid_size):
            if assigned[row][col]:
                continue

            # Decide cage size
            # Prefer sizes 2-4, occasionally allow singletons
            if singleton_count < max_singletons and random.random() < 0.1:
                target_size = 1
            else:
                target_size = random.randint(2, max_cage_size)

            # Grow the cage
            cage_cells = grow_cage(row, col, target_size)

            if len(cage_cells) == 1:
                singleton_count += 1

            # Create the cage
            cage = {"cells": cage_cells, "op": "", "target": 0}
            cage["op"] = generate_ops(cage, grid)
            cage["target"] = get_target(cage, grid)
            puzzle.append(cage)

    return puzzle


def generate_puzzle_from_solution(grid, grid_size, max_cage_size, max_singletons):
    """Generate a KenKen puzzle from a given solution grid."""
    return generate_puzzle_from_solution_simple(grid, grid_size, max_cage_size, max_singletons)


# ============================================================
# BOARD IMAGE GENERATION
# ============================================================

def add_border(grid, padding):
    """Add black border to grid."""
    for i in range(padding):
        grid[i] = 0
        grid[-i] = 0
        grid[:, i] = 0
        grid[:, -i] = 0
    return grid


def make_board(size):
    """Create blank KenKen board with grid lines."""
    grid = np.ones((900, 900))
    grid = add_border(grid, 16)
    square_size = 900 // size

    for i in range(1, size):
        dim = i * square_size
        thickness = 2 if size < 7 else 1
        for j in range(-thickness, 2):
            cur = dim + j
            grid[cur] = 0
            grid[-cur] = 0
            grid[:, cur] = 0
            grid[:, -cur] = 0

    return grid


def add_side_wall(grid, size, row, col, side):
    """Draw thick wall on one side of a cell."""
    square_size = 900 // size
    top_left = (row * square_size, col * square_size)
    bottom_right = ((row + 1) * square_size, (col + 1) * square_size)

    if side == "left":
        if row == 0:
            for j in range(-8, 8):
                grid[top_left[0]:bottom_right[0]+8, top_left[1]+j] = 0
        elif row == size-1:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0], top_left[1]+j] = 0
        else:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0]+8, top_left[1]+j] = 0

    elif side == "right":
        if row == 0:
            for j in range(-8, 8):
                grid[top_left[0]:bottom_right[0]+8, bottom_right[1]+j] = 0
        elif row == size-1:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0], bottom_right[1]+j] = 0
        else:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0]+8, bottom_right[1]+j] = 0

    elif side == "top":
        if col == 0:
            for j in range(-8, 8):
                grid[top_left[0]+j, top_left[1]:bottom_right[1]+8] = 0
        elif col == size-1:
            for j in range(-8, 8):
                grid[top_left[0]+j, top_left[1]-8:bottom_right[1]] = 0
        else:
            for j in range(-8, 8):
                grid[top_left[0]+j, top_left[1]-8:bottom_right[1]+8] = 0

    else:  # bottom
        if col == 0:
            for j in range(-8, 8):
                grid[bottom_right[0]+j, top_left[1]:bottom_right[1]+8] = 0
        elif col == size-1:
            for j in range(-8, 8):
                grid[bottom_right[0]+j, top_left[1]-8:bottom_right[1]] = 0
        else:
            for j in range(-8, 8):
                grid[bottom_right[0]+j, top_left[1]-8:bottom_right[1]+8] = 0

    return grid


def draw_cage(grid, size, cage):
    """Draw walls around a cage."""
    for cell in cage["cells"]:
        row, col = cell

        if [row, col+1] not in cage["cells"] and col+1 < size:
            grid = add_side_wall(grid, size, row, col, "right")
        if [row, col-1] not in cage["cells"] and col-1 >= 0:
            grid = add_side_wall(grid, size, row, col, "left")
        if [row+1, col] not in cage["cells"] and row+1 < size:
            grid = add_side_wall(grid, size, row, col, "bottom")
        if [row-1, col] not in cage["cells"] and row-1 >= 0:
            grid = add_side_wall(grid, size, row, col, "top")

    return grid


class BoardImageGenerator:
    """Class to handle board image generation with loaded fonts."""

    def __init__(self):
        # Load TMNIST font data
        df = pd.read_csv(os.path.join(KENKEN_DIR, "symbols/TMNIST_NotoSans.csv"))
        noto_sans = df[df['names'].str.contains('notosans', case=False, na=False)]
        noto_sans = noto_sans[noto_sans['names'].str.contains('NotoSans-Regular', case=False, na=False)]
        noto_sans = noto_sans.sort_values(by='labels')

        self.image_arrays = [
            (255 - row.values.reshape(28, 28)) / 255.0
            for _, row in noto_sans.drop(columns=['names', 'labels']).iterrows()
        ]
        self.labels = noto_sans['labels'].tolist()

    def normalize_symbol(self, file_path):
        """Load and normalize operator symbol."""
        im = Image.open(os.path.join(KENKEN_DIR, "symbols/operators/", file_path)).convert("RGBA")
        rgba = np.array(im)
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        alpha = a / 255.0
        composited = gray * alpha + 255 * (1 - alpha)
        normalized = composited / 255.0
        return normalized

    def insert_number_in_cage(self, grid, size, cell, number, position):
        """Insert a digit into a cage."""
        square_size = 900 // size
        number_size = square_size // 4

        pil_img = Image.fromarray(self.image_arrays[number])
        resized_img = pil_img.resize((number_size, number_size), resample=Image.NEAREST)
        cropped = np.array(resized_img.crop((resized_img.width // 5, 0, resized_img.width * 4 // 5, resized_img.height)))

        height, width = cropped.shape
        col_position = 20 + position * width

        top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)
        grid[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width] = cropped
        return grid

    def insert_symbol_in_cage(self, grid, size, cell, symbol, position):
        """Insert an operator symbol into a cage."""
        square_size = 900 // size
        number_size = square_size // 4

        symbol_img = Image.fromarray(self.normalize_symbol(symbol + ".png"))
        resized_img = symbol_img.resize((number_size, number_size), resample=Image.NEAREST)
        cropped = np.array(resized_img.crop((resized_img.width // 6, 0, resized_img.width * 5 // 6, resized_img.height)))

        height, width = cropped.shape
        col_position = 20 + position * width

        top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)
        grid[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width] = cropped
        return grid

    def write_to_cage(self, board, size, cage):
        """Write target and operator to a cage."""
        # Find top-left cell
        start_cell = cage["cells"][0]
        for cell in cage["cells"][1:]:
            if cell[0] < start_cell[0]:
                start_cell = cell
            elif cell[0] == start_cell[0] and cell[1] < start_cell[1]:
                start_cell = cell

        # Get digits of target
        digits = []
        temp = int(cage["target"])
        if temp == 0:
            digits = [0]
        else:
            while temp > 0:
                digits.append(int(temp % 10))
                temp //= 10
            digits = digits[::-1]

        # Write digits
        for i, digit in enumerate(digits):
            board = self.insert_number_in_cage(board, size, start_cell, digit, i)

        # Write operator
        if cage["op"] != "":
            board = self.insert_symbol_in_cage(board, size, start_cell, cage["op"], len(digits))

        return board

    def make_board_full(self, size, puzzle):
        """Create complete board image."""
        board = make_board(size)
        for cage in puzzle:
            board = draw_cage(board, size, cage)
            board = self.write_to_cage(board, size, cage)
        return board

    def make_and_save(self, size, puzzle, iter):
        """Generate and save board image."""
        board = self.make_board_full(size, puzzle)
        array_uint8 = (board * 255).astype(np.uint8)
        image = Image.fromarray(array_uint8, mode='L')
        image.save(os.path.join(KENKEN_DIR, f'board_images/board{size}_{iter}.png'))


# ============================================================
# EVALUATION PIPELINE
# ============================================================

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
    """Find horizontal borders between cells."""
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
    """Find vertical borders between cells."""
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


def make_cage_from_borders(start_cell, visited, h_borders, v_borders):
    """Construct a cage from border detection."""
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
    """Construct all cages from border matrices."""
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
    """Get border thickness from line detection."""
    v_lines = lines[lines['x1'] == lines['x2']]
    return min(v_lines['x1'])


def find_size_and_borders_with_size(filename, size):
    """Detect borders using provided size (bypasses Grid_CNN)."""
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


def get_contours(img):
    """Get character contours from image."""
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
    """Extract and normalize a character from image."""
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
    """Extract characters from a cell."""
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
    """Get character predictions from CNN model."""
    predictions = []
    with torch.no_grad():
        for c in characters:
            im = torch.tensor(c, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            output = model(im)
            predictions.append(torch.argmax(output, dim=1).item())
    return predictions


def update_puzzle_from_predictions(puzzle, predictions):
    """Update puzzle with predicted target and operator."""
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
    """Construct puzzle from image using OCR."""
    img = Image.open(filename).convert('L')
    grid = np.array(img)
    puzzle = []
    for cage in cages:
        puzzle.append({"cells": cage, "op": "", "target": 0})
        characters = segment_cell(grid, size, border_thickness + 5, cage[0][0], cage[0][1])
        predictions = get_predictions(characters, model)
        puzzle[-1] = update_puzzle_from_predictions(puzzle[-1], predictions)
    return puzzle


def run_evaluation(model):
    """Run evaluation on all 9x9 puzzles."""
    results = []
    csv_path = os.path.join(KENKEN_DIR, 'results/9x9_evaluation.csv')

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
        filename = os.path.join(KENKEN_DIR, f"board_images/board{size}_{i}.png")

        start_time = time.time()
        size_detected = size
        size_correct = True  # N/A - we provided the size
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

        # Progress indicator
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100...")
            sys.stdout.flush()

    size_end = time.time()
    size_time = size_end - size_start
    avg_time = size_time / 100 * 1000

    print(f"  Done: {solved_count}/100 solved ({solved_count}%) - Avg time: {avg_time:.0f}ms")
    print()
    sys.stdout.flush()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Print summary
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


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("9x9 KENKEN PUZZLE GENERATION AND EVALUATION")
    print("=" * 70)
    print()

    # Step 1: Load Sudoku solutions
    print("Step 1: Loading Sudoku solutions...")
    sudoku_path = os.path.join(SUDOKU_DIR, 'puzzles/puzzles_dict.json')
    with open(sudoku_path, 'r') as f:
        sudoku_data = json.load(f)

    solutions_9x9 = [puzzle['solution'] for puzzle in sudoku_data['9']]
    print(f"  Loaded {len(solutions_9x9)} unique 9x9 Sudoku solutions")
    print()

    # Step 2: Generate KenKen puzzles from solutions
    # Note: We skip Z3 validation during generation since puzzles are derived
    # from valid Sudoku solutions, so they are guaranteed to be solvable.
    print("Step 2: Generating KenKen puzzles from Sudoku solutions...")
    kenken_puzzles = []

    for i, solution in enumerate(solutions_9x9):
        puzzle = generate_puzzle_from_solution(solution, GRID_SIZE, MAX_CAGE_SIZE, MAX_SINGLETONS)

        # Convert numpy types to Python types for JSON serialization
        for cage in puzzle:
            if isinstance(cage["op"], np.str_):
                cage["op"] = str(cage["op"])
            if isinstance(cage["target"], (np.integer, np.floating)):
                cage["target"] = float(cage["target"]) if cage["op"] == "div" else int(cage["target"])

        kenken_puzzles.append(puzzle)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/100 puzzles...")

    print(f"  Successfully generated {len(kenken_puzzles)} KenKen puzzles")
    print()

    # Step 3: Save puzzles to JSON
    print("Step 3: Saving puzzles to puzzles_dict.json...")
    kenken_puzzles_path = os.path.join(KENKEN_DIR, 'puzzles/puzzles_dict.json')
    with open(kenken_puzzles_path, 'r') as f:
        puzzles_dict = json.load(f)

    puzzles_dict['9'] = kenken_puzzles

    with open(kenken_puzzles_path, 'w') as f:
        json.dump(puzzles_dict, f, indent=2)

    print(f"  Saved to {kenken_puzzles_path}")
    print()

    # Step 4: Generate board images
    print("Step 4: Generating board images...")
    generator = BoardImageGenerator()

    for i, puzzle in enumerate(kenken_puzzles):
        generator.make_and_save(GRID_SIZE, puzzle, i)
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/100 images...")

    print(f"  Successfully generated 100 board images")
    print()

    # Step 5: Run evaluation
    print("Step 5: Running solver evaluation...")
    print()

    # Load character recognition model
    character_model = CNN_v2(output_dim=14)
    model_path = os.path.join(KENKEN_DIR, 'models/character_recognition_v2_model_weights.pth')
    state_dict = torch.load(model_path, weights_only=False)
    character_model.load_state_dict(state_dict)
    character_model.eval()

    results_df = run_evaluation(character_model)

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
