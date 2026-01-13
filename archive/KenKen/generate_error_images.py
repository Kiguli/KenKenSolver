#!/usr/bin/env python3
"""
Generate board images showing differences between DETECTED and EXPECTED puzzles.
Things that match are shown in black, differences are highlighted in RED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2 as cv
import pandas as pd
from torchvision import transforms
import os
import json

# Constants
BOARD_SIZE = 900
IMG_SIZE = 28
SCALE_FACTOR = 2

# ============================================================
# MODEL DEFINITIONS (same as run_full_evaluation.py)
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
    im = Image.open(filename).convert("RGBA")
    im = transform(im).unsqueeze(0)
    output = grid_detection(im)
    prediction = torch.argmax(output, dim=1).item()
    return prediction + 3


# ============================================================
# BORDER DETECTION
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
    h_borders = find_h_borders(h_lines, size, border_thickness, border_thickness)
    v_borders = find_v_borders(v_lines, size, border_thickness, border_thickness)
    cages = construct_cages(h_borders, v_borders)
    return size, cages, border_thickness // SCALE_FACTOR, h_borders, v_borders


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

    if len(ctrs) == 0:
        return []

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
    if len(predictions) == 0:
        puzzle["target"] = 0
        puzzle["op"] = ""
    elif len(predictions) == 1:
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
# COMPARISON UTILITIES
# ============================================================

def get_expected_borders(expected_puzzle, size):
    """Convert expected puzzle cages to h_borders and v_borders format."""
    h_borders = np.zeros((size - 1, size))
    v_borders = np.zeros((size, size - 1))

    # For each cage, mark internal borders as 0 (no wall) and external as 1 (wall)
    # Start with all internal walls present
    h_borders.fill(1)
    v_borders.fill(1)

    # Remove walls within cages
    for cage in expected_puzzle:
        cells = [tuple(c) for c in cage["cells"]]
        for cell in cells:
            row, col = cell
            # Check if neighbor to the right is in same cage
            if col < size - 1 and (row, col + 1) in cells:
                v_borders[row][col] = 0
            # Check if neighbor below is in same cage
            if row < size - 1 and (row + 1, col) in cells:
                h_borders[row][col] = 0

    return h_borders, v_borders


def find_cage_for_cell(puzzle, row, col):
    """Find which cage contains a given cell."""
    for i, cage in enumerate(puzzle):
        for cell in cage["cells"]:
            if cell[0] == row and cell[1] == col:
                return i
    return -1


def cages_match(detected_cage, expected_cage):
    """Check if two cages have the same cells, target, and operation."""
    detected_cells = set(tuple(c) for c in detected_cage["cells"])
    expected_cells = set(tuple(c) for c in expected_cage["cells"])

    return (detected_cells == expected_cells and
            detected_cage["target"] == expected_cage["target"] and
            detected_cage["op"] == expected_cage["op"])


# ============================================================
# BOARD IMAGE GENERATION WITH DIFFERENCE HIGHLIGHTING
# ============================================================

def add_border(grid, padding, color):
    """Add border with specified color (RGB tuple or single value for grayscale-like)."""
    r, g, b = color
    for i in range(padding):
        grid[i, :, 0] = r
        grid[i, :, 1] = g
        grid[i, :, 2] = b
        grid[-i-1, :, 0] = r
        grid[-i-1, :, 1] = g
        grid[-i-1, :, 2] = b
        grid[:, i, 0] = r
        grid[:, i, 1] = g
        grid[:, i, 2] = b
        grid[:, -i-1, 0] = r
        grid[:, -i-1, 1] = g
        grid[:, -i-1, 2] = b
    return grid


def make_board_base(size):
    """Create a base board with thin black grid lines."""
    grid = np.ones((900, 900, 3)) * 255
    grid = add_border(grid, 16, (0, 0, 0))
    square_size = 900 // size

    for i in range(1, size):
        dim = i * square_size
        thickness = 2 if size < 7 else 1
        for j in range(-thickness, 2):
            cur = dim + j
            if 0 <= cur < 900:
                grid[cur, :, :] = 0
                grid[:, cur, :] = 0

    return grid


def add_side_wall(grid, size, row, col, side, color):
    """Draw a cage wall on the specified side of a cell with given color."""
    r, g, b = color
    square_size = 900 // size
    top_left = (row * square_size, col * square_size)
    bottom_right = ((row + 1) * square_size, (col + 1) * square_size)

    def set_color(grid, r_slice, c_slice):
        grid[r_slice, c_slice, 0] = r
        grid[r_slice, c_slice, 1] = g
        grid[r_slice, c_slice, 2] = b

    if side == "left":
        if row == 0:
            for j in range(-8, 8):
                col_idx = top_left[1] + j
                if 0 <= col_idx < 900:
                    set_color(grid, slice(top_left[0], min(bottom_right[0] + 8, 900)), col_idx)
        elif row == size - 1:
            for j in range(-8, 8):
                col_idx = top_left[1] + j
                if 0 <= col_idx < 900:
                    set_color(grid, slice(max(top_left[0] - 8, 0), bottom_right[0]), col_idx)
        else:
            for j in range(-8, 8):
                col_idx = top_left[1] + j
                if 0 <= col_idx < 900:
                    set_color(grid, slice(max(top_left[0] - 8, 0), min(bottom_right[0] + 8, 900)), col_idx)

    elif side == "right":
        if row == 0:
            for j in range(-8, 8):
                col_idx = bottom_right[1] + j
                if 0 <= col_idx < 900:
                    set_color(grid, slice(top_left[0], min(bottom_right[0] + 8, 900)), col_idx)
        elif row == size - 1:
            for j in range(-8, 8):
                col_idx = bottom_right[1] + j
                if 0 <= col_idx < 900:
                    set_color(grid, slice(max(top_left[0] - 8, 0), bottom_right[0]), col_idx)
        else:
            for j in range(-8, 8):
                col_idx = bottom_right[1] + j
                if 0 <= col_idx < 900:
                    set_color(grid, slice(max(top_left[0] - 8, 0), min(bottom_right[0] + 8, 900)), col_idx)

    elif side == "top":
        if col == 0:
            for j in range(-8, 8):
                row_idx = top_left[0] + j
                if 0 <= row_idx < 900:
                    set_color(grid, row_idx, slice(top_left[1], min(bottom_right[1] + 8, 900)))
        elif col == size - 1:
            for j in range(-8, 8):
                row_idx = top_left[0] + j
                if 0 <= row_idx < 900:
                    set_color(grid, row_idx, slice(max(top_left[1] - 8, 0), bottom_right[1]))
        else:
            for j in range(-8, 8):
                row_idx = top_left[0] + j
                if 0 <= row_idx < 900:
                    set_color(grid, row_idx, slice(max(top_left[1] - 8, 0), min(bottom_right[1] + 8, 900)))

    else:  # bottom
        if col == 0:
            for j in range(-8, 8):
                row_idx = bottom_right[0] + j
                if 0 <= row_idx < 900:
                    set_color(grid, row_idx, slice(top_left[1], min(bottom_right[1] + 8, 900)))
        elif col == size - 1:
            for j in range(-8, 8):
                row_idx = bottom_right[0] + j
                if 0 <= row_idx < 900:
                    set_color(grid, row_idx, slice(max(top_left[1] - 8, 0), bottom_right[1]))
        else:
            for j in range(-8, 8):
                row_idx = bottom_right[0] + j
                if 0 <= row_idx < 900:
                    set_color(grid, row_idx, slice(max(top_left[1] - 8, 0), min(bottom_right[1] + 8, 900)))

    return grid


def draw_cage_walls(grid, size, cage, color):
    """Draw cage boundaries with specified color."""
    for cell in cage["cells"]:
        row, col = cell

        if [row, col + 1] not in cage["cells"] and col + 1 < size:
            grid = add_side_wall(grid, size, row, col, "right", color)
        if [row, col - 1] not in cage["cells"] and col - 1 >= 0:
            grid = add_side_wall(grid, size, row, col, "left", color)
        if [row + 1, col] not in cage["cells"] and row + 1 < size:
            grid = add_side_wall(grid, size, row, col, "bottom", color)
        if [row - 1, col] not in cage["cells"] and row - 1 >= 0:
            grid = add_side_wall(grid, size, row, col, "top", color)

    return grid


def insert_number_in_cage(grid, size, cell, number, position, image_arrays, color):
    """Insert a number with specified color."""
    if number < 0 or number >= len(image_arrays):
        return grid

    square_size = 900 // size
    number_size = square_size // 4
    r, g, b = color

    pil_img = Image.fromarray(image_arrays[number])
    resized_img = pil_img.resize((number_size, number_size), resample=Image.NEAREST)
    cropped = np.array(resized_img.crop((resized_img.width // 5, 0, resized_img.width * 4 // 5, resized_img.height)))

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)

    end_row = min(top_left[0] + height, 900)
    end_col = min(top_left[1] + width, 900)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        char_slice = cropped[:actual_height, :actual_width]
        mask = char_slice < 0.5

        grid[top_left[0]:end_row, top_left[1]:end_col, 0] = np.where(mask, r, grid[top_left[0]:end_row, top_left[1]:end_col, 0])
        grid[top_left[0]:end_row, top_left[1]:end_col, 1] = np.where(mask, g, grid[top_left[0]:end_row, top_left[1]:end_col, 1])
        grid[top_left[0]:end_row, top_left[1]:end_col, 2] = np.where(mask, b, grid[top_left[0]:end_row, top_left[1]:end_col, 2])

    return grid


def normalize_symbol(file_path):
    """Load and normalize an operator symbol."""
    im = Image.open("./symbols/operators/" + file_path).convert("RGBA")
    rgba = np.array(im)
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    alpha = a / 255.0
    composited = gray * alpha + 255 * (1 - alpha)
    normalized = composited / 255.0
    return normalized


def insert_symbol_in_cage(grid, size, cell, symbol, position, color):
    """Insert an operator symbol with specified color."""
    if not symbol:
        return grid

    square_size = 900 // size
    number_size = square_size // 4
    r, g, b = color

    sym_img = Image.fromarray(normalize_symbol(symbol + ".png"))
    resized_img = sym_img.resize((number_size, number_size), resample=Image.NEAREST)
    cropped = np.array(resized_img.crop((resized_img.width // 6, 0, resized_img.width * 5 // 6, resized_img.height)))

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)

    end_row = min(top_left[0] + height, 900)
    end_col = min(top_left[1] + width, 900)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        char_slice = cropped[:actual_height, :actual_width]
        mask = char_slice < 0.5

        grid[top_left[0]:end_row, top_left[1]:end_col, 0] = np.where(mask, r, grid[top_left[0]:end_row, top_left[1]:end_col, 0])
        grid[top_left[0]:end_row, top_left[1]:end_col, 1] = np.where(mask, g, grid[top_left[0]:end_row, top_left[1]:end_col, 1])
        grid[top_left[0]:end_row, top_left[1]:end_col, 2] = np.where(mask, b, grid[top_left[0]:end_row, top_left[1]:end_col, 2])

    return grid


def write_to_cage(board, size, cage, image_arrays, color):
    """Write target and operator to cage with specified color."""
    start_cell = cage["cells"][0]

    if len(cage["cells"]) > 1:
        for cell in cage["cells"][1:]:
            if cell[0] < start_cell[0]:
                start_cell = cell
            elif cell[0] == start_cell[0] and cell[1] < start_cell[1]:
                start_cell = cell

    digits = []
    temp = cage["target"]
    if temp == 0:
        digits = [0]
    else:
        while temp > 0:
            digits.append(int(temp % 10))
            temp //= 10
        digits = digits[::-1]

    for i, digit in enumerate(digits):
        board = insert_number_in_cage(board, size, start_cell, digit, i, image_arrays, color)

    if cage["op"] != "":
        board = insert_symbol_in_cage(board, size, start_cell, cage["op"], len(digits), color)

    return board


def load_digit_images():
    """Load digit images from TMNIST dataset."""
    df = pd.read_csv("./symbols/TMNIST_NotoSans.csv")
    noto_sans = df[df['names'].str.contains('notosans', case=False, na=False)]
    noto_sans = noto_sans[noto_sans['names'].str.contains('NotoSans-Regular', case=False, na=False)]
    noto_sans = noto_sans.sort_values(by='labels')

    image_arrays = [
        (255 - row.values.reshape(28, 28)) / 255.0
        for _, row in noto_sans.drop(columns=['names', 'labels']).iterrows()
    ]
    return image_arrays


def make_diff_board(size, detected_puzzle, expected_puzzle, detected_h_borders, detected_v_borders, image_arrays):
    """
    Create a board showing differences between detected and expected puzzles.
    - Black: things that match
    - Red: things that are different (extra walls, wrong text, etc.)
    """
    board = make_board_base(size)

    # Get expected borders
    expected_h_borders, expected_v_borders = get_expected_borders(expected_puzzle, size)

    # Build a map from cell to expected cage info
    expected_cage_map = {}
    for cage in expected_puzzle:
        cells_tuple = frozenset(tuple(c) for c in cage["cells"])
        for cell in cage["cells"]:
            expected_cage_map[tuple(cell)] = {
                "cells": cells_tuple,
                "target": cage["target"],
                "op": cage["op"]
            }

    # Draw horizontal borders - compare detected vs expected
    for i in range(size - 1):
        for j in range(size):
            has_detected = detected_h_borders[i][j] == 1
            has_expected = expected_h_borders[i][j] == 1

            if has_detected and has_expected:
                # Both have wall - black
                board = add_side_wall(board, size, i, j, "bottom", (0, 0, 0))
            elif has_detected and not has_expected:
                # Detected has wall but expected doesn't - RED (extra wall)
                board = add_side_wall(board, size, i, j, "bottom", (255, 0, 0))
            elif not has_detected and has_expected:
                # Expected has wall but detected doesn't - RED (missing wall)
                # Draw in red to show what should be there but isn't
                board = add_side_wall(board, size, i, j, "bottom", (255, 0, 0))

    # Draw vertical borders - compare detected vs expected
    for i in range(size):
        for j in range(size - 1):
            has_detected = detected_v_borders[i][j] == 1
            has_expected = expected_v_borders[i][j] == 1

            if has_detected and has_expected:
                # Both have wall - black
                board = add_side_wall(board, size, i, j, "right", (0, 0, 0))
            elif has_detected and not has_expected:
                # Detected has wall but expected doesn't - RED (extra wall)
                board = add_side_wall(board, size, i, j, "right", (255, 0, 0))
            elif not has_detected and has_expected:
                # Expected has wall but detected doesn't - RED (missing wall)
                board = add_side_wall(board, size, i, j, "right", (255, 0, 0))

    # Now draw the text for each detected cage
    # Compare against expected to determine color
    for detected_cage in detected_puzzle:
        detected_cells = frozenset(tuple(c) for c in detected_cage["cells"])

        # Find if there's a matching expected cage
        matching_expected = None
        for expected_cage in expected_puzzle:
            expected_cells = frozenset(tuple(c) for c in expected_cage["cells"])
            if detected_cells == expected_cells:
                matching_expected = expected_cage
                break

        if matching_expected:
            # Cells match - check if target and op match
            target_matches = detected_cage["target"] == matching_expected["target"]
            op_matches = detected_cage["op"] == matching_expected["op"]

            if target_matches and op_matches:
                color = (0, 0, 0)  # Black - everything matches
            else:
                color = (255, 0, 0)  # Red - target or op mismatch
        else:
            # No matching cage (cells don't match any expected cage)
            color = (255, 0, 0)  # Red

        board = write_to_cage(board, size, detected_cage, image_arrays, color)

    return board


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GENERATING DIFFERENCE BOARD IMAGES")
    print("=" * 60)
    print()

    # Failed 7x7 puzzle indices
    failed_indices = [13, 14, 31, 35, 45, 49, 69, 85, 87]

    # Load expected puzzles
    print("Loading expected puzzles from puzzles_dict.json...")
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        puzzles_dict = json.load(f)

    # Load digit images
    print("Loading digit images...")
    image_arrays = load_digit_images()

    # Create output directory if needed
    os.makedirs('./board_image_errors', exist_ok=True)

    for idx in failed_indices:
        filename = f"./board_images/board7_{idx}.png"
        print(f"Processing puzzle {idx}...")

        try:
            # Get expected puzzle
            expected_puzzle = puzzles_dict["7"][idx]
            print(f"  Expected cages: {len(expected_puzzle)}")

            # Detect what the solver sees
            size, cages, border, h_borders, v_borders = find_size_and_borders(filename)
            detected_puzzle = make_puzzle(size, border, cages, filename)

            print(f"  Detected size: {size}")
            print(f"  Detected cages: {len(detected_puzzle)}")

            # Generate difference board image
            board = make_diff_board(size, detected_puzzle, expected_puzzle, h_borders, v_borders, image_arrays)

            # Save as RGB image
            array_uint8 = board.astype(np.uint8)
            image = Image.fromarray(array_uint8, mode='RGB')
            output_path = f'./board_image_errors/board7_{idx}.png'
            image.save(output_path)
            print(f"  Saved to {output_path}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
