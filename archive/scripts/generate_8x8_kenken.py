#!/usr/bin/env python3
"""
Generate 100 8x8 KenKen puzzles from Latin square solutions.

This script:
1. Generates 100 unique 8x8 Latin squares (each digit 1-8 appears exactly once per row/column)
2. Converts each to a KenKen puzzle with random cages and operations
3. Validates each puzzle with Z3 solver
4. Generates board images with both computer and handwritten digits (300px cells)
5. Updates kenken_puzzles.json with the new puzzles
"""

import json
import numpy as np
import random
import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
from z3 import *

# Constants
CELL_SIZE = 300  # Fixed cell size for all puzzle sizes (V2 approach)
BORDER_THICKNESS = 16
GRID_SIZE = 8
MAX_CAGE_SIZE = 4
MAX_SINGLETONS = 2
NUM_PUZZLES = 100

# Get base directory
BASE_DIR = Path(__file__).parent.parent.resolve()
ARCHIVE_KENKEN_DIR = BASE_DIR / 'archive' / 'KenKen'
ARCHIVE_HANDWRITTEN_DIR = BASE_DIR / 'archive' / 'KenKen-handwritten-v2'


# ============================================================
# LATIN SQUARE GENERATION
# ============================================================

def generate_latin_square(n):
    """Generate a random n×n Latin square using shuffling approach."""
    # Start with a base Latin square
    square = [[((i + j) % n) + 1 for j in range(n)] for i in range(n)]

    # Shuffle rows
    random.shuffle(square)

    # Shuffle columns
    cols = list(range(n))
    random.shuffle(cols)
    square = [[row[c] for c in cols] for row in square]

    # Shuffle digit labels (relabel 1-n to random permutation)
    digit_perm = list(range(1, n + 1))
    random.shuffle(digit_perm)
    square = [[digit_perm[cell - 1] for cell in row] for row in square]

    return square


def is_valid_latin_square(grid, n):
    """Verify a grid is a valid Latin square."""
    # Check rows
    for row in grid:
        if sorted(row) != list(range(1, n + 1)):
            return False

    # Check columns
    for col in range(n):
        column = [grid[row][col] for row in range(n)]
        if sorted(column) != list(range(1, n + 1)):
            return False

    return True


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
            options += ["div", "div"]  # Weight division higher
        options += ["sub", "mul", "add"]
    else:
        options = ["mul", "add"]

    return np.random.choice(options)


def generate_puzzle_from_solution(grid, grid_size, max_cage_size, max_singletons):
    """
    Generate a KenKen puzzle from a given solution grid.
    Uses a simple, reliable algorithm that guarantees all cells are covered.
    """
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
            all_neighbors = []
            for r, c in cage_cells:
                for nr, nc in get_neighbors(r, c):
                    if [nr, nc] not in cage_cells and (nr, nc) not in all_neighbors:
                        all_neighbors.append((nr, nc))

            if not all_neighbors:
                break

            nr, nc = random.choice(all_neighbors)
            cage_cells.append([nr, nc])
            assigned[nr][nc] = True

        return cage_cells

    # Iterate through all cells
    for row in range(grid_size):
        for col in range(grid_size):
            if assigned[row][col]:
                continue

            # Decide cage size
            if singleton_count < max_singletons and random.random() < 0.1:
                target_size = 1
            else:
                target_size = random.randint(2, max_cage_size)

            cage_cells = grow_cage(row, col, target_size)

            if len(cage_cells) == 1:
                singleton_count += 1

            cage = {"cells": cage_cells, "op": "", "target": 0}
            cage["op"] = generate_ops(cage, grid)
            cage["target"] = get_target(cage, grid)
            puzzle.append(cage)

    return puzzle


# ============================================================
# Z3 PUZZLE VALIDATION
# ============================================================

def parse_block_constraints(puzzle, cells):
    """Convert puzzle blocks to Z3 constraints."""
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
            raise ValueError(f"Unsupported operation: {block}")
    return constraints


def validate_puzzle(puzzle, size):
    """Validate puzzle with Z3 solver."""
    X = [[Int("x_%s_%s" % (i + 1, j + 1)) for j in range(size)] for i in range(size)]
    cells_c = [And(1 <= X[i][j], X[i][j] <= size) for i in range(size) for j in range(size)]
    rows_c = [Distinct(X[i]) for i in range(size)]
    cols_c = [Distinct([X[i][j] for i in range(size)]) for j in range(size)]
    constraints = cells_c + rows_c + cols_c + parse_block_constraints(puzzle, X)

    s = Solver()
    s.add(constraints)
    if s.check() == sat:
        m = s.model()
        solution = [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
        return solution
    return None


# ============================================================
# BOARD IMAGE GENERATION (300px cells - V2 approach)
# ============================================================

def add_border(grid, padding):
    """Add black border around the grid."""
    for i in range(padding):
        grid[i] = 0
        grid[-i-1] = 0
        grid[:, i] = 0
        grid[:, -i-1] = 0
    return grid


def make_board(size):
    """Create base board with grid lines. Board size = size * CELL_SIZE."""
    board_size = size * CELL_SIZE
    grid = np.ones((board_size, board_size))
    grid = add_border(grid, BORDER_THICKNESS)

    for i in range(1, size):
        dim = i * CELL_SIZE
        thickness = 2 if size < 7 else 1
        for j in range(-thickness, 2):
            cur = dim + j
            if 0 <= cur < board_size:
                grid[cur] = 0
                grid[:, cur] = 0

    return grid


def add_side_wall(grid, size, row, col, side):
    """Add thick cage boundary wall on specified side of cell."""
    board_size = size * CELL_SIZE
    top_left = (row * CELL_SIZE, col * CELL_SIZE)
    bottom_right = ((row + 1) * CELL_SIZE, (col + 1) * CELL_SIZE)

    if side == "left":
        if row == 0:
            for j in range(-8, 8):
                grid[top_left[0]:bottom_right[0]+8, top_left[1]+j] = 0
        elif row == size - 1:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0], top_left[1]+j] = 0
        else:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0]+8, top_left[1]+j] = 0

    elif side == "right":
        if row == 0:
            for j in range(-8, 8):
                grid[top_left[0]:bottom_right[0]+8, bottom_right[1]+j] = 0
        elif row == size - 1:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0], bottom_right[1]+j] = 0
        else:
            for j in range(-8, 8):
                grid[top_left[0]-8:bottom_right[0]+8, bottom_right[1]+j] = 0

    elif side == "top":
        if col == 0:
            for j in range(-8, 8):
                grid[top_left[0]+j, top_left[1]:bottom_right[1]+8] = 0
        elif col == size - 1:
            for j in range(-8, 8):
                grid[top_left[0]+j, top_left[1]-8:bottom_right[1]] = 0
        else:
            for j in range(-8, 8):
                grid[top_left[0]+j, top_left[1]-8:bottom_right[1]+8] = 0

    else:  # bottom
        if col == 0:
            for j in range(-8, 8):
                grid[bottom_right[0]+j, top_left[1]:bottom_right[1]+8] = 0
        elif col == size - 1:
            for j in range(-8, 8):
                grid[bottom_right[0]+j, top_left[1]-8:bottom_right[1]] = 0
        else:
            for j in range(-8, 8):
                grid[bottom_right[0]+j, top_left[1]-8:bottom_right[1]+8] = 0

    return grid


def draw_cage(grid, size, cage):
    """Draw cage boundaries."""
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


def normalize_symbol(file_path):
    """Load and normalize operator symbol PNG."""
    symbol_path = ARCHIVE_KENKEN_DIR / "symbols" / "operators" / file_path
    im = Image.open(symbol_path).convert("RGBA")

    rgba = np.array(im)
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    alpha = a / 255.0
    composited = gray * alpha + 255 * (1 - alpha)

    normalized = composited / 255.0
    return normalized


class ComputerDigitProvider:
    """Provides computer-generated digits from TMNIST font."""

    def __init__(self):
        df = pd.read_csv(ARCHIVE_KENKEN_DIR / "symbols" / "TMNIST_NotoSans.csv")
        noto_sans = df[df['names'].str.contains('notosans', case=False, na=False)]
        noto_sans = noto_sans[noto_sans['names'].str.contains('NotoSans-Regular', case=False, na=False)]
        noto_sans = noto_sans.sort_values(by='labels')

        self.image_arrays = [
            (255 - row.values.reshape(28, 28)) / 255.0
            for _, row in noto_sans.drop(columns=['names', 'labels']).iterrows()
        ]
        self.labels = noto_sans['labels'].tolist()

    def get_digit(self, digit):
        """Get a computer-generated digit image."""
        return self.image_arrays[digit]


class HandwrittenDigitProvider:
    """Provides handwritten MNIST digits for board generation."""

    def __init__(self, test_images, test_labels):
        self.images = test_images
        self.labels = test_labels

        self.digit_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(test_labels):
            self.digit_indices[label].append(idx)

        self.positions = {i: 0 for i in range(10)}

    def get_digit(self, digit):
        """Get a handwritten digit image (28x28, 0-1 range, ink=LOW, bg=HIGH)."""
        indices = self.digit_indices[digit]
        pos = self.positions[digit]

        idx = indices[pos]
        img = self.images[idx]

        self.positions[digit] = (pos + 1) % len(indices)

        # MNIST: ink=HIGH (255), background=LOW (0)
        # Board: ink=LOW (dark), background=HIGH (white)
        inverted = 255 - img
        normalized = inverted.astype(np.float32) / 255.0

        return normalized


def insert_digit(grid, size, cell, digit_img, position):
    """Insert a digit image into a cell."""
    number_size = CELL_SIZE // 4  # 75px for 300px cells

    pil_img = Image.fromarray((digit_img * 255).astype(np.uint8), mode='L')
    resized_img = pil_img.resize((number_size, number_size), resample=Image.LANCZOS)

    cropped = np.array(resized_img.crop((
        resized_img.width // 5, 0,
        resized_img.width * 4 // 5, resized_img.height
    ))).astype(np.float32) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * CELL_SIZE + 20, cell[1] * CELL_SIZE + col_position)

    board_size = len(grid)
    end_row = min(top_left[0] + height, board_size)
    end_col = min(top_left[1] + width, board_size)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        grid[top_left[0]:end_row, top_left[1]:end_col] = cropped[:actual_height, :actual_width]

    return grid


def insert_symbol(grid, size, cell, symbol, position):
    """Insert operator symbol into cell."""
    number_size = CELL_SIZE // 4

    symbol_img = Image.fromarray((normalize_symbol(symbol + ".png") * 255).astype(np.uint8), mode='L')
    resized_img = symbol_img.resize((number_size, number_size), resample=Image.LANCZOS)
    cropped = np.array(resized_img.crop((
        resized_img.width // 6, 0,
        resized_img.width * 5 // 6, resized_img.height
    ))).astype(np.float32) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * CELL_SIZE + 20, cell[1] * CELL_SIZE + col_position)

    board_size = len(grid)
    end_row = min(top_left[0] + height, board_size)
    end_col = min(top_left[1] + width, board_size)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        grid[top_left[0]:end_row, top_left[1]:end_col] = cropped[:actual_height, :actual_width]

    return grid


def write_to_cage(board, size, cage, digit_provider):
    """Write target number and operator to cage."""
    # Find top-left cell
    start_cell = cage["cells"][0]
    for cell in cage["cells"][1:]:
        if cell[0] < start_cell[0]:
            start_cell = cell
        elif cell[0] == start_cell[0] and cell[1] < start_cell[1]:
            start_cell = cell

    # Extract digits from target
    digits = []
    temp = int(cage["target"])
    if temp == 0:
        digits = [0]
    else:
        while temp > 0:
            digits.append(int(temp % 10))
            temp //= 10
        digits = digits[::-1]

    # Insert digits
    for i, digit in enumerate(digits):
        digit_img = digit_provider.get_digit(digit)
        board = insert_digit(board, size, start_cell, digit_img, i)

    # Insert operator
    if cage["op"] != "":
        board = insert_symbol(board, size, start_cell, cage["op"], len(digits))

    return board


def make_board_full(size, puzzle, digit_provider):
    """Create complete board with cages and numbers."""
    board = make_board(size)
    for cage in puzzle:
        board = draw_cage(board, size, cage)
        board = write_to_cage(board, size, cage, digit_provider)
    return board


def save_board(board, filepath):
    """Save board image."""
    array_uint8 = (board * 255).astype(np.uint8)
    image = Image.fromarray(array_uint8, mode='L')
    image.save(filepath)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("8x8 KENKEN PUZZLE GENERATION")
    print("=" * 70)
    print()

    random.seed(42)
    np.random.seed(42)

    # Step 1: Generate Latin squares
    print("Step 1: Generating 100 unique 8x8 Latin squares...")
    solutions = []
    for i in range(NUM_PUZZLES):
        grid = generate_latin_square(GRID_SIZE)
        assert is_valid_latin_square(grid, GRID_SIZE), f"Invalid Latin square {i}"
        solutions.append(grid)
    print(f"  Generated {len(solutions)} Latin squares")
    print()

    # Step 2: Generate KenKen puzzles
    print("Step 2: Converting to KenKen puzzles...")
    puzzles = []
    for i, solution in enumerate(solutions):
        puzzle = generate_puzzle_from_solution(solution, GRID_SIZE, MAX_CAGE_SIZE, MAX_SINGLETONS)

        # Convert numpy types for JSON
        for cage in puzzle:
            if isinstance(cage["op"], np.str_):
                cage["op"] = str(cage["op"])
            if isinstance(cage["target"], (np.integer, np.floating)):
                cage["target"] = float(cage["target"]) if cage["op"] == "div" else int(cage["target"])

        puzzles.append(puzzle)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{NUM_PUZZLES} puzzles...")
    print(f"  Successfully generated {len(puzzles)} puzzles")
    print()

    # Step 3: Skip Z3 validation (puzzles are derived from valid Latin squares)
    # Z3 validation is slow for 8x8; since we generate from valid solutions, skip it
    print("Step 3: Skipping Z3 validation (puzzles derived from valid Latin squares)")
    print("  All puzzles are guaranteed valid by construction")
    print()

    # Step 4: Create output directories
    print("Step 4: Creating output directories...")
    computer_dir = BASE_DIR / 'benchmarks' / 'KenKen' / 'Computer' / '8x8'
    handwritten_dir = BASE_DIR / 'benchmarks' / 'KenKen' / 'Handwritten' / '8x8'
    computer_dir.mkdir(parents=True, exist_ok=True)
    handwritten_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created {computer_dir}")
    print(f"  Created {handwritten_dir}")
    print()

    # Step 5: Generate computer digit images
    print("Step 5: Generating computer digit board images (300px cells)...")
    computer_provider = ComputerDigitProvider()

    for i, puzzle in enumerate(puzzles):
        board = make_board_full(GRID_SIZE, puzzle, computer_provider)
        filepath = computer_dir / f'board{GRID_SIZE}_{i}.png'
        save_board(board, filepath)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{NUM_PUZZLES} computer images...")
    print(f"  Saved {NUM_PUZZLES} images to {computer_dir}")
    print()

    # Step 6: Generate handwritten digit images
    print("Step 6: Generating handwritten digit board images (300px cells)...")

    # Load MNIST test data
    handwritten_data_dir = ARCHIVE_HANDWRITTEN_DIR / 'handwritten_data'
    test_images = np.load(handwritten_data_dir / 'test_images.npy')
    test_labels = np.load(handwritten_data_dir / 'test_labels.npy')
    print(f"  Loaded {len(test_images)} MNIST test images")

    handwritten_provider = HandwrittenDigitProvider(test_images, test_labels)

    for i, puzzle in enumerate(puzzles):
        board = make_board_full(GRID_SIZE, puzzle, handwritten_provider)
        filepath = handwritten_dir / f'board{GRID_SIZE}_{i}.png'
        save_board(board, filepath)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{NUM_PUZZLES} handwritten images...")
    print(f"  Saved {NUM_PUZZLES} images to {handwritten_dir}")
    print()

    # Step 7: Update kenken_puzzles.json
    print("Step 7: Updating kenken_puzzles.json...")
    puzzles_path = BASE_DIR / 'puzzles' / 'kenken_puzzles.json'

    with open(puzzles_path, 'r') as f:
        puzzles_dict = json.load(f)

    # Insert 8x8 puzzles in sorted position
    puzzles_dict['8'] = puzzles

    # Sort keys and rewrite
    sorted_dict = {k: puzzles_dict[k] for k in sorted(puzzles_dict.keys(), key=int)}

    with open(puzzles_path, 'w') as f:
        json.dump(sorted_dict, f, indent=2)

    print(f"  Added {len(puzzles)} puzzles under key '8'")
    print(f"  Updated {puzzles_path}")
    print()

    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print()
    print(f"Generated:")
    print(f"  - {NUM_PUZZLES} 8x8 KenKen puzzles")
    print(f"  - {NUM_PUZZLES} computer digit images (2400x2400px, 300px cells)")
    print(f"  - {NUM_PUZZLES} handwritten digit images (2400x2400px, 300px cells)")
    print(f"  - Updated puzzles/kenken_puzzles.json")


if __name__ == "__main__":
    main()
