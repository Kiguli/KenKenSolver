# -*- coding: utf-8 -*-
"""
Generate KenKen board images with 300px cells for all sizes.

Board sizes:
- 3x3: 900x900
- 4x4: 1200x1200
- 5x5: 1500x1500
- 6x6: 1800x1800
- 7x7: 2100x2100
- 9x9: 2700x2700
"""

import os
import json
import numpy as np
from PIL import Image
import pandas as pd

# Constants
CELL_SIZE = 300  # Fixed cell size for all puzzle sizes
BORDER_THICKNESS = 16


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
    im = Image.open("./symbols/operators/" + file_path).convert("RGBA")

    rgba = np.array(im)
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    alpha = a / 255.0
    composited = gray * alpha + 255 * (1 - alpha)

    normalized = composited / 255.0
    return normalized


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


def insert_number_in_cage(grid, size, cell, digit_img, position):
    """Insert a digit image into a cell."""
    number_size = CELL_SIZE // 4  # 75px for 300px cells

    pil_img = Image.fromarray((digit_img * 255).astype(np.uint8), mode='L')
    resized_img = pil_img.resize((number_size, number_size), resample=Image.NEAREST)
    cropped = np.array(resized_img.crop((
        resized_img.width // 5, 0,
        resized_img.width * 4 // 5, resized_img.height
    ))) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * CELL_SIZE + 20, cell[1] * CELL_SIZE + col_position)

    # Bounds check
    board_size = len(grid)
    end_row = min(top_left[0] + height, board_size)
    end_col = min(top_left[1] + width, board_size)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        grid[top_left[0]:end_row, top_left[1]:end_col] = cropped[:actual_height, :actual_width]

    return grid


def insert_symbol_in_cage(grid, size, cell, symbol, position):
    """Insert operator symbol into cell."""
    number_size = CELL_SIZE // 4

    symbol_img = Image.fromarray((normalize_symbol(symbol + ".png") * 255).astype(np.uint8), mode='L')
    resized_img = symbol_img.resize((number_size, number_size), resample=Image.NEAREST)
    cropped = np.array(resized_img.crop((
        resized_img.width // 6, 0,
        resized_img.width * 5 // 6, resized_img.height
    ))) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * CELL_SIZE + 20, cell[1] * CELL_SIZE + col_position)

    # Bounds check
    board_size = len(grid)
    end_row = min(top_left[0] + height, board_size)
    end_col = min(top_left[1] + width, board_size)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        grid[top_left[0]:end_row, top_left[1]:end_col] = cropped[:actual_height, :actual_width]

    return grid


def write_to_cage(board, size, cage, digit_images):
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
    temp = cage["target"]
    if temp == 0:
        digits = [0]
    else:
        while temp > 0:
            digits.append(int(temp % 10))
            temp //= 10
        digits = digits[::-1]

    # Insert digits
    for i, digit in enumerate(digits):
        board = insert_number_in_cage(board, size, start_cell, digit_images[digit], i)

    # Insert operator
    if cage["op"] != "":
        board = insert_symbol_in_cage(board, size, start_cell, cage["op"], len(digits))

    return board


def make_board_full(size, puzzle, digit_images):
    """Create complete board with cages and numbers."""
    board = make_board(size)
    for cage in puzzle:
        board = draw_cage(board, size, cage)
        board = write_to_cage(board, size, cage, digit_images)
    return board


def make_and_save(size, puzzle, idx, digit_images, output_dir):
    """Generate and save a board image."""
    board = make_board_full(size, puzzle, digit_images)
    array_uint8 = (board * 255).astype(np.uint8)
    image = Image.fromarray(array_uint8, mode='L')
    filepath = os.path.join(output_dir, f'board{size}_{idx}.png')
    image.save(filepath)


def main():
    # Load digit images
    print("Loading digit images from TMNIST...")
    digit_images = load_digit_images()
    print(f"Loaded {len(digit_images)} digit images")

    # Load puzzles
    print("\nLoading puzzles...")
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        puzzles_dict = json.load(f)

    # Create output directory
    output_dir = './board_images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for each size
    sizes = [3, 4, 5, 6, 7, 9]
    total_generated = 0

    for size in sizes:
        if str(size) not in puzzles_dict:
            print(f"No puzzles for size {size}")
            continue

        puzzles = puzzles_dict[str(size)]
        num_puzzles = min(100, len(puzzles))

        board_size = size * CELL_SIZE
        print(f"\nGenerating {num_puzzles} {size}x{size} puzzles ({board_size}x{board_size}px)...")

        for i in range(num_puzzles):
            puzzle = puzzles[i]
            # Handle both formats
            if isinstance(puzzle, dict) and 'cages' in puzzle:
                cages = puzzle['cages']
            else:
                cages = puzzle

            make_and_save(size, cages, i, digit_images, output_dir)
            total_generated += 1

        print(f"  Generated {num_puzzles} images")

    print(f"\nTotal: {total_generated} board images saved to {output_dir}/")


if __name__ == '__main__':
    main()
