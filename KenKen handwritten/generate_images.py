# -*- coding: utf-8 -*-
"""
Generate KenKen board images with handwritten MNIST digits.
Uses TEST set digits (unseen by CNN) for fair evaluation.
Operators remain computer-generated (same as original KenKen).
"""

import os
import json
import numpy as np
from PIL import Image
import random

# Constants
BOARD_SIZE = 900
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
    """Create base board with grid lines."""
    grid = np.ones((BOARD_SIZE, BOARD_SIZE))
    grid = add_border(grid, BORDER_THICKNESS)
    square_size = BOARD_SIZE // size

    for i in range(1, size):
        dim = i * square_size
        thickness = 2 if size < 7 else 1
        for j in range(-thickness, 2):
            cur = dim + j
            if 0 <= cur < BOARD_SIZE:
                grid[cur] = 0
                grid[:, cur] = 0
            if 0 <= BOARD_SIZE - cur < BOARD_SIZE:
                grid[BOARD_SIZE - cur - 1] = 0
                grid[:, BOARD_SIZE - cur - 1] = 0

    return grid


def add_side_wall(grid, size, row, col, side):
    """Add thick cage boundary wall on specified side of cell."""
    square_size = BOARD_SIZE // size
    top_left = (row * square_size, col * square_size)
    bottom_right = ((row + 1) * square_size, (col + 1) * square_size)

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


class HandwrittenDigitProvider:
    """Provides handwritten MNIST digits for board generation."""

    def __init__(self, test_images, test_labels):
        self.images = test_images
        self.labels = test_labels

        # Group indices by digit class
        self.digit_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(test_labels):
            self.digit_indices[label].append(idx)

        # Current position for each digit class (round-robin)
        self.positions = {i: 0 for i in range(10)}

        print("HandwrittenDigitProvider initialized:")
        for digit in range(10):
            print(f"  Digit {digit}: {len(self.digit_indices[digit])} samples")

    def get_digit(self, digit):
        """Get a handwritten digit image (28x28, 0-1 range, ink=LOW, bg=HIGH)."""
        indices = self.digit_indices[digit]
        pos = self.positions[digit]

        # Get image at current position
        idx = indices[pos]
        img = self.images[idx]

        # Move to next position (wrap around)
        self.positions[digit] = (pos + 1) % len(indices)

        # MNIST: ink=HIGH (255), background=LOW (0)
        # Board convention: ink=LOW (dark), background=HIGH (white)
        # So we invert: 255 - img
        inverted = 255 - img
        normalized = inverted.astype(np.float32) / 255.0

        return normalized


def insert_handwritten_digit(grid, size, cell, digit_img, position):
    """Insert a handwritten digit image into a cell."""
    square_size = BOARD_SIZE // size
    number_size = square_size // 4

    # Resize digit to appropriate size
    pil_img = Image.fromarray((digit_img * 255).astype(np.uint8), mode='L')
    resized_img = pil_img.resize((number_size, number_size), resample=Image.LANCZOS)

    # Crop to remove whitespace on sides (similar to original)
    cropped = np.array(resized_img.crop((
        resized_img.width // 5, 0,
        resized_img.width * 4 // 5, resized_img.height
    ))).astype(np.float32) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)

    # Bounds check
    end_row = min(top_left[0] + height, BOARD_SIZE)
    end_col = min(top_left[1] + width, BOARD_SIZE)
    actual_height = end_row - top_left[0]
    actual_width = end_col - top_left[1]

    if actual_height > 0 and actual_width > 0:
        grid[top_left[0]:end_row, top_left[1]:end_col] = cropped[:actual_height, :actual_width]

    return grid


def insert_symbol_in_cage(grid, size, cell, symbol, position):
    """Insert operator symbol (computer-generated) into cell."""
    square_size = BOARD_SIZE // size
    number_size = square_size // 4

    symbol_img = Image.fromarray((normalize_symbol(symbol + ".png") * 255).astype(np.uint8), mode='L')
    resized_img = symbol_img.resize((number_size, number_size), resample=Image.LANCZOS)
    cropped = np.array(resized_img.crop((
        resized_img.width // 6, 0,
        resized_img.width * 5 // 6, resized_img.height
    ))).astype(np.float32) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)

    # Bounds check
    end_row = min(top_left[0] + height, BOARD_SIZE)
    end_col = min(top_left[1] + width, BOARD_SIZE)
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
    temp = cage["target"]
    if temp == 0:
        digits = [0]
    else:
        while temp > 0:
            digits.append(int(temp % 10))
            temp //= 10
        digits = digits[::-1]

    # Insert handwritten digits
    for i, digit in enumerate(digits):
        digit_img = digit_provider.get_digit(digit)
        board = insert_handwritten_digit(board, size, start_cell, digit_img, i)

    # Insert operator (computer-generated)
    if cage["op"] != "":
        board = insert_symbol_in_cage(board, size, start_cell, cage["op"], len(digits))

    return board


def make_board_full(size, puzzle, digit_provider):
    """Create complete board with cages and numbers."""
    board = make_board(size)
    for cage in puzzle:
        board = draw_cage(board, size, cage)
        board = write_to_cage(board, size, cage, digit_provider)
    return board


def make_and_save(size, puzzle, idx, digit_provider, output_dir):
    """Generate and save a board image."""
    board = make_board_full(size, puzzle, digit_provider)
    array_uint8 = (board * 255).astype(np.uint8)
    image = Image.fromarray(array_uint8, mode='L')
    filepath = os.path.join(output_dir, f'board{size}_{idx}.png')
    image.save(filepath)


def main():
    random.seed(42)
    np.random.seed(42)

    # Load test MNIST data
    print("Loading test MNIST data...")
    test_images = np.load('./handwritten_data/test_images.npy')
    test_labels = np.load('./handwritten_data/test_labels.npy')
    print(f"Loaded {len(test_images)} test images")

    # Create digit provider
    digit_provider = HandwrittenDigitProvider(test_images, test_labels)

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

        print(f"\nGenerating {num_puzzles} {size}x{size} puzzles...")

        for i in range(num_puzzles):
            puzzle = puzzles[i]
            # Handle both formats
            if isinstance(puzzle, dict) and 'cages' in puzzle:
                cages = puzzle['cages']
            else:
                cages = puzzle

            make_and_save(size, cages, i, digit_provider, output_dir)
            total_generated += 1

        print(f"  Generated {num_puzzles} images")

    print(f"\nTotal: {total_generated} board images saved to {output_dir}/")


if __name__ == '__main__':
    main()
