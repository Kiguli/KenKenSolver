# -*- coding: utf-8 -*-
"""
Generate KenKen board images using handwritten TEST samples.

CRITICAL: Uses ONLY the test split of MNIST, which the CNN never saw during training.
This tests true generalization to unseen handwritten styles.

BOARD SIZE: Fixed 900x900 to match original KenKen exactly.
This enables fair comparison - only the digit source differs.

Image convention (matching original KenKen):
- Board background: WHITE (255, value 1.0)
- Characters: DARK (0, black ink, value 0.0)
- MNIST stores ink as HIGH values, so we INVERT when placing on board
"""

import json
import os
import numpy as np
from PIL import Image

# Constants - MATCH ORIGINAL KENKEN EXACTLY
BOARD_SIZE = 900  # Fixed 900x900 like original


def get_cell_size(size):
    """Get cell size for a given puzzle size (same as original)."""
    return BOARD_SIZE // size  # 900//4=225, 900//5=180, 900//6=150, etc.


def load_test_samples():
    """
    Load test set handwritten samples (never seen during CNN training).

    Returns:
        dict: {class_label: [list of 28x28 numpy arrays]}
    """
    data_dir = "./handwritten_data"

    if not os.path.exists(f"{data_dir}/test_images.npy"):
        raise FileNotFoundError(
            f"Test data not found in {data_dir}/. "
            "Run download_datasets.py first."
        )

    test_images = np.load(f"{data_dir}/test_images.npy")
    test_labels = np.load(f"{data_dir}/test_labels.npy")

    print(f"Loaded {len(test_images)} test samples")

    # Organize by class
    samples_by_class = {}
    for img, lbl in zip(test_images, test_labels):
        lbl = int(lbl)
        if lbl not in samples_by_class:
            samples_by_class[lbl] = []
        samples_by_class[lbl].append(img)

    print(f"Classes available: {sorted(samples_by_class.keys())}")
    for cls in sorted(samples_by_class.keys()):
        print(f"  Class {cls} (digit {cls}): {len(samples_by_class[cls])} samples")

    return samples_by_class


def load_operator_images():
    """
    Load operator images from the symbols folder.

    Returns grayscale normalized images matching the original KenKen processing.
    """
    operators = {}
    op_dir = "./symbols/operators"

    op_files = {
        "add": "add.png",
        "sub": "sub.png",
        "mul": "mul.png",
        "div": "div.png"
    }

    for op_name, filename in op_files.items():
        filepath = os.path.join(op_dir, filename)
        if os.path.exists(filepath):
            # Load and process exactly like original KenKen (normalize_symbol)
            im = Image.open(filepath).convert("RGBA")
            rgba = np.array(im)
            r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
            gray = 0.299 * r + 0.587 * g + 0.114 * b
            alpha = a / 255.0
            composited = gray * alpha + 255 * (1 - alpha)
            normalized = composited / 255.0  # 0.0 = black, 1.0 = white
            operators[op_name] = normalized
        else:
            print(f"Warning: Operator image not found: {filepath}")

    return operators


def add_border(grid, padding):
    """Add outer border (same as original)."""
    for i in range(padding):
        grid[i] = 0
        grid[-i-1] = 0
        grid[:, i] = 0
        grid[:, -i-1] = 0
    return grid


def make_board(size):
    """
    Create a blank KenKen board with grid lines.

    EXACTLY matches original KenKen BoardImageGeneration.ipynb
    """
    grid = np.ones((BOARD_SIZE, BOARD_SIZE))
    grid = add_border(grid, 16)
    square_size = BOARD_SIZE // size

    for i in range(1, size):
        dim = i * square_size
        thickness = 2 if size < 7 else 1
        for j in range(-thickness, 2):
            cur = dim + j
            if 0 <= cur < BOARD_SIZE:
                grid[cur] = 0
                grid[:, cur] = 0
            cur_neg = BOARD_SIZE - cur - 1
            if 0 <= cur_neg < BOARD_SIZE:
                grid[cur_neg] = 0
                grid[:, cur_neg] = 0

    return grid


def add_side_wall(grid, size, row, col, side):
    """
    Add a thick wall on one side of a cell.
    EXACTLY matches original KenKen BoardImageGeneration.ipynb
    """
    square_size = BOARD_SIZE // size
    top_left = (row * square_size, col * square_size)
    bottom_right = ((row + 1) * square_size, (col + 1) * square_size)

    if side == "left":
        if row == 0:
            for j in range(-8, 8):
                if 0 <= top_left[1] + j < BOARD_SIZE:
                    grid[top_left[0]:bottom_right[0]+8, top_left[1]+j] = 0
        elif row == size - 1:
            for j in range(-8, 8):
                if 0 <= top_left[1] + j < BOARD_SIZE:
                    grid[top_left[0]-8:bottom_right[0], top_left[1]+j] = 0
        else:
            for j in range(-8, 8):
                if 0 <= top_left[1] + j < BOARD_SIZE:
                    grid[top_left[0]-8:bottom_right[0]+8, top_left[1]+j] = 0

    elif side == "right":
        if row == 0:
            for j in range(-8, 8):
                if 0 <= bottom_right[1] + j < BOARD_SIZE:
                    grid[top_left[0]:bottom_right[0]+8, bottom_right[1]+j] = 0
        elif row == size - 1:
            for j in range(-8, 8):
                if 0 <= bottom_right[1] + j < BOARD_SIZE:
                    grid[top_left[0]-8:bottom_right[0], bottom_right[1]+j] = 0
        else:
            for j in range(-8, 8):
                if 0 <= bottom_right[1] + j < BOARD_SIZE:
                    grid[top_left[0]-8:bottom_right[0]+8, bottom_right[1]+j] = 0

    elif side == "top":
        if col == 0:
            for j in range(-8, 8):
                if 0 <= top_left[0] + j < BOARD_SIZE:
                    grid[top_left[0]+j, top_left[1]:bottom_right[1]+8] = 0
        elif col == size - 1:
            for j in range(-8, 8):
                if 0 <= top_left[0] + j < BOARD_SIZE:
                    grid[top_left[0]+j, top_left[1]-8:bottom_right[1]] = 0
        else:
            for j in range(-8, 8):
                if 0 <= top_left[0] + j < BOARD_SIZE:
                    grid[top_left[0]+j, top_left[1]-8:bottom_right[1]+8] = 0

    else:  # bottom
        if col == 0:
            for j in range(-8, 8):
                if 0 <= bottom_right[0] + j < BOARD_SIZE:
                    grid[bottom_right[0]+j, top_left[1]:bottom_right[1]+8] = 0
        elif col == size - 1:
            for j in range(-8, 8):
                if 0 <= bottom_right[0] + j < BOARD_SIZE:
                    grid[bottom_right[0]+j, top_left[1]-8:bottom_right[1]] = 0
        else:
            for j in range(-8, 8):
                if 0 <= bottom_right[0] + j < BOARD_SIZE:
                    grid[bottom_right[0]+j, top_left[1]-8:bottom_right[1]+8] = 0

    return grid


def draw_cage(grid, size, cage):
    """
    Draw thick cage borders on the board.
    EXACTLY matches original KenKen BoardImageGeneration.ipynb
    """
    for cell in cage['cells']:
        row, col = cell

        if [row, col + 1] not in cage['cells'] and col + 1 < size:
            grid = add_side_wall(grid, size, row, col, "right")
        if [row, col - 1] not in cage['cells'] and col - 1 >= 0:
            grid = add_side_wall(grid, size, row, col, "left")
        if [row + 1, col] not in cage['cells'] and row + 1 < size:
            grid = add_side_wall(grid, size, row, col, "bottom")
        if [row - 1, col] not in cage['cells'] and row - 1 >= 0:
            grid = add_side_wall(grid, size, row, col, "top")

    return grid


def render_handwritten_digit(samples_dict, digit, target_size):
    """
    Randomly select and prepare a handwritten sample for the given digit.

    The sample is INVERTED (MNIST ink=HIGH -> board ink=LOW/dark)
    to match the board convention (dark on white).

    Returns:
        2D numpy array (target_size x target_size) with values in [0,1]
        0.0 = dark ink, 1.0 = white background
    """
    if digit not in samples_dict or len(samples_dict[digit]) == 0:
        print(f"Warning: No samples for digit {digit}")
        return None

    # Random selection for variety
    idx = np.random.randint(len(samples_dict[digit]))
    sample = samples_dict[digit][idx]

    # Convert from [0,1] float to proper format
    if sample.max() <= 1.0:
        img_array = sample
    else:
        img_array = sample / 255.0

    # INVERT: MNIST has ink=HIGH (near 1.0), we need ink=LOW (near 0.0)
    # After inversion: 0.0 = dark ink, 1.0 = white background
    img_array = 1.0 - img_array

    # Convert to PIL for resizing
    pil_img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    resized = pil_img.resize((target_size, target_size), Image.Resampling.NEAREST)

    # Return as float array in [0,1]
    return np.array(resized).astype(np.float32) / 255.0


def insert_number_in_cage(grid, size, cell, samples_dict, digit, position):
    """
    Insert a handwritten digit in a cage.

    MATCHES original KenKen insert_number_in_cage layout exactly.
    """
    square_size = BOARD_SIZE // size
    number_size = square_size // 4

    # Render handwritten digit
    digit_img = render_handwritten_digit(samples_dict, digit, number_size)
    if digit_img is None:
        return grid

    # Crop to 3/5 width (same as original: 1/5 to 4/5)
    pil_img = Image.fromarray((digit_img * 255).astype(np.uint8))
    cropped = np.array(pil_img.crop((pil_img.width // 5, 0, pil_img.width * 4 // 5, pil_img.height)))
    cropped = cropped.astype(np.float32) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    # Position: 20px from top and left of cell (same as original)
    top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)

    # Blend using minimum (dark ink wins)
    y1, x1 = top_left
    y2, x2 = y1 + height, x1 + width

    if y2 <= BOARD_SIZE and x2 <= BOARD_SIZE:
        grid[y1:y2, x1:x2] = np.minimum(grid[y1:y2, x1:x2], cropped)

    return grid


def insert_symbol_in_cage(grid, size, cell, operators_dict, op_name, position):
    """
    Insert an operator symbol in a cage.

    MATCHES original KenKen insert_symbol_in_cage layout exactly.
    """
    if op_name not in operators_dict:
        return grid

    square_size = BOARD_SIZE // size
    number_size = square_size // 4

    # Get operator image and resize
    op_img = operators_dict[op_name]
    pil_img = Image.fromarray((op_img * 255).astype(np.uint8))
    resized = pil_img.resize((number_size, number_size), Image.Resampling.NEAREST)

    # Crop to 4/6 width (same as original: 1/6 to 5/6)
    cropped = np.array(resized.crop((resized.width // 6, 0, resized.width * 5 // 6, resized.height)))
    cropped = cropped.astype(np.float32) / 255.0

    height, width = cropped.shape
    col_position = 20 + position * width

    # Position: 20px from top and left of cell (same as original)
    top_left = (cell[0] * square_size + 20, cell[1] * square_size + col_position)

    # Blend using minimum (dark ink wins)
    y1, x1 = top_left
    y2, x2 = y1 + height, x1 + width

    if y2 <= BOARD_SIZE and x2 <= BOARD_SIZE:
        grid[y1:y2, x1:x2] = np.minimum(grid[y1:y2, x1:x2], cropped)

    return grid


def write_to_cage(board, size, cage, samples_dict, operators_dict):
    """
    Write the target and operator for a cage in its top-left cell.
    MATCHES original KenKen write_to_cage logic exactly.
    """
    # Find top-left cell of the cage
    start_cell = cage["cells"][0]
    if len(cage["cells"]) > 1:
        for cell in cage["cells"][1:]:
            if cell[0] < start_cell[0]:
                start_cell = cell
            elif cell[0] == start_cell[0] and cell[1] < start_cell[1]:
                start_cell = cell

    # Convert target to digits
    digits = []
    temp = int(cage["target"])
    if temp == 0:
        digits = [0]
    else:
        while temp > 0:
            digits.append(temp % 10)
            temp //= 10
        digits = digits[::-1]

    # Insert each digit
    for i, digit in enumerate(digits):
        board = insert_number_in_cage(board, size, start_cell, samples_dict, digit, i)

    # Insert operator if present
    if cage["op"] and cage["op"] != "":
        board = insert_symbol_in_cage(board, size, start_cell, operators_dict, cage["op"], len(digits))

    return board


def make_board_full(size, puzzle, samples_dict, operators_dict):
    """Generate a complete KenKen board image."""
    board = make_board(size)
    for cage in puzzle:
        board = draw_cage(board, size, cage)
        board = write_to_cage(board, size, cage, samples_dict, operators_dict)
    return board


def main():
    """Main function to generate all board images."""
    print("=" * 60)
    print("Handwritten KenKen Board Image Generation")
    print("=" * 60)
    print(f"\nBoard size: {BOARD_SIZE}x{BOARD_SIZE} (matching original KenKen)")

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load puzzles
    print("\n1. Loading puzzles...")
    puzzles_path = "./puzzles/puzzles_dict.json"

    if not os.path.exists(puzzles_path):
        raise FileNotFoundError(
            f"Puzzles not found at {puzzles_path}. "
            "Copy from KenKen/puzzles/ first."
        )

    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    # Load test samples (unseen by CNN)
    print("\n2. Loading handwritten test samples...")
    samples_dict = load_test_samples()

    # Load operator images
    print("\n3. Loading operator images...")
    operators_dict = load_operator_images()
    print(f"   Operators loaded: {list(operators_dict.keys())}")

    # Create output directory
    output_dir = "./board_images"
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for each size
    total_generated = 0

    # Sizes to generate: 3, 4, 5, 6, 7, 9 (matching the KenKen puzzles)
    sizes_to_generate = ['3', '4', '5', '6', '7', '9']

    for size_str in sizes_to_generate:
        size = int(size_str)
        puzzles = puzzles_ds.get(size_str, [])

        if not puzzles:
            print(f"\nNo puzzles found for size {size}x{size}")
            continue

        cell_size = get_cell_size(size)
        print(f"\n4. Generating {len(puzzles)} board images for {size}x{size} puzzles...")
        print(f"   Board: {BOARD_SIZE}x{BOARD_SIZE}, Cell: {cell_size}x{cell_size}")
        print(f"   (Using handwritten TEST samples - never seen during CNN training)")
        print("-" * 60)

        for i, puzzle in enumerate(puzzles):
            board = make_board_full(size, puzzle, samples_dict, operators_dict)

            # Save as PNG
            array_uint8 = (board * 255).astype(np.uint8)
            img = Image.fromarray(array_uint8, mode='L')
            filename = f"{output_dir}/board{size}_{i}.png"
            img.save(filename)

            if (i + 1) % 25 == 0:
                print(f"   Generated {i + 1}/{len(puzzles)} {size}x{size} images")

            total_generated += 1

        print(f"   Completed {len(puzzles)} {size}x{size} images")

    print("-" * 60)
    print(f"\nTotal images generated: {total_generated}")
    print(f"Output directory: {output_dir}/")

    print("\n" + "=" * 60)
    print("Board image generation complete!")
    print("=" * 60)
    print("\nNext step: Run evaluate.py to test the neuro-symbolic pipeline")


if __name__ == '__main__':
    main()
