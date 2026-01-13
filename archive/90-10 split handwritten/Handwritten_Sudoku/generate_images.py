"""
Generate Sudoku board images using handwritten TEST samples.

Critical: Uses ONLY the test split of MNIST, which the CNN
never saw during training. This tests true generalization.

Each board randomly samples different handwritten instances for each cell,
ensuring variety and realistic handwritten appearance.

Supports two puzzle sizes:
- 4×4: 2×2 boxes, 225px cells
- 9×9: 3×3 boxes, 100px cells

Image convention:
- Board background: WHITE (255)
- Characters: DARK (0, black ink)
- MNIST stores ink as HIGH values, so we INVERT when placing on board
"""

import json
import os
import numpy as np
from PIL import Image

# Constants
BOARD_PIXELS = 900  # Board size (matching Sudoku folder)


def get_cell_size(size):
    """Get cell size in pixels for a given puzzle size."""
    return BOARD_PIXELS // size


def get_box_size(size):
    """Get box size for a given puzzle size."""
    return 2 if size == 4 else 3


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
        print(f"  Class {cls} ({cls}): {len(samples_by_class[cls])} samples")

    return samples_by_class


def make_sudoku_board(size):
    """
    Create a blank Sudoku board with grid lines.

    Args:
        size: Grid size (4 or 9)

    Returns:
        numpy array of shape (BOARD_PIXELS, BOARD_PIXELS) with values 0-255
        White background (255), black lines (0)
    """
    cell_size = get_cell_size(size)
    box_size = get_box_size(size)

    # Start with white background
    grid = np.ones((BOARD_PIXELS, BOARD_PIXELS), dtype=np.uint8) * 255

    # Draw outer border (bold, 8px)
    border_thickness = 8
    for i in range(border_thickness):
        grid[i, :] = 0
        grid[-i-1, :] = 0
        grid[:, i] = 0
        grid[:, -i-1] = 0

    # Draw internal grid lines
    for i in range(1, size):
        pos = i * cell_size

        # Bold lines for box boundaries
        if i % box_size == 0:
            thickness = 6
        else:
            thickness = 2

        for j in range(-thickness, thickness):
            if 0 <= pos + j < BOARD_PIXELS:
                grid[pos + j, :] = 0  # Horizontal line
                grid[:, pos + j] = 0  # Vertical line

    return grid


def render_handwritten_digit(samples_dict, digit, target_size):
    """
    Randomly select and prepare a handwritten sample for the given digit.

    The sample is inverted (MNIST ink=high → board ink=low/black) and resized.

    Args:
        samples_dict: Dictionary of {class: [samples]}
        digit: Digit value (1-9)
        target_size: Output size in pixels

    Returns:
        PIL Image in mode 'L' (grayscale) with white background, dark digit
        Or None if digit not found
    """
    if digit not in samples_dict or len(samples_dict[digit]) == 0:
        print(f"Warning: No samples for digit {digit}")
        return None

    # Random selection for variety
    idx = np.random.randint(len(samples_dict[digit]))
    sample = samples_dict[digit][idx]

    # Convert from [0,1] float to [0,255] uint8
    if sample.max() <= 1.0:
        img_array = (sample * 255).astype(np.uint8)
    else:
        img_array = sample.astype(np.uint8)

    # INVERT: MNIST has ink=HIGH (white on black), we need ink=LOW (black on white)
    img_array = 255 - img_array

    # Convert to PIL and resize
    pil_img = Image.fromarray(img_array, mode='L')
    resized = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return resized


def insert_handwritten_digit(board, row, col, digit_img, size):
    """
    Paste a handwritten digit image into a cell on the board.

    Uses minimum blending so dark pixels (ink) override white background.

    Args:
        board: numpy array of the board (modified in place)
        row: Row index
        col: Column index
        digit_img: PIL Image of the digit
        size: Grid size (4 or 9)

    Returns:
        Modified board array
    """
    if digit_img is None:
        return board

    cell_size = get_cell_size(size)

    # Get digit as numpy array
    digit_array = np.array(digit_img)
    digit_h, digit_w = digit_array.shape

    # Calculate centered position within cell
    cell_x = col * cell_size
    cell_y = row * cell_size

    offset_x = (cell_size - digit_w) // 2
    offset_y = (cell_size - digit_h) // 2

    # Target region
    y1 = cell_y + offset_y
    y2 = y1 + digit_h
    x1 = cell_x + offset_x
    x2 = x1 + digit_w

    # Blend using minimum (dark ink wins over white background)
    board[y1:y2, x1:x2] = np.minimum(board[y1:y2, x1:x2], digit_array)

    return board


def generate_board_image(size, puzzle, samples_dict):
    """
    Generate a single board image with handwritten digits.

    Args:
        size: Grid size (4 or 9)
        puzzle: size×size list of puzzle values (0 = empty)
        samples_dict: Dictionary of handwritten samples by class

    Returns:
        numpy array of shape (BOARD_PIXELS, BOARD_PIXELS)
    """
    cell_size = get_cell_size(size)
    # Target digit size is 2/3 of cell size
    digit_size = cell_size * 2 // 3

    # Create blank board with grid lines
    board = make_sudoku_board(size)

    # Insert handwritten digits for each clue
    for row in range(size):
        for col in range(size):
            value = puzzle[row][col]
            if value != 0:
                digit_img = render_handwritten_digit(samples_dict, value, digit_size)
                if digit_img is not None:
                    board = insert_handwritten_digit(board, row, col, digit_img, size)

    return board


def main():
    """Main function to generate all board images."""
    print("=" * 60)
    print("Handwritten Sudoku Board Image Generation")
    print("=" * 60)

    # Load puzzles
    print("\n1. Loading puzzles...")
    puzzles_path = "./puzzles/puzzles_dict.json"

    if not os.path.exists(puzzles_path):
        raise FileNotFoundError(
            f"Puzzles not found at {puzzles_path}. "
            "Copy from Sudoku/puzzles/ first."
        )

    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    # Load test samples (unseen by CNN)
    print("\n2. Loading handwritten test samples...")
    samples_dict = load_test_samples()

    # Create output directory
    output_dir = "./board_images"
    os.makedirs(output_dir, exist_ok=True)

    # Generate images for each size
    total_generated = 0

    for size_str in ['4', '9']:
        size = int(size_str)
        puzzles = puzzles_ds.get(size_str, [])

        if not puzzles:
            print(f"\nNo puzzles found for size {size}×{size}")
            continue

        print(f"\n3. Generating {len(puzzles)} board images for {size}×{size} puzzles...")
        print(f"   (Using handwritten TEST samples - never seen during CNN training)")
        print("-" * 60)

        for i, puzzle_data in enumerate(puzzles):
            puzzle = puzzle_data['puzzle']
            board = generate_board_image(size, puzzle, samples_dict)

            # Save as PNG
            img = Image.fromarray(board, mode='L')
            filename = f"{output_dir}/board{size}_{i}.png"
            img.save(filename)

            if (i + 1) % 25 == 0:
                print(f"   Generated {i + 1}/{len(puzzles)} {size}×{size} images")

            total_generated += 1

        print(f"   Completed {len(puzzles)} {size}×{size} images")

    print("-" * 60)
    print(f"\nTotal images generated: {total_generated}")
    print(f"Output directory: {output_dir}/")

    print("\n" + "=" * 60)
    print("Board image generation complete!")
    print("=" * 60)
    print("\nNext step: Run evaluate.py to test the neuro-symbolic pipeline")


if __name__ == '__main__':
    main()
