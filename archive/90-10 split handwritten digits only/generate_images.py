"""
Generate HexaSudoku board images using MNIST digits only.

Key difference from letter-based version:
- Values 1-9: Single digit rendered in center
- Values 10-16: Two digits rendered side-by-side (e.g., "16" = "1" + "6")

Uses TEST samples (never seen during CNN training).

Image convention:
- Board background: WHITE (255)
- Characters: DARK (0, black ink)
- MNIST stores ink as HIGH values, so we INVERT when placing on board
"""

import json
import os
import numpy as np
from PIL import Image

# Constants (matching HexaSudoku)
SIZE = 16
BOX_SIZE = 4
BOARD_PIXELS = 1600
CELL_SIZE = BOARD_PIXELS // SIZE  # 100px per cell

# Single digit size (for values 1-9)
SINGLE_DIGIT_SIZE = 70

# Two-digit layout (for values 10-16)
TWO_DIGIT_SIZE = 40  # Each digit is 40x40
TWO_DIGIT_GAP = 10   # Gap between digits
TWO_DIGIT_LEFT_X = 10   # Left digit x position
TWO_DIGIT_RIGHT_X = 50  # Right digit x position
TWO_DIGIT_Y = 30        # Y position for both digits

RANDOM_SEED = 9010
np.random.seed(RANDOM_SEED)


def load_test_samples():
    """
    Load test set handwritten samples (never seen during CNN training).

    Returns:
        dict: {digit: [list of 28x28 numpy arrays]}
    """
    data_dir = "./data"

    if not os.path.exists(f"{data_dir}/test_images.npy"):
        raise FileNotFoundError(
            f"Test data not found in {data_dir}/. "
            "Run download_datasets.py first."
        )

    test_images = np.load(f"{data_dir}/test_images.npy")
    test_labels = np.load(f"{data_dir}/test_labels.npy")

    print(f"Loaded {len(test_images)} test samples")

    # Organize by digit (0-9)
    samples_by_digit = {}
    for img, lbl in zip(test_images, test_labels):
        lbl = int(lbl)
        if lbl not in samples_by_digit:
            samples_by_digit[lbl] = []
        samples_by_digit[lbl].append(img)

    print(f"Digits available: {sorted(samples_by_digit.keys())}")
    for digit in sorted(samples_by_digit.keys()):
        print(f"  Digit {digit}: {len(samples_by_digit[digit])} samples")

    return samples_by_digit


def make_hexasudoku_board():
    """
    Create a blank 16x16 Sudoku board with grid lines.

    Returns:
        numpy array of shape (BOARD_PIXELS, BOARD_PIXELS) with values 0-255
        White background (255), black lines (0)
    """
    grid = np.ones((BOARD_PIXELS, BOARD_PIXELS), dtype=np.uint8) * 255

    # Draw outer border (bold, 8px)
    border_thickness = 8
    for i in range(border_thickness):
        grid[i, :] = 0
        grid[-i-1, :] = 0
        grid[:, i] = 0
        grid[:, -i-1] = 0

    # Draw internal grid lines
    for i in range(1, SIZE):
        pos = i * CELL_SIZE

        if i % BOX_SIZE == 0:
            thickness = 6
        else:
            thickness = 2

        for j in range(-thickness, thickness):
            if 0 <= pos + j < BOARD_PIXELS:
                grid[pos + j, :] = 0
                grid[:, pos + j] = 0

    return grid


def render_digit(samples_dict, digit, target_size):
    """
    Randomly select and prepare a handwritten digit sample.

    Args:
        samples_dict: Dictionary of {digit: [samples]}
        digit: Digit to render (0-9)
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


def render_single_digit_cell(samples_dict, value):
    """
    Render a single digit (1-9) centered in a cell-sized image.

    Args:
        samples_dict: Dictionary of handwritten samples
        value: Digit value (1-9)

    Returns:
        numpy array of shape (CELL_SIZE, CELL_SIZE)
    """
    # Create white cell
    cell = np.ones((CELL_SIZE, CELL_SIZE), dtype=np.uint8) * 255

    # Render the digit
    digit_img = render_digit(samples_dict, value, SINGLE_DIGIT_SIZE)
    if digit_img is None:
        return cell

    digit_array = np.array(digit_img)

    # Center the digit in the cell
    offset_x = (CELL_SIZE - SINGLE_DIGIT_SIZE) // 2
    offset_y = (CELL_SIZE - SINGLE_DIGIT_SIZE) // 2

    # Paste using minimum (dark ink wins)
    cell[offset_y:offset_y + SINGLE_DIGIT_SIZE,
         offset_x:offset_x + SINGLE_DIGIT_SIZE] = np.minimum(
             cell[offset_y:offset_y + SINGLE_DIGIT_SIZE,
                  offset_x:offset_x + SINGLE_DIGIT_SIZE],
             digit_array
         )

    return cell


def render_two_digit_cell(samples_dict, value):
    """
    Render a two-digit number (10-16) with digits side-by-side.

    Args:
        samples_dict: Dictionary of handwritten samples
        value: Value (10-16)

    Returns:
        numpy array of shape (CELL_SIZE, CELL_SIZE)
    """
    # Create white cell
    cell = np.ones((CELL_SIZE, CELL_SIZE), dtype=np.uint8) * 255

    # Split into tens and ones
    tens_digit = value // 10  # Always 1 for 10-16
    ones_digit = value % 10   # 0-6

    # Render tens digit
    tens_img = render_digit(samples_dict, tens_digit, TWO_DIGIT_SIZE)
    if tens_img is not None:
        tens_array = np.array(tens_img)
        cell[TWO_DIGIT_Y:TWO_DIGIT_Y + TWO_DIGIT_SIZE,
             TWO_DIGIT_LEFT_X:TWO_DIGIT_LEFT_X + TWO_DIGIT_SIZE] = np.minimum(
                 cell[TWO_DIGIT_Y:TWO_DIGIT_Y + TWO_DIGIT_SIZE,
                      TWO_DIGIT_LEFT_X:TWO_DIGIT_LEFT_X + TWO_DIGIT_SIZE],
                 tens_array
             )

    # Render ones digit
    ones_img = render_digit(samples_dict, ones_digit, TWO_DIGIT_SIZE)
    if ones_img is not None:
        ones_array = np.array(ones_img)
        cell[TWO_DIGIT_Y:TWO_DIGIT_Y + TWO_DIGIT_SIZE,
             TWO_DIGIT_RIGHT_X:TWO_DIGIT_RIGHT_X + TWO_DIGIT_SIZE] = np.minimum(
                 cell[TWO_DIGIT_Y:TWO_DIGIT_Y + TWO_DIGIT_SIZE,
                      TWO_DIGIT_RIGHT_X:TWO_DIGIT_RIGHT_X + TWO_DIGIT_SIZE],
                 ones_array
             )

    return cell


def insert_cell_image(board, row, col, cell_img):
    """
    Paste a cell image into the board.

    Args:
        board: numpy array of the board (modified in place)
        row: Row index (0-15)
        col: Column index (0-15)
        cell_img: numpy array of the cell (CELL_SIZE x CELL_SIZE)

    Returns:
        Modified board array
    """
    cell_y = row * CELL_SIZE
    cell_x = col * CELL_SIZE

    # Blend using minimum (dark ink wins over white background and grid lines)
    board[cell_y:cell_y + CELL_SIZE,
          cell_x:cell_x + CELL_SIZE] = np.minimum(
              board[cell_y:cell_y + CELL_SIZE,
                    cell_x:cell_x + CELL_SIZE],
              cell_img
          )

    return board


def generate_board_image(puzzle, samples_dict):
    """
    Generate a single board image with handwritten digits.

    Args:
        puzzle: 16x16 list of puzzle values (0 = empty, 1-16 = values)
        samples_dict: Dictionary of handwritten samples by digit

    Returns:
        numpy array of shape (BOARD_PIXELS, BOARD_PIXELS)
    """
    # Create blank board with grid lines
    board = make_hexasudoku_board()

    # Insert handwritten digits for each clue
    for row in range(SIZE):
        for col in range(SIZE):
            value = puzzle[row][col]
            if value == 0:
                continue  # Empty cell

            if value <= 9:
                # Single digit
                cell_img = render_single_digit_cell(samples_dict, value)
            else:
                # Two digits (10-16)
                cell_img = render_two_digit_cell(samples_dict, value)

            board = insert_cell_image(board, row, col, cell_img)

    return board


def main():
    """Main function to generate all board images."""
    print("=" * 60)
    print("HexaSudoku Board Image Generation (Digits Only)")
    print("Values 10-16 rendered as two-digit numbers")
    print("=" * 60)

    # Load puzzles
    print("\n1. Loading puzzles...")
    puzzles_path = "./puzzles/unique_puzzles_dict.json"

    if not os.path.exists(puzzles_path):
        raise FileNotFoundError(
            f"Puzzles not found at {puzzles_path}. "
            "Copy from 90-10 split detect handwritten digit errors/puzzles/ first."
        )

    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"   Loaded {len(puzzles)} puzzles")

    # Load test samples
    print("\n2. Loading handwritten test samples...")
    samples_dict = load_test_samples()

    # Verify we have digit 0 (needed for 10, 20, etc.)
    if 0 not in samples_dict:
        raise ValueError("Digit 0 not found in samples. Required for rendering 10, 20, etc.")

    # Create output directory
    output_dir = "./board_images"
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    print(f"\n3. Generating {len(puzzles)} board images...")
    print("   Values 1-9: Single digit")
    print("   Values 10-16: Two digits side-by-side")
    print("-" * 60)

    for i, puzzle_data in enumerate(puzzles):
        puzzle = puzzle_data['puzzle']
        board = generate_board_image(puzzle, samples_dict)

        # Save as PNG
        img = Image.fromarray(board, mode='L')
        filename = f"{output_dir}/board16_{i}.png"
        img.save(filename)

        if (i + 1) % 10 == 0:
            print(f"   Generated {i + 1}/{len(puzzles)} images")

    print("-" * 60)
    print(f"\nTotal images generated: {len(puzzles)}")
    print(f"Output directory: {output_dir}/")

    print("\n" + "=" * 60)
    print("Board image generation complete!")
    print("=" * 60)
    print("\nNext step: Run detect_errors.py or predict_digits.py to evaluate")


if __name__ == '__main__':
    main()
