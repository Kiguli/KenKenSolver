"""
Generate HexaSudoku board images using handwritten TEST samples.

Critical: Uses ONLY the test split of MNIST/EMNIST, which the CNN
never saw during training. This tests true generalization.

Each board randomly samples different handwritten instances for each cell,
ensuring variety and realistic handwritten appearance.

Image convention:
- Board background: WHITE (255)
- Characters: DARK (0, black ink)
- MNIST/EMNIST store ink as HIGH values, so we INVERT when placing on board
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
CHAR_SIZE = 70  # Size to render character within cell (smaller than cell for margin)


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
        if cls == 0:
            name = "Empty"
        elif cls <= 9:
            name = str(cls)
        else:
            name = chr(ord('A') + cls - 10)
        print(f"  Class {cls:2d} ({name}): {len(samples_by_class[cls])} samples")

    return samples_by_class


def value_to_class(value):
    """
    Convert puzzle value (1-16) to CNN class.

    HexaSudoku values:
    - 1-9: digits → classes 1-9
    - 10-15: displayed as A-F → classes 10-15
    - 16: displayed as G → class 16 (but rarely used in puzzles)
    - 0: empty → not rendered

    Args:
        value: Puzzle value (0-16)

    Returns:
        CNN class label, or None if empty (0)
    """
    if value == 0:
        return None
    return value  # Direct mapping: value 1-16 → class 1-16


def make_hexasudoku_board():
    """
    Create a blank 16x16 Sudoku board with grid lines.

    Returns:
        numpy array of shape (BOARD_PIXELS, BOARD_PIXELS) with values 0-255
        White background (255), black lines (0)
    """
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
    for i in range(1, SIZE):
        pos = i * CELL_SIZE

        # Bold lines for box boundaries (every 4 cells)
        if i % BOX_SIZE == 0:
            thickness = 6
        else:
            thickness = 2

        for j in range(-thickness, thickness):
            if 0 <= pos + j < BOARD_PIXELS:
                grid[pos + j, :] = 0  # Horizontal line
                grid[:, pos + j] = 0  # Vertical line

    return grid


def render_handwritten_char(samples_dict, class_label, target_size=CHAR_SIZE):
    """
    Randomly select and prepare a handwritten sample for the given class.

    The sample is inverted (MNIST ink=high → board ink=low/black) and resized.

    Args:
        samples_dict: Dictionary of {class: [samples]}
        class_label: CNN class label
        target_size: Output size in pixels

    Returns:
        PIL Image in mode 'L' (grayscale) with white background, dark character
        Or None if class not found
    """
    if class_label not in samples_dict or len(samples_dict[class_label]) == 0:
        print(f"Warning: No samples for class {class_label}")
        return None

    # Random selection for variety
    idx = np.random.randint(len(samples_dict[class_label]))
    sample = samples_dict[class_label][idx]

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


def insert_handwritten_char(board, row, col, char_img):
    """
    Paste a handwritten character image into a cell on the board.

    Uses minimum blending so dark pixels (ink) override white background.

    Args:
        board: numpy array of the board (modified in place)
        row: Row index (0-15)
        col: Column index (0-15)
        char_img: PIL Image of the character

    Returns:
        Modified board array
    """
    if char_img is None:
        return board

    # Get character as numpy array
    char_array = np.array(char_img)
    char_h, char_w = char_array.shape

    # Calculate centered position within cell
    cell_x = col * CELL_SIZE
    cell_y = row * CELL_SIZE

    offset_x = (CELL_SIZE - char_w) // 2
    offset_y = (CELL_SIZE - char_h) // 2

    # Target region
    y1 = cell_y + offset_y
    y2 = y1 + char_h
    x1 = cell_x + offset_x
    x2 = x1 + char_w

    # Blend using minimum (dark ink wins over white background)
    board[y1:y2, x1:x2] = np.minimum(board[y1:y2, x1:x2], char_array)

    return board


def generate_board_image(puzzle, samples_dict):
    """
    Generate a single board image with handwritten characters.

    Args:
        puzzle: 16x16 list of puzzle values (0 = empty)
        samples_dict: Dictionary of handwritten samples by class

    Returns:
        numpy array of shape (BOARD_PIXELS, BOARD_PIXELS)
    """
    # Create blank board with grid lines
    board = make_hexasudoku_board()

    # Insert handwritten characters for each clue
    for row in range(SIZE):
        for col in range(SIZE):
            value = puzzle[row][col]
            if value != 0:
                class_label = value_to_class(value)
                if class_label is not None:
                    char_img = render_handwritten_char(samples_dict, class_label)
                    if char_img is not None:
                        board = insert_handwritten_char(board, row, col, char_img)

    return board


def main():
    """Main function to generate all board images."""
    print("=" * 60)
    print("Handwritten HexaSudoku Board Image Generation")
    print("=" * 60)

    # Load puzzles
    print("\n1. Loading puzzles...")
    puzzles_path = "./puzzles/unique_puzzles_dict.json"

    if not os.path.exists(puzzles_path):
        raise FileNotFoundError(
            f"Puzzles not found at {puzzles_path}. "
            "Copy from HexaSudoku/puzzles/ first."
        )

    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"   Loaded {len(puzzles)} puzzles")

    # Load test samples (unseen by CNN)
    print("\n2. Loading handwritten test samples...")
    samples_dict = load_test_samples()

    # Create output directory
    output_dir = "./board_images"
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    print(f"\n3. Generating {len(puzzles)} board images...")
    print("   (Using handwritten TEST samples - never seen during CNN training)")
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
    print("\nNext step: Run evaluate.py to test the neuro-symbolic pipeline")


if __name__ == '__main__':
    main()
