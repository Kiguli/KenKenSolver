"""
Generate 1600x1600 board images for HexaSudoku puzzles using NUMERIC notation.

Values 10-16 are displayed as two-digit numbers (10, 11, 12, 13, 14, 15, 16)
instead of hex letters (A, B, C, D, E, F, G).

The two-digit numbers are rendered centered in the cell (like single characters)
so they can be recognized as single classes by the CNN.
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SIZE = 16
BOX_SIZE = 4
BOARD_PIXELS = 1600
CELL_SIZE = BOARD_PIXELS // SIZE  # 100px per cell


def value_to_char(value):
    """Convert puzzle value (0-16) to display string (numeric notation)."""
    if value == 0:
        return ''
    return str(value)  # Display 10, 11, 12, ..., 16 as two-digit text


def make_hexasudoku_board():
    """Create a blank 16x16 Sudoku board with grid lines."""
    grid = np.ones((BOARD_PIXELS, BOARD_PIXELS))

    # Draw outer border (bold)
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


def insert_character(img, row, col, value, font, font_small):
    """Insert a character into a cell.

    Both single digits and two-digit numbers are centered in the cell.
    Two-digit numbers use a smaller font to fit.
    """
    if value == 0:
        return img

    draw = ImageDraw.Draw(img)

    # Cell boundaries
    cell_x = col * CELL_SIZE
    cell_y = row * CELL_SIZE

    # Get display string
    char = value_to_char(value)

    # Use smaller font for two-digit numbers
    if value >= 10:
        current_font = font_small
    else:
        current_font = font

    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), char, font=current_font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text in cell
    x = cell_x + (CELL_SIZE - text_width) // 2
    y = cell_y + (CELL_SIZE - text_height) // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=current_font)

    return img


def make_board_full(puzzle, font, font_small):
    """Create a complete HexaSudoku board image with given clues."""
    grid = make_hexasudoku_board()
    img = Image.fromarray((grid * 255).astype(np.uint8), mode='L')

    for row in range(SIZE):
        for col in range(SIZE):
            value = puzzle[row][col]
            if value != 0:
                img = insert_character(img, row, col, value, font, font_small)

    return np.array(img) / 255.0


def main():
    # Load puzzles
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"Loaded {len(puzzles)} puzzles")

    # Load fonts - regular size for single digits, smaller for two-digit numbers
    font_size = CELL_SIZE * 2 // 3  # ~67px for 100px cells
    font_size_small = CELL_SIZE * 2 // 5  # ~40px for two-digit numbers

    font = None
    font_small = None
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                font_small = ImageFont.truetype(path, font_size_small)
                print(f"Using font: {path}")
                break
            except:
                pass

    if font is None:
        font = ImageFont.load_default()
        font_small = font
        print("Using default font")

    # Create output directory
    output_dir = './board_images_numeric'
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    print(f"\nGenerating {len(puzzles)} board images with NUMERIC notation...")
    for i, puzzle_data in enumerate(puzzles):
        puzzle = puzzle_data['puzzle']
        board = make_board_full(puzzle, font, font_small)

        array_uint8 = (board * 255).astype(np.uint8)
        image = Image.fromarray(array_uint8, mode='L')

        filename = f"{output_dir}/board16_{i}.png"
        image.save(filename)

        if (i + 1) % 25 == 0:
            print(f"  Generated {i + 1}/{len(puzzles)}")

    print(f"\nTotal images generated: {len(puzzles)}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
