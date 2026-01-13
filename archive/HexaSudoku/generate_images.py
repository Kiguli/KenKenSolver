"""
Generate 1600x1600 board images for HexaSudoku puzzles.
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
    """Convert puzzle value (0-16) to display character."""
    if value == 0:
        return ''
    if value <= 9:
        return str(value)
    return chr(ord('A') + value - 10)  # 10→A, 11→B, ..., 15→F


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


def insert_character(img, row, col, value, font):
    """Insert a character into a cell."""
    if value == 0:
        return img

    char = value_to_char(value)
    draw = ImageDraw.Draw(img)

    # Calculate centered position
    cell_center_x = col * CELL_SIZE + CELL_SIZE // 2
    cell_center_y = row * CELL_SIZE + CELL_SIZE // 2

    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = cell_center_x - text_width // 2
    y = cell_center_y - text_height // 2 - bbox[1]

    draw.text((x, y), char, fill=0, font=font)

    return img


def make_board_full(puzzle, font):
    """Create a complete HexaSudoku board image with given clues."""
    grid = make_hexasudoku_board()
    img = Image.fromarray((grid * 255).astype(np.uint8), mode='L')

    for row in range(SIZE):
        for col in range(SIZE):
            value = puzzle[row][col]
            if value != 0:
                img = insert_character(img, row, col, value, font)

    return np.array(img) / 255.0


def main():
    # Load puzzles
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"Loaded {len(puzzles)} puzzles")

    # Load font
    font_size = CELL_SIZE * 2 // 3  # ~67px for 100px cells
    font = None
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
                print(f"Using font: {path}")
                break
            except:
                pass

    if font is None:
        font = ImageFont.load_default()
        print("Using default font")

    # Create output directory
    output_dir = './board_images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate images
    print(f"\nGenerating {len(puzzles)} board images...")
    for i, puzzle_data in enumerate(puzzles):
        puzzle = puzzle_data['puzzle']
        board = make_board_full(puzzle, font)

        array_uint8 = (board * 255).astype(np.uint8)
        image = Image.fromarray(array_uint8, mode='L')

        filename = f"{output_dir}/board16_{i}.png"
        image.save(filename)

        if (i + 1) % 25 == 0:
            print(f"  Generated {i + 1}/{len(puzzles)}")

    print(f"\nTotal images generated: {len(puzzles)}")


if __name__ == '__main__':
    main()
