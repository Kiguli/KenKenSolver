"""
Generate error visualization images for Sudoku detection mismatches.

For each puzzle where detected clues don't match expected:
- Black digits: Correctly detected (position and value match)
- Red digits: Incorrectly detected (wrong position or wrong value)
- Gray X: Missed digits (expected but not detected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import pandas as pd
import json
import os

# Constants
BOARD_SIZE = 900
IMG_SIZE = 28

# Colors (RGB)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)


class CNN_v2(nn.Module):
    """CNN for character recognition."""
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


def load_model():
    """Load the character recognition CNN model."""
    model_path = '../KenKen/models/character_recognition_v2_model_weights.pth'
    model = CNN_v2(output_dim=14)
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def extract_cell(img_array, size, row, col, border=10):
    """Extract a single cell from the grid."""
    height, width = img_array.shape
    cell_h = height // size
    cell_w = width // size

    y1 = row * cell_h + border
    y2 = (row + 1) * cell_h - border
    x1 = col * cell_w + border
    x2 = (col + 1) * cell_w - border

    return img_array[y1:y2, x1:x2]


def is_cell_empty(cell_img, threshold=0.98):
    """Check if a cell is empty (mostly white)."""
    white_pixels = np.sum(cell_img > 200)
    total_pixels = cell_img.size
    return (white_pixels / total_pixels) > threshold


def preprocess_cell(cell_img):
    """Preprocess cell image for CNN input."""
    cell_pil = Image.fromarray(cell_img)
    resized = cell_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    normalized = np.array(resized).astype(np.float32) / 255.0
    return normalized


def recognize_digit(cell_img, model):
    """Recognize digit in a cell using CNN."""
    with torch.no_grad():
        tensor = torch.tensor(cell_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()
        if prediction <= 9:
            return prediction
        else:
            return 0


def detect_cells(filename, size, model):
    """Detect all digits in an image, return dict of {(row, col): digit}."""
    img = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {filename}")

    detected = {}
    for row in range(size):
        for col in range(size):
            cell = extract_cell(img, size, row, col)
            if not is_cell_empty(cell):
                processed = preprocess_cell(cell)
                digit = recognize_digit(processed, model)
                if digit > 0:
                    detected[(row, col)] = digit

    return detected


def make_rgb_board(size):
    """Create a 900x900 RGB board with grid lines."""
    # Start with white background
    grid = np.ones((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8) * 255

    cell_size = BOARD_SIZE // size
    box_size = 2 if size == 4 else 3

    # Draw outer border (8 pixels)
    border = 8
    grid[:border, :, :] = 0
    grid[-border:, :, :] = 0
    grid[:, :border, :] = 0
    grid[:, -border:, :] = 0

    # Draw internal grid lines
    for i in range(1, size):
        pos = i * cell_size

        # Bold lines at box boundaries (6 pixels), thin lines elsewhere (2 pixels)
        if i % box_size == 0:
            thickness = 3
        else:
            thickness = 1

        # Horizontal and vertical lines
        for j in range(-thickness, thickness + 1):
            if 0 <= pos + j < BOARD_SIZE:
                grid[pos + j, :, :] = 0
                grid[:, pos + j, :] = 0

    return grid


def draw_digit(grid, size, row, col, digit, color):
    """Draw a digit at the specified cell position with given color."""
    cell_size = BOARD_SIZE // size

    # Create PIL image from grid
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)

    # Calculate font size and position
    font_size = cell_size * 2 // 3
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # Calculate centered position
    cell_center_x = col * cell_size + cell_size // 2
    cell_center_y = row * cell_size + cell_size // 2

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = cell_center_x - text_width // 2
    y = cell_center_y - text_height // 2 - bbox[1]

    draw.text((x, y), text, fill=color, font=font)

    return np.array(img)


def draw_x_mark(grid, size, row, col, color):
    """Draw an X mark at the specified cell position."""
    cell_size = BOARD_SIZE // size

    # Calculate cell boundaries with margin
    margin = cell_size // 4
    x1 = col * cell_size + margin
    y1 = row * cell_size + margin
    x2 = (col + 1) * cell_size - margin
    y2 = (row + 1) * cell_size - margin

    # Create PIL image and draw
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)

    # Draw X with thick lines
    line_width = max(4, cell_size // 20)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
    draw.line([(x1, y2), (x2, y1)], fill=color, width=line_width)

    return np.array(img)


def generate_error_visualization(size, puzzle_index, expected_puzzle, detected_cells, output_path):
    """Generate and save error visualization for a puzzle."""
    # Create base board
    grid = make_rgb_board(size)

    # Process each cell
    for row in range(size):
        for col in range(size):
            expected_val = expected_puzzle[row][col]
            detected_val = detected_cells.get((row, col), 0)

            if expected_val != 0 and (row, col) in detected_cells and detected_val == expected_val:
                # Correct detection - black digit
                grid = draw_digit(grid, size, row, col, detected_val, BLACK)

            elif expected_val != 0 and (row, col) not in detected_cells:
                # Missed digit - gray X
                grid = draw_x_mark(grid, size, row, col, GRAY)

            elif (row, col) in detected_cells and (expected_val == 0 or detected_val != expected_val):
                # False positive or wrong value - red digit
                grid = draw_digit(grid, size, row, col, detected_val, RED)

    # Save image
    img = Image.fromarray(grid)
    img.save(output_path)


def main():
    """Main function to generate all error visualizations."""
    # Load data
    print("Loading evaluation CSV...")
    eval_df = pd.read_csv('./results/detailed_evaluation.csv')

    print("Loading puzzles JSON...")
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        puzzles_ds = json.load(f)

    print("Loading CNN model...")
    model = load_model()

    # Create output directory
    output_dir = './board_image_errors'
    os.makedirs(output_dir, exist_ok=True)

    # Find puzzles with clue mismatches
    mismatch_df = eval_df[eval_df['clues_match'] == False]
    print(f"\nFound {len(mismatch_df)} puzzles with detection mismatches")

    # Generate visualizations
    for idx, row in mismatch_df.iterrows():
        size = row['size']
        puzzle_index = row['puzzle_index']
        filename = row['filename']

        # Get expected puzzle from JSON
        expected_puzzle = puzzles_ds[str(size)][puzzle_index]['puzzle']

        # Re-run detection
        detected_cells = detect_cells(filename, size, model)

        # Generate visualization
        output_path = os.path.join(output_dir, f'board{size}_{puzzle_index}_error.png')
        generate_error_visualization(size, puzzle_index, expected_puzzle, detected_cells, output_path)

        print(f"  Generated: {output_path}")

    print(f"\nDone! Generated {len(mismatch_df)} error visualizations in {output_dir}/")


if __name__ == '__main__':
    main()
