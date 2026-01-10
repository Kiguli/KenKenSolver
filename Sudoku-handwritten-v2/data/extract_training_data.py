# -*- coding: utf-8 -*-
"""
Extract training data from Sudoku and HexaSudoku board images for unified CNN training.

This script extracts characters from:
1. Sudoku 4x4 boards (digits 1-4)
2. Sudoku 9x9 boards (digits 1-9)
3. HexaSudoku A-G boards (digits 0-9 + letters A-G)
4. HexaSudoku numeric boards (digits 0-9, with two-digit numbers 10-16)

Output classes (17 total):
- 0-9: Digits
- 10-16: Letters A-G (mapped as A=10, B=11, ..., G=16)

The extraction uses the same preprocessing pipeline that will be used during inference
to ensure domain consistency (avoiding the MNIST→KenKen domain mismatch issue).
"""

import os
import sys
import json
import numpy as np
import cv2 as cv
from PIL import Image
from collections import Counter
import random

# Constants
IMG_SIZE = 28  # Output character size

# Board configurations
SUDOKU_CONFIG = {
    'board_size': 900,
    '4': {'cell_size': 225, 'num_puzzles': 100},
    '9': {'cell_size': 100, 'num_puzzles': 100}
}

HEXASUDOKU_CONFIG = {
    'board_size': 1600,
    '16': {'cell_size': 100, 'num_puzzles': 100}
}


def get_contours(threshold_image):
    """Find contours in thresholded image."""
    kernel = np.ones((2, 2), np.uint8)
    img_dilate = cv.dilate(threshold_image, kernel, iterations=1)
    contours, _ = cv.findContours(img_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def extract_character_image(cell_img, contour):
    """Extract and normalize a single character from contour."""
    x, y, w, h = cv.boundingRect(contour)

    # Filter too small contours
    if w < 5 or h < 5:
        return None

    char_img = cell_img[y:y+h, x:x+w]

    if char_img.size == 0:
        return None

    # Resize preserving aspect ratio
    char_pil = Image.fromarray((char_img * 255).astype(np.uint8))
    aspect = w / h

    if aspect > 1:
        new_w = IMG_SIZE - 4  # Leave margin
        new_h = max(1, int((IMG_SIZE - 4) / aspect))
    else:
        new_h = IMG_SIZE - 4
        new_w = max(1, int((IMG_SIZE - 4) * aspect))

    resized = char_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return np.array(canvas).astype(np.float32) / 255.0


def extract_single_character_from_cell(grid, row, col, cell_size, border=5):
    """
    Extract a single character from a Sudoku/HexaSudoku cell.

    Used for single-digit cells (Sudoku and HexaSudoku A-G).

    Returns: (image, confidence) or None if cell is empty
    """
    # Calculate cell region
    y_start = row * cell_size + border
    y_end = (row + 1) * cell_size - border
    x_start = col * cell_size + border
    x_end = (col + 1) * cell_size - border

    # Bounds check
    if y_end > grid.shape[0] or x_end > grid.shape[1]:
        return None

    cell_img = grid[y_start:y_end, x_start:x_end]

    if cell_img.size == 0:
        return None

    # Normalize to 0-1
    cell_img = cell_img.astype(np.float32) / 255.0

    # Threshold for contour detection
    img_uint8 = (cell_img * 255).astype(np.uint8)
    _, thresh = cv.threshold(img_uint8, 127, 255, cv.THRESH_BINARY_INV)

    # Check if cell has content
    if np.sum(thresh) < 100:  # Mostly empty
        return None

    # Find contours
    contours = get_contours(thresh)

    if not contours:
        return None

    # Find largest contour by area
    largest_contour = max(contours, key=cv.contourArea)

    # Extract character
    char_img = extract_character_image(cell_img, largest_contour)

    return char_img


def extract_two_digit_number_from_cell(grid, row, col, cell_size, border=5):
    """
    Extract two-digit number from a HexaSudoku numeric cell.

    For values 10-16, there are two digit images side by side.

    Returns: list of (image, position) tuples, or None if cell is empty
    """
    # Calculate cell region
    y_start = row * cell_size + border
    y_end = (row + 1) * cell_size - border
    x_start = col * cell_size + border
    x_end = (col + 1) * cell_size - border

    # Bounds check
    if y_end > grid.shape[0] or x_end > grid.shape[1]:
        return None

    cell_img = grid[y_start:y_end, x_start:x_end]

    if cell_img.size == 0:
        return None

    # Normalize to 0-1
    cell_img = cell_img.astype(np.float32) / 255.0

    # Threshold for contour detection
    img_uint8 = (cell_img * 255).astype(np.uint8)
    _, thresh = cv.threshold(img_uint8, 127, 255, cv.THRESH_BINARY_INV)

    # Check if cell has content
    if np.sum(thresh) < 100:  # Mostly empty
        return None

    # Find contours
    contours = get_contours(thresh)

    if not contours:
        return None

    # Get bounding boxes and sort left to right
    boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w >= 5 and h >= 8:  # Filter small noise
            boxes.append((x, contour))

    if not boxes:
        return None

    boxes.sort(key=lambda b: b[0])

    # Extract characters
    characters = []
    for pos, (x, contour) in enumerate(boxes):
        char_img = extract_character_image(cell_img, contour)
        if char_img is not None:
            characters.append((char_img, pos))

    return characters if characters else None


def extract_from_sudoku_board(board_path, puzzle_data, grid_size):
    """
    Extract character samples from a Sudoku board image.

    Args:
        board_path: Path to board image
        puzzle_data: Dict with 'puzzle' (0=empty) and 'solution' keys
        grid_size: 4 or 9

    Returns: List of (image, label) tuples
    """
    if not os.path.exists(board_path):
        return []

    # Load board image
    img = Image.open(board_path).convert('L')
    grid = np.array(img)

    cell_size = SUDOKU_CONFIG[str(grid_size)]['cell_size']
    samples = []

    puzzle = puzzle_data['puzzle']
    solution = puzzle_data['solution']

    for row in range(grid_size):
        for col in range(grid_size):
            # Only extract from cells with given values (not empty)
            if puzzle[row][col] != 0:
                char_img = extract_single_character_from_cell(grid, row, col, cell_size)

                if char_img is not None:
                    label = puzzle[row][col]  # Label is the digit value (1-9)
                    samples.append((char_img, label))

    return samples


def extract_from_hexasudoku_ag_board(board_path, puzzle_data):
    """
    Extract character samples from a HexaSudoku A-G board image.

    Values 1-9 are digits, values 10-16 are letters A-G.

    Args:
        board_path: Path to board image
        puzzle_data: Dict with 'puzzle' (0=empty) and 'solution' keys

    Returns: List of (image, label) tuples
    """
    if not os.path.exists(board_path):
        return []

    # Load board image
    img = Image.open(board_path).convert('L')
    grid = np.array(img)

    cell_size = HEXASUDOKU_CONFIG['16']['cell_size']
    samples = []

    puzzle = puzzle_data['puzzle']

    for row in range(16):
        for col in range(16):
            value = puzzle[row][col]

            # Only extract from cells with given values (not empty)
            if value != 0:
                char_img = extract_single_character_from_cell(grid, row, col, cell_size)

                if char_img is not None:
                    # Values 1-9 stay as digits, 10-16 are letters A-G
                    # But we map them directly since our output classes include 10-16
                    label = value
                    samples.append((char_img, label))

    return samples


def extract_from_hexasudoku_numeric_board(board_path, puzzle_data):
    """
    Extract character samples from a HexaSudoku numeric board image.

    Values 1-9 are single digits, values 10-16 are two-digit numbers.
    For two-digit numbers, we extract each digit separately.

    Args:
        board_path: Path to board image
        puzzle_data: Dict with 'puzzle' (0=empty) and 'solution' keys

    Returns: List of (image, label) tuples
    """
    if not os.path.exists(board_path):
        return []

    # Load board image
    img = Image.open(board_path).convert('L')
    grid = np.array(img)

    cell_size = HEXASUDOKU_CONFIG['16']['cell_size']
    samples = []

    puzzle = puzzle_data['puzzle']

    for row in range(16):
        for col in range(16):
            value = puzzle[row][col]

            # Only extract from cells with given values (not empty)
            if value != 0:
                if value <= 9:
                    # Single digit
                    char_img = extract_single_character_from_cell(grid, row, col, cell_size)
                    if char_img is not None:
                        samples.append((char_img, value))
                else:
                    # Two-digit number (10-16)
                    chars = extract_two_digit_number_from_cell(grid, row, col, cell_size)
                    if chars and len(chars) == 2:
                        # First digit is tens (always 1)
                        samples.append((chars[0][0], 1))
                        # Second digit is ones (0-6)
                        ones_digit = value - 10
                        samples.append((chars[1][0], ones_digit))

    return samples


def extract_all_training_data(base_dir):
    """
    Extract training data from all puzzle types.

    Returns: (images, labels) tuple
    """
    all_samples = []

    # 1. Sudoku 4x4 and 9x9
    print("\n1. Extracting from Sudoku boards...")
    sudoku_dir = os.path.join(base_dir, '90-10 split handwritten', 'Handwritten_Sudoku')
    sudoku_puzzles_path = os.path.join(sudoku_dir, 'puzzles', 'puzzles_dict.json')

    if os.path.exists(sudoku_puzzles_path):
        with open(sudoku_puzzles_path, 'r') as f:
            sudoku_puzzles = json.load(f)

        for grid_size in ['4', '9']:
            if grid_size in sudoku_puzzles:
                puzzles = sudoku_puzzles[grid_size]
                num_puzzles = min(100, len(puzzles))
                size_samples = []

                for idx in range(num_puzzles):
                    board_path = os.path.join(sudoku_dir, 'board_images', f'board{grid_size}_{idx}.png')
                    samples = extract_from_sudoku_board(board_path, puzzles[idx], int(grid_size))
                    size_samples.extend(samples)

                print(f"   Sudoku {grid_size}x{grid_size}: {len(size_samples)} samples from {num_puzzles} puzzles")
                all_samples.extend(size_samples)
    else:
        print(f"   Warning: {sudoku_puzzles_path} not found")

    # 2. HexaSudoku A-G
    print("\n2. Extracting from HexaSudoku A-G boards...")
    hexa_ag_dir = os.path.join(base_dir, '90-10 split handwritten', 'Handwritten_HexaSudoku')
    hexa_ag_puzzles_path = os.path.join(hexa_ag_dir, 'puzzles', 'unique_puzzles_dict.json')

    if os.path.exists(hexa_ag_puzzles_path):
        with open(hexa_ag_puzzles_path, 'r') as f:
            hexa_ag_puzzles = json.load(f)

        if '16' in hexa_ag_puzzles:
            puzzles = hexa_ag_puzzles['16']
            num_puzzles = min(100, len(puzzles))
            hexa_samples = []

            for idx in range(num_puzzles):
                board_path = os.path.join(hexa_ag_dir, 'board_images', f'board16_{idx}.png')
                samples = extract_from_hexasudoku_ag_board(board_path, puzzles[idx])
                hexa_samples.extend(samples)

            print(f"   HexaSudoku A-G: {len(hexa_samples)} samples from {num_puzzles} puzzles")
            all_samples.extend(hexa_samples)
    else:
        print(f"   Warning: {hexa_ag_puzzles_path} not found")

    # 3. HexaSudoku Numeric
    print("\n3. Extracting from HexaSudoku Numeric boards...")
    hexa_num_dir = os.path.join(base_dir, '90-10 split handwritten digits only')
    hexa_num_puzzles_path = os.path.join(hexa_num_dir, 'puzzles', 'unique_puzzles_dict.json')

    if os.path.exists(hexa_num_puzzles_path):
        with open(hexa_num_puzzles_path, 'r') as f:
            hexa_num_puzzles = json.load(f)

        if '16' in hexa_num_puzzles:
            puzzles = hexa_num_puzzles['16']
            num_puzzles = min(100, len(puzzles))
            hexa_samples = []

            for idx in range(num_puzzles):
                board_path = os.path.join(hexa_num_dir, 'board_images', f'board16_{idx}.png')
                samples = extract_from_hexasudoku_numeric_board(board_path, puzzles[idx])
                hexa_samples.extend(samples)

            print(f"   HexaSudoku Numeric: {len(hexa_samples)} samples from {num_puzzles} puzzles")
            all_samples.extend(hexa_samples)
    else:
        print(f"   Warning: {hexa_num_puzzles_path} not found")

    # Convert to arrays
    images = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    return images, labels


def analyze_and_balance_classes(images, labels, target_per_class=None):
    """
    Analyze class distribution and balance by oversampling minority classes.
    """
    label_counts = Counter(labels)

    print("\nClass distribution before balancing:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        if label <= 9:
            char = str(label)
        else:
            char = chr(ord('A') + label - 10)
        print(f"   Class {label:2d} ({char}): {count:5d} samples")

    if target_per_class is None:
        # Use median count as target
        target_per_class = int(np.median(list(label_counts.values())))

    print(f"\nBalancing to {target_per_class} samples per class...")

    # Group by label
    by_label = {}
    for img, label in zip(images, labels):
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(img)

    # Balance
    balanced_images = []
    balanced_labels = []

    for label in sorted(by_label.keys()):
        class_images = by_label[label]
        count = len(class_images)

        if count >= target_per_class:
            # Undersample
            selected = random.sample(class_images, target_per_class)
        else:
            # Oversample
            selected = class_images.copy()
            while len(selected) < target_per_class:
                selected.append(random.choice(class_images))

        balanced_images.extend(selected)
        balanced_labels.extend([label] * len(selected))

    # Shuffle
    combined = list(zip(balanced_images, balanced_labels))
    random.shuffle(combined)
    balanced_images, balanced_labels = zip(*combined)

    print(f"   Balanced dataset: {len(balanced_images)} total samples")

    return list(balanced_images), list(balanced_labels)


def save_training_data(output_dir, images, labels):
    """Save extracted training data to numpy files."""
    os.makedirs(output_dir, exist_ok=True)

    images_path = os.path.join(output_dir, 'extracted_images.npy')
    labels_path = os.path.join(output_dir, 'extracted_labels.npy')

    np.save(images_path, np.array(images))
    np.save(labels_path, np.array(labels))

    print(f"\nSaved to:")
    print(f"   {images_path}")
    print(f"   {labels_path}")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # Get base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))  # KenKenSolver root

    print("=" * 60)
    print("Unified Training Data Extraction")
    print("=" * 60)
    print(f"\nBase directory: {base_dir}")

    # Extract all training data
    images, labels = extract_all_training_data(base_dir)

    print(f"\n{'=' * 60}")
    print(f"Total extracted: {len(images)} samples")

    # Balance classes
    images, labels = analyze_and_balance_classes(images, labels)

    # Save
    output_dir = os.path.join(script_dir)
    save_training_data(output_dir, images, labels)

    print("\nDone!")
