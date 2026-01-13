# -*- coding: utf-8 -*-
"""
Extract training data from actual board images for improved CNN training.

Key insight: Training on MNIST alone creates domain mismatch.
Instead, extract characters from actual board images using the same
preprocessing pipeline used during inference.

Sources:
1. KenKen-300px-handwritten/board_images (handwritten digits)
2. KenKen-300px/board_images (computer-generated - for baseline)
"""

import os
import sys
import json
import numpy as np
import cv2 as cv
from PIL import Image
from collections import Counter
import random

# Add solver to path for character extraction functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'solver'))

# Constants matching solve_all_sizes.py
CELL_SIZE = 300
IMG_SIZE = 28
SCALE_FACTOR = 2


def get_contours(threshold_image):
    """Find contours in thresholded image."""
    kernel = np.ones((2, 2), np.uint8)
    img_dilate = cv.dilate(threshold_image, kernel, iterations=1)
    contours, _ = cv.findContours(img_dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_character(cell_img, contour):
    """Extract and normalize a single character from contour."""
    x, y, w, h = cv.boundingRect(contour)

    if w < 5 or h < 5:
        return None

    char_img = cell_img[y:y+h, x:x+w]

    if char_img.size == 0:
        return None

    # Resize preserving aspect ratio
    char_pil = Image.fromarray((char_img * 255).astype(np.uint8))
    aspect = w / h

    if aspect > 1:
        new_w = IMG_SIZE
        new_h = max(1, int(IMG_SIZE / aspect))
    else:
        new_h = IMG_SIZE
        new_w = max(1, int(IMG_SIZE * aspect))

    resized = char_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    paste_x = (IMG_SIZE - new_w) // 2
    paste_y = (IMG_SIZE - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return np.array(canvas).astype(np.float32) / 255.0


def extract_characters_from_cell(grid, size, row, col, border_thickness):
    """
    Extract character images from a single cell.

    Returns list of (image, position) tuples.
    """
    # Calculate cell region (top portion where numbers are)
    height_factor = 0.5 if size <= 6 else (0.6 if size == 7 else 0.7)
    y_start = row * CELL_SIZE + border_thickness + 5
    y_end = int(row * CELL_SIZE + CELL_SIZE * height_factor)
    x_start = col * CELL_SIZE + border_thickness + 5
    x_end = (col + 1) * CELL_SIZE - border_thickness

    # Bounds check
    if y_end > grid.shape[0] or x_end > grid.shape[1]:
        return []

    cell_img = grid[y_start:y_end, x_start:x_end]

    if cell_img.size == 0:
        return []

    # Normalize to 0-1
    cell_img = cell_img.astype(np.float32) / 255.0

    # Threshold for contour detection
    img_uint8 = (cell_img * 255).astype(np.uint8)
    _, thresh = cv.threshold(img_uint8, 127, 255, cv.THRESH_BINARY_INV)

    # Find contours
    contours = get_contours(thresh)

    if not contours:
        return []

    # Get bounding boxes and sort left to right
    boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if w >= 5 and h >= 5:  # Filter small noise
            boxes.append((x, contour))

    boxes.sort(key=lambda b: b[0])

    # Extract characters
    characters = []
    for pos, (x, contour) in enumerate(boxes):
        char_img = get_character(cell_img, contour)
        if char_img is not None:
            characters.append((char_img, pos))

    return characters


def extract_from_board(board_path, puzzle, size, border_thickness=16):
    """
    Extract all character images from a single board.

    Returns list of (image, label, metadata) tuples.
    """
    if not os.path.exists(board_path):
        return []

    # Load board image
    img = Image.open(board_path).convert('L')
    grid = np.array(img)

    # Operator mapping
    op_to_label = {'add': 10, 'div': 11, 'mul': 12, 'sub': 13}

    samples = []

    for cage in puzzle:
        target = cage['target']
        op = cage['op']
        cells = cage['cells']

        # Find top-left cell (where target is written)
        start_cell = cells[0]
        for cell in cells[1:]:
            if cell[0] < start_cell[0] or (cell[0] == start_cell[0] and cell[1] < start_cell[1]):
                start_cell = cell

        # Get expected labels
        digit_labels = []
        temp = int(target)
        if temp == 0:
            digit_labels = [0]
        else:
            while temp > 0:
                digit_labels.append(temp % 10)
                temp //= 10
            digit_labels = digit_labels[::-1]

        # Add operator if present
        if op and op in op_to_label:
            digit_labels.append(op_to_label[op])

        # Extract characters from cell
        row, col = start_cell
        characters = extract_characters_from_cell(grid, size, row, col, border_thickness)

        # Match characters to labels
        if len(characters) == len(digit_labels):
            for (char_img, pos), label in zip(characters, digit_labels):
                samples.append({
                    'image': char_img,
                    'label': label,
                    'size': size,
                    'position': pos,
                    'is_operator': label >= 10,
                    'num_digits': len([l for l in digit_labels if l < 10])
                })

    return samples


def extract_from_directory(board_dir, puzzles_path, sizes, num_per_size=100, source_type='handwritten'):
    """
    Extract training data from all boards in a directory.

    Args:
        board_dir: Path to board_images directory
        puzzles_path: Path to puzzles_dict.json
        sizes: List of puzzle sizes to process
        num_per_size: Max puzzles per size
        source_type: 'handwritten' or 'computer' for metadata

    Returns:
        List of sample dictionaries
    """
    with open(puzzles_path, 'r') as f:
        puzzles_dict = json.load(f)

    all_samples = []

    for size in sizes:
        if str(size) not in puzzles_dict:
            print(f"  Size {size}: not found in puzzles")
            continue

        puzzles = puzzles_dict[str(size)]
        num_puzzles = min(num_per_size, len(puzzles))

        size_samples = []
        for puzzle_idx in range(num_puzzles):
            board_path = os.path.join(board_dir, f'board{size}_{puzzle_idx}.png')

            puzzle = puzzles[puzzle_idx]
            if isinstance(puzzle, dict) and 'cages' in puzzle:
                cages = puzzle['cages']
            else:
                cages = puzzle

            samples = extract_from_board(board_path, cages, size)
            for sample in samples:
                sample['source'] = source_type
                sample['puzzle_idx'] = puzzle_idx

            size_samples.extend(samples)

        print(f"  Size {size}: extracted {len(size_samples)} samples from {num_puzzles} puzzles")
        all_samples.extend(size_samples)

    return all_samples


def balance_classes(samples, target_per_class=None):
    """
    Balance dataset by oversampling minority classes.

    Args:
        samples: List of sample dictionaries
        target_per_class: Target samples per class (default: median count)

    Returns:
        Balanced list of samples
    """
    # Count per class
    label_counts = Counter(s['label'] for s in samples)

    if target_per_class is None:
        target_per_class = int(np.median(list(label_counts.values())))

    print(f"\nBalancing classes (target: {target_per_class} per class)")

    # Group by label
    by_label = {label: [] for label in range(14)}
    for sample in samples:
        by_label[sample['label']].append(sample)

    # Oversample/undersample to target
    balanced = []
    for label in range(14):
        class_samples = by_label[label]
        count = len(class_samples)

        if count == 0:
            print(f"  Warning: No samples for class {label}")
            continue

        if count >= target_per_class:
            # Undersample
            selected = random.sample(class_samples, target_per_class)
        else:
            # Oversample
            selected = class_samples.copy()
            while len(selected) < target_per_class:
                selected.append(random.choice(class_samples))

        balanced.extend(selected)

    random.shuffle(balanced)
    return balanced


def extract_training_data(handwritten_ratio=0.7, num_per_size=100, balance=True):
    """
    Main function to extract training data from both sources.

    Args:
        handwritten_ratio: Fraction of data from handwritten boards
        num_per_size: Number of puzzles per size to process
        balance: Whether to balance classes

    Returns:
        (images, labels) tuple
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.dirname(base_dir)

    sizes = [3, 4, 5, 6, 7, 9]

    print("Extracting training data...")
    print("=" * 50)

    # Extract from handwritten boards
    print("\n1. Handwritten boards:")
    handwritten_dir = os.path.join(parent_dir, 'KenKen-300px-handwritten', 'board_images')
    handwritten_puzzles = os.path.join(parent_dir, 'KenKen-300px', 'puzzles', 'puzzles_dict.json')

    if os.path.exists(handwritten_dir):
        handwritten_samples = extract_from_directory(
            handwritten_dir, handwritten_puzzles, sizes, num_per_size, 'handwritten'
        )
    else:
        print(f"  Warning: {handwritten_dir} not found")
        handwritten_samples = []

    # Extract from computer-generated boards
    print("\n2. Computer-generated boards:")
    computer_dir = os.path.join(parent_dir, 'KenKen-300px', 'board_images')
    computer_puzzles = os.path.join(parent_dir, 'KenKen-300px', 'puzzles', 'puzzles_dict.json')

    if os.path.exists(computer_dir):
        computer_samples = extract_from_directory(
            computer_dir, computer_puzzles, sizes, num_per_size, 'computer'
        )
    else:
        print(f"  Warning: {computer_dir} not found")
        computer_samples = []

    # Combine based on ratio
    print("\n3. Combining sources...")
    all_samples = []

    # Add handwritten samples
    n_handwritten = int(len(handwritten_samples) * handwritten_ratio / (handwritten_ratio + 0.3))
    if len(handwritten_samples) > n_handwritten:
        handwritten_selected = random.sample(handwritten_samples, n_handwritten)
    else:
        handwritten_selected = handwritten_samples

    all_samples.extend(handwritten_selected)
    print(f"  Handwritten: {len(handwritten_selected)} samples")

    # Add computer samples to make up ratio
    n_computer = int(len(handwritten_selected) * (1 - handwritten_ratio) / handwritten_ratio)
    if len(computer_samples) > n_computer:
        computer_selected = random.sample(computer_samples, n_computer)
    else:
        computer_selected = computer_samples

    all_samples.extend(computer_selected)
    print(f"  Computer: {len(computer_selected)} samples")

    # Balance if requested
    if balance and all_samples:
        all_samples = balance_classes(all_samples)

    # Convert to arrays
    images = [s['image'] for s in all_samples]
    labels = [s['label'] for s in all_samples]

    print(f"\nFinal dataset: {len(images)} samples")
    print(f"Label distribution: {Counter(labels)}")

    return images, labels


def save_extracted_data(output_path, images, labels):
    """Save extracted data to numpy files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path + '_images.npy', np.array(images))
    np.save(output_path + '_labels.npy', np.array(labels))

    print(f"Saved to {output_path}_images.npy and {output_path}_labels.npy")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    # Extract data
    images, labels = extract_training_data(
        handwritten_ratio=0.7,
        num_per_size=100,
        balance=True
    )

    # Save
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'extracted_training'
    )
    save_extracted_data(output_path, images, labels)
