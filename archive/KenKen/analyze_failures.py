# -*- coding: utf-8 -*-
"""
Analyze KenKen solver failures and create visualizations.
Identifies which puzzles fail at baseline vs after correction,
and creates annotated images showing errors in red.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv
import pandas as pd
from z3 import Int, Solver, And, Or, Distinct, Sum, sat, Bool, Then, Tactic
import os
import json
from collections import defaultdict

# Import from solve_all_sizes
BOARD_SIZE = 900
IMG_SIZE = 28
SCALE_FACTOR = 2

# =============================================================================
# Neural Network Models (same as solve_all_sizes.py)
# =============================================================================

class Grid_CNN(nn.Module):
    def __init__(self, output_dim):
        super(Grid_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(262144, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_v2(nn.Module):
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
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load models
def load_models():
    grid_model = Grid_CNN(output_dim=6)
    grid_model.load_state_dict(torch.load('./models/grid_detection_model_weights.pth',
                                          map_location=torch.device('cpu'), weights_only=True))
    grid_model.eval()

    char_model = CNN_v2(output_dim=14)
    char_model.load_state_dict(torch.load('./models/character_recognition_v2_model_weights.pth',
                                          map_location=torch.device('cpu'), weights_only=True))
    char_model.eval()

    return grid_model, char_model


# Import the solver functions
from solve_all_sizes import (
    get_size, find_size_and_borders, make_puzzle, make_puzzle_with_alternatives,
    solve_kenken, attempt_error_correction, LABEL_TO_SIZE
)


def load_ground_truth(size):
    """Load ground truth puzzles from JSON."""
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        puzzles_dict = json.load(f)
    return puzzles_dict.get(str(size), [])


def compare_puzzle_to_ground_truth(detected_puzzle, ground_truth):
    """Compare detected puzzle to ground truth, return differences."""
    errors = []

    gt_cages = {tuple(tuple(c) for c in cage['cells']): cage for cage in ground_truth}

    for det_cage in detected_puzzle:
        cage_key = tuple(tuple(c) for c in det_cage['cells'])
        if cage_key in gt_cages:
            gt_cage = gt_cages[cage_key]
            if det_cage['target'] != gt_cage['target']:
                errors.append({
                    'type': 'target',
                    'cells': det_cage['cells'],
                    'detected': det_cage['target'],
                    'expected': gt_cage['target']
                })
            if det_cage['op'] != gt_cage['op']:
                errors.append({
                    'type': 'operator',
                    'cells': det_cage['cells'],
                    'detected': det_cage['op'],
                    'expected': gt_cage['op']
                })
        else:
            errors.append({
                'type': 'cage_mismatch',
                'cells': det_cage['cells'],
                'detected': det_cage,
                'expected': None
            })

    return errors


def create_error_visualization(image_path, errors, output_path, size):
    """Create visualization with errors highlighted in red."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    cell_size = BOARD_SIZE // size

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        small_font = font

    for error in errors:
        cells = error['cells']
        # Get top-left cell for annotation
        min_row = min(c[0] for c in cells)
        min_col = min(c[1] for c in cells)

        # Draw red rectangle around the cage
        for cell in cells:
            row, col = cell
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Add error annotation
        x = min_col * cell_size + 5
        y = min_row * cell_size + cell_size // 2

        if error['type'] == 'target':
            text = f"Got: {error['detected']}\nExp: {error['expected']}"
        elif error['type'] == 'operator':
            text = f"Op: '{error['detected']}' vs '{error['expected']}'"
        else:
            text = "Cage mismatch"

        # Draw text background
        draw.rectangle([x-2, y-2, x+80, y+35], fill='white', outline='red')
        draw.text((x, y), text, fill='red', font=small_font)

    img.save(output_path)
    return len(errors)


def analyze_size(size, num_puzzles, grid_model, char_model, output_dir):
    """Analyze puzzles of a given size and create error visualizations."""
    print(f"\nAnalyzing {size}x{size} puzzles...")

    ground_truth_list = load_ground_truth(size)

    base_failures = []
    corrected_failures = []
    all_errors = []

    for i in range(min(num_puzzles, len(ground_truth_list))):
        filename = f'./board_images/board{size}_{i}.png'
        if not os.path.exists(filename):
            continue

        ground_truth = ground_truth_list[i]

        # Detect puzzle
        try:
            detected_size, cages, border_thickness = find_size_and_borders(filename, grid_model)
        except Exception as e:
            base_failures.append((i, f"Detection error: {e}"))
            corrected_failures.append((i, f"Detection error: {e}"))
            continue

        if not cages or detected_size != size:
            base_failures.append((i, f"Size/cage detection failed: detected {detected_size}, {len(cages) if cages else 0} cages"))
            corrected_failures.append((i, f"Size/cage detection failed"))
            continue

        # Extract puzzle
        puzzle = make_puzzle(size, border_thickness, cages, filename, char_model, invert=False)

        # Compare to ground truth
        errors = compare_puzzle_to_ground_truth(puzzle, ground_truth)

        # Try to solve
        solution = solve_kenken(puzzle, size)

        if solution is None:
            base_failures.append((i, errors if errors else "Unsolvable (no errors detected)"))

            # Try error correction
            puzzle, alternatives = make_puzzle_with_alternatives(
                size, border_thickness, cages, filename, char_model, k=4, invert=False
            )
            max_errors = 5 if size >= 9 else 3
            correction = attempt_error_correction(puzzle, alternatives, size, max_errors=max_errors, max_k=4)

            if not correction.success:
                corrected_failures.append((i, errors if errors else "Uncorrectable"))
                all_errors.append((i, errors))
        else:
            # Base solved - check if there are still OCR errors (invisible errors)
            if errors:
                all_errors.append((i, errors))

    # Create visualizations
    error_viz_dir = os.path.join(output_dir, f'{size}x{size}_errors')
    os.makedirs(error_viz_dir, exist_ok=True)

    for puzzle_idx, errors in all_errors:
        if errors:
            filename = f'./board_images/board{size}_{puzzle_idx}.png'
            output_path = os.path.join(error_viz_dir, f'board{size}_{puzzle_idx}_errors.png')
            create_error_visualization(filename, errors, output_path, size)

    return {
        'size': size,
        'total': min(num_puzzles, len(ground_truth_list)),
        'base_failures': base_failures,
        'corrected_failures': corrected_failures,
        'base_accuracy': 1 - len(base_failures) / min(num_puzzles, len(ground_truth_list)),
        'corrected_accuracy': 1 - len(corrected_failures) / min(num_puzzles, len(ground_truth_list))
    }


def main():
    print("Loading models...")
    grid_model, char_model = load_models()
    print("Models loaded.")

    output_dir = './failure_analysis'
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for size in [7, 9]:
        result = analyze_size(size, 100, grid_model, char_model, output_dir)
        results[size] = result

        print(f"\n{size}x{size} Results:")
        print(f"  Base accuracy: {result['base_accuracy']*100:.0f}%")
        print(f"  Corrected accuracy: {result['corrected_accuracy']*100:.0f}%")
        print(f"  Base failures ({len(result['base_failures'])}):")
        for idx, err in result['base_failures'][:10]:
            print(f"    Puzzle {idx}: {err}")
        if len(result['base_failures']) > 10:
            print(f"    ... and {len(result['base_failures']) - 10} more")

        print(f"  Still failing after correction ({len(result['corrected_failures'])}):")
        for idx, err in result['corrected_failures']:
            print(f"    Puzzle {idx}: {err}")

    # Save summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        for size, result in results.items():
            f.write(f"\n{size}x{size} Results:\n")
            f.write(f"  Base accuracy: {result['base_accuracy']*100:.0f}%\n")
            f.write(f"  Corrected accuracy: {result['corrected_accuracy']*100:.0f}%\n")
            f.write(f"  Base failures: {[x[0] for x in result['base_failures']]}\n")
            f.write(f"  Corrected failures: {[x[0] for x in result['corrected_failures']]}\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
