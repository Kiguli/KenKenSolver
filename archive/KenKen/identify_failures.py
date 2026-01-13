# -*- coding: utf-8 -*-
"""
Identify KenKen solver failures and create visualizations with errors highlighted in red.
"""

import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import sys

# Add current dir to path
sys.path.insert(0, '.')

from solve_all_sizes import (
    Grid_CNN, CNN_v2, find_size_and_borders, make_puzzle, make_puzzle_with_alternatives,
    solve_kenken, attempt_error_correction, BOARD_SIZE, LABEL_TO_SIZE
)
from torchvision import transforms

grid_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def load_models():
    """Load the CNN models."""
    grid_model = Grid_CNN(output_dim=6)
    grid_model.load_state_dict(torch.load('./models/grid_detection_model_weights.pth',
                                          map_location=torch.device('cpu'), weights_only=True))
    grid_model.eval()

    char_model = CNN_v2(output_dim=14)
    char_model.load_state_dict(torch.load('./models/character_recognition_v2_model_weights.pth',
                                          map_location=torch.device('cpu'), weights_only=True))
    char_model.eval()

    return grid_model, char_model


def load_ground_truth(size):
    """Load ground truth puzzles from JSON."""
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        puzzles_dict = json.load(f)
    puzzles = puzzles_dict.get(str(size), [])
    # Handle both formats: list of cages directly, or list of dicts with 'cages' key
    if puzzles and isinstance(puzzles[0], dict) and 'cages' in puzzles[0]:
        return [p['cages'] for p in puzzles]
    return puzzles


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

    # Check if any ground truth cages are missing
    det_cages = {tuple(tuple(c) for c in cage['cells']): cage for cage in detected_puzzle}
    for gt_key, gt_cage in gt_cages.items():
        if gt_key not in det_cages:
            errors.append({
                'type': 'missing_cage',
                'cells': [list(c) for c in gt_key],
                'detected': None,
                'expected': gt_cage
            })

    return errors


def create_error_visualization(image_path, errors, output_path, size, detected_puzzle, ground_truth):
    """Create visualization with errors highlighted in red."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    cell_size = BOARD_SIZE // size

    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    for error in errors:
        cells = error['cells']
        if not cells:
            continue

        # Draw red rectangle around each cell in the cage
        for cell in cells:
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                row, col = cell
                x1 = col * cell_size + 2
                y1 = row * cell_size + 2
                x2 = (col + 1) * cell_size - 2
                y2 = (row + 1) * cell_size - 2
                draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

        # Add error annotation at top-left cell
        if cells:
            first_cell = cells[0]
            if isinstance(first_cell, (list, tuple)) and len(first_cell) == 2:
                row, col = first_cell
                x = col * cell_size + 5
                y = row * cell_size + cell_size // 2 + 10

                if error['type'] == 'target':
                    text = f"{error['detected']}->{error['expected']}"
                elif error['type'] == 'operator':
                    det_op = error['detected'] if error['detected'] else '?'
                    exp_op = error['expected'] if error['expected'] else '?'
                    text = f"op:{det_op}->{exp_op}"
                elif error['type'] == 'cage_mismatch':
                    text = "CAGE ERR"
                else:
                    text = "MISSING"

                # Draw text with white background
                bbox = draw.textbbox((x, y), text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white')
                draw.text((x, y), text, fill='red', font=font)

    img.save(output_path)


def analyze_puzzles(size, num_puzzles, grid_model, char_model, output_dir):
    """Analyze puzzles and identify failures."""
    print(f"\nAnalyzing {size}x{size} puzzles...")

    ground_truth_list = load_ground_truth(size)
    if not ground_truth_list:
        print(f"  No ground truth found for size {size}")
        return None

    base_failures = []
    correction_needed = []  # Puzzles that failed base but were corrected
    still_failing = []  # Puzzles that failed even after correction

    error_dir = os.path.join(output_dir, f'{size}x{size}_errors')
    os.makedirs(error_dir, exist_ok=True)

    for i in range(min(num_puzzles, len(ground_truth_list))):
        filename = f'./board_images/board{size}_{i}.png'
        if not os.path.exists(filename):
            print(f"  Puzzle {i}: File not found")
            continue

        ground_truth = ground_truth_list[i]

        # Detect puzzle
        try:
            detected_size, cages, border_thickness = find_size_and_borders(filename, grid_model)
        except Exception as e:
            base_failures.append((i, f"Detection error: {e}", []))
            still_failing.append((i, f"Detection error: {e}", []))
            continue

        if not cages or detected_size != size:
            err_msg = f"Size/cage detection failed: detected {detected_size}, {len(cages) if cages else 0} cages"
            base_failures.append((i, err_msg, []))
            still_failing.append((i, err_msg, []))
            continue

        # Extract puzzle
        puzzle = make_puzzle(size, border_thickness, cages, filename, char_model, invert=False)

        # Compare to ground truth
        errors = compare_puzzle_to_ground_truth(puzzle, ground_truth)

        # Try to solve
        solution = solve_kenken(puzzle, size)

        if solution is None:
            base_failures.append((i, "Unsolvable", errors))

            # Try error correction
            puzzle_alt, alternatives = make_puzzle_with_alternatives(
                size, border_thickness, cages, filename, char_model, k=4, invert=False
            )
            max_errors = 5 if size >= 9 else 3
            correction = attempt_error_correction(puzzle_alt, alternatives, size, max_errors=max_errors, max_k=4)

            if correction.success:
                correction_needed.append((i, correction.correction_type, errors))
            else:
                still_failing.append((i, "Uncorrectable", errors))

            # Create visualization for this failure
            if errors:
                output_path = os.path.join(error_dir, f'board{size}_{i}_errors.png')
                create_error_visualization(filename, errors, output_path, size, puzzle, ground_truth)

    total = min(num_puzzles, len(ground_truth_list))
    base_accuracy = (total - len(base_failures)) / total * 100
    corrected_accuracy = (total - len(still_failing)) / total * 100

    return {
        'size': size,
        'total': total,
        'base_failures': base_failures,
        'correction_needed': correction_needed,
        'still_failing': still_failing,
        'base_accuracy': base_accuracy,
        'corrected_accuracy': corrected_accuracy
    }


def main():
    print("Loading models...")
    grid_model, char_model = load_models()
    print("Models loaded.")

    output_dir = './failure_analysis'
    os.makedirs(output_dir, exist_ok=True)

    for size in [7, 9]:
        result = analyze_puzzles(size, 100, grid_model, char_model, output_dir)
        if result is None:
            continue

        print(f"\n{'='*60}")
        print(f"{size}x{size} RESULTS")
        print(f"{'='*60}")
        print(f"Base accuracy: {result['base_accuracy']:.0f}%")
        print(f"Corrected accuracy: {result['corrected_accuracy']:.0f}%")

        print(f"\nBase failures ({len(result['base_failures'])} puzzles):")
        for idx, reason, errors in result['base_failures']:
            err_summary = ""
            if errors:
                err_details = []
                for e in errors:
                    if e['type'] == 'target':
                        err_details.append(f"target:{e['detected']}->{e['expected']}")
                    elif e['type'] == 'operator':
                        err_details.append(f"op:{e['detected']}->{e['expected']}")
                    else:
                        err_details.append(e['type'])
                err_summary = f" [{', '.join(err_details)}]"
            print(f"  Puzzle {idx}: {reason}{err_summary}")

        print(f"\nCorrected (needed error correction): {len(result['correction_needed'])} puzzles")
        for idx, correction_type, errors in result['correction_needed'][:10]:
            print(f"  Puzzle {idx}: Fixed with {correction_type}")
        if len(result['correction_needed']) > 10:
            print(f"  ... and {len(result['correction_needed']) - 10} more")

        print(f"\nStill failing after correction: {len(result['still_failing'])} puzzles")
        for idx, reason, errors in result['still_failing']:
            print(f"  Puzzle {idx}: {reason}")

        # Save detailed report
        report_path = os.path.join(output_dir, f'{size}x{size}_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"{size}x{size} KenKen Failure Analysis\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Base accuracy: {result['base_accuracy']:.0f}%\n")
            f.write(f"Corrected accuracy: {result['corrected_accuracy']:.0f}%\n\n")

            f.write(f"Base failures ({len(result['base_failures'])} puzzles):\n")
            for idx, reason, errors in result['base_failures']:
                f.write(f"  Puzzle {idx}: {reason}\n")
                for err in errors:
                    f.write(f"    - {err}\n")

            f.write(f"\nCorrected with error correction ({len(result['correction_needed'])} puzzles):\n")
            for idx, correction_type, errors in result['correction_needed']:
                f.write(f"  Puzzle {idx}: Fixed with {correction_type}\n")

            f.write(f"\nStill failing ({len(result['still_failing'])} puzzles):\n")
            for idx, reason, errors in result['still_failing']:
                f.write(f"  Puzzle {idx}: {reason}\n")

    print(f"\nError visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
