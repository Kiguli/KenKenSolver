# -*- coding: utf-8 -*-
"""
Analyze KenKen handwritten solver failures.
Compares detected puzzle to ground truth, identifies error types,
and creates visualizations with errors highlighted in red.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# Add parent KenKen directory to import solver functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'KenKen'))
from solve_all_sizes import (
    find_size_and_borders, make_puzzle,
    Grid_CNN, CNN_v2
)

BOARD_SIZE = 900


def load_models():
    """Load Grid CNN and our custom Character CNN."""
    grid_model = Grid_CNN(output_dim=6)
    grid_model.load_state_dict(torch.load('./models/grid_detection_model_weights.pth',
                                          map_location='cpu', weights_only=True))
    grid_model.eval()

    char_model = CNN_v2(output_dim=14)
    char_model.load_state_dict(torch.load('./models/unified_kenken_cnn.pth',
                                          map_location='cpu', weights_only=True))
    char_model.eval()

    return grid_model, char_model


def load_ground_truth():
    """Load ground truth puzzles from JSON."""
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        data = json.load(f)

    # Normalize structure: 9x9 has 'cages' key, others are just lists
    normalized = {}
    for size_key, puzzles in data.items():
        normalized[size_key] = []
        for puzzle in puzzles:
            if isinstance(puzzle, dict) and 'cages' in puzzle:
                # 9x9 format: {'cages': [...], 'solution': [...]}
                normalized[size_key].append(puzzle['cages'])
            else:
                # 3-7 format: [cage1, cage2, ...]
                normalized[size_key].append(puzzle)

    return normalized


def compare_puzzle_to_ground_truth(detected_puzzle, ground_truth):
    """
    Compare detected puzzle to ground truth, return differences.

    Returns list of errors with type, cells, detected value, expected value.
    """
    errors = []

    # Create lookup by cells (normalized to sorted tuple of tuples)
    def normalize_cells(cells):
        return tuple(sorted(tuple(c) for c in cells))

    gt_cages = {normalize_cells(cage['cells']): cage for cage in ground_truth}
    detected_cages = {normalize_cells(cage['cells']): cage for cage in detected_puzzle}

    # Check each detected cage against ground truth
    for cage_key, det_cage in detected_cages.items():
        if cage_key in gt_cages:
            gt_cage = gt_cages[cage_key]

            # Check target value
            if det_cage['target'] != gt_cage['target']:
                errors.append({
                    'type': 'target',
                    'cells': det_cage['cells'],
                    'detected': det_cage['target'],
                    'expected': gt_cage['target']
                })

            # Check operator
            if det_cage['op'] != gt_cage['op']:
                errors.append({
                    'type': 'operator',
                    'cells': det_cage['cells'],
                    'detected': det_cage['op'],
                    'expected': gt_cage['op']
                })
        else:
            # Cage cells don't match any ground truth cage
            target = det_cage['target']
            op = det_cage['op']
            errors.append({
                'type': 'cage_boundary',
                'cells': det_cage['cells'],
                'detected': f"{target}{op}",
                'expected': 'Different cage boundaries'
            })

    # Check for missing cages (in ground truth but not detected)
    for cage_key, gt_cage in gt_cages.items():
        if cage_key not in detected_cages:
            target = gt_cage['target']
            op = gt_cage['op']
            errors.append({
                'type': 'missing_cage',
                'cells': gt_cage['cells'],
                'detected': None,
                'expected': f"{target}{op}"
            })

    return errors


def create_error_visualization(image_path, errors, output_path, size):
    """Create visualization with errors highlighted in red."""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    cell_size = BOARD_SIZE // size

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()

    for error in errors:
        cells = error['cells']

        # Get top-left cell for annotation
        min_row = min(c[0] for c in cells)
        min_col = min(c[1] for c in cells)

        # Draw red rectangle around error cells
        for cell in cells:
            row, col = cell
            x1 = col * cell_size + 2
            y1 = row * cell_size + 2
            x2 = x1 + cell_size - 4
            y2 = y1 + cell_size - 4
            draw.rectangle([x1, y1, x2, y2], outline='red', width=4)

        # Add error annotation
        x = min_col * cell_size + 5
        y = min_row * cell_size + cell_size // 2 + 10

        if error['type'] == 'target':
            text = f"Got:{error['detected']} Exp:{error['expected']}"
        elif error['type'] == 'operator':
            text = f"Op:'{error['detected']}' vs '{error['expected']}'"
        elif error['type'] == 'cage_boundary':
            text = "Wrong cage"
        else:
            text = "Missing"

        # Draw text with background
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill='white', outline='red')
        draw.text((x, y), text, fill='red', font=font)

    img.save(output_path)


def analyze_puzzles(sizes, num_per_size, grid_model, char_model):
    """Analyze all puzzles and identify errors."""
    ground_truth = load_ground_truth()

    all_results = []
    error_summary = defaultdict(lambda: {'target': 0, 'operator': 0, 'cage_boundary': 0, 'missing_cage': 0})
    confusion_matrix = defaultdict(lambda: defaultdict(int))  # expected -> detected -> count

    os.makedirs('./failure_analysis', exist_ok=True)

    for size in sizes:
        print(f"\nAnalyzing {size}x{size} puzzles...")

        gt_puzzles = ground_truth.get(str(size), [])
        error_viz_dir = f'./failure_analysis/{size}x{size}_errors'
        os.makedirs(error_viz_dir, exist_ok=True)

        puzzles_with_errors = 0
        total_errors = 0

        for idx in range(min(num_per_size, len(gt_puzzles))):
            filename = f'./board_images/board{size}_{idx}.png'
            if not os.path.exists(filename):
                continue

            try:
                detected_size, cages, border_thickness = find_size_and_borders(filename, grid_model)

                if not cages or detected_size != size:
                    all_results.append({
                        'puzzle_id': f'{size}_{idx}',
                        'size': size,
                        'error_type': 'detection_failed',
                        'error_count': 1,
                        'details': f'Size: {detected_size}, Cages: {len(cages) if cages else 0}'
                    })
                    continue

                puzzle = make_puzzle(size, border_thickness, cages, filename, char_model, invert=False)
                gt_puzzle = gt_puzzles[idx]

                errors = compare_puzzle_to_ground_truth(puzzle, gt_puzzle)

                if errors:
                    puzzles_with_errors += 1
                    total_errors += len(errors)

                    # Track error types
                    for error in errors:
                        error_summary[size][error['type']] += 1

                        # Track confusion matrix for target errors
                        if error['type'] == 'target':
                            confusion_matrix[error['expected']][error['detected']] += 1

                    # Create visualization
                    output_path = os.path.join(error_viz_dir, f'board{size}_{idx}_errors.png')
                    create_error_visualization(filename, errors, output_path, size)

                    all_results.append({
                        'puzzle_id': f'{size}_{idx}',
                        'size': size,
                        'error_type': 'ocr_errors',
                        'error_count': len(errors),
                        'details': ', '.join([f"{e['type']}:{e['detected']}->{e['expected']}" for e in errors])
                    })
                else:
                    all_results.append({
                        'puzzle_id': f'{size}_{idx}',
                        'size': size,
                        'error_type': 'none',
                        'error_count': 0,
                        'details': ''
                    })

            except Exception as e:
                all_results.append({
                    'puzzle_id': f'{size}_{idx}',
                    'size': size,
                    'error_type': 'exception',
                    'error_count': 1,
                    'details': str(e)[:100]
                })

        print(f"  Puzzles with errors: {puzzles_with_errors}")
        print(f"  Total errors: {total_errors}")
        print(f"  Error breakdown: {dict(error_summary[size])}")

    return all_results, dict(error_summary), dict(confusion_matrix)


def main():
    print("Loading models...")
    grid_model, char_model = load_models()
    print("Models loaded.")

    sizes = [3, 4, 5, 6, 7, 9]
    num_per_size = 100

    print("\nAnalyzing puzzles for errors...")
    results, error_summary, confusion_matrix = analyze_puzzles(sizes, num_per_size, grid_model, char_model)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('./failure_analysis/error_details.csv', index=False)

    # Create summary report
    print("\n" + "="*70)
    print("FAILURE ANALYSIS SUMMARY")
    print("="*70)

    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        total = len(size_results)
        with_errors = sum(1 for r in size_results if r['error_count'] > 0)
        total_errors = sum(r['error_count'] for r in size_results)

        print(f"\n{size}x{size}:")
        print(f"  Puzzles analyzed: {total}")
        print(f"  Puzzles with errors: {with_errors} ({100*with_errors/total:.0f}%)")
        print(f"  Total errors: {total_errors}")

        if size in error_summary:
            print(f"  Error types:")
            for error_type, count in error_summary[size].items():
                print(f"    {error_type}: {count}")

    # Print confusion matrix for target digit errors
    print("\n" + "="*70)
    print("DIGIT CONFUSION MATRIX (Expected -> Detected)")
    print("="*70)

    if confusion_matrix:
        all_digits = sorted(set(list(confusion_matrix.keys()) +
                               [d for exp in confusion_matrix.values() for d in exp.keys()]))

        # Header
        header = "Exp\\Det" + "".join(f"{d:>5}" for d in all_digits[:15])
        print(header)

        for expected in all_digits[:15]:
            row = f"{expected:>7}"
            for detected in all_digits[:15]:
                count = confusion_matrix.get(expected, {}).get(detected, 0)
                row += f"{count:>5}" if count > 0 else "    ."
            print(row)

    # Save summary
    with open('./failure_analysis/summary.txt', 'w') as f:
        f.write("KENKEN HANDWRITTEN FAILURE ANALYSIS\n")
        f.write("="*50 + "\n\n")

        for size in sizes:
            size_results = [r for r in results if r['size'] == size]
            total = len(size_results)
            with_errors = sum(1 for r in size_results if r['error_count'] > 0)
            total_errors = sum(r['error_count'] for r in size_results)

            f.write(f"\n{size}x{size}:\n")
            f.write(f"  Puzzles analyzed: {total}\n")
            f.write(f"  Puzzles with errors: {with_errors} ({100*with_errors/total:.0f}%)\n")
            f.write(f"  Total errors: {total_errors}\n")

            if size in error_summary:
                f.write(f"  Error types:\n")
                for error_type, count in error_summary[size].items():
                    f.write(f"    {error_type}: {count}\n")

    print(f"\nResults saved to ./failure_analysis/")
    print("  - error_details.csv: Per-puzzle error breakdown")
    print("  - summary.txt: Analysis summary")
    print("  - {size}x{size}_errors/: Visualization images")


if __name__ == '__main__':
    main()
