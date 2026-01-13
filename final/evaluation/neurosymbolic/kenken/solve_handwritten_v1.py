# -*- coding: utf-8 -*-
"""
Baseline evaluation of KenKen solver on handwritten digit board images.
Pipeline: Grid_CNN → OpenCV cages → Character CNN → Z3 Solver
No error correction applied - this measures raw pipeline accuracy.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image
from z3 import Int, Solver, And, Or, Distinct, Sum, sat, Then, Tactic

# Add parent KenKen directory to import solver functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'KenKen'))
from solve_all_sizes import (
    find_size_and_borders, make_puzzle, solve_kenken,
    Grid_CNN, CNN_v2, LABEL_TO_SIZE
)

# Constants
BOARD_SIZE = 900
IMG_SIZE = 28


def load_models():
    """Load Grid CNN and our custom Character CNN."""
    grid_model = Grid_CNN(output_dim=6)
    grid_model.load_state_dict(torch.load('./models/grid_detection_model_weights.pth',
                                          map_location='cpu', weights_only=True))
    grid_model.eval()

    # Use our unified handwritten model instead of the original
    char_model = CNN_v2(output_dim=14)
    char_model.load_state_dict(torch.load('./models/unified_kenken_cnn.pth',
                                          map_location='cpu', weights_only=True))
    char_model.eval()

    return grid_model, char_model


def evaluate_puzzles(sizes, num_per_size, grid_model, char_model):
    """Run baseline evaluation on all puzzles."""
    results = []

    for size in sizes:
        print(f"\nEvaluating {size}x{size} puzzles...")

        solved_count = 0
        for idx in range(num_per_size):
            filename = f'./board_images/board{size}_{idx}.png'
            if not os.path.exists(filename):
                continue

            start_time = time.time()

            try:
                detected_size, cages, border_thickness = find_size_and_borders(filename, grid_model)

                size_correct = (detected_size == size)
                num_cages = len(cages)

                if not size_correct or not cages:
                    results.append({
                        'puzzle_id': f'{size}_{idx}',
                        'size': size,
                        'size_detected': detected_size,
                        'size_correct': size_correct,
                        'num_cages': num_cages,
                        'solved': False,
                        'solve_time_ms': int((time.time() - start_time) * 1000),
                        'error': 'size_detection' if not size_correct else 'no_cages'
                    })
                    continue

                puzzle = make_puzzle(size, border_thickness, cages, filename, char_model, invert=False)
                solution = solve_kenken(puzzle, size)

                solved = solution is not None
                if solved:
                    solved_count += 1

                results.append({
                    'puzzle_id': f'{size}_{idx}',
                    'size': size,
                    'size_detected': detected_size,
                    'size_correct': size_correct,
                    'num_cages': num_cages,
                    'solved': solved,
                    'solve_time_ms': int((time.time() - start_time) * 1000),
                    'error': None if solved else 'unsolvable'
                })

            except Exception as e:
                results.append({
                    'puzzle_id': f'{size}_{idx}',
                    'size': size,
                    'size_detected': None,
                    'size_correct': False,
                    'num_cages': 0,
                    'solved': False,
                    'solve_time_ms': int((time.time() - start_time) * 1000),
                    'error': str(e)
                })

        print(f"  {size}x{size}: {solved_count}/{num_per_size} solved ({100*solved_count/num_per_size:.0f}%)")

    return results


def main():
    print("Loading models...")
    grid_model, char_model = load_models()
    print("Models loaded.")

    sizes = [3, 4, 5, 6, 7, 9]
    num_per_size = 100

    print("\nRunning baseline evaluation...")
    results = evaluate_puzzles(sizes, num_per_size, grid_model, char_model)

    # Save results
    os.makedirs('./results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('./results/baseline_evaluation.csv', index=False)

    # Create summary
    summary = []
    for size in sizes:
        size_results = [r for r in results if r['size'] == size]
        total = len(size_results)
        solved = sum(1 for r in size_results if r['solved'])
        summary.append({
            'size': size,
            'total': total,
            'solved': solved,
            'accuracy': f"{100*solved/total:.0f}%" if total > 0 else "N/A"
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('./results/baseline_summary.csv', index=False)

    print("\n" + "="*60)
    print("BASELINE EVALUATION RESULTS")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("\nResults saved to ./results/")


if __name__ == '__main__':
    main()
