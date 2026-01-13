#!/usr/bin/env python3
"""
Full evaluation of optimized KenKen solver on 3x3-7x7 puzzles.
Uses the unified solver from solve_all_sizes.py with:
1. Pre-filled singletons
2. Integer-only division
3. Domain tightening
4. Solver tactics
5. Operator inference for cages with missing/incorrect operators
"""

import json
import time
import sys
import os
import pandas as pd

# Change to script directory first
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import from solve_all_sizes
from solve_all_sizes import (
    Grid_CNN, CNN_v2,
    find_size_and_borders, make_puzzle,
    solve_kenken, solve_with_operator_inference
)
import torch


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 70)
    print("KenKen Solver Full Evaluation (3x3 to 7x7) - With Operator Inference")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")
    character_model = CNN_v2(output_dim=14)
    state_dict = torch.load('./models/character_recognition_v2_model_weights.pth', weights_only=False)
    character_model.load_state_dict(state_dict)
    character_model.eval()

    grid_detection = Grid_CNN(output_dim=6)
    state_dict = torch.load('./models/grid_detection_model_weights.pth', weights_only=False)
    grid_detection.load_state_dict(state_dict)
    grid_detection.eval()
    print("  Models loaded successfully")
    print()

    # Load puzzle data
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    # Results tracking
    results = []
    accuracy = {}
    avg_time = {}

    # Run evaluation for sizes 3-7
    for size in range(3, 8):
        print(f"Evaluating {size}x{size} puzzles...")
        size_start = time.time()
        solved_count = 0
        puzzle_count = len(puzzles_ds[str(size)])

        for i in range(puzzle_count):
            filepath = f"./board_images/board{size}_{i}.png"
            start = time.time()
            correction_type = "none"

            try:
                s, cages, b_t = find_size_and_borders(filepath, grid_detection)
                puzzle = make_puzzle(s, b_t, cages, filepath, character_model, invert=False)

                # Try basic solve first
                solution = solve_kenken(puzzle, s)

                if solution is None:
                    # Try operator inference
                    solution, inferences = solve_with_operator_inference(puzzle, s)
                    if solution is not None:
                        correction_type = "operator_inference"

                if solution:
                    solved_count += 1
                    solved = True
                else:
                    solved = False
            except Exception as e:
                solved = False
                correction_type = f"error: {str(e)[:30]}"

            end = time.time()
            time_ms = (end - start) * 1000

            results.append({
                'size': size,
                'puzzle_index': i,
                'solved': solved,
                'correction_type': correction_type,
                'solve_time_ms': time_ms
            })

            # Progress indicator every 10 puzzles
            if (i + 1) % 10 == 0 or i == puzzle_count - 1:
                print(f"  [{i+1}/{puzzle_count}] solved: {solved_count}")
                sys.stdout.flush()

        size_end = time.time()
        size_time = size_end - size_start

        accuracy[size] = solved_count / puzzle_count * 100
        avg_time[size] = size_time / puzzle_count * 1000

        print(f"  Done: {solved_count}/{puzzle_count} ({accuracy[size]:.1f}%) - Avg time: {avg_time[size]:.0f}ms")
        print()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('./results/optimized_evaluation.csv', index=False)

    # Save summary
    summary_df = pd.DataFrame({
        'size': list(accuracy.keys()),
        'accuracy': [accuracy[s] for s in accuracy.keys()],
        'avg_time_ms': [avg_time[s] for s in avg_time.keys()]
    })
    summary_df.to_csv('./results/optimized_summary.csv', index=False)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Size':<6} {'Accuracy':<12} {'Avg Time':<12}")
    print("-" * 30)
    for size in range(3, 8):
        print(f"{size}x{size:<4} {accuracy[size]:.1f}%{'':<6} {avg_time[size]:.0f}ms")
    print()
    print(f"Results saved to ./results/optimized_evaluation.csv")
    print(f"Summary saved to ./results/optimized_summary.csv")


if __name__ == "__main__":
    main()
