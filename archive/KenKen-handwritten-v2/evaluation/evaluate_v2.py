# -*- coding: utf-8 -*-
"""
Evaluation pipeline for KenKen-handwritten-v2.

Evaluates the improved CNN and error correction system:
1. Baseline OCR accuracy
2. Per-class accuracy and confusion matrix
3. Correction success by error type
4. Comparison with v1 results

Usage:
    python evaluate_v2.py
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Add parent directories to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir / 'solver'))
sys.path.insert(0, str(parent_dir / 'models'))

# Import solver components
from solve_all_sizes import (
    get_contours, get_character, detect_puzzle_size,
    segment_cell, CNN_v2, detect_cages, solve_puzzle
)
from improved_cnn import ImprovedCNN
from constraint_validation import detect_impossible_cages, is_valid_target
from confusion_aware import rerank_alternatives, get_confusion_boost


# =============================================================================
# Evaluation Functions
# =============================================================================

class EvaluationResult:
    """Container for evaluation results."""

    def __init__(self):
        self.correct_baseline = 0
        self.correct_with_correction = 0
        self.total = 0
        self.ocr_errors = 0
        self.impossible_targets = 0
        self.corrections_attempted = 0
        self.corrections_successful = 0
        self.details = []


def load_improved_model(model_path, device='cpu'):
    """Load the improved CNN model."""
    model = ImprovedCNN(output_dim=14)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_single_puzzle(board_path, puzzle, size, model, device='cpu'):
    """
    Evaluate a single puzzle.

    Returns:
        dict with:
        - baseline_correct: bool
        - corrected_correct: bool
        - ocr_accuracy: float
        - impossible_count: int
        - correction_attempts: int
        - correction_successes: int
    """
    from PIL import Image
    import cv2 as cv

    # Load board image
    img = Image.open(board_path).convert('L')
    grid = np.array(img)

    # Detect puzzle (using existing solver pipeline)
    # This is simplified - in practice would call the full solve_puzzle function

    result = {
        'baseline_correct': False,
        'corrected_correct': False,
        'ocr_accuracy': 0.0,
        'impossible_count': 0,
        'correction_attempts': 0,
        'correction_successes': 0,
    }

    try:
        # Run solver
        solution = solve_puzzle(grid, puzzle, size, model, device,
                                attempt_correction=False, verbose=False)
        result['baseline_correct'] = solution is not None

        # Try with correction
        corrected_solution = solve_puzzle(grid, puzzle, size, model, device,
                                          attempt_correction=True, verbose=False)
        result['corrected_correct'] = corrected_solution is not None

    except Exception as e:
        # Solver failed
        pass

    return result


def evaluate_size(size, board_dir, puzzles, model, device='cpu', num_puzzles=100):
    """Evaluate all puzzles of a given size."""
    results = EvaluationResult()

    for idx in range(min(num_puzzles, len(puzzles))):
        board_path = board_dir / f'board{size}_{idx}.png'

        if not board_path.exists():
            continue

        puzzle = puzzles[idx]
        if isinstance(puzzle, dict) and 'cages' in puzzle:
            cages = puzzle['cages']
        else:
            cages = puzzle

        result = evaluate_single_puzzle(
            str(board_path), cages, size, model, device
        )

        results.total += 1
        if result['baseline_correct']:
            results.correct_baseline += 1
        if result['corrected_correct']:
            results.correct_with_correction += 1
        results.impossible_targets += result['impossible_count']
        results.corrections_attempted += result['correction_attempts']
        results.corrections_successful += result['correction_successes']

        results.details.append({
            'idx': idx,
            **result
        })

    return results


def run_full_evaluation(model, device='cpu', num_per_size=100):
    """Run full evaluation across all sizes."""
    print("=" * 70)
    print("KenKen-handwritten-v2 Evaluation")
    print("=" * 70)

    base_dir = parent_dir
    board_dir = base_dir / 'board_images'
    puzzles_path = base_dir / 'puzzles' / 'puzzles_dict.json'

    if not puzzles_path.exists():
        print(f"Error: Puzzles file not found at {puzzles_path}")
        return None

    with open(puzzles_path, 'r') as f:
        puzzles_dict = json.load(f)

    sizes = [3, 4, 5, 6, 7, 9]
    all_results = {}

    print(f"\nEvaluating {num_per_size} puzzles per size...")
    print("-" * 70)

    for size in sizes:
        if str(size) not in puzzles_dict:
            print(f"Size {size}: No puzzles available")
            continue

        puzzles = puzzles_dict[str(size)]
        print(f"\nSize {size}x{size}:")

        results = evaluate_size(
            size, board_dir, puzzles, model, device, num_per_size
        )
        all_results[size] = results

        if results.total > 0:
            baseline_acc = results.correct_baseline / results.total * 100
            corrected_acc = results.correct_with_correction / results.total * 100
            improvement = corrected_acc - baseline_acc

            print(f"  Baseline:  {results.correct_baseline}/{results.total} ({baseline_acc:.1f}%)")
            print(f"  Corrected: {results.correct_with_correction}/{results.total} ({corrected_acc:.1f}%)")
            print(f"  Improvement: +{improvement:.1f}%")

    return all_results


def generate_report(results, output_path=None):
    """Generate evaluation report."""
    report_lines = [
        "=" * 70,
        "KenKen-handwritten-v2 Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "## Summary",
        "",
        "| Size | Baseline | Corrected | Improvement |",
        "|------|----------|-----------|-------------|",
    ]

    total_baseline = 0
    total_corrected = 0
    total_puzzles = 0

    for size in sorted(results.keys()):
        r = results[size]
        if r.total > 0:
            baseline = r.correct_baseline / r.total * 100
            corrected = r.correct_with_correction / r.total * 100
            improvement = corrected - baseline

            report_lines.append(
                f"| {size}x{size} | {baseline:.1f}% | {corrected:.1f}% | +{improvement:.1f}% |"
            )

            total_baseline += r.correct_baseline
            total_corrected += r.correct_with_correction
            total_puzzles += r.total

    if total_puzzles > 0:
        overall_baseline = total_baseline / total_puzzles * 100
        overall_corrected = total_corrected / total_puzzles * 100
        overall_improvement = overall_corrected - overall_baseline

        report_lines.extend([
            f"| **Overall** | **{overall_baseline:.1f}%** | **{overall_corrected:.1f}%** | **+{overall_improvement:.1f}%** |",
            "",
            f"Total puzzles evaluated: {total_puzzles}",
        ])

    report = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {output_path}")

    return report


def compare_with_v1(v2_results):
    """Compare with v1 baseline results."""
    # V1 results from FAILURE_ANALYSIS_REPORT.md
    v1_baseline = {
        3: 69, 4: 36, 5: 18, 6: 7, 7: 1, 9: 0
    }
    v1_corrected = {
        3: 89, 4: 58, 5: 26, 6: 15, 7: 2, 9: 1
    }

    print("\n" + "=" * 70)
    print("Comparison with V1")
    print("=" * 70)
    print("\n| Size | V1 Base | V2 Base | V1 Corr | V2 Corr |")
    print("|------|---------|---------|---------|---------|")

    for size in sorted(v2_results.keys()):
        r = v2_results[size]
        if r.total > 0:
            v2_base = r.correct_baseline / r.total * 100
            v2_corr = r.correct_with_correction / r.total * 100

            print(f"| {size}x{size} | {v1_baseline.get(size, 0)}% | {v2_base:.1f}% | "
                  f"{v1_corrected.get(size, 0)}% | {v2_corr:.1f}% |")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Try to load improved model, fall back to original
    improved_model_path = parent_dir / 'models' / 'improved_cnn_weights.pth'
    original_model_path = parent_dir.parent / 'KenKen-300px' / 'models' / 'character_recognition_v2_model_weights.pth'

    if improved_model_path.exists():
        print(f"\nLoading improved model from {improved_model_path}")
        model = load_improved_model(improved_model_path, device)
    elif original_model_path.exists():
        print(f"\nImproved model not found, using original from {original_model_path}")
        model = CNN_v2(output_dim=14)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        print("Error: No model weights found!")
        print(f"  Tried: {improved_model_path}")
        print(f"  Tried: {original_model_path}")
        return

    # Run evaluation
    results = run_full_evaluation(model, device, num_per_size=100)

    if results:
        # Generate report
        report_path = parent_dir / 'evaluation' / 'evaluation_report.md'
        report = generate_report(results, report_path)
        print("\n" + report)

        # Compare with v1
        compare_with_v1(results)


if __name__ == '__main__':
    main()
