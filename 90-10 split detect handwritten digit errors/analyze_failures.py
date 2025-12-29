# -*- coding: utf-8 -*-
"""
Failure Analysis for Handwritten Puzzle Error Correction.

Analyzes why puzzles fail to be solved by examining:
1. Where the true digit ranks in CNN's probability distribution
2. Common confusion patterns between characters
3. Whether failures are due to K being too small or fundamental CNN errors

Outputs:
- failure_analysis.csv: Detailed per-error analysis
- Summary statistics printed to console
"""

import json
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CNN Model (same as predict_digits.py)
# =============================================================================

class CNN_v2(nn.Module):
    """CNN for digit/character recognition."""

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


def load_model(model_path, num_classes):
    """Load trained CNN model."""
    model = CNN_v2(output_dim=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


# =============================================================================
# Cell Extraction and Recognition
# =============================================================================

def extract_cell(board_img, row, col, cell_size, target_size=28):
    """Extract and preprocess a single cell from board image."""
    margin = 8
    y_start = row * cell_size + margin
    y_end = (row + 1) * cell_size - margin
    x_start = col * cell_size + margin
    x_end = (col + 1) * cell_size - margin

    cell = board_img[y_start:y_end, x_start:x_end]

    cell_pil = Image.fromarray((cell * 255).astype(np.uint8))
    cell_pil = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(cell_pil) / 255.0


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


def recognize_with_full_ranking(cell_img, model, num_classes):
    """
    Get full probability ranking for all classes.

    Returns:
        {
            'prediction': int,           # Top prediction (class index)
            'confidence': float,         # Confidence of top prediction
            'full_ranking': List[Tuple], # [(class, prob), ...] sorted by prob desc
            'all_probs': np.array,       # Raw probability array
        }
    """
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    full_ranking = [(sorted_indices[i].item(), sorted_probs[i].item())
                    for i in range(num_classes)]

    return {
        'prediction': sorted_indices[0].item(),
        'confidence': sorted_probs[0].item(),
        'full_ranking': full_ranking,
        'all_probs': probs.numpy()
    }


def digit_to_display(digit, size):
    """Convert digit to display character."""
    if size <= 9:
        return str(digit) if digit > 0 else '0'
    else:
        # HexaSudoku: 1-9 as digits, 10-16 as A-G
        if digit == 0:
            return '0'
        elif digit <= 9:
            return str(digit)
        else:
            return chr(ord('A') + digit - 10)


def find_true_digit_rank(full_ranking, true_digit):
    """Find where the true digit appears in the ranking (1-indexed)."""
    for rank, (digit, prob) in enumerate(full_ranking, start=1):
        if digit == true_digit:
            return rank, prob
    return len(full_ranking) + 1, 0.0


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_cell(cell_img, model, true_digit, num_classes, size):
    """
    Analyze a single cell's CNN prediction vs ground truth.

    Returns dict with analysis results.
    """
    recog = recognize_with_full_ranking(cell_img, model, num_classes)

    cnn_prediction = recog['prediction']
    cnn_confidence = recog['confidence']
    is_misclassified = (cnn_prediction != true_digit)

    true_digit_rank, true_digit_prob = find_true_digit_rank(recog['full_ranking'], true_digit)

    # Create confusion string
    if is_misclassified:
        pred_display = digit_to_display(cnn_prediction, size)
        true_display = digit_to_display(true_digit, size)
        confusion_type = true_display + "->" + pred_display
    else:
        confusion_type = "correct"

    return {
        'cnn_prediction': cnn_prediction,
        'cnn_confidence': cnn_confidence,
        'true_digit': true_digit,
        'true_digit_rank': true_digit_rank,
        'true_digit_prob': true_digit_prob,
        'is_misclassified': is_misclassified,
        'confusion_type': confusion_type,
        'full_ranking': recog['full_ranking'][:5]  # Top 5 for reference
    }


def analyze_puzzle(image_path, model, ground_truth_puzzle, size, board_pixels, num_classes):
    """
    Analyze all clue cells in a puzzle.

    Returns list of analysis results for misclassified cells.
    """
    cell_size = board_pixels // size

    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    results = []

    for row in range(size):
        for col in range(size):
            true_digit = ground_truth_puzzle[row][col]

            # Skip empty cells in ground truth
            if true_digit == 0:
                continue

            cell = extract_cell(board_img, row, col, cell_size)

            # Skip cells detected as empty by the CNN
            if is_cell_empty(cell):
                # This is an error if ground truth has a digit
                results.append({
                    'position': (row, col),
                    'cnn_prediction': 0,
                    'cnn_confidence': 1.0,
                    'true_digit': true_digit,
                    'true_digit_rank': -1,  # Special: detected as empty
                    'true_digit_prob': 0.0,
                    'is_misclassified': True,
                    'confusion_type': digit_to_display(true_digit, size) + "->empty",
                    'full_ranking': []
                })
                continue

            analysis = analyze_cell(cell, model, true_digit, num_classes, size)
            analysis['position'] = (row, col)

            # Only include misclassified cells
            if analysis['is_misclassified']:
                results.append(analysis)

    return results


def categorize_puzzle_failure(error_analyses, final_status):
    """
    Categorize why a puzzle failed based on error analysis.

    Categories:
    - "no_errors": No misclassifications
    - "all_rank_1-4": All true digits were in top-4
    - "some_rank_5+": At least one true digit ranked 5 or worse
    - "some_rank_10+": At least one true digit ranked 10 or worse
    """
    if not error_analyses:
        return "no_errors"

    ranks = [e['true_digit_rank'] for e in error_analyses if e['true_digit_rank'] > 0]

    if not ranks:
        return "detection_issue"

    max_rank = max(ranks)

    if max_rank <= 4:
        return "all_rank_1-4"
    elif max_rank <= 10:
        return "some_rank_5-10"
    else:
        return "some_rank_11+"


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("Failure Analysis for Handwritten Puzzle Error Correction")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)

    sudoku_model_path = os.path.join(parent_dir, "90-10 split handwritten/Handwritten_Sudoku/models/handwritten_sudoku_cnn.pth")
    hex_model_path = os.path.join(parent_dir, "90-10 split handwritten/Handwritten_HexaSudoku/models/handwritten_hex_cnn.pth")

    sudoku_image_dir = os.path.join(parent_dir, "90-10 split handwritten/Handwritten_Sudoku/board_images")
    hex_image_dir = os.path.join(parent_dir, "90-10 split handwritten/Handwritten_HexaSudoku/board_images")

    puzzles_dir = os.path.join(base_dir, "puzzles")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load existing prediction results to get final status
    prediction_results = {}
    pred_csv_path = os.path.join(results_dir, "prediction_results.csv")
    if os.path.exists(pred_csv_path):
        with open(pred_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['puzzle_type'], int(row['puzzle_idx']))
                prediction_results[key] = row

    all_errors = []
    puzzle_summaries = []

    # ==========================================================================
    # Analyze Sudoku (4x4 and 9x9)
    # ==========================================================================
    if os.path.exists(sudoku_model_path):
        print(f"\n{'='*70}")
        print("Analyzing Sudoku (4x4 and 9x9)")
        print("=" * 70)

        model = load_model(sudoku_model_path, num_classes=10)

        with open(os.path.join(puzzles_dir, "puzzles_dict.json"), "r") as f:
            puzzles_ds = json.load(f)

        for size_str in ['4', '9']:
            size = int(size_str)
            puzzles = puzzles_ds.get(size_str, [])
            puzzle_type = f"sudoku_{size}x{size}"

            print(f"\nAnalyzing {len(puzzles)} {size}x{size} puzzles...")

            for idx, puzzle_data in enumerate(puzzles):
                image_path = os.path.join(sudoku_image_dir, f"board{size}_{idx}.png")

                if not os.path.exists(image_path):
                    continue

                ground_truth = puzzle_data['puzzle']

                errors = analyze_puzzle(
                    image_path, model, ground_truth, size,
                    board_pixels=900, num_classes=10
                )

                # Get final status from prediction results
                key = (puzzle_type, idx)
                final_status = prediction_results.get(key, {}).get('correction_type', 'unknown')

                # Categorize failure
                failure_category = categorize_puzzle_failure(errors, final_status)

                # Add puzzle info to each error
                for err in errors:
                    err['puzzle_type'] = puzzle_type
                    err['puzzle_idx'] = idx
                    err['final_status'] = final_status
                    all_errors.append(err)

                puzzle_summaries.append({
                    'puzzle_type': puzzle_type,
                    'puzzle_idx': idx,
                    'num_errors': len(errors),
                    'final_status': final_status,
                    'failure_category': failure_category,
                    'max_true_rank': max([e['true_digit_rank'] for e in errors], default=0)
                })

            # Print summary for this size
            errors_this_size = [e for e in all_errors if e['puzzle_type'] == puzzle_type]
            print(f"  {size}x{size}: {len(errors_this_size)} total misclassifications")

    # ==========================================================================
    # Analyze HexaSudoku (16x16)
    # ==========================================================================
    if os.path.exists(hex_model_path):
        print(f"\n{'='*70}")
        print("Analyzing HexaSudoku (16x16)")
        print("=" * 70)

        model = load_model(hex_model_path, num_classes=17)

        with open(os.path.join(puzzles_dir, "unique_puzzles_dict.json"), "r") as f:
            puzzles_ds = json.load(f)

        puzzles = puzzles_ds.get('16', [])
        puzzle_type = "hexasudoku_16x16"

        print(f"\nAnalyzing {len(puzzles)} 16x16 puzzles...")

        for idx, puzzle_data in enumerate(puzzles):
            image_path = os.path.join(hex_image_dir, f"board16_{idx}.png")

            if not os.path.exists(image_path):
                continue

            ground_truth = puzzle_data['puzzle']

            errors = analyze_puzzle(
                image_path, model, ground_truth, 16,
                board_pixels=1600, num_classes=17
            )

            # Get final status
            key = (puzzle_type, idx)
            final_status = prediction_results.get(key, {}).get('correction_type', 'unknown')

            # Categorize failure
            failure_category = categorize_puzzle_failure(errors, final_status)

            # Add puzzle info to each error
            for err in errors:
                err['puzzle_type'] = puzzle_type
                err['puzzle_idx'] = idx
                err['final_status'] = final_status
                all_errors.append(err)

            puzzle_summaries.append({
                'puzzle_type': puzzle_type,
                'puzzle_idx': idx,
                'num_errors': len(errors),
                'final_status': final_status,
                'failure_category': failure_category,
                'max_true_rank': max([e['true_digit_rank'] for e in errors], default=0)
            })

            if (idx + 1) % 25 == 0:
                print(f"  Processed {idx+1}/{len(puzzles)} puzzles")

        errors_hex = [e for e in all_errors if e['puzzle_type'] == puzzle_type]
        print(f"  16x16: {len(errors_hex)} total misclassifications")

    # ==========================================================================
    # Generate Statistics
    # ==========================================================================
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 70)

    for ptype in ['sudoku_4x4', 'sudoku_9x9', 'hexasudoku_16x16']:
        errors = [e for e in all_errors if e['puzzle_type'] == ptype]
        summaries = [s for s in puzzle_summaries if s['puzzle_type'] == ptype]

        if not summaries:
            continue

        print(f"\n{ptype}:")
        print(f"  Total puzzles: {len(summaries)}")
        print(f"  Total misclassifications: {len(errors)}")

        if errors:
            # Rank distribution
            rank_counts = Counter()
            for e in errors:
                rank = e['true_digit_rank']
                if rank == -1:
                    rank_counts['empty'] += 1
                elif rank == 1:
                    rank_counts['rank_1'] += 1
                elif rank <= 4:
                    rank_counts['rank_2-4'] += 1
                elif rank <= 10:
                    rank_counts['rank_5-10'] += 1
                else:
                    rank_counts['rank_11+'] += 1

            print(f"\n  True Digit Rank Distribution:")
            for category in ['rank_1', 'rank_2-4', 'rank_5-10', 'rank_11+', 'empty']:
                count = rank_counts.get(category, 0)
                pct = count / len(errors) * 100 if errors else 0
                marker = ""
                if category == 'rank_5-10':
                    marker = " ← Beyond K=4"
                elif category == 'rank_11+':
                    marker = " ← CNN fundamentally wrong"
                print(f"    {category}: {count} ({pct:.1f}%){marker}")

            # Confusion patterns
            confusion_counts = Counter(e['confusion_type'] for e in errors if e['confusion_type'] != 'correct')
            print(f"\n  Top Confusions:")
            for confusion, count in confusion_counts.most_common(10):
                avg_rank = np.mean([e['true_digit_rank'] for e in errors
                                   if e['confusion_type'] == confusion and e['true_digit_rank'] > 0])
                print(f"    {confusion}: {count} occurrences (avg true rank: {avg_rank:.1f})")

        # Puzzle-level failure categories
        uncorrectable = [s for s in summaries if s['final_status'] == 'uncorrectable']
        if uncorrectable:
            print(f"\n  Uncorrectable Puzzles ({len(uncorrectable)}):")
            category_counts = Counter(s['failure_category'] for s in uncorrectable)
            for cat, count in category_counts.most_common():
                print(f"    {cat}: {count}")

    # ==========================================================================
    # Save CSV
    # ==========================================================================
    if all_errors:
        csv_path = os.path.join(results_dir, "failure_analysis.csv")
        fieldnames = [
            'puzzle_type', 'puzzle_idx', 'final_status',
            'position', 'cnn_prediction', 'cnn_confidence',
            'true_digit', 'true_digit_rank', 'true_digit_prob',
            'confusion_type', 'full_ranking'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for err in all_errors:
                row = err.copy()
                row['position'] = str(row['position'])
                row['full_ranking'] = str(row.get('full_ranking', []))
                writer.writerow(row)

        print(f"\nDetailed results saved to {csv_path}")

        # Save puzzle summaries
        summary_csv_path = os.path.join(results_dir, "puzzle_failure_summary.csv")
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=puzzle_summaries[0].keys())
            writer.writeheader()
            writer.writerows(puzzle_summaries)

        print(f"Puzzle summaries saved to {summary_csv_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
