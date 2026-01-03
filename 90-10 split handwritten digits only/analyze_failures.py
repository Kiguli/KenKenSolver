# -*- coding: utf-8 -*-
"""
Failure Analysis for Handwritten Puzzle Error Correction (Digits Only).

Analyzes why puzzles fail to be solved by examining:
1. Where the true digit ranks in CNN's probability distribution
2. Common confusion patterns between digits
3. Whether failures are due to K being too small or fundamental CNN errors
4. Specific analysis for two-digit cells (10-16)

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
# Constants
# =============================================================================

SIZE = 16
BOX_SIZE = 4
BOARD_PIXELS = 1600
CELL_SIZE = BOARD_PIXELS // SIZE  # 100px

# Two-digit extraction parameters (matching generate_images.py)
TWO_DIGIT_SIZE = 40
TWO_DIGIT_LEFT_X = 10
TWO_DIGIT_RIGHT_X = 50
TWO_DIGIT_Y = 30

# Cell classification parameters
CELL_MARGIN = 8
SINGLE_INK_THRESHOLD = 0.01
TWO_DIGIT_RATIO_THRESHOLD = 0.15
TWO_DIGIT_MIN_INK = 0.005


# =============================================================================
# CNN Model
# =============================================================================

class CNN_v2(nn.Module):
    """CNN for digit recognition."""

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
# Cell Extraction and Classification
# =============================================================================

def extract_cell(board_img, row, col):
    """Extract a single cell from board image."""
    y_start = row * CELL_SIZE + CELL_MARGIN
    y_end = (row + 1) * CELL_SIZE - CELL_MARGIN
    x_start = col * CELL_SIZE + CELL_MARGIN
    x_end = (col + 1) * CELL_SIZE - CELL_MARGIN

    return board_img[y_start:y_end, x_start:x_end]


def get_ink_density(cell_img):
    """Calculate ink density (dark pixels) in cell."""
    return np.mean(1.0 - cell_img)


def classify_cell_type(cell_img):
    """
    Classify cell as 'empty', 'single', or 'double' digit.

    Key insight: Single digits (70x70) are centered with high ink concentration.
    Double digits (40x40 each) spread ink across the cell with lower center density.
    """
    total_ink = get_ink_density(cell_img)

    if total_ink < SINGLE_INK_THRESHOLD:
        return 'empty'

    h, w = cell_img.shape

    # Check center region ink density (middle 50% of cell)
    center = cell_img[:, int(w*0.25):int(w*0.75)]
    center_ink = get_ink_density(center)

    # Single digits have high center ink density
    # Double digits have lower center ink density
    center_ratio = center_ink / total_ink if total_ink > 0 else 0

    # Threshold at 1.7 separates single vs double digit cells
    if center_ratio > 1.7:
        return 'single'
    else:
        return 'double'


def extract_single_digit(cell_img, target_size=28):
    """Extract single digit from cell, centered."""
    cell_pil = Image.fromarray((cell_img * 255).astype(np.uint8))
    resized = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def extract_left_digit(cell_img, target_size=28):
    """Extract left digit from two-digit cell."""
    h, w = cell_img.shape
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = 0
    x2 = w // 2
    left_region = cell_img[y1:y2, x1:x2]
    cell_pil = Image.fromarray((left_region * 255).astype(np.uint8))
    resized = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


def extract_right_digit(cell_img, target_size=28):
    """Extract right digit from two-digit cell."""
    h, w = cell_img.shape
    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    x1 = w // 2
    x2 = w
    right_region = cell_img[y1:y2, x1:x2]
    cell_pil = Image.fromarray((right_region * 255).astype(np.uint8))
    resized = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return np.array(resized) / 255.0


# =============================================================================
# Recognition with Full Ranking
# =============================================================================

def recognize_digit_with_full_ranking(digit_img, model, num_classes):
    """Get full probability ranking for a single digit image."""
    digit_inverted = 1.0 - digit_img

    with torch.no_grad():
        tensor = torch.tensor(digit_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
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


def find_true_digit_rank(full_ranking, true_digit):
    """Find where the true digit appears in the ranking (1-indexed)."""
    for rank, (digit, prob) in enumerate(full_ranking, start=1):
        if digit == true_digit:
            return rank, prob
    return len(full_ranking) + 1, 0.0


# =============================================================================
# Cell Analysis
# =============================================================================

def analyze_cell(cell_img, model, true_value, num_classes):
    """
    Analyze a single cell's CNN prediction vs ground truth.

    Handles both single-digit (1-9) and two-digit (10-16) cells.
    """
    cell_type = classify_cell_type(cell_img)

    if cell_type == 'empty':
        if true_value != 0:
            return {
                'cnn_prediction': 0,
                'cnn_confidence': 1.0,
                'true_value': true_value,
                'true_digit_rank': -1,  # Special: detected as empty
                'true_digit_prob': 0.0,
                'is_misclassified': True,
                'confusion_type': f"{true_value}->empty",
                'cell_type': 'empty',
                'full_ranking': [],
                'digit_errors': None
            }
        else:
            return None  # Correct empty cell

    elif cell_type == 'single':
        digit_img = extract_single_digit(cell_img)
        recog = recognize_digit_with_full_ranking(digit_img, model, num_classes)

        cnn_pred = recog['prediction']
        if cnn_pred == 10:  # Empty class predicted
            cnn_pred = 0

        is_misclassified = (cnn_pred != true_value)
        true_rank, true_prob = find_true_digit_rank(recog['full_ranking'], true_value)

        if is_misclassified:
            return {
                'cnn_prediction': cnn_pred,
                'cnn_confidence': recog['confidence'],
                'true_value': true_value,
                'true_digit_rank': true_rank,
                'true_digit_prob': true_prob,
                'is_misclassified': True,
                'confusion_type': f"{true_value}->{cnn_pred}",
                'cell_type': 'single',
                'full_ranking': recog['full_ranking'][:5],
                'digit_errors': None
            }
        return None  # Correct

    else:  # double
        left_img = extract_left_digit(cell_img)
        right_img = extract_right_digit(cell_img)

        left_recog = recognize_digit_with_full_ranking(left_img, model, num_classes)
        right_recog = recognize_digit_with_full_ranking(right_img, model, num_classes)

        # Get predicted value
        tens_pred = left_recog['prediction']
        ones_pred = right_recog['prediction']
        if tens_pred == 10:
            tens_pred = 0
        if ones_pred == 10:
            ones_pred = 0

        cnn_pred = tens_pred * 10 + ones_pred

        # Get true tens/ones digits
        true_tens = true_value // 10
        true_ones = true_value % 10

        is_misclassified = (cnn_pred != true_value)

        if is_misclassified:
            # Analyze which digit(s) are wrong
            tens_wrong = (tens_pred != true_tens)
            ones_wrong = (ones_pred != true_ones)

            tens_rank, tens_prob = find_true_digit_rank(left_recog['full_ranking'], true_tens)
            ones_rank, ones_prob = find_true_digit_rank(right_recog['full_ranking'], true_ones)

            # Determine error type
            if tens_wrong and ones_wrong:
                error_type = 'both'
                confusion = f"{true_value}->{cnn_pred} (both digits)"
                max_rank = max(tens_rank, ones_rank)
            elif tens_wrong:
                error_type = 'tens'
                confusion = f"{true_value}->{cnn_pred} (tens: {true_tens}->{tens_pred})"
                max_rank = tens_rank
            else:
                error_type = 'ones'
                confusion = f"{true_value}->{cnn_pred} (ones: {true_ones}->{ones_pred})"
                max_rank = ones_rank

            return {
                'cnn_prediction': cnn_pred,
                'cnn_confidence': (left_recog['confidence'] + right_recog['confidence']) / 2,
                'true_value': true_value,
                'true_digit_rank': max_rank,  # Worst rank among wrong digits
                'true_digit_prob': min(tens_prob, ones_prob),
                'is_misclassified': True,
                'confusion_type': confusion,
                'cell_type': 'double',
                'full_ranking': [],  # Complex for two-digit
                'digit_errors': {
                    'type': error_type,
                    'tens': {
                        'predicted': tens_pred,
                        'true': true_tens,
                        'rank': tens_rank,
                        'wrong': tens_wrong
                    },
                    'ones': {
                        'predicted': ones_pred,
                        'true': true_ones,
                        'rank': ones_rank,
                        'wrong': ones_wrong
                    }
                }
            }

        return None  # Correct


def analyze_puzzle(image_path, model, ground_truth_puzzle, num_classes):
    """Analyze all clue cells in a puzzle."""
    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    results = []
    cell_type_counts = {'single': 0, 'double': 0, 'empty': 0}

    for row in range(SIZE):
        for col in range(SIZE):
            true_value = ground_truth_puzzle[row][col]

            # Skip empty cells in ground truth
            if true_value == 0:
                continue

            cell = extract_cell(board_img, row, col)
            cell_type = classify_cell_type(cell)
            cell_type_counts[cell_type] += 1

            analysis = analyze_cell(cell, model, true_value, num_classes)

            if analysis is not None and analysis['is_misclassified']:
                analysis['position'] = (row, col)
                results.append(analysis)

    return results, cell_type_counts


def categorize_puzzle_failure(error_analyses, final_status):
    """Categorize why a puzzle failed based on error analysis."""
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
    print("Failure Analysis for Handwritten HexaSudoku (Digits Only)")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "models/handwritten_digit_cnn.pth")
    image_dir = os.path.join(base_dir, "board_images")
    puzzles_path = os.path.join(base_dir, "puzzles/unique_puzzles_dict.json")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train_cnn.py first.")
        return

    print(f"\nLoading model from {model_path}")
    model = load_model(model_path, num_classes=11)  # 0-9 + empty

    # Load puzzles
    if not os.path.exists(puzzles_path):
        print(f"Error: Puzzles not found at {puzzles_path}")
        return

    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"Loaded {len(puzzles)} puzzles")

    # Load existing prediction results
    prediction_results = {}
    pred_csv_path = os.path.join(results_dir, "prediction_results.csv")
    if os.path.exists(pred_csv_path):
        with open(pred_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prediction_results[int(row['puzzle_idx'])] = row

    # Analyze puzzles
    print(f"\n{'='*70}")
    print("Analyzing HexaSudoku 16x16 (Digits Only)")
    print("=" * 70)

    all_errors = []
    puzzle_summaries = []
    total_cell_types = {'single': 0, 'double': 0, 'empty': 0}

    for idx, puzzle_data in enumerate(puzzles):
        image_path = os.path.join(image_dir, f"board16_{idx}.png")

        if not os.path.exists(image_path):
            continue

        ground_truth = puzzle_data['puzzle']

        errors, cell_types = analyze_puzzle(
            image_path, model, ground_truth, num_classes=11
        )

        # Accumulate cell type counts
        for ct, count in cell_types.items():
            total_cell_types[ct] += count

        # Get final status
        final_status = prediction_results.get(idx, {}).get('correction_type', 'unknown')

        # Categorize failure
        failure_category = categorize_puzzle_failure(errors, final_status)

        # Add puzzle info to each error
        for err in errors:
            err['puzzle_idx'] = idx
            err['final_status'] = final_status
            all_errors.append(err)

        puzzle_summaries.append({
            'puzzle_idx': idx,
            'num_errors': len(errors),
            'final_status': final_status,
            'failure_category': failure_category,
            'max_true_rank': max([e['true_digit_rank'] for e in errors], default=0),
            'single_digit_cells': cell_types['single'],
            'double_digit_cells': cell_types['double']
        })

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx+1}/{len(puzzles)} puzzles")

    # Generate statistics
    print(f"\n{'='*70}")
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nCell Type Distribution (across all puzzles):")
    print(f"  Single-digit cells: {total_cell_types['single']}")
    print(f"  Double-digit cells: {total_cell_types['double']}")
    print(f"  Empty detected: {total_cell_types['empty']}")

    print(f"\nTotal puzzles: {len(puzzle_summaries)}")
    print(f"Total misclassifications: {len(all_errors)}")

    if all_errors:
        # Rank distribution
        rank_counts = Counter()
        for e in all_errors:
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

        print(f"\nTrue Value Rank Distribution:")
        for category in ['rank_1', 'rank_2-4', 'rank_5-10', 'rank_11+', 'empty']:
            count = rank_counts.get(category, 0)
            pct = count / len(all_errors) * 100
            marker = ""
            if category == 'rank_5-10':
                marker = " <- Beyond K=4"
            elif category == 'rank_11+':
                marker = " <- CNN fundamentally wrong"
            print(f"  {category}: {count} ({pct:.1f}%){marker}")

        # Error type for double-digit cells
        double_errors = [e for e in all_errors if e['cell_type'] == 'double']
        single_errors = [e for e in all_errors if e['cell_type'] == 'single']

        print(f"\nErrors by Cell Type:")
        print(f"  Single-digit cell errors: {len(single_errors)}")
        print(f"  Double-digit cell errors: {len(double_errors)}")

        if double_errors:
            digit_error_types = Counter()
            for e in double_errors:
                if e['digit_errors']:
                    digit_error_types[e['digit_errors']['type']] += 1

            print(f"\n  Double-digit error breakdown:")
            for etype, count in digit_error_types.most_common():
                pct = count / len(double_errors) * 100
                print(f"    {etype} digit wrong: {count} ({pct:.1f}%)")

        # Confusion patterns
        confusion_counts = Counter(e['confusion_type'] for e in all_errors
                                  if e['confusion_type'] != 'correct')
        print(f"\nTop Confusions:")
        for confusion, count in confusion_counts.most_common(15):
            avg_rank = np.mean([e['true_digit_rank'] for e in all_errors
                               if e['confusion_type'] == confusion and e['true_digit_rank'] > 0])
            print(f"  {confusion}: {count} (avg true rank: {avg_rank:.1f})")

    # Puzzle-level failure categories
    uncorrectable = [s for s in puzzle_summaries if s['final_status'] == 'uncorrectable']
    if uncorrectable:
        print(f"\nUncorrectable Puzzles ({len(uncorrectable)}):")
        category_counts = Counter(s['failure_category'] for s in uncorrectable)
        for cat, count in category_counts.most_common():
            print(f"  {cat}: {count}")

        # Error count distribution for uncorrectable
        error_counts = Counter(s['num_errors'] for s in uncorrectable)
        print(f"\n  Errors per uncorrectable puzzle:")
        for n_errors in sorted(error_counts.keys()):
            count = error_counts[n_errors]
            print(f"    {n_errors} errors: {count} puzzles")

    # Save CSV
    if all_errors:
        csv_path = os.path.join(results_dir, "failure_analysis.csv")
        fieldnames = [
            'puzzle_idx', 'final_status', 'position',
            'cnn_prediction', 'cnn_confidence', 'true_value',
            'true_digit_rank', 'true_digit_prob',
            'confusion_type', 'cell_type', 'digit_errors'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for err in all_errors:
                row = err.copy()
                row['position'] = str(row['position'])
                row['digit_errors'] = str(row.get('digit_errors', ''))
                writer.writerow(row)

        print(f"\nDetailed results saved to {csv_path}")

        # Save puzzle summaries
        if puzzle_summaries:
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
