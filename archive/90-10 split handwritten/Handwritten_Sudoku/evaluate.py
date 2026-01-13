"""
Evaluate handwritten Sudoku puzzles using the neuro-symbolic pipeline.

Pipeline: Image → CNN (digit recognition) → Z3 (constraint solving) → Solution

This evaluates the model's ability to recognize real handwritten digits
(from MNIST test set) that were never seen during training.

Supports two puzzle sizes:
- 4×4: 2×2 boxes
- 9×9: 3×3 boxes
"""

import json
import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from z3 import Int, Solver, And, Distinct, sat

# Constants
BOARD_PIXELS = 900
NUM_CLASSES = 10  # 0-9


class CNN_v2(nn.Module):
    """CNN for digit recognition (same architecture as training)."""

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


def load_model(model_path):
    """Load trained CNN model."""
    model = CNN_v2(output_dim=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def get_cell_size(size):
    """Get cell size in pixels for a given puzzle size."""
    return BOARD_PIXELS // size


def get_box_size(size):
    """Get box size for a given puzzle size."""
    return 2 if size == 4 else 3


def extract_cell(board_img, row, col, size, target_size=28):
    """
    Extract and preprocess a single cell from board image.

    Args:
        board_img: numpy array of board (0-1 float, 0=black, 1=white)
        row: Row index
        col: Column index
        size: Grid size (4 or 9)
        target_size: Output size (default 28x28 for CNN)

    Returns:
        numpy array of shape (target_size, target_size) with values in [0,1]
    """
    cell_size = get_cell_size(size)
    margin = 8  # Avoid grid lines

    y_start = row * cell_size + margin
    y_end = (row + 1) * cell_size - margin
    x_start = col * cell_size + margin
    x_end = (col + 1) * cell_size - margin

    cell = board_img[y_start:y_end, x_start:x_end]

    # Resize to target size
    cell_pil = Image.fromarray((cell * 255).astype(np.uint8))
    cell_pil = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(cell_pil) / 255.0


def is_cell_empty(cell_img, threshold=0.98):
    """
    Check if cell is empty (mostly white).

    Args:
        cell_img: numpy array with values in [0,1]
        threshold: Mean intensity threshold (higher = whiter)

    Returns:
        True if cell appears empty
    """
    return np.mean(cell_img) > threshold


def recognize_digit(cell_img, model):
    """
    Recognize digit in cell using CNN.

    Args:
        cell_img: numpy array of cell (0-1, 0=black ink, 1=white bg)
        model: Trained CNN model

    Returns:
        Predicted class (0-9)
    """
    # Invert so ink is high value (MNIST convention for training)
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction


def extract_puzzle_from_image(image_path, model, size):
    """
    Extract puzzle values from board image using CNN.

    Args:
        image_path: Path to board image
        model: Trained CNN model
        size: Grid size (4 or 9)

    Returns:
        size×size list of predicted values (0 = empty, 1-9 = digits)
    """
    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * size for _ in range(size)]

    for row in range(size):
        for col in range(size):
            cell = extract_cell(board_img, row, col, size)

            if is_cell_empty(cell):
                puzzle[row][col] = 0
            else:
                predicted = recognize_digit(cell, model)
                puzzle[row][col] = predicted

    return puzzle


def solve_sudoku(size, given_cells):
    """
    Solve Sudoku using Z3 constraint solver.

    Args:
        size: Grid size (4 or 9)
        given_cells: dict of {(row, col): value} for known cells

    Returns:
        size×size solution grid, or None if unsatisfiable
    """
    box_size = get_box_size(size)
    X = [[Int(f"x_{i}_{j}") for j in range(size)] for i in range(size)]
    s = Solver()

    # Cell range constraints: 1 <= X[i][j] <= size
    for i in range(size):
        for j in range(size):
            s.add(And(X[i][j] >= 1, X[i][j] <= size))

    # Given cell constraints
    for (i, j), val in given_cells.items():
        s.add(X[i][j] == val)

    # Row uniqueness
    for i in range(size):
        s.add(Distinct([X[i][j] for j in range(size)]))

    # Column uniqueness
    for j in range(size):
        s.add(Distinct([X[i][j] for i in range(size)]))

    # Box uniqueness
    for box_row in range(size // box_size):
        for box_col in range(size // box_size):
            cells = []
            for i in range(box_size):
                for j in range(box_size):
                    cells.append(X[box_row * box_size + i][box_col * box_size + j])
            s.add(Distinct(cells))

    # Solve
    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(size)] for i in range(size)]
    return None


def solve_puzzle(size, puzzle):
    """Convert puzzle to given_cells format and solve."""
    given_cells = {}
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]
    return solve_sudoku(size, given_cells)


def verify_solution(size, solution):
    """Verify that a solution is a valid Sudoku."""
    if solution is None:
        return False

    box_size = get_box_size(size)

    # Check rows
    for row in solution:
        if sorted(row) != list(range(1, size + 1)):
            return False

    # Check columns
    for col in range(size):
        if sorted([solution[row][col] for row in range(size)]) != list(range(1, size + 1)):
            return False

    # Check boxes
    for box_row in range(size // box_size):
        for box_col in range(size // box_size):
            cells = []
            for i in range(box_size):
                for j in range(box_size):
                    cells.append(solution[box_row * box_size + i][box_col * box_size + j])
            if sorted(cells) != list(range(1, size + 1)):
                return False

    return True


def solution_matches_puzzle(size, solution, puzzle):
    """Check if solution matches all given clues in puzzle."""
    if solution is None:
        return False
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0 and puzzle[i][j] != solution[i][j]:
                return False
    return True


def compare_solutions(size, computed, expected):
    """Compare two solutions and return number of matching cells."""
    if computed is None:
        return 0, size * size
    correct = sum(1 for i in range(size) for j in range(size)
                  if computed[i][j] == expected[i][j])
    return correct, size * size


def main():
    print("=" * 60)
    print("Handwritten Sudoku Neuro-Symbolic Evaluation")
    print("=" * 60)

    # Load model
    model_path = "./models/handwritten_sudoku_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train_cnn.py first to train the model.")
        return

    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load ground truth
    puzzles_path = "./puzzles/puzzles_dict.json"
    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    # Check for board images
    image_dir = "./board_images"
    if not os.path.exists(image_dir):
        print(f"Error: Board images not found at {image_dir}")
        print("Run generate_images.py first to create board images.")
        return

    # Prepare results
    results = []
    overall_stats = {4: {'correct': 0, 'total': 0, 'extraction_sum': 0},
                     9: {'correct': 0, 'total': 0, 'extraction_sum': 0}}

    print(f"\nEvaluating handwritten puzzles...")
    print("-" * 60)

    for size_str in ['4', '9']:
        size = int(size_str)
        puzzles = puzzles_ds.get(size_str, [])

        if not puzzles:
            continue

        print(f"\nEvaluating {len(puzzles)} {size}×{size} puzzles...")

        for idx in range(len(puzzles)):
            image_path = os.path.join(image_dir, f"board{size}_{idx}.png")

            if not os.path.exists(image_path):
                continue

            expected_solution = puzzles[idx]['solution']
            expected_puzzle = puzzles[idx]['puzzle']

            # Extract puzzle from image (CNN recognition)
            start_time = time.time()
            extracted_puzzle = extract_puzzle_from_image(image_path, model, size)
            extract_time = time.time() - start_time

            # Count extraction accuracy
            extraction_matches = sum(1 for i in range(size) for j in range(size)
                                      if extracted_puzzle[i][j] == expected_puzzle[i][j])
            extraction_accuracy = extraction_matches / (size * size)

            # Solve with Z3
            start_time = time.time()
            computed_solution = solve_puzzle(size, extracted_puzzle)
            solve_time = time.time() - start_time

            # Verify solution
            is_valid = verify_solution(size, computed_solution)
            matches_clues = solution_matches_puzzle(size, computed_solution, extracted_puzzle)
            is_correct = is_valid and matches_clues

            # Compare to expected solution
            correct_cells, total_cells = compare_solutions(size, computed_solution, expected_solution)
            exact_match = correct_cells == total_cells

            # Update stats
            overall_stats[size]['total'] += 1
            overall_stats[size]['extraction_sum'] += extraction_accuracy
            if exact_match:
                overall_stats[size]['correct'] += 1

            results.append({
                'size': size,
                'puzzle_idx': idx,
                'extraction_accuracy': extraction_accuracy,
                'is_valid_solution': is_valid,
                'matches_clues': matches_clues,
                'is_correct': is_correct,
                'exact_match_expected': exact_match,
                'cells_match_expected': correct_cells,
                'total_cells': total_cells,
                'extract_time': extract_time,
                'solve_time': solve_time
            })

            if (idx + 1) % 25 == 0:
                acc = overall_stats[size]['correct'] / overall_stats[size]['total'] * 100
                ext = overall_stats[size]['extraction_sum'] / overall_stats[size]['total'] * 100
                print(f"  {size}×{size}: {idx + 1}/{len(puzzles)} - "
                      f"Exact: {acc:.1f}%, Avg Extraction: {ext:.1f}%")

    # Summary
    print("-" * 60)
    print(f"\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for size in [4, 9]:
        stats = overall_stats[size]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            ext = stats['extraction_sum'] / stats['total'] * 100
            print(f"\n{size}×{size} Puzzles:")
            print(f"  Total evaluated: {stats['total']}")
            print(f"  Exact matches: {stats['correct']} ({acc:.1f}%)")
            print(f"  Average extraction accuracy: {ext:.2f}%")

    # Overall
    total_puzzles = sum(s['total'] for s in overall_stats.values())
    total_correct = sum(s['correct'] for s in overall_stats.values())
    total_extraction = sum(s['extraction_sum'] for s in overall_stats.values())

    print(f"\nOverall:")
    print(f"  Total puzzles: {total_puzzles}")
    print(f"  Exact matches: {total_correct} ({total_correct/total_puzzles*100:.1f}%)")
    print(f"  Average extraction accuracy: {total_extraction/total_puzzles*100:.2f}%")

    # Timing
    avg_extract = np.mean([r['extract_time'] for r in results])
    avg_solve = np.mean([r['solve_time'] for r in results])
    print(f"\nTiming:")
    print(f"  Average extraction time: {avg_extract:.3f}s")
    print(f"  Average solve time: {avg_solve:.3f}s")

    # Save results
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    csv_path = f"{results_dir}/detailed_evaluation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results saved to {csv_path}")

    # Show failed puzzles
    failed = [r for r in results if not r['exact_match_expected']]
    if failed:
        print(f"\nPuzzles not matching expected solution ({len(failed)}):")
        for r in failed[:10]:
            print(f"  {r['size']}×{r['size']} Puzzle {r['puzzle_idx']}: "
                  f"extraction={r['extraction_accuracy']*100:.1f}%, "
                  f"valid={r['is_valid_solution']}, "
                  f"cells_match={r['cells_match_expected']}/{r['total_cells']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
