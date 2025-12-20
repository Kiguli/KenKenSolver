"""
Evaluate handwritten HexaSudoku puzzles using the neuro-symbolic pipeline.

Pipeline: Image → CNN (character recognition) → Z3 (constraint solving) → Solution

This evaluates the model's ability to recognize real handwritten characters
(from MNIST/EMNIST test sets) that were never seen during training.
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
SIZE = 16
BOX_SIZE = 4
BOARD_PIXELS = 1600
CELL_SIZE = BOARD_PIXELS // SIZE  # 100px
NUM_CLASSES = 17


class CNN_v2(nn.Module):
    """CNN for character recognition (same architecture as training)."""

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


def extract_cell(board_img, row, col, target_size=28):
    """
    Extract and preprocess a single cell from board image.

    Args:
        board_img: numpy array of board (0-1 float, 0=black, 1=white)
        row: Row index (0-15)
        col: Column index (0-15)
        target_size: Output size (default 28x28 for CNN)

    Returns:
        numpy array of shape (target_size, target_size) with values in [0,1]
    """
    margin = 8  # Avoid grid lines
    y_start = row * CELL_SIZE + margin
    y_end = (row + 1) * CELL_SIZE - margin
    x_start = col * CELL_SIZE + margin
    x_end = (col + 1) * CELL_SIZE - margin

    cell = board_img[y_start:y_end, x_start:x_end]

    # Resize to target size
    cell_pil = Image.fromarray((cell * 255).astype(np.uint8))
    cell_pil = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(cell_pil) / 255.0


def is_cell_empty(cell_img, threshold=0.98):
    """
    Check if cell is empty (mostly white).

    Handwritten characters may have more noise, so threshold might need tuning.

    Args:
        cell_img: numpy array with values in [0,1]
        threshold: Mean intensity threshold (higher = whiter)

    Returns:
        True if cell appears empty
    """
    return np.mean(cell_img) > threshold


def recognize_character(cell_img, model):
    """
    Recognize character in cell using CNN.

    Args:
        cell_img: numpy array of cell (0-1, 0=black ink, 1=white bg)
        model: Trained CNN model

    Returns:
        Predicted class (0-16)
    """
    # Invert so ink is high value (MNIST convention for training)
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction


def extract_puzzle_from_image(image_path, model):
    """
    Extract puzzle values from board image using CNN.

    Args:
        image_path: Path to board image
        model: Trained CNN model

    Returns:
        16x16 list of predicted values (0 = empty, 1-15 = digits/letters)
    """
    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    puzzle = [[0] * SIZE for _ in range(SIZE)]

    for row in range(SIZE):
        for col in range(SIZE):
            cell = extract_cell(board_img, row, col)

            if is_cell_empty(cell):
                puzzle[row][col] = 0
            else:
                predicted = recognize_character(cell, model)
                puzzle[row][col] = predicted

    return puzzle


def solve_hexasudoku(given_cells):
    """
    Solve 16x16 Sudoku using Z3 constraint solver.

    Args:
        given_cells: dict of {(row, col): value} for known cells

    Returns:
        16x16 solution grid, or None if unsatisfiable
    """
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    # Cell range constraints: 1 <= X[i][j] <= 16
    for i in range(SIZE):
        for j in range(SIZE):
            s.add(And(X[i][j] >= 1, X[i][j] <= SIZE))

    # Given cell constraints
    for (i, j), val in given_cells.items():
        s.add(X[i][j] == val)

    # Row uniqueness
    for i in range(SIZE):
        s.add(Distinct([X[i][j] for j in range(SIZE)]))

    # Column uniqueness
    for j in range(SIZE):
        s.add(Distinct([X[i][j] for i in range(SIZE)]))

    # 4x4 box uniqueness
    for box_row in range(BOX_SIZE):
        for box_col in range(BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

    # Solve
    if s.check() == sat:
        m = s.model()
        return [[m.evaluate(X[i][j]).as_long() for j in range(SIZE)] for i in range(SIZE)]
    return None


def solve_puzzle(puzzle):
    """Convert puzzle to given_cells format and solve."""
    given_cells = {}
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0:
                given_cells[(i, j)] = puzzle[i][j]
    return solve_hexasudoku(given_cells)


def verify_solution(solution):
    """Verify that a solution is a valid 16x16 Sudoku."""
    if solution is None:
        return False

    # Check rows
    for row in solution:
        if sorted(row) != list(range(1, SIZE + 1)):
            return False

    # Check columns
    for col in range(SIZE):
        if sorted([solution[row][col] for row in range(SIZE)]) != list(range(1, SIZE + 1)):
            return False

    # Check 4x4 boxes
    for box_row in range(BOX_SIZE):
        for box_col in range(BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(solution[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            if sorted(cells) != list(range(1, SIZE + 1)):
                return False

    return True


def solution_matches_puzzle(solution, puzzle):
    """Check if solution matches all given clues in puzzle."""
    if solution is None:
        return False
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0 and puzzle[i][j] != solution[i][j]:
                return False
    return True


def compare_solutions(computed, expected):
    """Compare two solutions and return number of matching cells."""
    if computed is None:
        return 0, SIZE * SIZE
    correct = sum(1 for i in range(SIZE) for j in range(SIZE)
                  if computed[i][j] == expected[i][j])
    return correct, SIZE * SIZE


def analyze_extraction_errors(extracted, expected):
    """
    Analyze character recognition errors by class.

    Returns dict of {class: {correct, total, confused_with}}
    """
    errors = {}

    for i in range(SIZE):
        for j in range(SIZE):
            exp_val = expected[i][j]
            ext_val = extracted[i][j]

            if exp_val not in errors:
                errors[exp_val] = {'correct': 0, 'total': 0, 'confused': {}}

            errors[exp_val]['total'] += 1
            if ext_val == exp_val:
                errors[exp_val]['correct'] += 1
            else:
                if ext_val not in errors[exp_val]['confused']:
                    errors[exp_val]['confused'][ext_val] = 0
                errors[exp_val]['confused'][ext_val] += 1

    return errors


def main():
    print("=" * 60)
    print("Handwritten HexaSudoku Neuro-Symbolic Evaluation")
    print("=" * 60)

    # Load model
    model_path = "./models/handwritten_hex_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Run train_cnn.py first to train the model.")
        return

    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load ground truth
    puzzles_path = "./puzzles/unique_puzzles_dict.json"
    with open(puzzles_path, "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"Loaded {len(puzzles)} puzzles from JSON.")

    # Find images
    image_dir = "./board_images"
    if not os.path.exists(image_dir):
        print(f"Error: Board images not found at {image_dir}")
        print("Run generate_images.py first to create board images.")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} board images.")

    # Evaluate
    results = []
    total_correct = 0
    total_exact_match = 0
    total_puzzles = min(len(puzzles), len(image_files))
    all_errors = {}

    print(f"\nEvaluating {total_puzzles} handwritten puzzles...")
    print("-" * 60)

    for idx in range(total_puzzles):
        image_path = os.path.join(image_dir, f"board16_{idx}.png")
        expected_solution = puzzles[idx]['solution']
        expected_puzzle = puzzles[idx]['puzzle']

        # Extract puzzle from image (CNN recognition)
        start_time = time.time()
        extracted_puzzle = extract_puzzle_from_image(image_path, model)
        extract_time = time.time() - start_time

        # Count extraction accuracy
        extraction_matches = sum(1 for i in range(SIZE) for j in range(SIZE)
                                  if extracted_puzzle[i][j] == expected_puzzle[i][j])
        extraction_accuracy = extraction_matches / (SIZE * SIZE)

        # Analyze errors for this puzzle
        puzzle_errors = analyze_extraction_errors(extracted_puzzle, expected_puzzle)
        for cls, data in puzzle_errors.items():
            if cls not in all_errors:
                all_errors[cls] = {'correct': 0, 'total': 0, 'confused': {}}
            all_errors[cls]['correct'] += data['correct']
            all_errors[cls]['total'] += data['total']
            for conf_cls, count in data['confused'].items():
                if conf_cls not in all_errors[cls]['confused']:
                    all_errors[cls]['confused'][conf_cls] = 0
                all_errors[cls]['confused'][conf_cls] += count

        # Solve with Z3
        start_time = time.time()
        computed_solution = solve_puzzle(extracted_puzzle)
        solve_time = time.time() - start_time

        # Verify solution
        is_valid = verify_solution(computed_solution)
        matches_clues = solution_matches_puzzle(computed_solution, extracted_puzzle)
        is_correct = is_valid and matches_clues

        # Compare to expected solution
        correct_cells, total_cells = compare_solutions(computed_solution, expected_solution)
        exact_match = correct_cells == total_cells

        if is_correct:
            total_correct += 1
        if exact_match:
            total_exact_match += 1

        results.append({
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

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total_puzzles} - "
                  f"Valid: {total_correct/(idx+1)*100:.1f}%, "
                  f"Exact: {total_exact_match/(idx+1)*100:.1f}%, "
                  f"Avg Extraction: {np.mean([r['extraction_accuracy'] for r in results])*100:.1f}%")

    # Summary
    print("-" * 60)
    print(f"\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nPipeline Performance:")
    print(f"  Total puzzles evaluated: {total_puzzles}")
    print(f"  Valid solutions found: {total_correct} ({total_correct/total_puzzles*100:.1f}%)")
    print(f"  Exact matches with expected: {total_exact_match} ({total_exact_match/total_puzzles*100:.1f}%)")

    avg_extraction = np.mean([r['extraction_accuracy'] for r in results])
    min_extraction = np.min([r['extraction_accuracy'] for r in results])
    max_extraction = np.max([r['extraction_accuracy'] for r in results])
    print(f"\nCharacter Recognition:")
    print(f"  Average extraction accuracy: {avg_extraction*100:.2f}%")
    print(f"  Min extraction accuracy: {min_extraction*100:.2f}%")
    print(f"  Max extraction accuracy: {max_extraction*100:.2f}%")

    avg_extract_time = np.mean([r['extract_time'] for r in results])
    avg_solve_time = np.mean([r['solve_time'] for r in results])
    print(f"\nTiming:")
    print(f"  Average extraction time: {avg_extract_time:.3f}s")
    print(f"  Average solve time: {avg_solve_time:.3f}s")

    # Per-class accuracy
    print(f"\nPer-Class Recognition Accuracy:")
    for cls in sorted(all_errors.keys()):
        data = all_errors[cls]
        if data['total'] > 0:
            acc = data['correct'] / data['total']
            if cls == 0:
                name = "Empty"
            elif cls <= 9:
                name = str(cls)
            else:
                name = chr(ord('A') + cls - 10)

            conf_str = ""
            if data['confused']:
                top_conf = sorted(data['confused'].items(), key=lambda x: -x[1])[:3]
                conf_str = " | confused with: " + ", ".join(
                    [f"{c}({n})" for c, n in top_conf]
                )
            print(f"  Class {cls:2d} ({name:5s}): {acc*100:.1f}% ({data['correct']}/{data['total']}){conf_str}")

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
            print(f"  Puzzle {r['puzzle_idx']}: extraction={r['extraction_accuracy']*100:.1f}%, "
                  f"valid={r['is_valid_solution']}, cells_match={r['cells_match_expected']}/{r['total_cells']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
