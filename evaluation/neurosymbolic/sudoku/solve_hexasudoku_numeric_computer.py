"""
HexaSudoku Neuro-Symbolic Solver Evaluation for NUMERIC notation.

Pipeline: Image → CNN (character recognition) → Z3 (constraint solving) → Solution

Uses a 17-class CNN that recognizes values 0-16 directly:
- Single digits (1-9): recognized as-is
- Two-digit numbers (10-16): recognized as single classes
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
CELL_SIZE = BOARD_PIXELS // SIZE
NUM_CLASSES = 17  # 0-16 values (0=empty, 1-9 digits, 10-16 two-digit numbers)


class CNN_v2(nn.Module):
    """CNN for character recognition."""
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
    """Extract and preprocess a single cell from board image."""
    # Calculate cell boundaries with margin
    margin = 8
    y_start = row * CELL_SIZE + margin
    y_end = (row + 1) * CELL_SIZE - margin
    x_start = col * CELL_SIZE + margin
    x_end = (col + 1) * CELL_SIZE - margin

    # Extract cell
    cell = board_img[y_start:y_end, x_start:x_end]

    # Resize to target size
    cell_pil = Image.fromarray((cell * 255).astype(np.uint8))
    cell_pil = cell_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)

    return np.array(cell_pil) / 255.0


def is_cell_empty(cell_img, threshold=0.98):
    """Check if cell is empty (mostly white)."""
    return np.mean(cell_img) > threshold


def recognize_character(cell_img, model):
    """Recognize character in cell using CNN."""
    # Invert so ink is high value (like training data)
    cell_inverted = 1.0 - cell_img

    with torch.no_grad():
        tensor = torch.tensor(cell_inverted, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction


def extract_puzzle_from_image(image_path, model):
    """Extract puzzle values from board image using CNN."""
    # Load image
    img = Image.open(image_path).convert('L')
    board_img = np.array(img) / 255.0

    # Extract each cell
    puzzle = [[0] * SIZE for _ in range(SIZE)]

    for row in range(SIZE):
        for col in range(SIZE):
            cell = extract_cell(board_img, row, col)

            if is_cell_empty(cell):
                puzzle[row][col] = 0
            else:
                predicted = recognize_character(cell, model)
                # Model directly predicts values 0-16
                if predicted == 0:
                    puzzle[row][col] = 0
                else:
                    puzzle[row][col] = predicted

    return puzzle


def solve_hexasudoku(given_cells):
    """Solve 16x16 Sudoku using Z3."""
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    # Cell range: 1 <= X[i][j] <= 16
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

    # 4x4 Box uniqueness
    for box_row in range(BOX_SIZE):
        for box_col in range(BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

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
    # Check rows have all 1-16
    for row in solution:
        if sorted(row) != list(range(1, SIZE + 1)):
            return False
    # Check columns
    for col in range(SIZE):
        if sorted([solution[row][col] for row in range(SIZE)]) != list(range(1, SIZE + 1)):
            return False
    # Check boxes
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


def value_to_char(value):
    """Convert value to display character (numeric notation)."""
    if value == 0:
        return '.'
    return str(value)


def main():
    print("=" * 60)
    print("HexaSudoku Neuro-Symbolic Solver Evaluation (NUMERIC notation)")
    print("=" * 60)

    # Load model
    model_path = "./models/numeric_character_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Load ground truth
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"Loaded {len(puzzles)} puzzles from JSON.")

    # Find images
    image_dir = "./board_images_numeric"
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"Found {len(image_files)} board images in {image_dir}.")

    # Evaluate
    results = []
    total_correct = 0
    total_puzzles = min(len(puzzles), len(image_files))

    print(f"\nEvaluating {total_puzzles} puzzles...")
    print("-" * 60)

    for idx in range(total_puzzles):
        image_path = os.path.join(image_dir, f"board16_{idx}.png")
        expected_solution = puzzles[idx]['solution']
        expected_puzzle = puzzles[idx]['puzzle']

        # Extract puzzle from image
        start_time = time.time()
        extracted_puzzle = extract_puzzle_from_image(image_path, model)
        extract_time = time.time() - start_time

        # Count extraction accuracy
        extraction_matches = sum(1 for i in range(SIZE) for j in range(SIZE)
                                  if extracted_puzzle[i][j] == expected_puzzle[i][j])
        extraction_accuracy = extraction_matches / (SIZE * SIZE)

        # Solve with Z3
        start_time = time.time()
        computed_solution = solve_puzzle(extracted_puzzle)
        solve_time = time.time() - start_time

        # Verify solution is valid and matches puzzle clues
        is_valid = verify_solution(computed_solution)
        matches_clues = solution_matches_puzzle(computed_solution, extracted_puzzle)
        is_correct = is_valid and matches_clues

        # Also compare to stored solution
        correct_cells, total_cells = compare_solutions(computed_solution, expected_solution)

        if is_correct:
            total_correct += 1

        results.append({
            'puzzle_idx': idx,
            'extraction_accuracy': extraction_accuracy,
            'is_valid_solution': is_valid,
            'matches_clues': matches_clues,
            'is_correct': is_correct,
            'cells_match_expected': correct_cells,
            'total_cells': total_cells,
            'extract_time': extract_time,
            'solve_time': solve_time
        })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total_puzzles} - Running accuracy: {total_correct/(idx+1)*100:.1f}%")

    # Summary
    print("-" * 60)
    print(f"\nResults Summary:")
    print(f"  Total puzzles: {total_puzzles}")
    print(f"  Correct solutions: {total_correct}")
    print(f"  Accuracy: {total_correct/total_puzzles*100:.1f}%")

    avg_extraction = np.mean([r['extraction_accuracy'] for r in results])
    print(f"  Average extraction accuracy: {avg_extraction*100:.2f}%")

    avg_extract_time = np.mean([r['extract_time'] for r in results])
    avg_solve_time = np.mean([r['solve_time'] for r in results])
    print(f"  Average extraction time: {avg_extract_time:.3f}s")
    print(f"  Average solve time: {avg_solve_time:.3f}s")

    # Save results
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    csv_path = f"{results_dir}/numeric_evaluation.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    # Show additional stats
    valid_solutions = sum(1 for r in results if r['is_valid_solution'])
    matching_clues = sum(1 for r in results if r['matches_clues'])
    print(f"\n  Valid Sudoku solutions: {valid_solutions}/{total_puzzles}")
    print(f"  Solutions matching clues: {matching_clues}/{total_puzzles}")

    # Show failed puzzles if any
    failed = [r for r in results if not r['is_correct']]
    if failed:
        print(f"\nFailed puzzles ({len(failed)}):")
        for r in failed[:5]:  # Show first 5
            print(f"  Puzzle {r['puzzle_idx']}: valid={r['is_valid_solution']}, "
                  f"matches_clues={r['matches_clues']}, extraction={r['extraction_accuracy']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
