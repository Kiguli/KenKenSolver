"""
Generate HexaSudoku puzzles with guaranteed unique solutions.

Uses Z3 to verify uniqueness during generation by:
1. Generating a complete valid grid
2. Removing cells one by one
3. After each removal, checking if the puzzle still has exactly one solution
4. Only keeping removals that maintain uniqueness
"""
import json
import time
import random
import numpy as np
from z3 import Int, Solver, And, Or, Distinct, sat, unsat

SIZE = 16
BOX_SIZE = 4
NUM_PUZZLES = 100
MIN_CLUES = 120  # Minimum clues to keep (more clues = faster uniqueness check)


def generate_complete_grid_fast():
    """
    Generate a valid 16x16 Sudoku grid using a permutation-based approach.
    """
    base = list(range(1, SIZE + 1))
    grid = []

    for row in range(SIZE):
        shift = (row // BOX_SIZE) + (row % BOX_SIZE) * BOX_SIZE
        shifted = base[shift:] + base[:shift]
        grid.append(shifted)

    grid = np.array(grid)

    # Shuffle rows within each band
    for band in range(BOX_SIZE):
        rows = list(range(band * BOX_SIZE, (band + 1) * BOX_SIZE))
        random.shuffle(rows)
        grid[band * BOX_SIZE:(band + 1) * BOX_SIZE] = grid[rows]

    # Shuffle columns within each stack
    for stack in range(BOX_SIZE):
        cols = list(range(stack * BOX_SIZE, (stack + 1) * BOX_SIZE))
        random.shuffle(cols)
        grid[:, stack * BOX_SIZE:(stack + 1) * BOX_SIZE] = grid[:, cols]

    # Shuffle bands
    bands = list(range(BOX_SIZE))
    random.shuffle(bands)
    new_grid = np.zeros_like(grid)
    for i, band in enumerate(bands):
        new_grid[i * BOX_SIZE:(i + 1) * BOX_SIZE] = grid[band * BOX_SIZE:(band + 1) * BOX_SIZE]
    grid = new_grid

    # Shuffle stacks
    stacks = list(range(BOX_SIZE))
    random.shuffle(stacks)
    new_grid = np.zeros_like(grid)
    for i, stack in enumerate(stacks):
        new_grid[:, i * BOX_SIZE:(i + 1) * BOX_SIZE] = grid[:, stack * BOX_SIZE:(stack + 1) * BOX_SIZE]
    grid = new_grid

    # Randomly relabel values
    perm = list(range(1, SIZE + 1))
    random.shuffle(perm)
    relabeled = np.zeros_like(grid)
    for i in range(SIZE):
        for j in range(SIZE):
            relabeled[i][j] = perm[grid[i][j] - 1]

    return relabeled.tolist()


def has_unique_solution(puzzle):
    """
    Check if a 16x16 Sudoku puzzle has exactly one solution.
    """
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    # Cell range constraints
    for i in range(SIZE):
        for j in range(SIZE):
            s.add(And(X[i][j] >= 1, X[i][j] <= SIZE))

    # Given cell constraints
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0:
                s.add(X[i][j] == puzzle[i][j])

    # Row constraints
    for i in range(SIZE):
        s.add(Distinct([X[i][j] for j in range(SIZE)]))

    # Column constraints
    for j in range(SIZE):
        s.add(Distinct([X[i][j] for i in range(SIZE)]))

    # Box constraints
    for box_row in range(BOX_SIZE):
        for box_col in range(BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

    # Find first solution
    if s.check() != sat:
        return False

    first_solution = s.model()

    # Block first solution
    block = []
    for i in range(SIZE):
        for j in range(SIZE):
            block.append(X[i][j] != first_solution.evaluate(X[i][j]))
    s.add(Or(block))

    # If no second solution, puzzle is unique
    return s.check() == unsat


def generate_unique_puzzle(min_clues=MIN_CLUES):
    """
    Generate a puzzle with a guaranteed unique solution.

    Strategy: Start with full grid, remove cells while maintaining uniqueness.
    """
    solution = generate_complete_grid_fast()
    puzzle = [row[:] for row in solution]

    # Get all cells and shuffle for random removal order
    all_cells = [(i, j) for i in range(SIZE) for j in range(SIZE)]
    random.shuffle(all_cells)

    current_clues = SIZE * SIZE  # 256

    for (i, j) in all_cells:
        if current_clues <= min_clues:
            break

        # Try removing this cell
        old_value = puzzle[i][j]
        puzzle[i][j] = 0

        # Check if still unique
        if has_unique_solution(puzzle):
            current_clues -= 1
        else:
            # Put it back - removal breaks uniqueness
            puzzle[i][j] = old_value

    return puzzle, solution


def main():
    print("=" * 60, flush=True)
    print("Generating Unique HexaSudoku Puzzles", flush=True)
    print("=" * 60, flush=True)
    print(f"Target: {NUM_PUZZLES} puzzles with unique solutions", flush=True)
    print(f"Minimum clues: {MIN_CLUES}", flush=True)
    print("This will take a long time...\n", flush=True)

    puzzles = []
    start_total = time.time()

    for i in range(NUM_PUZZLES):
        start = time.time()
        puzzle, solution = generate_unique_puzzle()
        elapsed = time.time() - start

        clues = sum(1 for row in puzzle for cell in row if cell != 0)
        puzzles.append({'puzzle': puzzle, 'solution': solution})

        print(f"Generated puzzle {i+1}/{NUM_PUZZLES} ({clues} clues, {elapsed:.1f}s)", flush=True)

    total_time = time.time() - start_total

    # Save to JSON
    output_data = {'16': puzzles}
    output_path = './puzzles/unique_puzzles_dict.json'

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Statistics
    clue_counts = [sum(1 for row in p['puzzle'] for cell in row if cell != 0) for p in puzzles]

    print("\n" + "=" * 60, flush=True)
    print("Summary:", flush=True)
    print(f"  Total puzzles: {len(puzzles)}", flush=True)
    print(f"  Clue range: {min(clue_counts)} - {max(clue_counts)}", flush=True)
    print(f"  Average clues: {np.mean(clue_counts):.1f}", flush=True)
    print(f"  Total time: {total_time:.1f}s ({total_time/NUM_PUZZLES:.1f}s per puzzle)", flush=True)
    print(f"\nSaved to {output_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
