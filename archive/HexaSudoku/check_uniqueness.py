"""
Check which HexaSudoku puzzles have unique solutions.

Uses Z3 to verify uniqueness by:
1. Finding the first solution
2. Blocking that solution and checking for a second
3. If no second solution exists, the puzzle is unique
"""
import json
import time
import sys
from z3 import Int, Solver, And, Or, Distinct, sat, unsat

SIZE = 16
BOX_SIZE = 4


def has_unique_solution(puzzle):
    """
    Check if a 16x16 Sudoku puzzle has exactly one solution.

    Args:
        puzzle: 16x16 2D list where 0 = empty cell, 1-16 = filled

    Returns:
        bool: True if puzzle has exactly one solution
    """
    X = [[Int(f"x_{i}_{j}") for j in range(SIZE)] for i in range(SIZE)]
    s = Solver()

    # Cell range constraints: 1 <= X[i][j] <= 16
    for i in range(SIZE):
        for j in range(SIZE):
            s.add(And(X[i][j] >= 1, X[i][j] <= SIZE))

    # Given cell constraints
    for i in range(SIZE):
        for j in range(SIZE):
            if puzzle[i][j] != 0:
                s.add(X[i][j] == puzzle[i][j])

    # Row constraints: each row has distinct values
    for i in range(SIZE):
        s.add(Distinct([X[i][j] for j in range(SIZE)]))

    # Column constraints: each column has distinct values
    for j in range(SIZE):
        s.add(Distinct([X[i][j] for i in range(SIZE)]))

    # Box constraints: each 4x4 box has distinct values
    for box_row in range(BOX_SIZE):
        for box_col in range(BOX_SIZE):
            cells = []
            for i in range(BOX_SIZE):
                for j in range(BOX_SIZE):
                    cells.append(X[box_row * BOX_SIZE + i][box_col * BOX_SIZE + j])
            s.add(Distinct(cells))

    # Find first solution
    if s.check() != sat:
        return False  # No solution exists

    first_solution = s.model()

    # Block first solution and check for another
    block = []
    for i in range(SIZE):
        for j in range(SIZE):
            block.append(X[i][j] != first_solution.evaluate(X[i][j]))
    s.add(Or(block))

    # If no second solution exists, puzzle is unique
    return s.check() == unsat


def main():
    print("=" * 60, flush=True)
    print("HexaSudoku Puzzle Uniqueness Check", flush=True)
    print("=" * 60, flush=True)

    # Load puzzles
    with open("./puzzles/puzzles_dict.json", "r") as f:
        puzzles_ds = json.load(f)

    puzzles = puzzles_ds.get('16', [])
    print(f"\nLoaded {len(puzzles)} puzzles to check.", flush=True)
    print("This may take several hours...\n", flush=True)

    unique_indices = []
    start_total = time.time()

    for idx, puzzle_data in enumerate(puzzles):
        puzzle = puzzle_data['puzzle']
        clues = sum(1 for row in puzzle for cell in row if cell != 0)

        start = time.time()
        is_unique = has_unique_solution(puzzle)
        elapsed = time.time() - start

        status = "UNIQUE" if is_unique else "multiple solutions"
        print(f"Puzzle {idx:3d}: {status} ({clues} clues, {elapsed:.1f}s)", flush=True)

        if is_unique:
            unique_indices.append(idx)

    total_time = time.time() - start_total

    # Write results to file
    output_path = "./unique_puzzles.txt"
    with open(output_path, "w") as f:
        f.write("HexaSudoku Unique Puzzle Analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total puzzles analyzed: {len(puzzles)}\n")
        f.write(f"Puzzles with unique solutions: {len(unique_indices)}\n")
        f.write(f"Total analysis time: {total_time:.1f}s\n")
        f.write("\n")

        if unique_indices:
            f.write("Unique puzzle indices:\n")
            for idx in unique_indices:
                f.write(f"{idx}\n")
        else:
            f.write("No puzzles with unique solutions found.\n")

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("Summary:", flush=True)
    print(f"  Total puzzles: {len(puzzles)}", flush=True)
    print(f"  Unique solutions: {len(unique_indices)}", flush=True)
    print(f"  Total time: {total_time:.1f}s ({total_time/len(puzzles):.1f}s per puzzle)", flush=True)
    print(f"\nResults saved to {output_path}", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
