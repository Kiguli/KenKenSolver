"""
Fast 16x16 HexaSudoku puzzle generator using permutation-based approach.
"""
import numpy as np
import random
import time
import json
import os
import sys

SIZE = 16
BOX_SIZE = 4
NUM_PUZZLES = 100
TARGET_CLUES = 120

print('Generating 100 16x16 HexaSudoku puzzles...', flush=True)


def generate_complete_grid_fast():
    """
    Generate a valid 16x16 Sudoku grid using a permutation-based approach.
    This is much faster than backtracking for generating random valid grids.
    """
    # Start with a valid base grid using Latin square construction
    base = list(range(1, SIZE + 1))
    grid = []

    for row in range(SIZE):
        # Create shifted version for each row
        shift = (row // BOX_SIZE) + (row % BOX_SIZE) * BOX_SIZE
        shifted = base[shift:] + base[:shift]
        grid.append(shifted)

    # Randomly permute the grid while maintaining validity
    grid = np.array(grid)

    # Shuffle rows within each band (group of BOX_SIZE rows)
    for band in range(BOX_SIZE):
        rows = list(range(band * BOX_SIZE, (band + 1) * BOX_SIZE))
        random.shuffle(rows)
        grid[band * BOX_SIZE:(band + 1) * BOX_SIZE] = grid[rows]

    # Shuffle columns within each stack (group of BOX_SIZE columns)
    for stack in range(BOX_SIZE):
        cols = list(range(stack * BOX_SIZE, (stack + 1) * BOX_SIZE))
        random.shuffle(cols)
        grid[:, stack * BOX_SIZE:(stack + 1) * BOX_SIZE] = grid[:, cols]

    # Shuffle bands (groups of BOX_SIZE rows)
    bands = list(range(BOX_SIZE))
    random.shuffle(bands)
    new_grid = np.zeros_like(grid)
    for i, band in enumerate(bands):
        new_grid[i * BOX_SIZE:(i + 1) * BOX_SIZE] = grid[band * BOX_SIZE:(band + 1) * BOX_SIZE]
    grid = new_grid

    # Shuffle stacks (groups of BOX_SIZE columns)
    stacks = list(range(BOX_SIZE))
    random.shuffle(stacks)
    new_grid = np.zeros_like(grid)
    for i, stack in enumerate(stacks):
        new_grid[:, i * BOX_SIZE:(i + 1) * BOX_SIZE] = grid[:, stack * BOX_SIZE:(stack + 1) * BOX_SIZE]
    grid = new_grid

    # Randomly relabel values (permute 1-16)
    perm = list(range(1, SIZE + 1))
    random.shuffle(perm)
    relabeled = np.zeros_like(grid)
    for i in range(SIZE):
        for j in range(SIZE):
            relabeled[i][j] = perm[grid[i][j] - 1]

    return relabeled.tolist()


def generate_puzzle(target_clues=TARGET_CLUES):
    """Generate puzzle by removing cells from complete grid."""
    solution = generate_complete_grid_fast()
    puzzle = [row[:] for row in solution]

    # Randomly select cells to remove
    all_cells = [(i, j) for i in range(SIZE) for j in range(SIZE)]
    cells_to_remove = random.sample(all_cells, 256 - target_clues)

    for (i, j) in cells_to_remove:
        puzzle[i][j] = 0

    return puzzle, solution


def verify_solution(solution):
    """Verify that a solution is valid."""
    # Check rows
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


# Generate puzzles
puzzles = []
start_total = time.time()

for i in range(NUM_PUZZLES):
    start = time.time()
    puzzle, solution = generate_puzzle()

    # Verify solution is valid
    assert verify_solution(solution), f"Invalid solution generated for puzzle {i}"

    elapsed = time.time() - start
    clues = sum(1 for row in puzzle for cell in row if cell != 0)
    puzzles.append({'puzzle': puzzle, 'solution': solution})

    if (i + 1) % 10 == 0:
        print(f'Generated puzzle {i+1}/{NUM_PUZZLES} ({clues} clues, {elapsed:.3f}s)', flush=True)

total_time = time.time() - start_total
print(f'\nTotal time: {total_time:.1f}s ({total_time/NUM_PUZZLES:.3f}s per puzzle)', flush=True)

# Save puzzles
output_data = {'16': puzzles}
output_path = './puzzles/puzzles_dict.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f'Saved {len(puzzles)} puzzles to {output_path}', flush=True)

# Show clue distribution
clue_counts = [sum(1 for row in p['puzzle'] for cell in row if cell != 0) for p in puzzles]
print(f'\nClue distribution: min={min(clue_counts)}, max={max(clue_counts)}, avg={np.mean(clue_counts):.1f}', flush=True)
