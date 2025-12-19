#!/usr/bin/env python3
"""Generate additional KenKen puzzles for sizes 7, 8, and 9."""

from z3 import *
import numpy as np
import math
import random
import time
import json
import sys

def parse_block_constraints(puzzle, cells):
    constraints = []
    for block in puzzle:
        op = block["op"]
        target = block["target"]
        vars_in_block = [cells[i][j] for i, j in block["cells"]]
        if op == "":
            constraints.append(vars_in_block[0] == target)
        elif op == "add":
            constraints.append(Sum(vars_in_block) == target)
        elif op == "mul":
            product = vars_in_block[0]
            for v in vars_in_block[1:]:
                product *= v
            constraints.append(product == target)
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a - b == target, b - a == target))
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a / b == target, b / a == target))
        else:
            raise ValueError(f"Unsupported operation or malformed block: {block}")
    return constraints

def evaluate_puzzle(puzzle, size):
    X = [ [ Int("x_%s_%s" % (i+1, j+1)) for j in range(size) ]
        for i in range(size) ]
    cells_c  = [ And(1 <= X[i][j], X[i][j] <= size)
                for i in range(size) for j in range(size) ]
    rows_c   = [ Distinct(X[i]) for i in range(size) ]
    cols_c   = [ Distinct([ X[i][j] for i in range(size) ])
                for j in range(size) ]
    constraints = cells_c + rows_c + cols_c + parse_block_constraints(puzzle, X)
    instance = [[0] * size] * size
    instance = [ If(instance[i][j] == 0,
                    True,
                    X[i][j] == instance[i][j])
                 for i in range(size) for j in range(size) ]
    s = Solver()
    problem = constraints + instance
    s.add(problem)
    if s.check() == sat:
        m = s.model()
        solution = [ [ m.evaluate(X[i][j]) for j in range(size) ]
          for i in range(size) ]
        return solution
    else:
        return None

def generate_sizes(grid_size, num_cages, max_cage_size, max_singletons):
    cage_sizes = []
    singleton_count = 0
    for _ in range(num_cages-1):
        cage_size = np.random.randint(1, max_cage_size+1)
        if cage_size == 1:
            if singleton_count >= max_singletons-1:
                cage_size = np.random.randint(2, max_cage_size+1)
            else:
                singleton_count += 1
        cage_sizes.append(cage_size)
    if sum(cage_sizes) < grid_size**2 and (grid_size**2 - sum(cage_sizes)) <= max_cage_size:
        cage_sizes.append(grid_size**2 - sum(cage_sizes))
    else:
        return None
    return cage_sizes

def generate_solution_grid(n):
    """Generate a random Latin square using fast permutation method.

    Creates a base Latin square then applies random row, column, and symbol permutations
    to generate variety. Much faster than Z3 for larger grids.
    """
    # Create base Latin square: row i has values [i, i+1, i+2, ... i+n-1] mod n, shifted to 1-indexed
    grid = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]

    # Random permutation of rows
    row_perm = list(range(n))
    random.shuffle(row_perm)
    grid = [grid[i] for i in row_perm]

    # Random permutation of columns
    col_perm = list(range(n))
    random.shuffle(col_perm)
    grid = [[row[j] for j in col_perm] for row in grid]

    # Random permutation of symbols (1 to n)
    symbols = list(range(1, n + 1))
    random.shuffle(symbols)
    symbol_map = {i + 1: symbols[i] for i in range(n)}
    grid = [[symbol_map[cell] for cell in row] for row in grid]

    return grid

def choose_unassigned(grid_size, assigned):
    if grid_size**2 == len(assigned):
        return None
    while True:
        row = np.random.randint(0, grid_size)
        col = np.random.randint(0, grid_size)
        if (row, col) not in assigned:
            return (row, col)

def get_target(cage, grid):
    if cage["op"] == "sub":
        return abs(grid[cage["cells"][0][0]][cage["cells"][0][1]] - grid[cage["cells"][1][0]][cage["cells"][1][1]])
    elif cage["op"] == "div":
        return max((grid[cage["cells"][0][0]][cage["cells"][0][1]] / grid[cage["cells"][1][0]][cage["cells"][1][1]]),
                  (grid[cage["cells"][1][0]][cage["cells"][1][1]] / grid[cage["cells"][0][0]][cage["cells"][0][1]]))
    elif cage["op"] == "mul":
        s = 1
        for cell in cage["cells"]:
            s *= grid[cell[0]][cell[1]]
        return s
    else:
        s = 0
        for cell in cage["cells"]:
            s += grid[cell[0]][cell[1]]
        return s

def get_unassigned_neighbors(grid_size, assigned, cell):
    row = cell[0]
    col = cell[1]
    neighbors = []
    if row > 0:
        neighbors.append((row-1, col))
    if row < grid_size-1:
        neighbors.append((row+1, col))
    if col > 0:
        neighbors.append((row, col-1))
    if col < grid_size-1:
        neighbors.append((row, col+1))
    unassigned_neighbors = [neighbor for neighbor in neighbors if neighbor not in assigned]
    return unassigned_neighbors

def make_cage(puzzle, grid, grid_size, assigned, cage_size, index):
    to_assign = set()
    start_cell = choose_unassigned(grid_size, assigned)
    puzzle.append({"cells": [], "op": "", "target": 0})
    num_attempts = 10
    if not start_cell:
        return
    if cage_size == 1:
        puzzle[index]["cells"].append(list(start_cell))
        assigned.add(start_cell)
        puzzle[index]["target"] = grid[start_cell[0]][start_cell[1]]
        return
    puzzle[index]["cells"].append(list(start_cell))
    to_assign.add(start_cell)
    while len(puzzle[index]["cells"]) < cage_size:
        neighbors = []
        combined = assigned | to_assign
        for cell in puzzle[index]["cells"]:
            neighbors.extend(get_unassigned_neighbors(grid_size, combined, (cell[0], cell[1])))
        if not neighbors:
            if num_attempts == 0:
                puzzle[index]["cells"] = []
                return
            num_attempts -= 1
            start_cell = choose_unassigned(grid_size, assigned)
            puzzle[index]["cells"] = [list(start_cell)]
            to_assign = {start_cell}
            continue
        neighbor = neighbors[np.random.randint(0, len(neighbors))]
        puzzle[index]["cells"].append(list(neighbor))
        to_assign.add(neighbor)
    assigned.update(to_assign)
    puzzle[index]["op"] = generate_ops(puzzle[index], grid)
    puzzle[index]["target"] = get_target(puzzle[index], grid)

def generate_ops(cage, grid):
    cage_size = len(cage["cells"])
    options = []
    if cage_size == 1:
        return ""
    if cage_size == 2:
        if grid[cage["cells"][0][0]][cage["cells"][0][1]] % grid[cage["cells"][1][0]][cage["cells"][1][1]]==0 or grid[cage["cells"][1][0]][cage["cells"][1][1]] % grid[cage["cells"][0][0]][cage["cells"][0][1]]==0:
            options+=["div", "div"]
        options+=["sub", "mul", "add"]
    else:
        options = ["mul", "add"]
    return np.random.choice(options)

def generate_puzzle(grid_size, max_cage_size, max_singletons):
    num_cages = np.random.randint(math.ceil(grid_size**2 / max_cage_size), min((max_singletons+((grid_size**2 - max_singletons)//2) + max_singletons), grid_size**2))
    cage_sizes = generate_sizes(grid_size, num_cages, max_cage_size, max_singletons)
    while not cage_sizes:
        num_cages = np.random.randint(math.ceil(grid_size**2 / max_cage_size), min((max_singletons+((grid_size**2 - max_singletons)//2) + max_singletons), grid_size**2))
        cage_sizes = generate_sizes(grid_size, num_cages, max_cage_size, max_singletons)
    grid = generate_solution_grid(grid_size)
    puzzle = []
    assigned = set()
    for index in range(num_cages):
        make_cage(puzzle, grid, grid_size, assigned, cage_sizes[index], index)
        if puzzle[index]["cells"] == []:
            return None, None
    # Return both puzzle and solution - no need to re-solve with Z3
    return puzzle, grid

def generate_puzzle_set(num_puzzles, grid_size, max_cage_size, max_singletons):
    puzzle_set = []
    solution_set = []
    num_puzzles_generated = 0
    while num_puzzles_generated < num_puzzles:
        puzzle, solution = generate_puzzle(grid_size, max_cage_size, max_singletons)
        if puzzle and solution:
            puzzle_set.append(puzzle)
            solution_set.append(solution)
            num_puzzles_generated += 1
            if num_puzzles_generated % 10 == 0:
                print(f"  Generated {num_puzzles_generated}/{num_puzzles} size {grid_size} puzzles")
                sys.stdout.flush()
    return puzzle_set, solution_set

def main():
    # Load existing puzzles
    with open('./puzzles/puzzles_dict.json', 'r') as f:
        puzzles_full = json.load(f)

    print("=" * 60)
    print("KENKEN PUZZLE GENERATION - EXTENSION")
    print("=" * 60)
    print()
    sys.stdout.flush()

    # Generate 70 more 7x7 puzzles
    print("[1/3] Generating 70 more 7x7 puzzles...")
    sys.stdout.flush()
    start = time.time()
    p7, s7 = generate_puzzle_set(70, 7, 4, 2)
    end = time.time()
    # Extend existing list
    puzzles_full["7"].extend(p7)
    print(f"  Done in {end-start:.1f}s")
    print(f"  Total 7x7 puzzles: {len(puzzles_full['7'])}")
    print()
    sys.stdout.flush()

    # Save intermediate results
    with open('./puzzles/puzzles_dict.json', 'w') as f:
        json.dump(puzzles_full, f, indent=2)
    print("  Saved intermediate results")
    print()
    sys.stdout.flush()

    # Generate 100 8x8 puzzles
    print("[2/3] Generating 100 8x8 puzzles...")
    sys.stdout.flush()
    start = time.time()
    p8, s8 = generate_puzzle_set(100, 8, 4, 2)
    end = time.time()
    puzzles_full["8"] = p8
    print(f"  Done in {end-start:.1f}s")
    print()
    sys.stdout.flush()

    # Save intermediate results
    with open('./puzzles/puzzles_dict.json', 'w') as f:
        json.dump(puzzles_full, f, indent=2)
    print("  Saved intermediate results")
    print()
    sys.stdout.flush()

    # Generate 100 9x9 puzzles
    print("[3/3] Generating 100 9x9 puzzles...")
    sys.stdout.flush()
    start = time.time()
    p9, s9 = generate_puzzle_set(100, 9, 5, 2)
    end = time.time()
    puzzles_full["9"] = p9
    print(f"  Done in {end-start:.1f}s")
    print()
    sys.stdout.flush()

    # Save final results
    with open('./puzzles/puzzles_dict.json', 'w') as f:
        json.dump(puzzles_full, f, indent=2)

    print("=" * 60)
    print("PUZZLE GENERATION COMPLETE")
    print("=" * 60)
    print()
    for size in sorted(puzzles_full.keys(), key=int):
        print(f"Size {size}: {len(puzzles_full[size])} puzzles")
    print()
    print("Saved to: ./puzzles/puzzles_dict.json")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
