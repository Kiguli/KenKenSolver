# -*- coding: utf-8 -*-
"""
Constraint propagation for KenKen puzzles.

Uses arc consistency (AC-3) to eliminate impossible values before
passing to Z3 solver. This reduces the search space and helps
filter OCR alternatives to only those that are mathematically valid.

Key insight: Many OCR alternatives can be eliminated purely through
Latin square and cage arithmetic constraints, without needing full
Z3 solving.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import deque
from itertools import permutations

from constraint_validation import (
    get_valid_cell_values,
    get_valid_combinations,
    is_valid_target
)


# =============================================================================
# Domain Representation
# =============================================================================

class CellDomain:
    """Domain of possible values for a single cell."""

    def __init__(self, row: int, col: int, size: int):
        self.row = row
        self.col = col
        self.size = size
        self.values = set(range(1, size + 1))

    def remove(self, value: int) -> bool:
        """Remove a value from domain. Returns True if removed."""
        if value in self.values:
            self.values.discard(value)
            return True
        return False

    def assign(self, value: int):
        """Assign a single value (remove all others)."""
        self.values = {value}

    def is_empty(self) -> bool:
        return len(self.values) == 0

    def is_singleton(self) -> bool:
        return len(self.values) == 1

    def get_value(self) -> Optional[int]:
        """Get the single value if domain is singleton."""
        if self.is_singleton():
            return next(iter(self.values))
        return None

    def intersect(self, valid_values: Set[int]) -> bool:
        """Intersect with valid values. Returns True if domain changed."""
        new_values = self.values & valid_values
        if new_values != self.values:
            self.values = new_values
            return True
        return False

    def copy(self) -> 'CellDomain':
        """Create a copy of this domain."""
        new_domain = CellDomain(self.row, self.col, self.size)
        new_domain.values = self.values.copy()
        return new_domain

    def __repr__(self):
        return f"({self.row},{self.col}):{self.values}"


class PuzzleDomains:
    """Domains for all cells in a puzzle."""

    def __init__(self, size: int):
        self.size = size
        self.domains = {}
        for r in range(size):
            for c in range(size):
                self.domains[(r, c)] = CellDomain(r, c, size)

    def get(self, row: int, col: int) -> CellDomain:
        return self.domains[(row, col)]

    def is_consistent(self) -> bool:
        """Check if all domains are non-empty."""
        return all(not d.is_empty() for d in self.domains.values())

    def is_complete(self) -> bool:
        """Check if all domains are singletons."""
        return all(d.is_singleton() for d in self.domains.values())

    def get_solution(self) -> Optional[List[List[int]]]:
        """Get solution if complete, else None."""
        if not self.is_complete():
            return None
        solution = [[0] * self.size for _ in range(self.size)]
        for (r, c), d in self.domains.items():
            solution[r][c] = d.get_value()
        return solution

    def copy(self) -> 'PuzzleDomains':
        """Deep copy all domains."""
        new_puzzle = PuzzleDomains(self.size)
        for key, domain in self.domains.items():
            new_puzzle.domains[key] = domain.copy()
        return new_puzzle


# =============================================================================
# Constraint Propagation
# =============================================================================

def apply_row_constraint(domains: PuzzleDomains, row: int) -> bool:
    """
    Apply row uniqueness constraint.

    If a cell has only one possible value, remove that value
    from all other cells in the row.

    Returns True if any domain changed.
    """
    changed = False
    size = domains.size

    for col in range(size):
        domain = domains.get(row, col)
        if domain.is_singleton():
            value = domain.get_value()
            # Remove from other cells in row
            for other_col in range(size):
                if other_col != col:
                    if domains.get(row, other_col).remove(value):
                        changed = True

    return changed


def apply_column_constraint(domains: PuzzleDomains, col: int) -> bool:
    """
    Apply column uniqueness constraint.

    Returns True if any domain changed.
    """
    changed = False
    size = domains.size

    for row in range(size):
        domain = domains.get(row, col)
        if domain.is_singleton():
            value = domain.get_value()
            # Remove from other cells in column
            for other_row in range(size):
                if other_row != row:
                    if domains.get(other_row, col).remove(value):
                        changed = True

    return changed


def apply_cage_constraint(
    domains: PuzzleDomains,
    cage: Dict[str, Any]
) -> bool:
    """
    Apply cage arithmetic constraint.

    Restricts cell domains to only values that can satisfy
    the cage's target with some valid combination.

    Returns True if any domain changed.
    """
    op = cage['op']
    target = cage['target']
    cells = [tuple(c) for c in cage['cells']]
    size = domains.size

    if not is_valid_target(op, target, len(cells), size):
        # Invalid target - can't satisfy this cage
        return False

    # Get valid cell values for this cage
    valid = get_valid_cell_values(op, target, len(cells), size)

    # Intersect with each cell's domain
    changed = False
    for cell in cells:
        if domains.get(*cell).intersect(valid):
            changed = True

    # For 2-cell cages, we can be more precise
    if len(cells) == 2:
        combos = get_valid_combinations(op, target, 2, size)
        if combos:
            # Get union of values that appear in valid combinations
            valid_for_cell = [set(), set()]
            for combo in combos:
                for i, v in enumerate(combo):
                    valid_for_cell[i].add(v)

            # This isn't quite right - we need to consider the domains
            # Let's filter to combos possible with current domains
            cell0_domain = domains.get(*cells[0]).values
            cell1_domain = domains.get(*cells[1]).values

            possible_combos = []
            for combo in combos:
                # Check both orderings
                if combo[0] in cell0_domain and combo[1] in cell1_domain:
                    possible_combos.append(combo)
                if combo[1] in cell0_domain and combo[0] in cell1_domain:
                    possible_combos.append((combo[1], combo[0]))

            if possible_combos:
                # Restrict to values in possible combos
                valid0 = {c[0] for c in possible_combos}
                valid1 = {c[1] for c in possible_combos}

                if domains.get(*cells[0]).intersect(valid0):
                    changed = True
                if domains.get(*cells[1]).intersect(valid1):
                    changed = True

    return changed


def apply_naked_singles(domains: PuzzleDomains) -> bool:
    """
    If a value appears in only one cell's domain in a row/column,
    assign it to that cell.

    Returns True if any domain changed.
    """
    changed = False
    size = domains.size

    # Check rows
    for row in range(size):
        for value in range(1, size + 1):
            # Find cells that can hold this value
            possible_cols = []
            for col in range(size):
                if value in domains.get(row, col).values:
                    possible_cols.append(col)

            if len(possible_cols) == 1:
                domain = domains.get(row, possible_cols[0])
                if not domain.is_singleton():
                    domain.assign(value)
                    changed = True

    # Check columns
    for col in range(size):
        for value in range(1, size + 1):
            # Find cells that can hold this value
            possible_rows = []
            for row in range(size):
                if value in domains.get(row, col).values:
                    possible_rows.append(row)

            if len(possible_rows) == 1:
                domain = domains.get(possible_rows[0], col)
                if not domain.is_singleton():
                    domain.assign(value)
                    changed = True

    return changed


def propagate(
    domains: PuzzleDomains,
    cages: List[Dict],
    max_iterations: int = 100
) -> bool:
    """
    Main propagation loop using AC-3 style algorithm.

    Iteratively applies constraints until no changes occur or
    inconsistency is detected.

    Args:
        domains: PuzzleDomains to modify in place
        cages: List of cage dictionaries
        max_iterations: Safety limit

    Returns:
        True if consistent state reached, False if inconsistent
    """
    for iteration in range(max_iterations):
        changed = False

        # Apply row constraints
        for row in range(domains.size):
            if apply_row_constraint(domains, row):
                changed = True
            if not domains.is_consistent():
                return False

        # Apply column constraints
        for col in range(domains.size):
            if apply_column_constraint(domains, col):
                changed = True
            if not domains.is_consistent():
                return False

        # Apply cage constraints
        for cage in cages:
            if apply_cage_constraint(domains, cage):
                changed = True
            if not domains.is_consistent():
                return False

        # Apply naked singles
        if apply_naked_singles(domains):
            changed = True
        if not domains.is_consistent():
            return False

        if not changed:
            break

    return domains.is_consistent()


# =============================================================================
# Alternative Filtering
# =============================================================================

def filter_alternatives_by_propagation(
    puzzle: List[Dict],
    alternatives: List[Dict],
    size: int
) -> List[Dict]:
    """
    Filter cage alternatives to only those that are consistent
    with constraint propagation.

    Args:
        puzzle: Current puzzle (list of cages)
        alternatives: List of alternative cage interpretations
        size: Puzzle size

    Returns:
        Filtered list of alternatives
    """
    valid_alternatives = []

    for alt in alternatives:
        # Create test puzzle with this alternative
        test_puzzle = []
        alt_cage_cells = set(tuple(c) for c in alt['cells'])

        for cage in puzzle:
            cage_cells = set(tuple(c) for c in cage['cells'])
            if cage_cells == alt_cage_cells:
                # Replace with alternative
                test_puzzle.append(alt)
            else:
                test_puzzle.append(cage)

        # Try propagation
        domains = PuzzleDomains(size)
        if propagate(domains, test_puzzle):
            valid_alternatives.append(alt)

    return valid_alternatives


def rank_alternatives_by_constraint_tightness(
    puzzle: List[Dict],
    alternatives: List[Dict],
    size: int
) -> List[Tuple[Dict, float]]:
    """
    Rank alternatives by how much they constrain the puzzle.

    More constrained = more likely to be correct (assuming valid).

    Args:
        puzzle: Current puzzle
        alternatives: List of alternative interpretations
        size: Puzzle size

    Returns:
        List of (alternative, score) tuples, sorted by score (descending)
    """
    scored = []

    for alt in alternatives:
        # Create test puzzle with this alternative
        test_puzzle = []
        alt_cage_cells = set(tuple(c) for c in alt['cells'])

        for cage in puzzle:
            cage_cells = set(tuple(c) for c in cage['cells'])
            if cage_cells == alt_cage_cells:
                test_puzzle.append(alt)
            else:
                test_puzzle.append(cage)

        # Run propagation
        domains = PuzzleDomains(size)
        if propagate(domains, test_puzzle):
            # Score = reduction in domain sizes
            total_remaining = sum(len(d.values) for d in domains.domains.values())
            max_possible = size * size * size
            reduction_score = 1.0 - (total_remaining / max_possible)

            # Bonus for more singletons
            singletons = sum(1 for d in domains.domains.values() if d.is_singleton())
            singleton_bonus = singletons / (size * size)

            score = reduction_score + 0.5 * singleton_bonus
            scored.append((alt, score))

    # Sort by score (descending)
    scored.sort(key=lambda x: -x[1])

    return scored


# =============================================================================
# Pre-solve Simplification
# =============================================================================

def simplify_puzzle(
    puzzle: List[Dict],
    size: int
) -> Tuple[PuzzleDomains, bool]:
    """
    Apply constraint propagation to simplify puzzle before Z3.

    Args:
        puzzle: List of cage dictionaries
        size: Puzzle size

    Returns:
        (simplified_domains, is_consistent)
    """
    domains = PuzzleDomains(size)
    is_consistent = propagate(domains, puzzle)
    return domains, is_consistent


def get_fixed_cells(domains: PuzzleDomains) -> List[Tuple[int, int, int]]:
    """
    Get cells that have been determined by propagation.

    Returns list of (row, col, value) tuples.
    """
    fixed = []
    for (row, col), domain in domains.domains.items():
        if domain.is_singleton():
            fixed.append((row, col, domain.get_value()))
    return fixed


def get_constrained_cells(
    domains: PuzzleDomains,
    max_domain_size: int = 3
) -> List[Tuple[int, int, Set[int]]]:
    """
    Get cells with small (but not singleton) domains.

    These are good candidates for branching in search.

    Returns list of (row, col, possible_values) tuples.
    """
    constrained = []
    for (row, col), domain in domains.domains.items():
        if 1 < len(domain.values) <= max_domain_size:
            constrained.append((row, col, domain.values.copy()))
    return constrained


# =============================================================================
# Integration with Error Correction
# =============================================================================

def validate_correction_with_propagation(
    original_puzzle: List[Dict],
    corrected_cage: Dict,
    size: int
) -> bool:
    """
    Check if a proposed correction is consistent with constraints.

    Args:
        original_puzzle: Original puzzle cages
        corrected_cage: Proposed corrected cage
        size: Puzzle size

    Returns:
        True if correction leads to consistent state
    """
    # Build puzzle with correction
    corrected_cells = set(tuple(c) for c in corrected_cage['cells'])
    test_puzzle = []

    for cage in original_puzzle:
        cage_cells = set(tuple(c) for c in cage['cells'])
        if cage_cells == corrected_cells:
            test_puzzle.append(corrected_cage)
        else:
            test_puzzle.append(cage)

    # Run propagation
    domains = PuzzleDomains(size)
    return propagate(domains, test_puzzle)


def find_correction_candidates(
    puzzle: List[Dict],
    cage_idx: int,
    possible_targets: List[int],
    size: int
) -> List[Tuple[int, float]]:
    """
    Find which target corrections are consistent with constraints.

    Args:
        puzzle: Current puzzle
        cage_idx: Index of cage to correct
        possible_targets: List of possible target values
        size: Puzzle size

    Returns:
        List of (target, score) tuples for valid corrections
    """
    original_cage = puzzle[cage_idx]
    valid_corrections = []

    for target in possible_targets:
        # Create corrected cage
        corrected = {
            'cells': original_cage['cells'],
            'op': original_cage['op'],
            'target': target
        }

        # Test with propagation
        if validate_correction_with_propagation(puzzle, corrected, size):
            # Score based on constraint tightness
            test_puzzle = puzzle.copy()
            test_puzzle[cage_idx] = corrected

            domains = PuzzleDomains(size)
            propagate(domains, test_puzzle)

            # Higher score = more constrained (better)
            total = sum(len(d.values) for d in domains.domains.values())
            max_total = size * size * size
            score = 1.0 - (total / max_total)

            valid_corrections.append((target, score))

    # Sort by score (descending)
    valid_corrections.sort(key=lambda x: -x[1])

    return valid_corrections


# =============================================================================
# Test Functions
# =============================================================================

if __name__ == '__main__':
    print("Constraint Propagation Tests")
    print("=" * 50)

    # Test with simple 3x3 puzzle
    size = 3
    puzzle = [
        {'cells': [[0, 0], [0, 1]], 'op': 'add', 'target': 4},
        {'cells': [[0, 2]], 'op': '', 'target': 2},
        {'cells': [[1, 0]], 'op': '', 'target': 2},
        {'cells': [[1, 1], [1, 2]], 'op': 'sub', 'target': 1},
        {'cells': [[2, 0], [2, 1]], 'op': 'mul', 'target': 3},
        {'cells': [[2, 2]], 'op': '', 'target': 1},
    ]

    print(f"\n1. Test propagation on 3x3 puzzle:")
    domains, consistent = simplify_puzzle(puzzle, size)
    print(f"   Consistent: {consistent}")
    print(f"   Fixed cells: {get_fixed_cells(domains)}")

    print("\n2. Domain state after propagation:")
    for row in range(size):
        row_str = "   "
        for col in range(size):
            d = domains.get(row, col)
            row_str += f"{d.values} "
        print(row_str)

    # Test validation
    print("\n3. Test correction validation:")
    # Valid correction
    valid_correction = {'cells': [[0, 0], [0, 1]], 'op': 'add', 'target': 4}
    print(f"   Valid (target=4): {validate_correction_with_propagation(puzzle, valid_correction, size)}")

    # Invalid correction
    invalid_correction = {'cells': [[0, 0], [0, 1]], 'op': 'add', 'target': 10}
    print(f"   Invalid (target=10): {validate_correction_with_propagation(puzzle, invalid_correction, size)}")

    # Test finding corrections
    print("\n4. Test finding correction candidates:")
    candidates = find_correction_candidates(puzzle, 0, [3, 4, 5, 6], size)
    print(f"   Candidates for cage 0 (add): {candidates}")
