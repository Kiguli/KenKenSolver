# -*- coding: utf-8 -*-
"""
Constraint-based validation and error detection for KenKen puzzles.

Uses mathematical rules to detect impossible target values and
guide error correction by filtering to only valid alternatives.
"""

from typing import Set, List, Tuple, Optional, Dict, Any
from itertools import product


# =============================================================================
# Lookup Tables for Valid Target Ranges
# =============================================================================

# Maximum valid subtraction target = size - 1
MAX_SUB_TARGET = {3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 8}

# Maximum valid division target = size
MAX_DIV_TARGET = {3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 9: 9}


# =============================================================================
# Target Validation Functions
# =============================================================================

def is_valid_target(op: str, target: int, num_cells: int, size: int) -> bool:
    """
    Check if a target value is mathematically possible for the given operation.

    Args:
        op: Operation type ('add', 'sub', 'mul', 'div', or '' for single cell)
        target: The target value to check
        num_cells: Number of cells in the cage
        size: Puzzle size (3-9)

    Returns:
        True if the target is valid, False if impossible
    """
    if target <= 0:
        return False

    if op == "sub":
        # Subtraction: max difference is size - 1 (e.g., 9 - 1 = 8)
        return 1 <= target <= size - 1

    elif op == "div":
        # Division: max quotient is size (e.g., 9 / 1 = 9)
        return 1 <= target <= size

    elif op == "mul":
        # Multiplication: must have at least one valid factorization
        # Check if any value in [1, size] is a factor of target
        return any(target % v == 0 for v in range(1, size + 1))

    elif op == "add":
        # Addition: sum must be in range [num_cells, num_cells * size]
        min_sum = num_cells
        max_sum = num_cells * size
        return min_sum <= target <= max_sum

    elif op == "":
        # Single cell: target must be a valid cell value
        return 1 <= target <= size

    return True


def detect_impossible_cages(puzzle: List[Dict], size: int) -> List[Tuple]:
    """
    Detect cages with impossible target values.

    Args:
        puzzle: List of cage dictionaries with 'op', 'target', 'cells'
        size: Puzzle size (3-9)

    Returns:
        List of (cage_idx, error_type, detected_target, valid_range) tuples
    """
    errors = []

    for idx, cage in enumerate(puzzle):
        op = cage['op']
        target = cage['target']
        num_cells = len(cage['cells'])

        if op == "sub":
            max_target = size - 1
            if target > max_target or target <= 0:
                errors.append((idx, 'impossible_sub', target, (1, max_target)))

        elif op == "div":
            if target > size or target <= 0:
                errors.append((idx, 'impossible_div', target, (1, size)))

        elif op == "mul":
            # Check if any factor of target is in valid range
            valid_factors = [v for v in range(1, size + 1) if target % v == 0]
            if not valid_factors or target <= 0:
                errors.append((idx, 'impossible_mul', target, valid_factors))

        elif op == "add":
            min_sum = num_cells
            max_sum = num_cells * size
            if target < min_sum or target > max_sum:
                errors.append((idx, 'impossible_add', target, (min_sum, max_sum)))

        elif op == "":
            # Single cell
            if num_cells == 1 and (target < 1 or target > size):
                errors.append((idx, 'impossible_single', target, (1, size)))

    return errors


# =============================================================================
# Valid Cell Values Functions
# =============================================================================

def get_valid_cell_values(op: str, target: int, num_cells: int, size: int) -> Set[int]:
    """
    Get all mathematically valid cell values for a cage.

    Args:
        op: Operation type
        target: Target value
        num_cells: Number of cells in cage
        size: Puzzle size

    Returns:
        Set of valid cell values, empty set if impossible
    """
    if op == "sub":
        # |a - b| = target → values v where v±target in [1, size]
        valid = set()
        for v in range(1, size + 1):
            if 1 <= v + target <= size:  # v is smaller, v+target is larger
                valid.add(v)
                valid.add(v + target)
            if 1 <= v - target:  # v is larger, v-target is smaller
                valid.add(v)
                valid.add(v - target)
        return valid

    elif op == "div":
        # a/b = target or b/a = target
        valid = set()
        for v in range(1, size + 1):
            if v * target <= size:  # v is smaller value
                valid.add(v)
                valid.add(v * target)
            if v % target == 0 and v // target >= 1:  # v is larger value
                valid.add(v)
                valid.add(v // target)
        return valid

    elif op == "mul":
        # All values must be divisors of target
        return {v for v in range(1, size + 1) if target % v == 0}

    elif op == "add":
        # Each cell: [max(1, T-(n-1)*size), min(size, T-(n-1))]
        max_val = min(size, target - (num_cells - 1))
        min_val = max(1, target - (num_cells - 1) * size)
        if max_val < min_val:
            return set()
        return set(range(min_val, max_val + 1))

    elif op == "":
        # Single cell: target must equal cell value
        if 1 <= target <= size:
            return {target}
        return set()

    return set(range(1, size + 1))


# =============================================================================
# Valid Combinations Functions
# =============================================================================

def get_valid_combinations(op: str, target: int, num_cells: int, size: int) -> List[Tuple[int, ...]]:
    """
    Get all valid value combinations for a cage.

    Args:
        op: Operation type
        target: Target value
        num_cells: Number of cells in cage
        size: Puzzle size

    Returns:
        List of tuples representing valid value combinations
    """
    if num_cells == 1:
        if 1 <= target <= size:
            return [(target,)]
        return []

    if num_cells == 2:
        return _get_valid_pairs(op, target, size)

    # For 3+ cells (mul and add only for KenKen)
    if num_cells >= 3:
        if op == "mul":
            return _get_mul_combinations(target, num_cells, size)
        elif op == "add":
            return _get_add_combinations(target, num_cells, size)

    return []


def _get_valid_pairs(op: str, target: int, size: int) -> List[Tuple[int, int]]:
    """Get all valid pairs for 2-cell cages."""
    pairs = []

    if op == "sub":
        # |a - b| = target
        for a in range(1, size + 1):
            b = a + target
            if 1 <= b <= size:
                pairs.append((a, b))
                pairs.append((b, a))

    elif op == "div":
        # a/b = target (a = b*target)
        for b in range(1, size + 1):
            a = b * target
            if 1 <= a <= size:
                pairs.append((a, b))
                pairs.append((b, a))

    elif op == "mul":
        # a * b = target
        for a in range(1, size + 1):
            if target % a == 0:
                b = target // a
                if 1 <= b <= size:
                    pairs.append((a, b))

    elif op == "add":
        # a + b = target
        for a in range(1, size + 1):
            b = target - a
            if 1 <= b <= size:
                pairs.append((a, b))

    return pairs


def _get_mul_combinations(target: int, num_cells: int, size: int,
                          current: List[int] = None) -> List[Tuple[int, ...]]:
    """Recursively find all multiplication combinations."""
    if current is None:
        current = []

    if num_cells == 1:
        if 1 <= target <= size:
            return [tuple(current + [target])]
        return []

    results = []
    start = current[-1] if current else 1  # Avoid duplicate permutations
    for v in range(start, size + 1):
        if target % v == 0:
            results.extend(_get_mul_combinations(
                target // v, num_cells - 1, size, current + [v]
            ))
    return results


def _get_add_combinations(target: int, num_cells: int, size: int,
                          current: List[int] = None) -> List[Tuple[int, ...]]:
    """Recursively find all addition combinations."""
    if current is None:
        current = []

    if num_cells == 1:
        if 1 <= target <= size:
            return [tuple(current + [target])]
        return []

    results = []
    start = current[-1] if current else 1  # Avoid duplicate permutations
    max_v = min(size, target - (num_cells - 1))  # Leave room for other cells
    for v in range(start, max_v + 1):
        results.extend(_get_add_combinations(
            target - v, num_cells - 1, size, current + [v]
        ))
    return results


# =============================================================================
# Target Inference from OCR Alternatives
# =============================================================================

def generate_target_candidates(predictions: List[int], top_k: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
    """
    Generate all possible target values from OCR predictions.

    Args:
        predictions: List of predicted digit/operator values
        top_k: List of (prediction, confidence) tuples for each position

    Returns:
        List of (target_value, confidence) tuples, sorted by confidence
    """
    if not predictions or not top_k:
        return []

    # Generate all combinations of digit predictions
    # Exclude operators (10=add, 11=div, 12=mul, 13=sub)
    digit_positions = []
    operator_position = None

    for i, pred in enumerate(predictions):
        if pred >= 10:  # Operator
            operator_position = i
        else:
            digit_positions.append(i)

    if not digit_positions:
        return []

    # Get digit alternatives (only digits 0-9)
    digit_alternatives = []
    for pos in digit_positions:
        alts = [(p, c) for p, c in top_k[pos] if p < 10]
        if not alts:
            alts = [(predictions[pos], 1.0)]
        digit_alternatives.append(alts)

    # Generate all combinations
    candidates = []
    for combo in product(*digit_alternatives):
        digits = [p for p, _ in combo]
        conf = sum(c for _, c in combo) / len(combo)

        # Convert digits to target number
        target = 0
        for d in digits:
            target = target * 10 + d

        candidates.append((target, conf))

    # Sort by confidence (descending)
    candidates.sort(key=lambda x: -x[1])

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for target, conf in candidates:
        if target not in seen:
            seen.add(target)
            unique.append((target, conf))

    return unique


def infer_correct_target(cage: Dict, alternatives: Dict, size: int) -> Optional[int]:
    """
    Given a cage with impossible target, infer the correct target.

    Uses top-K digit alternatives filtered by constraint validity.

    Args:
        cage: Cage dictionary with 'op', 'target', 'cells'
        alternatives: Dictionary with 'predictions' and 'top_k'
        size: Puzzle size

    Returns:
        Most likely valid target, or None if no valid candidate found
    """
    op = cage['op']
    num_cells = len(cage['cells'])

    predictions = alternatives.get('predictions', [])
    top_k = alternatives.get('top_k', [])

    if not predictions or not top_k:
        return None

    # Generate all candidate targets
    candidates = generate_target_candidates(predictions, top_k)

    # Filter to only valid targets
    for target, conf in candidates:
        if is_valid_target(op, target, num_cells, size):
            return target

    return None


# =============================================================================
# Constraint-Filtered Alternative Generation
# =============================================================================

def filter_alternatives_by_constraints(cage: Dict, alternatives: Dict, size: int) -> List[Dict]:
    """
    Filter OCR alternatives to only those producing valid targets.

    Args:
        cage: Cage dictionary
        alternatives: Dictionary with 'predictions', 'top_k', 'cage'
        size: Puzzle size

    Returns:
        List of valid alternative cage dictionaries
    """
    op = cage['op']
    num_cells = len(cage['cells'])
    cells = cage['cells']

    predictions = alternatives.get('predictions', [])
    top_k = alternatives.get('top_k', [])

    if not predictions or not top_k:
        return []

    # Generate candidate targets
    candidates = generate_target_candidates(predictions, top_k)

    # Build valid alternatives
    valid_alternatives = []
    for target, conf in candidates:
        if is_valid_target(op, target, num_cells, size):
            valid_alternatives.append({
                'cells': cells,
                'op': op,
                'target': target,
                'confidence': conf
            })

    return valid_alternatives


def get_cage_error_likelihood(cage: Dict, alternatives: Dict, size: int) -> float:
    """
    Estimate how likely a cage contains an OCR error.

    Based on:
    - Confidence of top prediction
    - Whether target is impossible
    - Number of digits in target (more digits = more error prone)

    Args:
        cage: Cage dictionary
        alternatives: Alternatives dictionary with top_k predictions
        size: Puzzle size

    Returns:
        Error likelihood score (higher = more likely error)
    """
    op = cage['op']
    target = cage['target']
    num_cells = len(cage['cells'])

    # Start with base score
    score = 0.0

    # Check if target is impossible (definite error)
    if not is_valid_target(op, target, num_cells, size):
        score += 10.0

    # Check prediction confidence
    top_k = alternatives.get('top_k', [])
    if top_k:
        for char_top_k in top_k:
            if char_top_k:
                top_conf = char_top_k[0][1]
                # Low confidence = higher error likelihood
                if top_conf < 0.5:
                    score += 2.0
                elif top_conf < 0.7:
                    score += 1.0
                elif top_conf < 0.9:
                    score += 0.5

    # Multi-digit targets are more error prone
    num_digits = len(str(target))
    if num_digits >= 2:
        score += 1.0 * (num_digits - 1)

    return score


# =============================================================================
# Test/Debug Functions
# =============================================================================

def print_valid_combinations_table(size: int):
    """Print lookup tables for debugging."""
    print(f"\n=== Valid Combinations for Size {size} ===\n")

    print("SUBTRACTION (2 cells):")
    for target in range(1, size):
        pairs = _get_valid_pairs("sub", target, size)
        values = get_valid_cell_values("sub", target, 2, size)
        print(f"  T={target}: pairs={pairs[:5]}{'...' if len(pairs) > 5 else ''}, values={values}")

    print("\nDIVISION (2 cells):")
    for target in range(1, size + 1):
        pairs = _get_valid_pairs("div", target, size)
        values = get_valid_cell_values("div", target, 2, size)
        print(f"  T={target}: pairs={pairs}, values={values}")

    print("\nMULTIPLICATION (2 cells, sample targets):")
    for target in [2, 6, 12, 18, 24]:
        if target <= size * size:
            pairs = _get_valid_pairs("mul", target, size)
            values = get_valid_cell_values("mul", target, 2, size)
            print(f"  T={target}: pairs={pairs}, values={values}")

    print("\nADDITION (2 cells, sample targets):")
    for target in [3, 5, 8, 10, 15]:
        if target <= 2 * size:
            pairs = _get_valid_pairs("add", target, size)
            values = get_valid_cell_values("add", target, 2, size)
            print(f"  T={target}: pairs={pairs[:5]}{'...' if len(pairs) > 5 else ''}, values={values}")


if __name__ == '__main__':
    # Test the lookup tables
    print_valid_combinations_table(9)

    # Test impossible cage detection
    test_puzzle = [
        {'cells': [[0, 0], [0, 1]], 'op': 'sub', 'target': 10},  # Impossible: 10 > 8
        {'cells': [[1, 0], [1, 1]], 'op': 'div', 'target': 12},  # Impossible: 12 > 9
        {'cells': [[2, 0]], 'op': '', 'target': 15},             # Impossible: 15 > 9
        {'cells': [[2, 1], [2, 2]], 'op': 'add', 'target': 3},   # Valid
    ]

    print("\n=== Testing Impossible Cage Detection ===")
    errors = detect_impossible_cages(test_puzzle, 9)
    for cage_idx, error_type, detected, valid_range in errors:
        print(f"  Cage {cage_idx}: {error_type}, detected={detected}, valid_range={valid_range}")
