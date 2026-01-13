# -*- coding: utf-8 -*-
"""
Confusion-aware scoring for handwritten digit recognition.

Uses known confusion patterns from failure analysis to improve
error correction by boosting alternatives that match common
misrecognition patterns.

Key insight: 1<->7 accounts for 37% of all errors, so when the
model predicts 1 with low confidence, 7 should be ranked higher
than other alternatives.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


# =============================================================================
# Confusion Matrix Priors
# =============================================================================

# Known confusion patterns from FAILURE_ANALYSIS_REPORT.md
# Format: (predicted, actual) -> frequency
CONFUSION_PRIORS = {
    # 1 <-> 7 (37% of all errors)
    (1, 7): 0.37,
    (7, 1): 0.37,

    # 6 <-> 8 (18% combined)
    (6, 8): 0.10,
    (8, 6): 0.08,

    # 3 <-> 8 (5%)
    (3, 8): 0.05,
    (8, 3): 0.05,

    # 5 <-> 6 (4%)
    (5, 6): 0.04,
    (6, 5): 0.04,

    # 4 <-> 9 (3%)
    (4, 9): 0.03,
    (9, 4): 0.03,

    # 2 <-> 7 (2%)
    (2, 7): 0.02,
    (7, 2): 0.02,

    # 0 <-> 6 (rare but exists)
    (0, 6): 0.01,
    (6, 0): 0.01,

    # Operator confusions (less common but impactful)
    # + <-> x (both have crossing strokes)
    (10, 12): 0.02,  # add confused with mul
    (12, 10): 0.02,

    # - <-> / (both are single strokes)
    (11, 13): 0.01,  # div confused with sub
    (13, 11): 0.01,
}

# Confusion groups: sets of commonly confused characters
CONFUSION_GROUPS = [
    {1, 7},      # Vertical stroke with optional angle
    {6, 8, 0},   # Loop-based digits
    {3, 8},      # Two-curve digits
    {5, 6},      # Similar curves
    {4, 9},      # Closed/open top
    {2, 7},      # Angular digits
]


# =============================================================================
# Scoring Functions
# =============================================================================

def get_confusion_boost(predicted: int, alternative: int) -> float:
    """
    Get confusion-based boost for an alternative.

    Args:
        predicted: Originally predicted class
        alternative: Alternative class to consider

    Returns:
        Boost factor (0.0 to 1.0)
    """
    return CONFUSION_PRIORS.get((predicted, alternative), 0.0)


def get_confusion_group(digit: int) -> Optional[set]:
    """
    Get the confusion group containing a digit.

    Args:
        digit: Digit to look up

    Returns:
        Set of commonly confused digits, or None
    """
    for group in CONFUSION_GROUPS:
        if digit in group:
            return group
    return None


def rerank_alternatives(
    predicted: int,
    top_k: List[Tuple[int, float]],
    confusion_weight: float = 0.3
) -> List[Tuple[int, float]]:
    """
    Rerank alternatives using confusion priors.

    Boosts alternatives that match known confusion patterns.

    Args:
        predicted: Originally predicted class
        top_k: List of (class, confidence) tuples from CNN
        confusion_weight: Weight for confusion boost (0-1)

    Returns:
        Reranked list of (class, adjusted_score) tuples
    """
    if not top_k:
        return []

    reranked = []
    for alt_class, confidence in top_k:
        # Base score is CNN confidence
        score = confidence

        # Add confusion boost if this is a known confusion
        boost = get_confusion_boost(predicted, alt_class)
        if boost > 0:
            # Scale boost by how low the original confidence was
            # Lower confidence = more weight to confusion prior
            uncertainty = 1.0 - top_k[0][1]  # Use top prediction confidence
            adjusted_boost = boost * confusion_weight * (0.5 + 0.5 * uncertainty)
            score += adjusted_boost

        reranked.append((alt_class, score))

    # Sort by adjusted score (descending)
    reranked.sort(key=lambda x: -x[1])

    # Normalize scores
    max_score = max(s for _, s in reranked) if reranked else 1.0
    if max_score > 0:
        reranked = [(c, s / max_score) for c, s in reranked]

    return reranked


def rerank_cage_alternatives(
    predictions: List[int],
    all_top_k: List[List[Tuple[int, float]]],
    confusion_weight: float = 0.3
) -> List[List[Tuple[int, float]]]:
    """
    Rerank alternatives for all characters in a cage.

    Args:
        predictions: List of predicted classes for each position
        all_top_k: List of top-K predictions for each position
        confusion_weight: Weight for confusion boost

    Returns:
        List of reranked alternatives for each position
    """
    reranked_all = []

    for pred, top_k in zip(predictions, all_top_k):
        reranked = rerank_alternatives(pred, top_k, confusion_weight)
        reranked_all.append(reranked)

    return reranked_all


# =============================================================================
# Alternative Generation with Confusion Awareness
# =============================================================================

def generate_confusion_alternatives(
    predictions: List[int],
    all_top_k: List[List[Tuple[int, float]]],
    max_alternatives: int = 10,
    min_confidence: float = 0.1
) -> List[Tuple[List[int], float]]:
    """
    Generate alternative interpretations prioritized by confusion patterns.

    Args:
        predictions: List of predicted classes
        all_top_k: Top-K predictions for each position
        max_alternatives: Maximum alternatives to generate
        min_confidence: Minimum confidence threshold

    Returns:
        List of (alternative_predictions, score) tuples
    """
    if not predictions or not all_top_k:
        return []

    # Rerank all positions
    reranked = rerank_cage_alternatives(predictions, all_top_k)

    # Generate alternatives by considering single-position changes first
    alternatives = []

    # Original prediction
    original_score = sum(rk[0][1] if rk else 0 for rk in reranked) / len(reranked)
    alternatives.append((list(predictions), original_score))

    # Single-position changes (most likely corrections)
    for pos in range(len(predictions)):
        if pos >= len(reranked) or not reranked[pos]:
            continue

        for alt_class, alt_score in reranked[pos][1:5]:  # Top 4 alternatives
            if alt_score < min_confidence:
                continue

            new_pred = list(predictions)
            new_pred[pos] = alt_class

            # Calculate combined score
            combined_score = 0
            for i, (rk, p) in enumerate(zip(reranked, new_pred)):
                for c, s in rk:
                    if c == p:
                        combined_score += s
                        break
            combined_score /= len(predictions)

            alternatives.append((new_pred, combined_score))

    # Sort by score and limit
    alternatives.sort(key=lambda x: -x[1])
    alternatives = alternatives[:max_alternatives]

    return alternatives


def get_likely_corrections(
    prediction: int,
    confidence: float,
    top_k: List[Tuple[int, float]]
) -> List[Tuple[int, float]]:
    """
    Get likely corrections for a low-confidence prediction.

    Args:
        prediction: Current prediction
        confidence: Confidence of current prediction
        top_k: All top-K predictions

    Returns:
        List of (correction, likelihood) tuples
    """
    corrections = []

    # If confidence is high, less likely to need correction
    if confidence > 0.9:
        return corrections

    # Get confusion group for this digit
    group = get_confusion_group(prediction)

    for alt_class, alt_conf in top_k:
        if alt_class == prediction:
            continue

        # Base likelihood from confidence ratio
        likelihood = alt_conf / max(confidence, 0.01)

        # Boost if in same confusion group
        if group and alt_class in group:
            likelihood *= 1.5

        # Boost based on known confusion prior
        prior = get_confusion_boost(prediction, alt_class)
        if prior > 0:
            likelihood *= (1.0 + prior * 2)

        if likelihood > 0.1:  # Threshold
            corrections.append((alt_class, min(likelihood, 1.0)))

    # Sort by likelihood
    corrections.sort(key=lambda x: -x[1])

    return corrections[:5]  # Top 5 corrections


# =============================================================================
# Confusion Matrix Learning
# =============================================================================

class ConfusionTracker:
    """
    Track and learn confusion patterns from solver feedback.

    Updates priors based on actual corrections that lead to valid solutions.
    """

    def __init__(self):
        self.confusion_counts = {}
        self.total_corrections = 0

    def record_correction(self, predicted: int, actual: int):
        """Record a successful correction."""
        key = (predicted, actual)
        self.confusion_counts[key] = self.confusion_counts.get(key, 0) + 1
        self.total_corrections += 1

    def get_learned_priors(self, min_count: int = 5) -> Dict[Tuple[int, int], float]:
        """
        Get learned confusion priors.

        Args:
            min_count: Minimum observations to include

        Returns:
            Dictionary of (predicted, actual) -> frequency
        """
        if self.total_corrections == 0:
            return {}

        priors = {}
        for key, count in self.confusion_counts.items():
            if count >= min_count:
                priors[key] = count / self.total_corrections

        return priors

    def merge_with_static_priors(
        self,
        learned_weight: float = 0.3
    ) -> Dict[Tuple[int, int], float]:
        """
        Merge learned priors with static priors.

        Args:
            learned_weight: Weight for learned priors (0-1)

        Returns:
            Combined priors dictionary
        """
        learned = self.get_learned_priors()

        if not learned:
            return CONFUSION_PRIORS.copy()

        merged = {}
        all_keys = set(CONFUSION_PRIORS.keys()) | set(learned.keys())

        for key in all_keys:
            static = CONFUSION_PRIORS.get(key, 0.0)
            dynamic = learned.get(key, 0.0)
            merged[key] = (1 - learned_weight) * static + learned_weight * dynamic

        return merged


# =============================================================================
# Digit Similarity Functions
# =============================================================================

def digit_similarity(d1: int, d2: int) -> float:
    """
    Compute visual similarity between two digits.

    Based on known confusion patterns and visual features.

    Args:
        d1, d2: Digits to compare (0-9)

    Returns:
        Similarity score (0-1)
    """
    if d1 == d2:
        return 1.0

    # Symmetric lookup
    if d1 > d2:
        d1, d2 = d2, d1

    # Similarity matrix (lower triangle, symmetric)
    similarities = {
        (0, 6): 0.4,   # Both have loops
        (0, 8): 0.3,   # Loop similarity
        (1, 4): 0.2,   # Vertical stroke
        (1, 7): 0.7,   # Most confused pair
        (2, 7): 0.4,   # Angular
        (3, 5): 0.3,   # Curve direction
        (3, 8): 0.5,   # Two curves
        (4, 9): 0.5,   # Top portion
        (5, 6): 0.5,   # Curve direction
        (5, 8): 0.3,   # Top curve
        (6, 8): 0.6,   # Loop based
        (6, 9): 0.4,   # Loop and tail
        (8, 9): 0.3,   # Top portion
    }

    return similarities.get((d1, d2), 0.1)


def operator_similarity(op1: int, op2: int) -> float:
    """
    Compute visual similarity between operators.

    Args:
        op1, op2: Operator codes (10=add, 11=div, 12=mul, 13=sub)

    Returns:
        Similarity score (0-1)
    """
    if op1 == op2:
        return 1.0

    # Operator similarities
    similarities = {
        (10, 12): 0.5,  # + and x (crossing strokes)
        (11, 13): 0.4,  # / and - (single strokes)
        (10, 11): 0.2,  # + and /
        (10, 13): 0.3,  # + and -
        (12, 13): 0.2,  # x and -
        (11, 12): 0.2,  # / and x
    }

    key = (min(op1, op2), max(op1, op2))
    return similarities.get(key, 0.1)


# =============================================================================
# Test Functions
# =============================================================================

if __name__ == '__main__':
    print("Confusion-Aware Scoring Tests")
    print("=" * 50)

    # Test reranking
    print("\n1. Test reranking with 1->7 confusion:")
    top_k = [(1, 0.45), (7, 0.35), (4, 0.10), (2, 0.05), (9, 0.05)]
    reranked = rerank_alternatives(1, top_k, confusion_weight=0.3)
    print(f"   Original: {top_k}")
    print(f"   Reranked: {reranked}")

    # Test alternative generation
    print("\n2. Test alternative generation:")
    predictions = [1, 2]  # "12"
    all_top_k = [
        [(1, 0.5), (7, 0.3), (4, 0.1), (2, 0.1)],
        [(2, 0.8), (7, 0.1), (3, 0.05), (8, 0.05)]
    ]
    alts = generate_confusion_alternatives(predictions, all_top_k)
    print(f"   Predictions: {predictions}")
    print(f"   Alternatives: {alts[:5]}")

    # Test likely corrections
    print("\n3. Test likely corrections for low-confidence 1:")
    corrections = get_likely_corrections(1, 0.45, top_k)
    print(f"   Corrections: {corrections}")

    # Test confusion tracker
    print("\n4. Test confusion tracker:")
    tracker = ConfusionTracker()
    tracker.record_correction(1, 7)
    tracker.record_correction(1, 7)
    tracker.record_correction(1, 7)
    tracker.record_correction(6, 8)
    tracker.record_correction(3, 8)
    print(f"   Learned priors: {tracker.get_learned_priors(min_count=1)}")

    # Test similarity
    print("\n5. Test digit similarity:")
    for d1, d2 in [(1, 7), (6, 8), (3, 8), (1, 9)]:
        sim = digit_similarity(d1, d2)
        print(f"   {d1} <-> {d2}: {sim:.2f}")
