# KenKen Handwritten Solver: V1 vs V2 Comparison Report

**Date:** January 2026
**Dataset:** KenKen-handwritten-v2 (300px cells, MNIST test set digits)

---

## Executive Summary

| Size | V1 Baseline | V2 Baseline | V1 Corrected | V2 Corrected | Absolute Gain |
|------|-------------|-------------|--------------|--------------|---------------|
| 3x3  | 69%         | **98%**     | 89%          | **100%**     | +11%          |
| 4x4  | 36%         | **86%**     | 58%          | **90%**      | +32%          |
| 5x5  | 18%         | **36%**     | 26%          | **74%**      | +48%          |
| 6x6  | 7%          | **32%**     | 15%          | **65%**      | +50%          |
| 7x7  | 1%          | **10%**     | 2%           | **43%**      | +41%          |
| 9x9  | 0%          | **13%**     | 1%           | **24%**      | +23%          |

**Key Achievement:** V2 exceeds V1's *corrected* performance using baseline OCR alone for sizes 3x3-5x5, demonstrating the power of improved character recognition.

---

## Part 1: Methodology Differences

### 1.1 CNN Architecture

| Aspect | V1 (CNN_v2) | V2 (ImprovedCNN) |
|--------|-------------|------------------|
| Conv Layers | 2 | 4 |
| BatchNorm | None | After every conv layer |
| FC Hidden Units | 128 | 256 |
| Dropout Rate | 0.25 | 0.40 |
| Total Parameters | ~400K | **872,558** |
| Architecture | Conv→Pool→Conv→Pool→FC | Conv→BN→Conv→BN→Pool→Conv→BN→Conv→BN→Pool→FC→BN |

**V2 Architecture Detail:**
```
Block 1: 1→32→32 channels, BatchNorm, MaxPool (28→14px)
Block 2: 32→64→64 channels, BatchNorm, MaxPool (14→7px)
FC: 3136→256→14 with BatchNorm and Dropout(0.4)
```

**Why This Matters:**
- Deeper networks capture more complex patterns in handwriting
- BatchNorm stabilizes training and enables higher learning rates
- More FC units provide better class separation
- Higher dropout prevents overfitting on limited handwritten data

### 1.2 Training Data

| Aspect | V1 | V2 |
|--------|-----|-----|
| Source | MNIST training set only | Actual board images + MNIST |
| Handwritten Samples | ~60,000 (MNIST) | 17,742 (extracted from boards) |
| Computer Samples | 0 | 19,474 (from KenKen-300px boards) |
| Total Training | ~60,000 | 25,345 → 19,058 (balanced) |
| Class Balance | Imbalanced | Balanced to median count |
| Augmentation | 1x | **10x** (with augmentation) |
| Final Training Size | ~60,000 | **152,460** |

**V2 Data Extraction:**
- Characters extracted using *same pipeline as inference* (segment_cell, get_contours, get_character)
- Eliminates domain gap between training and inference
- 70% handwritten / 30% computer-generated ratio

### 1.3 Data Augmentation

| Technique | V1 | V2 |
|-----------|-----|-----|
| Random Rotation | ±5° | **±20°** |
| Random Scale | 0.9-1.1 | **0.5-1.2** |
| Random Translation | ±2px | ±3px |
| Gaussian Noise | σ=0.02 | σ=0.03 |
| **Elastic Deformation** | None | **α=8, σ=3** |
| **Morphological Ops** | None | **Erosion/Dilation** |
| **Aspect Ratio** | None | **0.9-1.1** |

**Elastic Deformation:** Simulates natural handwriting variation by applying smooth spatial transformations, critical for generalizing to unseen handwriting styles.

**Morphological Operations:** Erosion (thinning) and dilation (thickening) simulate different pen widths and writing pressures.

### 1.4 Training Process

| Aspect | V1 | V2 |
|--------|-----|-----|
| Loss Function | Cross-Entropy | **Focal Loss** (γ=2.0) |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.0003) |
| Batch Size | 32 | 64 |
| Epochs | 50 | 100 (early stopping) |
| Early Stopping | None | **Patience=15** |
| Best Validation Acc | ~95% | **99.9%** |

**Focal Loss:** Down-weights well-classified examples, focusing training on hard cases. Formula: FL(p_t) = -α(1-p_t)^γ log(p_t)

### 1.5 Error Correction Methods

Both V1 and V2 use the same error correction framework from `solve_all_sizes.py`:

| Method | Description | Max Errors |
|--------|-------------|------------|
| `none` | Baseline solve succeeds | 0 |
| `single` | Try top-K alternatives for each cage | 1 |
| `two` | Try combinations of 2 cages | 2 |
| `three` | Try combinations of 3 cages | 3 |
| `uncorrectable` | Exceeds correction limit | >3 |

**Constraint Validation:** Both versions use mathematical constraints to:
1. Detect impossible targets (e.g., subtraction > size-1)
2. Filter alternatives to only valid values
3. Prioritize cages by error likelihood

---

## Part 2: Results Comparison

### 2.1 Baseline OCR Accuracy

| Size | V1 Baseline | V2 Baseline | Improvement |
|------|-------------|-------------|-------------|
| 3x3  | 69%         | 98%         | **+29%**    |
| 4x4  | 36%         | 86%         | **+50%**    |
| 5x5  | 18%         | 36%         | **+18%**    |
| 6x6  | 7%          | 32%         | **+25%**    |
| 7x7  | 1%          | 10%         | **+9%**     |
| 9x9  | 0%          | 13%         | **+13%**    |

**Analysis:** The improved CNN provides massive gains in baseline accuracy, particularly for smaller puzzles where a single OCR error is fatal.

### 2.2 Corrected Accuracy

| Size | V1 Corrected | V2 Corrected | Improvement |
|------|--------------|--------------|-------------|
| 3x3  | 89%          | 100%         | **+11%**    |
| 4x4  | 58%          | 90%          | **+32%**    |
| 5x5  | 26%          | 74%          | **+48%**    |
| 6x6  | 15%          | 65%          | **+50%**    |
| 7x7  | 2%           | 43%          | **+41%**    |
| 9x9  | 1%           | 24%          | **+23%**    |

### 2.3 Correction Method Breakdown (V2)

| Size | None | Single | Two | Three | Uncorrectable |
|------|------|--------|-----|-------|---------------|
| 3x3  | 98   | 2      | 0   | 0     | 0             |
| 4x4  | 86   | 4      | 0   | 0     | 10            |
| 5x5  | 36   | 31     | 5   | 2     | 26            |
| 6x6  | 32   | 30     | 3   | 0     | 35            |
| 7x7  | 10   | 25     | 7   | 1     | 57            |
| 9x9  | 13   | 8      | 3   | 0     | 76            |

**Key Observation:** Single-error correction is highly effective in V2 (70 additional puzzles solved), enabled by fewer OCR errors making single corrections sufficient.

### 2.4 CNN Validation Performance

**V2 ImprovedCNN Per-Class Accuracy:**

| Class | Accuracy | Errors |
|-------|----------|--------|
| 0     | 99.6%    | 1 (→2) |
| 1     | 99.7%    | 1 (→2) |
| 2     | 99.7%    | 1 (→8) |
| 3     | 99.7%    | 1 (→5) |
| 4     | 99.7%    | 1 (→0) |
| 5     | 98.2%    | 5 (→6,8,9) |
| 6     | 100.0%   | 0 |
| 7     | 100.0%   | 0 |
| 8     | 100.0%   | 0 |
| 9     | 100.0%   | 0 |
| +     | 100.0%   | 0 |
| ×     | 100.0%   | 0 |
| -     | 100.0%   | 0 |

**Notable:** The 1↔7 confusion that dominated V1 errors (37% of all errors) has been **completely eliminated** in V2 validation. This is the single biggest improvement from the deeper architecture and better augmentation.

---

## Part 3: Analysis of Remaining Errors

### 3.1 Uncorrectable Puzzle Distribution

| Size | Uncorrectable | % of Total |
|------|---------------|------------|
| 3x3  | 0             | 0%         |
| 4x4  | 10            | 10%        |
| 5x5  | 26            | 26%        |
| 6x6  | 35            | 35%        |
| 7x7  | 57            | 57%        |
| 9x9  | 76            | 76%        |
| **Total** | **204** | **34%** |

### 3.2 Root Causes of Remaining Failures

#### Cause 1: Cumulative Error Probability (Primary Factor)

Even with 99.9% per-character accuracy, larger puzzles have more characters:

| Size | Cells | Cages (~) | Characters (~) | P(no errors) |
|------|-------|-----------|----------------|--------------|
| 3x3  | 9     | 4-5       | 10-15          | 98.5%        |
| 4x4  | 16    | 6-8       | 16-24          | 97.6%        |
| 5x5  | 25    | 8-12      | 20-36          | 96.4%        |
| 6x6  | 36    | 12-16     | 30-48          | 95.2%        |
| 7x7  | 49    | 16-22     | 40-66          | 93.6%        |
| 9x9  | 81    | 25-35     | 60-105         | 90.0%        |

**Formula:** P(no errors) = (0.999)^n where n = number of characters

Even at 99.9% accuracy, a 9x9 puzzle with ~80 characters has only ~92% chance of zero OCR errors.

#### Cause 2: Multi-Digit Target Errors

Larger puzzles have more multi-digit targets (e.g., "24×", "36+"):

| Target Digits | Error Probability | Example |
|---------------|-------------------|---------|
| 1 digit       | 0.1%              | "6-"    |
| 2 digits      | 0.2%              | "12+"   |
| 3 digits      | 0.3%              | "126×"  |

Multi-digit targets have compounding error risk and more valid-but-wrong alternatives.

#### Cause 3: Error Correction Search Space

The correction algorithm has exponential complexity:

| Errors | Combinations (20 cages, 5 alternatives) |
|--------|----------------------------------------|
| 1      | 100                                    |
| 2      | 4,950                                  |
| 3      | 161,700                                |
| 4      | 3,921,225                              |

**Why 4+ errors are uncorrectable:** With >3 errors, the search space becomes intractable. V2's improved OCR ensures most puzzles have ≤3 errors, but some large puzzles still exceed this limit.

#### Cause 4: Constraint-Valid Wrong Alternatives

Some OCR errors produce targets that are mathematically valid but still wrong:

**Example:** Target "12×" misread as "72×"
- "72×" is valid (72 = 8×9, 72 = 6×12, etc.)
- But it's wrong for the actual puzzle
- Constraint filtering cannot catch this error

#### Cause 5: Missing Division Operator in Training

**Note:** The division operator (/) was not present in the training data extraction:
```
Classes in training: 0,1,2,3,4,5,6,7,8,9,+ (10),× (12),- (13)
Missing: / (11)
```

This means division operators rely on the model's generalization from MNIST training, not board-extracted examples. This could contribute to errors in division cages.

### 3.3 Solve Time Analysis

| Size | Baseline (avg) | Failed (avg) | Reason |
|------|----------------|--------------|--------|
| 3x3  | 133ms          | N/A          | No failures |
| 4x4  | 250ms          | ~1,500ms     | ~100 correction attempts |
| 5x5  | 546ms          | ~4,000ms     | ~500 correction attempts |
| 6x6  | 1,000ms        | ~8,000ms     | ~1,500 correction attempts |
| 7x7  | 4,709ms        | ~6,000ms     | ~2,000 correction attempts |
| 9x9  | 10,738ms       | ~12,000ms    | ~3,000 correction attempts |

---

## Part 4: Comparison Summary

### 4.1 What V2 Does Better

1. **Character Recognition:** 99.9% validation accuracy vs ~95% for V1
2. **1↔7 Confusion:** Completely eliminated (was 37% of V1 errors)
3. **Baseline Solving:** 98% vs 69% on 3x3; 86% vs 36% on 4x4
4. **Training Data:** Board-extracted characters eliminate domain gap
5. **Augmentation:** Elastic deformation simulates handwriting variation
6. **Training Stability:** BatchNorm enables deeper, more powerful networks

### 4.2 What V2 Does Differently

1. **Focal Loss:** Focuses training on hard examples
2. **Class Balancing:** Equal representation of all 13 classes
3. **Deeper Architecture:** 4 conv layers vs 2
4. **More Parameters:** 872K vs ~400K

### 4.3 What Remains Unchanged

1. **Error Correction Algorithm:** Same constraint-based approach
2. **Constraint Validation:** Same mathematical filters
3. **Z3 Solver:** Same symbolic reasoning engine
4. **Max Errors:** Still limited to 3 correctable errors

### 4.4 What Still Limits Performance

1. **Cumulative Error Probability:** Even 99.9% accuracy fails on 80+ character puzzles
2. **Multi-Error Correction:** Exponential search space limits to 3 errors
3. **Valid-but-Wrong Targets:** Constraint filtering cannot catch all errors
4. **Missing Division Data:** / operator not in board-extracted training set

---

## Part 5: Recommendations for Further Improvement

### 5.1 Short-Term (Incremental Gains)

1. **Add Division Operator to Training:** Extract / from boards or add synthetic examples
2. **Increase Error Limit:** Allow 4+ error correction for 9x9 puzzles
3. **Ensemble Models:** Combine ImprovedCNN with attention variant
4. **Confidence Thresholding:** Skip correction attempts when confidence is high

### 5.2 Medium-Term (Significant Gains)

1. **Confusion-Aware Correction:** Prioritize alternatives based on known confusions
2. **Constraint Propagation:** Use arc consistency to prune invalid alternatives before Z3
3. **Iterative Solving:** Solve partial puzzles to identify definite errors
4. **Per-Cell Re-OCR:** Re-run OCR with different preprocessing on suspected errors

### 5.3 Long-Term (Research Directions)

1. **End-to-End Training:** Train character recognition with puzzle-solving loss
2. **Transformer Architecture:** Self-attention for context-aware character recognition
3. **Synthetic Data Generation:** Generate unlimited handwritten training examples
4. **Active Learning:** Identify and label hard examples from failures

---

## Appendix: Uncorrectable Puzzle Lists

### V2 Uncorrectable Puzzles by Size

| Size | Count | Puzzle IDs |
|------|-------|------------|
| 3x3  | 0     | (none) |
| 4x4  | 10    | 7, 24, 30, 31, 43, 46, 68, 72, 77, 86 |
| 5x5  | 26    | 2, 3, 5, 8, 11, 13, 16, 17, 22, 26, 30, 34, 36, 42, 46, 52, 53, 61, 66, 70, 75, 80, 82, 87, 95, 97 |
| 6x6  | 35    | (see results CSV) |
| 7x7  | 57    | (see results CSV) |
| 9x9  | 76    | (see results CSV) |

---

## Part 6: Hybrid Error Correction (V2.1)

### 6.1 Methodology

Integrated previously unused modules into the error correction pipeline:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `constraint_propagation.py` | AC-3 pre-filtering | Prune inconsistent alternatives before Z3 |
| `confusion_aware.py` | Confusion-aware scoring | Boost 1↔7 (37%), 6↔8 (18%) alternatives |

**Hybrid Correction Pipeline:**
1. **Auto-correct** impossible cages using `detect_impossible_cages()`
2. **Score suspects** with hybrid scoring (confidence + confusion priors)
3. **Generate alternatives** with propagation pre-filtering
4. **Validate** with `propagate()` before expensive Z3 calls

**Configuration:**
```python
HYBRID_CONFIG = {
    'confusion_weight': 0.3,           # Weight for confusion priors
    'use_propagation_filter': True,    # Enable AC-3 pre-filtering
    'max_errors': {3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 9: 5},
    'max_k': {3: 4, 4: 5, 5: 5, 6: 6, 7: 6, 9: 8},
}
```

### 6.2 Results

| Size | V2 Baseline | V2 Original | V2.1 Hybrid | Change |
|------|-------------|-------------|-------------|--------|
| 3x3  | 98%         | 100%        | **99%**     | -1%    |
| 4x4  | 86%         | 90%         | **90%**     | 0%     |
| 5x5  | 36%         | 74%         | **42%**     | -32%   |
| 6x6  | 32%         | 65%         | **35%**     | -30%   |
| 7x7  | 10%         | 43%         | **12%**     | -31%   |
| 9x9  | 13%         | 24%         | **16%**     | -8%    |

### 6.3 Correction Method Breakdown (V2.1 Hybrid)

| Size | None | Auto | Single | Two | Three | Uncorrectable |
|------|------|------|--------|-----|-------|---------------|
| 3x3  | 98   | 1    | 0      | 0   | 0     | 1             |
| 4x4  | 86   | 0    | 4      | 0   | 0     | 10            |
| 5x5  | 36   | 0    | 2      | 3   | 1     | 58            |
| 6x6  | 32   | 1    | 2      | 0   | 0     | 65            |
| 7x7  | 10   | 0    | 2      | 0   | 0     | 88            |
| 9x9  | 13   | 0    | 3      | 0   | 0     | 84            |

### 6.4 Analysis: Why Hybrid Correction Underperformed

The hybrid correction performed **significantly worse** than the original V2 correction. Analysis:

#### Issue 1: Propagation Pre-filtering Too Aggressive

The AC-3 propagation was designed to eliminate inconsistent alternatives before Z3, but it appears to:
- Eliminate valid alternatives due to incomplete propagation
- Not account for the full constraint system that Z3 can handle
- Reject alternatives that would have led to correct solutions

#### Issue 2: Confusion Priors Outdated

The confusion priors were derived from V1 error analysis:
```python
CONFUSION_PRIORS = {
    (1, 7): 0.37,  # 1↔7 confusion (37% of V1 errors)
    (6, 8): 0.10,  # 6↔8 confusion
    ...
}
```

But V2's ImprovedCNN **eliminated the 1↔7 confusion entirely**. Boosting 7 alternatives when the model predicts 1 actually introduces errors rather than correcting them.

#### Issue 3: Hybrid Scoring Conflicts

The hybrid scoring combines:
- CNN confidence (accurate in V2)
- Confusion boost (outdated priors)
- Constraint tightness (not always correlated with correctness)

This combination can rank incorrect alternatives higher than correct ones.

### 6.5 Lessons Learned

1. **V2's improved CNN changed the error landscape** - priors from V1 are not applicable
2. **Propagation pre-filtering** needs more careful integration with the full constraint system
3. **The original V2 error correction** was already well-tuned for V2's CNN
4. **Confusion-aware correction** requires updated confusion analysis from V2 failures

### 6.6 Recommendations

To improve hybrid correction:

1. **Re-analyze V2 confusion patterns** - Extract actual confusion matrix from V2 failures
2. **Tune propagation parameters** - Less aggressive filtering, or only use for obvious contradictions
3. **Adaptive confusion weights** - Reduce confusion_weight since V2 has fewer confusions
4. **Ablation study** - Test propagation and confusion separately to identify which causes regression

### 6.7 Solve Time Comparison

| Size | V2 Original | V2.1 Hybrid | Change |
|------|-------------|-------------|--------|
| 3x3  | 133ms       | 138ms       | +4%    |
| 4x4  | 250ms       | 243ms       | -3%    |
| 5x5  | 546ms       | 1,385ms     | +154%  |
| 6x6  | 1,000ms     | 8,110ms     | +711%  |
| 7x7  | 4,709ms     | 40,753ms    | +765%  |
| 9x9  | 10,738ms    | 103,138ms   | +860%  |

**Note:** The dramatic increase in solve time for larger puzzles indicates the hybrid correction is doing significant additional work without proportional accuracy gains.

---

## Part 7: Final Summary

### Overall Performance Progression

| Size | V1 Corrected | V2 Corrected | V2.1 Hybrid | Best |
|------|--------------|--------------|-------------|------|
| 3x3  | 89%          | **100%**     | 99%         | V2   |
| 4x4  | 58%          | **90%**      | 90%         | V2   |
| 5x5  | 26%          | **74%**      | 42%         | V2   |
| 6x6  | 15%          | **65%**      | 35%         | V2   |
| 7x7  | 2%           | **43%**      | 12%         | V2   |
| 9x9  | 1%           | **24%**      | 16%         | V2   |

**Conclusion:** The V2 error correction (without hybrid modifications) remains the best approach. The CNN improvements provided the major gains, while the original constraint-based error correction is well-suited for V2's error patterns.

### Key Takeaways

1. **CNN quality is paramount** - V2's 99.9% accuracy drives most of the improvement
2. **Error correction is complementary** - Effective for 1-3 errors, limited returns beyond
3. **Priors must match the model** - V1 confusion priors are counterproductive for V2
4. **Simple often beats complex** - The original V2 correction outperforms the hybrid approach

---

*Report updated: January 2026*
*Data source: KenKen-handwritten-v2/solver/results/unified_solver_evaluation.csv*
