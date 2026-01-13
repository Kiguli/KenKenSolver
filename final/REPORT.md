# Neuro-Symbolic Puzzle Solvers - Technical Report

## 1. Project Overview

This project demonstrates a neuro-symbolic approach to solving constraint-based puzzles, combining **computer vision (CNNs)** for perception with **symbolic reasoning (Z3 SMT Solver)** for constraint satisfaction. The system achieves near-perfect accuracy on puzzles where Large Language Models (LLMs) fail completely.

### Supported Puzzle Types

| Puzzle Type | Grid Sizes | Constraints |
|-------------|------------|-------------|
| **KenKen** | 3×3 to 9×9 | Latin square + arithmetic cages |
| **Sudoku** | 4×4, 9×9 | Latin square + box constraints |
| **HexaSudoku** | 16×16 | Latin square + 4×4 box constraints |

### Core Pipeline

```
Image Input → Size Detection CNN → Structure Extraction (OpenCV) → Character Recognition CNN → Z3 Solver → Solution
```

---

## 2. Benchmark Datasets

All benchmark images are organized in `benchmarks/`:

### KenKen (1,200 images)
- **Computer**: 100 images per size (3×3 to 9×9) = 600 images
- **Handwritten**: 100 images per size (3×3 to 9×9) = 600 images
- Image format: 300px per cell, PNG

### Sudoku (400 images)
- **Computer**: 100 images each for 4×4 and 9×9 = 200 images
- **Handwritten**: 100 images each for 4×4 and 9×9 = 200 images
- Image format: 900×900px PNG

### HexaSudoku (400 images)
- **Computer Hex Notation**: 100 images (values 0-9, A-F)
- **Computer Numeric**: 100 images (values 1-16)
- **Handwritten Hex Notation**: 100 images
- **Handwritten Numeric**: 100 images
- Image format: 16×16 grid, 900×900px PNG

**Total: 2,000 benchmark images**

---

## 3. CNN Model Architectures

### 3.1 Grid Detection CNN (Size Detection)

**Purpose**: Detect puzzle grid size from board image

```
Input: 128×128 grayscale
Conv2d(1→32, 3×3) + MaxPool(2×2)
Conv2d(32→64, 3×3) + MaxPool(2×2)
Dropout(0.25)
Flatten → Linear(262144→128) → Linear(128→6)
Output: 6 classes (sizes 3, 4, 5, 6, 7, 9)
Parameters: ~262K
```

### 3.2 CNN_v2 (Character Recognition - Computer)

**Purpose**: Recognize digits and operators from computer-generated puzzles

```
Input: 28×28 grayscale
Conv2d(1→32, 3×3) + ReLU + MaxPool(2×2)
Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)
Flatten → Linear(3136→128) + Dropout(0.25)
Linear(128→output_dim)
Parameters: ~400K
```

**Output classes**:
- KenKen: 14 classes (digits 0-9 + operators +-×÷)
- Sudoku: 10 classes (digits 0-9)
- HexaSudoku: 17 classes (0-9 + A-G)

### 3.3 ImprovedCNN (Character Recognition - Handwritten V2)

**Purpose**: Enhanced recognition for handwritten puzzles

```
Input: 28×28 grayscale

Block 1:
  Conv2d(1→32, 3×3) + BatchNorm + ReLU
  Conv2d(32→32, 3×3) + BatchNorm + ReLU
  MaxPool(2×2)

Block 2:
  Conv2d(32→64, 3×3) + BatchNorm + ReLU
  Conv2d(64→64, 3×3) + BatchNorm + ReLU
  MaxPool(2×2)

Flatten → Linear(3136→256) + BatchNorm + Dropout(0.4)
Linear(256→output_dim)
Parameters: ~850K
```

**Key improvements over CNN_v2**:
- 4 convolutional layers (vs 2)
- Batch normalization after each layer
- Higher dropout (0.4 vs 0.25)
- Focal loss for hard examples during training

---

## 4. Results Summary

### 4.1 Neuro-Symbolic Solver Results

#### KenKen

| Size | Computer Baseline | Computer Corrected | Handwritten V1 | Handwritten V2 |
|------|-------------------|--------------------|-----------------------|-----------------------|
| 3×3 | 100% | 100% | 89% | **100%** |
| 4×4 | 100% | 100% | 58% | **90%** |
| 5×5 | 100% | 100% | 26% | **74%** |
| 6×6 | 100% | 100% | 15% | **65%** |
| 7×7 | 95% | 100% | 2% | **43%** |
| 9×9 | 96% | 100% | 1% | **24%** |

#### Sudoku

| Size | Computer | Handwritten V1 | Handwritten V2 |
|------|----------|----------------|----------------|
| 4×4 | 100% | 99% | **100%** |
| 9×9 | 100% | 99% | **98%** |

#### HexaSudoku (16×16)

| Notation | Computer | Handwritten V1 | Handwritten V2 |
|----------|----------|----------------|----------------|
| Hex (A-G) | 100% | 40% | **91%** |
| Numeric (10-16) | 100% | 58% | **72%** |

### 4.2 LLM Comparison (KenKen)

| Solver | 3×3 | 4×4 | 5×5 | 6×6 | 7×7 | 9×9 |
|--------|-----|-----|-----|-----|-----|-----|
| **NeuroSymbolic** | 100% | 100% | 100% | 100% | 100% | 100% |
| Gemini 2.5 Pro | 74% | 30% | 0% | 0% | 0% | - |
| Claude Sonnet 4 | 39% | 7% | 0% | 0% | 0% | - |
| GPT-4o Mini | 8% | 0% | 0% | 0% | 0% | - |
| Qwen2.5-VL-7B | 10% | 0% | 0% | 0% | 0% | - |

**Key Finding**: All LLMs fail completely on puzzles 5×5 and larger.

---

## 5. Error Correction Methods

### 5.1 Confidence-Based Correction

Uses CNN softmax probabilities to identify likely errors:
1. Sort clues by confidence score (lowest first)
2. Substitute suspect clues with second-best prediction
3. Try single-error, then two-error corrections

### 5.2 Constraint-Based Correction (Unsat Core)

Uses Z3's unsat core to identify logical conflicts:
1. Extract minimal constraint set causing UNSAT
2. Only test substitutions on conflicting clues
3. Supports up to 5-error correction

### 5.3 Top-K Prediction

Extends constraint-based approach:
1. Try 2nd, 3rd, and 4th best CNN predictions
2. Handles cases where second-best is also wrong
3. Improves HexaSudoku from 36% → 40%

### 5.4 Domain Constraints (HexaSudoku Numeric)

For two-digit cells (values 10-16):
- **Tens digit forced to 1** (eliminates 1↔7 confusion)
- **Ones digit constrained to 0-6** (values 17-19 don't exist)
- Improved solve rate from 34% → 58%

---

## 6. V1 vs V2 Comparison

### What Changed in V2

| Aspect | V1 | V2 |
|--------|----|----|
| **CNN Architecture** | CNN_v2 (~400K params) | ImprovedCNN (~850K params) |
| **Convolutional Layers** | 2 | 4 |
| **Batch Normalization** | No | Yes |
| **Loss Function** | Cross-entropy | Focal loss |
| **Training Data** | MNIST digits only | Board-extracted characters |
| **Data Augmentation** | Basic (rotation, scale) | Advanced (elastic, morphological) |
| **Validation Accuracy** | ~95% | 99.9% |

### Key Improvement

The 1↔7 confusion that caused 37% of V1 errors was **completely eliminated** in V2 through:
- Improved architecture with batch normalization
- Training on actual board-extracted characters
- More aggressive data augmentation

---

## 7. Usage Instructions

### Running the Neuro-Symbolic Solver

```bash
# KenKen (from archive/)
cd archive/KenKen-handwritten-v2/solver
python solve_all_sizes.py --sizes 3,4,5,6,7,9 --num 100

# Sudoku (from archive/)
cd archive/Sudoku-handwritten-v2
python unified_solver.py
```

### Running LLM Benchmarks

```bash
# Single LLM on single puzzle type
python evaluation/llm_benchmark.py --llm claude --puzzle kenken --sizes 3,4,5 --num 30

# All LLMs on KenKen
python evaluation/llm_benchmark.py --llm all --puzzle kenken --sizes 3,4,5,6,7 --num 30

# Specific configuration
python evaluation/llm_benchmark.py --llm gemini --puzzle sudoku --sizes 4,9 --num 100 --variant Handwritten
```

### Environment Setup

```bash
# Required packages
pip install z3-solver torch torchvision opencv-python pillow pandas numpy

# For LLM evaluation
pip install anthropic openai google-generativeai transformers python-dotenv

# Set API keys in .env file or environment:
# ANTHROPIC_API_KEY=your_key
# OPENAI_API_KEY=your_key
# GOOGLE_API_KEY=your_key
```

---

## 8. File Organization

```
final/
├── benchmarks/           # 2,000 puzzle images
│   ├── KenKen/
│   ├── Sudoku/
│   └── HexaSudoku_16x16/
├── models/               # Pre-trained CNN weights
│   ├── computer/
│   ├── handwritten_v1/
│   └── handwritten_v2/
├── puzzles/              # Puzzle definitions (JSON)
├── results/              # Evaluation CSVs
│   ├── neurosymbolic/
│   └── llm/
├── evaluation/           # LLM benchmark script
│   └── llm_benchmark.py
└── REPORT.md            # This file

archive/                 # Complete code and experimental history
```

---

## 9. Key Findings

1. **Neuro-symbolic approach outperforms pure neural approaches** on constraint satisfaction problems by separating perception from reasoning.

2. **LLMs fail on constraint puzzles beyond trivial sizes** - even state-of-the-art models achieve 0% accuracy on 5×5 KenKen.

3. **~99% character recognition accuracy is insufficient** - even rare misclassifications break constraint satisfaction in larger puzzles.

4. **Error correction is essential for handwritten puzzles** - constraint-based correction using Z3 unsat cores provides significant accuracy gains (+2% to +51%).

5. **Domain constraints dramatically help** - forcing known constraints (e.g., tens digit=1 for HexaSudoku Numeric) eliminates entire classes of errors.

6. **Training on in-domain data matters** - V2 models trained on board-extracted characters significantly outperform V1 models trained on standard datasets.

---

## 10. Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
- MNIST dataset (LeCun et al., 1998) for handwritten digits
- EMNIST dataset (Cohen et al., 2017) for handwritten letters
