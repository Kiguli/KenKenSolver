# Neuro-Symbolic Puzzle Solvers

AI systems that solve constraint-based puzzles by combining **computer vision (CNNs)** with **symbolic reasoning (Z3 SMT Solver)**. This hybrid approach achieves near-perfect accuracy on problems where LLMs fail completely.

**Pipeline**: Image → CNN (perception) → Z3 Solver (reasoning) → Solution

## Benchmark Files

All puzzle images are available in [`benchmark_files/`](benchmark_files/):
- **KenKen**: 3×3 to 9×9 (Computer & Handwritten)
- **Sudoku**: 4×4 and 9×9 (Computer & Handwritten)
- **HexaSudoku**: 16×16 with Hex (A-G) or Numeric (10-16) notation

## Results

| Puzzle | Size | Computer Baseline | Computer Corrected | Handwritten Baseline | Handwritten Corrected |
|--------|------|-------------------|--------------------|-----------------------|-----------------------|
| **KenKen** | 3×3 | 100% | 100% | 69% | 87% |
| **KenKen** | 4×4 | 100% | 100% | 41% | 72% |
| **KenKen** | 5×5 | 100% | 100% | 18% | 41% |
| **KenKen** | 6×6 | 100% | 100% | 2% | 14% |
| **KenKen** | 7×7 | 95% | 100% | 0% | 4% |
| **KenKen** | 9×9 | 96% | 100% | 0% | 0% |
| **Sudoku** | 4×4 | 100% | 100% | 92% | 99% |
| **Sudoku** | 9×9 | 100% | 100% | 75% | 99% |
| **HexaSudoku** | 16×16 (Hex) | 100% | 100% | 10% | 40% |
| **HexaSudoku** | 16×16 (Numeric) | 100% | 100% | 34% | 58% |

*Handwritten results use 90-10 train/test split with MNIST digits and multi-scale training augmentation (0.33-1.0 scale). Error correction uses unsat core detection, top-K alternatives, operator inference, and cage re-detection.*

## Why Neuro-Symbolic?

Large Language Models (GPT-4, Claude, Gemini) struggle with constraint satisfaction puzzles beyond trivial sizes. Our approach separates the problem into:

1. **Perception (Neural)**: CNNs extract puzzle structure from images
2. **Reasoning (Symbolic)**: Z3 solver computes valid solutions using formal constraints

This division of labor achieves near-perfect accuracy where LLMs fail completely.

## Architecture

```
Image Input (900×900)
        ↓
    Grid CNN → Detect puzzle size
        ↓
    OpenCV → Extract structure (borders, boxes)
        ↓
   Digit CNN → Read numbers/operators
        ↓
   Z3 Solver → Compute valid solution
        ↓
    Solution Output
```

## LLM Comparison (KenKen)

| Solver | 3×3 | 4×4 | 5×5 | 6×6 | 7×7 | 9×9 |
|--------|-----|-----|-----|-----|-----|-----|
| **NeuroSymbolic** | 100% | 100% | 100% | 100% | 96% | 100% |
| Gemini 2.5 Pro | 74% | 30% | 0% | 0% | 0% | 0% |
| Claude Sonnet 4 | 39% | 7% | 0% | 0% | 0% | 0% |
| GPT-4o Mini | 8% | 0% | 0% | 0% | 0% | 0% |

*KenKen 7×7 achieves 95% baseline, 100% with operator inference. 9×9 achieves 96% baseline, 100% with error correction.

**Key Finding**: All LLMs fail completely on puzzles 5×5 and larger.

## Error Correction Methods

For handwritten puzzles, CNN misclassifications can cause constraint conflicts. Three correction approaches were developed:

### 1. Confidence-Based
Uses CNN softmax probabilities to identify likely errors:
- Sorts clues by confidence score (lowest first = most likely wrong)
- Substitutes each suspect with its second-best CNN prediction
- Tries single-error, then two-error corrections exhaustively

### 2. Constraint-Based (Unsat Core)
Uses Z3's unsat core to identify clues causing logical conflicts:
- Extracts the minimal set of constraints causing UNSAT
- Only tests substitutions on clues involved in conflicts
- Supports up to 5-error correction with targeted search

### 3. Top-K Prediction
Extends constraint-based approach by trying 2nd, 3rd, and 4th best CNN predictions:
- Handles cases where second-best prediction is also wrong
- Improves HexaSudoku from 36% → 40%

### 4. Cage Re-detection (KenKen)
When cage detection fails validation (too many cages, too many single-cells), retry with stricter thresholds:
- Uses threshold multipliers (1.5x, 2.0x, 2.5x) to filter thin grid lines misdetected as cage walls
- Validates cages: cell count, single-cell count, total cage count
- Fixes 7×7 puzzles where integer division rounding causes false cage walls

### 5. Operator Inference (KenKen)
For multi-cell cages with missing operators and target > grid size:
- Subtraction max = size - 1, Division max = size → ruled out
- Automatically tries addition and multiplication operators
- Fixes 9×9 puzzles where multiplication symbol is truncated during character segmentation

### When to Use Each Method

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Speed critical | Unsat Core | 10-25x fewer solver calls |
| Maximum accuracy (9×9) | Confidence-Based | Catches errors that don't cause UNSAT |
| Maximum accuracy (16×16) | Top-K | Handles cases where 2nd-best is also wrong |
| Small puzzles (4×4) | Any | All achieve 99% |

## Handwritten Experiments

Multiple train/test split configurations were tested:

| Split | Training Data | Sudoku 4×4 | Sudoku 9×9 | HexaSudoku 16×16 |
|-------|---------------|------------|------------|------------------|
| 83-17 | 5,000/class | 96% | 91% | 30% |
| 90-10 | 5,400/class | 99% | 99% | 40% |
| 95-5 | 5,700/class | - | - | 37-42% |
| 100-0 | ALL (~7,000) | - | - | 68% |

### Digits-Only Experiment (HexaSudoku)

An alternative approach renders values 10-16 as two-digit numbers instead of letters (A-G):

| Approach | Solve Rate |
|----------|------------|
| Letters (A-G) | 40% |
| Digits Only (baseline) | 34% |
| Digits + Domain Constraints | **58%** |

**Domain Constraints Applied**:
- Tens digit forced to 1 (all values 10-16 start with "1")
- Ones digit constrained to 0-6 (values 17-19 don't exist)
- Error correction only considers valid values {10-16}

### Key Insights

1. **~99% character recognition ≠ puzzle solving**: Even rare misclassifications break constraint satisfaction
2. **Unsat core is faster but incomplete**: Only detects errors that directly cause logical conflicts
3. **Some errors are "invisible"**: If a wrong digit enables a different valid solution, it won't cause UNSAT
4. **Domain constraints dramatically help**: Forcing tens digit=1 eliminates 64% of errors, improving solve rate from 34% → 58%

## Installation

### Prerequisites
- Python 3.10+
- Git LFS (for model weights)

### Setup

```bash
# Clone the repository
git clone https://github.com/Kiguli/KenKenSolver.git
cd KenKenSolver

# Install Git LFS and pull model weights
brew install git-lfs  # macOS (use apt on Linux)
git lfs install
git lfs pull

# Install core dependencies
pip install z3-solver torch torchvision opencv-python pillow pandas numpy matplotlib

# For LLM evaluations (optional)
pip install anthropic openai google-generativeai transformers python-dotenv
```

## Quick Start

### KenKen Solver (Command Line)
```bash
cd KenKen
python solve_all_sizes.py --sizes 3,4,5,6,7,9 --num 100
```

### KenKen Solver (Jupyter Notebook)
```bash
cd KenKen
jupyter notebook NeuroSymbolicSolver.ipynb
```

### Sudoku Solver
```bash
cd Sudoku
jupyter notebook NeuroSymbolicSolver.ipynb
```

## Project Structure

```
KenKenSolver/
├── README.md                    # This file
├── CLAUDE.md                    # Development documentation
├── benchmark_files/             # All puzzle images organized by type
├── KenKen/                      # KenKen puzzle solver (computer-generated)
│   ├── solve_all_sizes.py      # Unified solver (3×3 to 9×9)
│   ├── NeuroSymbolicSolver.ipynb
│   ├── train_character_cnn.py  # CNN training script
│   ├── models/                 # Pre-trained CNN weights
│   ├── puzzles/                # Puzzle dataset (JSON)
│   └── board_images/           # Generated puzzle images
├── KenKen handwritten/          # KenKen with MNIST handwritten digits
│   ├── evaluate.py             # Baseline evaluation
│   ├── detect_errors.py        # Error correction pipeline
│   ├── analyze_failures.py     # Error analysis
│   └── board_images/           # Handwritten digit puzzle images
├── Sudoku/                      # Sudoku puzzle solver (4×4, 9×9)
├── HexaSudoku/                  # 16×16 Sudoku solver
├── 83-17 split handwritten/     # Handwritten experiments (83-17 split)
├── 90-10 split handwritten/     # Handwritten experiments (90-10 split)
├── 90-10 split detect handwritten digit errors/  # Constraint-based error detection
├── 90-10 split handwritten digits only/  # Digits-only experiment (58%)
├── 95-5 split detect handwritten digit errors/   # Test set comparison
└── 100-0 split detect handwritten digit errors/  # Upper bound experiment (68%)
```

## License

MIT License

## Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
- MNIST dataset (LeCun et al., 1998) for handwritten digits
- EMNIST dataset (Cohen et al., 2017) for handwritten letters
