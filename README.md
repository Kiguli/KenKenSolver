# Neuro-Symbolic Puzzle Solvers

A collection of AI systems that solve constraint-based puzzles by combining **computer vision (CNNs)** with **symbolic reasoning (Z3 SMT Solver)**. This hybrid approach demonstrates the power of neuro-symbolic AI for structured logical problems where pure LLMs fail.

## Why Neuro-Symbolic?

Large Language Models (GPT-4, Claude, Gemini) struggle with constraint satisfaction puzzles beyond trivial sizes. Our approach separates the problem into:

1. **Perception (Neural)**: CNNs extract puzzle structure from images
2. **Reasoning (Symbolic)**: Z3 solver computes valid solutions using formal constraints

This division of labor achieves near-perfect accuracy where LLMs fail completely.

## Supported Puzzles

| Puzzle | Status | Accuracy | Description |
|--------|--------|----------|-------------|
| [KenKen](KenKen/) | Complete | 93-100% | Latin square + arithmetic cages |
| [Sudoku](Sudoku/) | Complete | 100% | Latin square + box constraints (4×4, 9×9) |
| [HexaSudoku](HexaSudoku/) | Complete | 100% | 16×16 Sudoku with digits 1-9 and letters A-G |

### Handwritten Puzzle Experiments

Multiple train/test split configurations are available for handwritten digit recognition:

| Folder | Split | Description |
|--------|-------|-------------|
| [83-17 split handwritten](83-17%20split%20handwritten/) | 5000/1000 per class | Original experiments |
| [90-10 split handwritten](90-10%20split%20handwritten/) | 5400/600 per class | More training data |
| [90-10 split detect errors](90-10%20split%20detect%20handwritten%20digit%20errors/) | - | Constraint-based error detection |
| [90-10 split digits only](90-10%20split%20handwritten%20digits%20only/) | 5400/600 per class | Two-digit rendering + domain constraints (58%) |
| [95-5 split detect errors](95-5%20split%20detect%20handwritten%20digit%20errors/) | 5700/300 per class | Same augmentation, test set comparison |
| [100-0 split detect errors](100-0%20split%20detect%20handwritten%20digit%20errors/) | ALL data for train | Upper bound: training data for board images |

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

### KenKen Solver
```bash
cd KenKen
jupyter notebook NeuroSymbolicSolver.ipynb
```

### Sudoku Solver
```bash
cd Sudoku
jupyter notebook NeuroSymbolicSolver.ipynb
```

## Benchmark Results

### KenKen (430 puzzles, sizes 3×3 to 7×7)

| Solver | 3×3 | 4×4 | 5×5 | 6×6 | 7×7 |
|--------|-----|-----|-----|-----|-----|
| **NeuroSymbolic** | 100% | 100% | 100% | 100% | 93% |
| Gemini 2.5 Pro | 74% | 30% | 0% | 0% | 0% |
| Claude Sonnet 4 | 39% | 7% | 0% | 0% | 0% |
| GPT-4o Mini | 8% | 0% | 0% | 0% | 0% |

**Key Finding**: All LLMs fail completely on puzzles 5×5 and larger.

### Handwritten Digit Recognition (MNIST/EMNIST)

| Puzzle Type | Split | Extraction | Original Solve | + Error Correction |
|-------------|-------|------------|----------------|-------------------|
| **Sudoku 4×4** | 83-17 | 99.50% | 92% | 96% |
| **Sudoku 4×4** | 90-10 | 99.31% | 92% | **99%** |
| **Sudoku 9×9** | 83-17 | 99.67% | 79% | 91% |
| **Sudoku 9×9** | 90-10 | 99.64% | 75% | **99%** |
| **HexaSudoku 16×16** | 83-17 | 98.89% | 6% | 30% |
| **HexaSudoku 16×16** | 90-10 | 99.06% | 10% | **40%** |

### Error Correction Methods

Two approaches were developed to handle CNN misclassifications that cause puzzles to become unsolvable:

#### 1. Confidence-Based ([90-10 split handwritten](90-10%20split%20handwritten/Handwritten_Error_Correction/))

Uses CNN softmax probabilities to identify likely errors:
- Sorts clues by confidence score (lowest first = most likely wrong)
- Substitutes each suspect with its second-best CNN prediction
- Tries single-error, then two-error corrections exhaustively

#### 2. Constraint-Based ([90-10 split detect errors](90-10%20split%20detect%20handwritten%20digit%20errors/))

Uses Z3's **unsat core** to identify clues causing logical conflicts:
- Extracts the minimal set of constraints causing UNSAT
- Only tests substitutions on clues involved in conflicts
- Supports up to 5-error correction with targeted search

#### 3. Top-K Prediction ([90-10 split detect errors](90-10%20split%20detect%20handwritten%20digit%20errors/))

Extends constraint-based approach by trying **2nd, 3rd, and 4th** best CNN predictions:
- When second-best prediction is also wrong, tries additional alternatives
- Improves HexaSudoku from 36% → **40%**
- Trade-off: More solver calls needed

### Error Correction Comparison (90-10 Split)

| Aspect | Confidence-Based | Constraint-Based | + Top-K (2nd-4th) |
|--------|-----------------|------------------|-------------------|
| **Principle** | Statistical | Logical (unsat core) | Logical + multi-prediction |
| **Search Strategy** | Exhaustive | Targeted | Targeted + alternatives |
| **Max Errors** | 2 | 5 | 5 |
| **Sudoku 4×4** | 99% | 99% | 99% |
| **Sudoku 9×9** | **99%** | 85% | 84% |
| **HexaSudoku** | 37% | 36% | **40%** |
| **Avg Solve Calls** | 50-500 | **2-20** | 2-800 |

### When to Use Each Method

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| **Speed critical** | Unsat Core | 10-25x fewer solver calls |
| **Maximum accuracy (9×9)** | Confidence-Based | Catches errors that don't cause UNSAT |
| **Maximum accuracy (16×16)** | Top-K Prediction | Handles cases where 2nd-best is also wrong |
| **Small puzzles (4×4)** | Any | All achieve 99% |

### Train/Test Split Comparison (HexaSudoku 16x16)

| Split | Training Data | Board Images Use | Top-K Solve Rate | Avg Errors/Puzzle |
|-------|---------------|------------------|------------------|-------------------|
| 90-10 | 5,400/class | Test (unseen) | 40% | ~2.5 |
| 90-10 (digits only) | 5,400/class | Test (unseen) | 34% → **58%** (constrained) | 3.1 → ~1.5 |
| 95-5  | 5,700/class | Test (unseen) | 37-42% | 2.5 |
| **100-0** | ALL (~7,000) | **Training** | **68%** | **1.45** |

**Key finding**: Even using training data (CNN has memorized digits), solve rate is only 68%, not 100%. Data augmentation during training creates variations that don't exactly match board images.

### Digits-Only Experiment (90-10 Split)

An alternative approach renders values 10-16 as two-digit numbers (e.g., "16" instead of letter "G"):

| Aspect | Letters (A-G) | Digits Only (baseline) | + Domain Constraints |
|--------|---------------|------------------------|---------------------|
| Top-K Solve Rate | 40% | 34% | **58%** |
| Constraint-Based Solve | 36% | 22% | **49%** |
| Cell Error Rate (10-16) | ~1% | 4.7% | ~1.5% (estimated) |

**Domain Constraints Applied**:
- **Tens digit forced to 1**: All values 10-16 start with "1", eliminating 1→7 confusion (64% of errors)
- **Ones digit constrained to 0-6**: Values 17-19 don't exist in HexaSudoku
- **Alternative filtering**: Error correction only considers valid values {10-16}

**Finding**: Domain knowledge significantly improves the digits-only approach:
- Top-K solve rate improved from 34% → 58% (+71%)
- Constraint-based solve rate improved from 22% → 49% (+123%)
- The remaining gap vs letters (58% vs 40%) is due to ones digit confusion, which is harder to constrain

### Key Insights

1. **~99% character recognition ≠ puzzle solving**: Even rare misclassifications break constraint satisfaction
2. **Unsat core is faster but incomplete**: Only detects errors that directly cause logical conflicts
3. **Some errors are "invisible"**: If a wrong digit enables a different valid solution (not the intended one), it won't cause UNSAT and the unsat core approach will miss it
4. **Second-best prediction matters**: When the CNN's second-best guess is also wrong, trying 3rd/4th best helps (Top-K improves HexaSudoku by 4%)
5. **Upper bound ~68%**: Even with perfect familiarity (100-0 split), character confusions limit performance
6. **Domain constraints dramatically help**: Forcing tens digit=1 for two-digit cells eliminates 64% of errors, improving solve rate from 34% → 58%

## Project Structure

```
KenKenSolver/
├── README.md                    # This file
├── CLAUDE.md                    # Development documentation
├── KenKen/                      # KenKen puzzle solver
│   ├── README.md               # KenKen-specific documentation
│   ├── NeuroSymbolicSolver.ipynb
│   ├── SymbolicPuzzleGenerator.ipynb
│   ├── BoardImageGeneration.ipynb
│   ├── *Evaluation.ipynb       # LLM benchmarks
│   ├── models/                 # Pre-trained CNN weights
│   ├── puzzles/                # Puzzle dataset (JSON)
│   ├── board_images/           # Generated puzzle images
│   └── results/                # Evaluation results
├── Sudoku/                      # Sudoku puzzle solver
│   ├── README.md
│   ├── NeuroSymbolicSolver.ipynb
│   ├── SymbolicPuzzleGenerator.ipynb
│   ├── BoardImageGeneration.ipynb
│   ├── models/
│   ├── puzzles/
│   ├── board_images/
│   └── results/
├── HexaSudoku/                  # 16×16 Sudoku solver
│   └── ...
├── 83-17 split handwritten/     # Handwritten experiments (83-17 split)
│   ├── Handwritten_Sudoku/
│   ├── Handwritten_HexaSudoku/
│   └── Handwritten_Error_Correction/
├── 90-10 split handwritten/     # Handwritten experiments (90-10 split)
│   ├── Handwritten_Sudoku/
│   ├── Handwritten_HexaSudoku/
│   └── Handwritten_Error_Correction/
├── 90-10 split detect handwritten digit errors/  # Constraint-based error detection
│   ├── detect_errors.py         # Unsat core analysis (2nd-best only)
│   ├── predict_digits.py        # Top-K prediction (2nd-4th best)
│   └── results/                 # Detection results
├── 95-5 split detect handwritten digit errors/   # Test set comparison experiment
│   ├── download_datasets.py     # 95-5 split (seed configurable)
│   ├── train_cnn.py             # Same augmentation as 90-10
│   ├── detect_errors.py         # Constraint-based correction
│   ├── predict_digits.py        # Top-K prediction
│   └── results/                 # Detection results
├── 100-0 split detect handwritten digit errors/  # Upper bound experiment
│   ├── download_datasets.py     # Combines ALL data for training
│   ├── train_cnn.py             # Same augmentation
│   ├── generate_images.py       # Uses TRAINING data (key difference)
│   └── results/                 # Detection results (68% solve rate)
├── 90-10 split handwritten digits only/  # Digits-only experiment (no letters)
│   ├── download_datasets.py     # MNIST only (0-9)
│   ├── train_cnn.py             # 11 classes (0-9 + empty)
│   ├── generate_images.py       # Two-digit rendering for 10-16
│   ├── detect_errors.py         # Constraint-based + domain constraints
│   ├── predict_digits.py        # Top-K + domain constraints
│   └── results/                 # 58% solve rate (with domain constraints)
```

## License

MIT License

## Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
- MNIST dataset (LeCun et al., 1998) for handwritten digits
- EMNIST dataset (Cohen et al., 2017) for handwritten letters
