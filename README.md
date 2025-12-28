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

Two train/test split configurations are available for handwritten digit recognition:

| Folder | Split | Description |
|--------|-------|-------------|
| [83-17 split handwritten](83-17%20split%20handwritten/) | 5000/1000 per class | Original experiments |
| [90-10 split handwritten](90-10%20split%20handwritten/) | 5400/600 per class | More training data |

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
| **HexaSudoku 16×16** | 90-10 | 99.06% | 10% | **37%** |

**Error Correction**: When CNN misclassifies digits causing UNSAT, the system attempts:
1. **Single-error correction**: Substitute low-confidence predictions with second-best alternatives (O(n))
2. **Two-error correction**: If single fails, try pairs of low-confidence clues (O(n²))

**Key Finding**: Near-perfect character recognition (~99%) doesn't guarantee puzzle solving. Larger puzzles with more clues are highly sensitive to even rare misclassifications. Two-error correction significantly improves solve rates, achieving 99% on Sudoku and 37% on HexaSudoku.

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
```

## License

MIT License

## Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
- MNIST dataset (LeCun et al., 1998) for handwritten digits
- EMNIST dataset (Cohen et al., 2017) for handwritten letters
