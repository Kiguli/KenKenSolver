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
| [Sudoku](Sudoku/) | In Progress | - | Latin square + box constraints |

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
```

## License

MIT License

## Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
