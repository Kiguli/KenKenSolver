# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A collection of neuro-symbolic puzzle solvers that combine computer vision (CNNs) with symbolic constraint solving (Z3 SMT solver). Currently supports:

- **KenKen**: Latin square + arithmetic cages (100% accuracy on 3×3 to 9×9 with error correction)
- **Sudoku**: Latin square + box constraints (4×4 and 9×9)

## Repository Structure

```
KenKenSolver/
├── CLAUDE.md              # This file
├── README.md              # Project overview
├── KenKen/                # KenKen puzzle solver
│   ├── README.md
│   ├── NeuroSymbolicSolver.ipynb
│   ├── SymbolicPuzzleGenerator.ipynb
│   ├── BoardImageGeneration.ipynb
│   ├── *Evaluation.ipynb  # LLM benchmarks
│   ├── train_character_cnn.py  # CNN training with augmentation
│   ├── detect_errors_9x9.py    # 9x9 solver with error correction
│   ├── models/            # CNN weights (.pth)
│   ├── puzzles/           # puzzles_dict.json
│   ├── board_images/      # 900x900 PNGs
│   ├── symbols/           # TMNIST training data
│   └── results/           # Evaluation CSVs
├── Sudoku/                # Sudoku puzzle solver
│   ├── README.md
│   ├── NeuroSymbolicSolver.ipynb
│   ├── SymbolicPuzzleGenerator.ipynb
│   ├── BoardImageGeneration.ipynb
│   ├── models/
│   ├── puzzles/
│   ├── board_images/
│   └── results/
```

## Architecture

### KenKen Pipeline (KenKen/NeuroSymbolicSolver.ipynb)
```
Image → Grid_CNN (size) → OpenCV (cages) → CNN_v2 (digits+ops) → Z3 → Solution
```

### Sudoku Pipeline (Sudoku/NeuroSymbolicSolver.ipynb)
```
Image/JSON → Size Detection → Digit Recognition → Z3 → Solution
```

### Key CNN Models
- `Grid_CNN`: Board size detection (128×128 input)
- `CNN_v2`: Character recognition (28×28 input, 14 classes for KenKen, 10 for Sudoku)

## Z3 Constraints

### KenKen
```python
1 ≤ X[i][j] ≤ size              # Cell range
Distinct(row), Distinct(col)     # Latin square
Sum(cage) == target              # Addition
Product(cage) == target          # Multiplication
|a - b| == target                # Subtraction
max(a/b, b/a) == target          # Division
```

### Sudoku
```python
1 ≤ X[i][j] ≤ size              # Cell range
Distinct(row), Distinct(col)     # Latin square
Distinct(box)                    # Box constraint (2×2 or 3×3)
```

## Setup

```bash
# Core dependencies
pip install z3-solver torch torchvision opencv-python pillow pandas numpy matplotlib

# For LLM evaluations
pip install anthropic openai google-generativeai transformers python-dotenv

# Git LFS for model weights
brew install git-lfs  # macOS
git lfs pull
```

## Data Formats

### KenKen Puzzle JSON
```json
{
  "cells": [[0,0], [0,1]],
  "op": "add",
  "target": 5
}
```

### Sudoku Puzzle JSON
```json
{
  "puzzle": [[0,2,0,4], ...],   // 0 = empty
  "solution": [[1,2,3,4], ...]
}
```

## File Path Notes

All notebooks use relative paths with `./` prefix (e.g., `./puzzles/puzzles_dict.json`). Ensure you run notebooks from their respective directories (KenKen/ or Sudoku/).

## Training Character Recognition

To retrain the character recognition CNN:
```bash
cd KenKen && python train_character_cnn.py
```

This extracts training data from actual board images (ensuring consistency with inference) and applies data augmentation for robustness.

## 9x9 KenKen Evaluation

To evaluate 9x9 KenKen puzzles with error correction:
```bash
cd KenKen && python detect_errors_9x9.py
```

The solver uses:
- Z3 SMT solver with optimized tactics (simplify → propagate-values → solve-eqs → smt)
- Domain tightening for multiplication/division cages
- Top-K OCR predictions with multi-error correction
- Achieves 95% base accuracy, 100% with error correction
