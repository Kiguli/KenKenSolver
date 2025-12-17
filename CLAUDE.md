# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A neuro-symbolic KenKen puzzle solver that combines computer vision (CNNs for grid detection and character recognition), symbolic constraint solving (Z3 SMT solver), and LLM evaluation benchmarks. The solver achieves 100% accuracy on puzzles of size 3x3 to 7x7, significantly outperforming pure LLM approaches.

## Architecture

### Core Pipeline (NeuroSymbolicSolver.ipynb)
```
Image → Grid Detection (CNN) → Border/Cage Detection (OpenCV) →
Character Recognition (CNN) → Constraint Solving (Z3) → Solution
```

**Key Components:**
- `Grid_CNN`: Detects board size (3-7) from 128x128 input
- `CNN_v2`: Character recognition (14 classes: 0-9, +, -, *, /) from 28x28 input
- Z3 constraints: Latin square rules + cage arithmetic constraints

### Data Flow Functions
- `find_size_and_borders(img)` → extracts grid size, cage structure, border thickness
- `make_puzzle(img, ...)` → segments cells, runs OCR, builds puzzle specification
- `evaluate_puzzle(puzzle)` → Z3 solver returns solution grid

### Other Notebooks
- `SymbolicPuzzleGenerator.ipynb`: Generates valid KenKen puzzles using Z3
- `BoardImageGeneration.ipynb`: Creates 900x900px puzzle images from JSON data
- `*Evaluation.ipynb`: Benchmarks various LLMs (Claude, GPT, Gemini, Qwen) against the neuro-symbolic approach

## Setup

```bash
pip install z3-solver torch torchvision anthropic openai opencv-python pillow pandas matplotlib numpy
```

**Note:** File paths in notebooks reference Google Colab paths (`/content/drive/MyDrive/...`). Adjust for local use.

## Key Files

| File | Purpose |
|------|---------|
| `models/*.pth` | Pre-trained PyTorch weights (Git LFS) |
| `puzzles/puzzles_dict.json` | Dataset of 290 puzzles (3x3 to 7x7) |
| `board_images/` | Generated puzzle PNGs |
| `symbols/TMNIST_NotoSans.csv` | Character training data (28x28 images) |
| `results/*.csv` | Evaluation results for each solver |

## Puzzle Data Format

```json
{
  "puzzles": [{
    "cells": [[0,0], [0,1]],
    "op": "+",
    "target": 5
  }, ...],
  "solution": [[1,2,3], [3,1,2], [2,3,1]]
}
```

## Z3 Constraint Structure

- Cell values: `1 ≤ X[i][j] ≤ grid_size`
- Row/column: `Distinct()` constraint (Latin square)
- Cage operations: `+` (sum), `*` (product), `-` (absolute difference), `/` (max ratio)
