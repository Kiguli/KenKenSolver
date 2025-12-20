# HexaSudoku Solver

A neuro-symbolic solver for 16×16 Sudoku puzzles using hexadecimal notation.

## Overview

HexaSudoku puzzles are 16×16 grids divided into 4×4 boxes. Values range from 1-16, displayed as:
- **1-9**: Standard digits
- **A-F**: Hexadecimal for values 10-15

## Pipeline

```
Image (1600×1600) → CNN (character recognition) → Z3 (constraint solving) → Solution
```

## Files

| File | Description |
|------|-------------|
| `SymbolicPuzzleGenerator.ipynb` | Generate 16×16 puzzles using Z3 |
| `BoardImageGeneration.ipynb` | Create 1600×1600 PNG board images |
| `train_cnn.py` | Train CNN for 0-9 + A-F recognition |
| `NeuroSymbolicSolver.ipynb` | Main solver pipeline and evaluation |

## Directories

| Directory | Contents |
|-----------|----------|
| `puzzles/` | `puzzles_dict.json` - Generated puzzles |
| `board_images/` | PNG images (1600×1600) |
| `models/` | Trained CNN weights |
| `symbols/` | Training data CSV |
| `results/` | Evaluation results |

## Quick Start

### 1. Generate Puzzles
```bash
# Run SymbolicPuzzleGenerator.ipynb
# Generates 100 puzzles in puzzles/puzzles_dict.json
```

### 2. Generate Board Images
```bash
# Run BoardImageGeneration.ipynb
# Creates board_images/board16_*.png
```

### 3. Train CNN
```bash
python train_cnn.py
# Generates symbols/hex_characters.csv
# Trains and saves models/hex_character_cnn.pth
```

### 4. Run Solver
```bash
# Run NeuroSymbolicSolver.ipynb
# Evaluates on images and saves results/detailed_evaluation.csv
```

## Z3 Constraints

```python
# Cell range: 1 ≤ X[i][j] ≤ 16
# Row uniqueness: Distinct(row)
# Column uniqueness: Distinct(column)
# 4×4 Box uniqueness: Distinct(box)
```

## CNN Architecture

- Input: 28×28 grayscale images
- Output: 16 classes (0-9 digits, A-F letters)
- Architecture: CNN_v2 (same as KenKen/Sudoku)

## Character Mapping

| Value | Display | CNN Class |
|-------|---------|-----------|
| 0 | (empty) | 0 |
| 1-9 | 1-9 | 1-9 |
| 10 | A | 10 |
| 11 | B | 11 |
| 12 | C | 12 |
| 13 | D | 13 |
| 14 | E | 14 |
| 15 | F | 15 |
