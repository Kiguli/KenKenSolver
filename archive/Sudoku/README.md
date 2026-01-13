# Sudoku Puzzle Solver

A neuro-symbolic AI system that solves Sudoku puzzles by combining computer vision (CNNs) with symbolic constraint solving (Z3 SMT Solver).

## Supported Puzzle Sizes

| Size | Box Size | Description |
|------|----------|-------------|
| 4×4 | 2×2 | Mini Sudoku (good for testing) |
| 9×9 | 3×3 | Standard Sudoku |

## Project Structure

```
Sudoku/
├── README.md                      # This file
├── NeuroSymbolicSolver.ipynb      # Main solver pipeline
├── SymbolicPuzzleGenerator.ipynb  # Generate valid puzzles with Z3
├── BoardImageGeneration.ipynb     # Create puzzle images from JSON
├── models/                        # CNN model weights (if trained)
├── puzzles/
│   └── puzzles_dict.json          # Generated puzzle dataset
├── board_images/                  # Generated puzzle PNGs
└── results/                       # Evaluation results
```

## Usage

### 1. Generate Puzzles

```bash
jupyter notebook SymbolicPuzzleGenerator.ipynb
```

This generates `puzzles/puzzles_dict.json` containing 100 puzzles each for 4×4 and 9×9 sizes.

**Puzzle format:**
```json
{
  "4": [
    {
      "puzzle": [[0,2,0,4], [3,0,1,0], ...],
      "solution": [[1,2,3,4], [3,4,1,2], ...]
    }
  ],
  "9": [...]
}
```

Where `0` indicates an empty cell.

### 2. Generate Board Images

```bash
jupyter notebook BoardImageGeneration.ipynb
```

Creates 900×900px PNG images in `board_images/`.

### 3. Solve Puzzles

```bash
jupyter notebook NeuroSymbolicSolver.ipynb
```

The solver can work in two modes:

**From JSON (no image processing):**
```python
solution = solve_from_json(puzzle)
```

**From Image (requires trained CNN):**
```python
solution = solve_from_image('board_images/board9_0.png', model)
```

## Architecture

### Pipeline

```
Image (900×900) or JSON
        ↓
   Grid Detection → Size (4 or 9)
        ↓
   Digit Recognition → Given cells
        ↓
   Z3 Solver → Complete solution
```

### Z3 Constraints

```python
# Cell values: 1 to size
1 ≤ X[i][j] ≤ size

# Row uniqueness
Distinct(X[row])

# Column uniqueness
Distinct([X[i][col] for i in range(size)])

# Box uniqueness (2×2 or 3×3)
Distinct(box_cells)
```

### Key Differences from KenKen

| Aspect | KenKen | Sudoku |
|--------|--------|--------|
| Structure | Irregular cages | Fixed boxes |
| Constraints | Arithmetic operations | Uniqueness only |
| Symbols | Digits + operators | Digits only |
| Detection | Cage borders (complex) | Box lines (simple) |

## Stretch Goals: Variant Sudokus

### Killer Sudoku

Combines Sudoku with KenKen-style cages:
- Standard Sudoku rules (row/col/box uniqueness)
- Plus: Cage sum constraints
- Plus: No repeated digits within cage

**Additional Z3 constraint:**
```python
for cage in cages:
    s.add(Sum(cage_cells) == target)
    s.add(Distinct(cage_cells))
```

### Arrow Sudoku

- Standard Sudoku rules
- Plus: Circle digit = sum of arrow path digits

**Additional Z3 constraint:**
```python
s.add(circle_cell == Sum(arrow_path_cells))
```

## Dependencies

```bash
pip install z3-solver torch torchvision opencv-python pillow pandas numpy matplotlib
```

## Model Training (Optional)

The Sudoku solver can work without image processing by solving directly from JSON. For image-based solving, you can either:

1. **Reuse KenKen models** - The character recognition model already handles digits 0-9
2. **Train Sudoku-specific models** - Using TMNIST dataset in `../KenKen/symbols/`

## License

MIT License
