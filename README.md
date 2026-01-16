# Neuro-Symbolic Puzzle Solvers

AI systems that solve constraint-based puzzles by combining **computer vision (CNNs)** with **symbolic reasoning (Z3 SMT Solver)**. This hybrid approach achieves near-perfect accuracy on problems where LLMs fail completely.

## Why Neuro-Symbolic?

Large Language Models (GPT-4, Claude, Gemini) struggle with constraint satisfaction puzzles beyond trivial sizes. Our approach separates the problem into:

1. **Perception (Neural)**: CNNs extract puzzle structure from images
2. **Reasoning (Symbolic)**: Z3 solver computes valid solutions using formal constraints

This division of labor achieves near-perfect accuracy where LLMs fail completely.

## Benchmark Dataset

| Category | Puzzle Type | Sizes | Images/Size | Total | Location |
|----------|-------------|-------|-------------|-------|----------|
| Computer | KenKen | 3,4,5,6,7,8,9 | 100 | 700 | `benchmarks/KenKen/Computer/` |
| Computer | Sudoku | 4,9 | 100 | 200 | `benchmarks/Sudoku/Computer/` |
| Computer | Sudoku (Hex) | 16 | 100 | 100 | `benchmarks/HexaSudoku_16x16/Computer_Hex_Notation/` |
| Computer | Sudoku (Numeric) | 16 | 100 | 100 | `benchmarks/HexaSudoku_16x16/Computer_Numeric/` |
| Handwritten | KenKen | 3,4,5,6,7,8,9 | 100 | 700 | `benchmarks/KenKen/Handwritten/` |
| Handwritten | Sudoku | 4,9 | 100 | 200 | `benchmarks/Sudoku/Handwritten/` |
| Handwritten | Sudoku (Hex) | 16 | 100 | 100 | `benchmarks/HexaSudoku_16x16/Handwritten_Hex_Notation/` |
| Handwritten | Sudoku (Numeric) | 16 | 100 | 100 | `benchmarks/HexaSudoku_16x16/Handwritten_Numeric/` |
| **Total** | | | | **2,200** | |

## Evaluation Results

### KenKen Results (Computer-Generated)

| Model | 3×3 | 4×4 | 5×5 | 6×6 | 7×7 | 8×8 | 9×9 |
|-------|-----|-----|-----|-----|-----|-----|-----|
| NeuroSymbolic | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| Gemini 2.5 Pro | 69% | 35% | 0% | - | - | - | - |
| Claude Sonnet 4 | 41% | 6% | 0% | - | - | - | - |
| Qwen 2.5-VL | 24% | 0% | - | - | - | - | - |
| GPT-4o | 21% | 0% | - | - | - | - | - |
| GPT-4o Mini | 5% | 0% | - | - | - | - | - |

*All LLMs fail completely on KenKen puzzles 5×5 and larger. The neuro-symbolic approach achieves 100% accuracy without error correction due to perfect digit recognition.*

### Sudoku Results (Computer-Generated)

| Model | 4×4 | 9×9 | 16×16 (Hex) | 16×16 (Numeric) |
|-------|-----|-----|-------------|-----------------|
| NeuroSymbolic | **100%** | **100%** | **100%** | **100%** |
| Gemini 2.5 Pro | 99% | 0% | - | - |
| Claude Sonnet 4 | 73% | 0% | - | - |
| Qwen 2.5-VL | 51% | 0% | - | - |
| GPT-4o | 75% | 8% | 0% | 0% |
| GPT-4o Mini | 65% | 1% | 0% | 0% |

*LLMs struggle with 9×9 Sudoku (GPT-4o: 8%, others: ≤1%) and fail completely on 16×16 Sudoku. The neuro-symbolic approach achieves 100% accuracy without error correction due to perfect digit recognition.*

### Handwritten Results (Baseline vs Error-Corrected)

| Model | KenKen 3×3 | KenKen 4×4 | KenKen 5×5 | KenKen 6×6 | KenKen 7×7 | KenKen 8×8 | KenKen 9×9 |
|-------|------------|------------|------------|------------|------------|------------|------------|
| V1 Baseline | 69% | 36% | 18% | 7% | 1% | 1% | 0% |
| V1 Corrected | 89% | 58% | 26% | 15% | 2% | 1% | 1% |
| V2 Baseline | 98% | 86% | 36% | 32% | 10% | 13% | 13% |
| V2 Corrected | **100%** | **94%** | **74%** | **66%** | **41%** | **46%** | **26%** |
| GPT-4o | 20% | 0% | - | - | - | - | - |
| Gemini 2.5 Pro | 68% | 30% | 0% | - | - | - | - |

| Model | Sudoku 4×4 | Sudoku 9×9 | Sudoku 16×16 (Hex) | Sudoku 16×16 (Numeric) |
|-------|------------|------------|---------------------|------------------------|
| V1 Baseline | 90% | 75% | 10% | 30% |
| V1 Corrected | 92% | 85% | 10% | 32% |
| V2 Baseline | 100% | 96% | 77% | 60% |
| V2 Corrected | **100%** | **98%** | **91%** | **72%** |
| GPT-4o | 63% | 9% | 0% | 0% |
| Gemini 2.5 Pro | 100% | 2% | - | - |

*V1: 90-10 train/test split with MNIST digits. V2: ImprovedCNN trained on board-extracted characters with augmentation. V1 Sudoku baselines derived from extraction accuracy.*

## Error Correction

For handwritten puzzles, CNN misclassifications can cause constraint conflicts. Five correction approaches were developed:

### Correction Methods

**1. Confidence-Based**
Uses CNN softmax probabilities to identify likely errors:
- Sorts clues by confidence score (lowest first = most likely wrong)
- Substitutes each suspect with its second-best CNN prediction
- Tries single-error, then two-error corrections exhaustively

**2. Constraint-Based (Unsat Core)**
Uses Z3's unsat core to identify clues causing logical conflicts:
- Extracts the minimal set of constraints causing UNSAT
- Only tests substitutions on clues involved in conflicts
- Supports up to 5-error correction with targeted search

**3. Top-K Prediction**
Extends constraint-based approach by trying 2nd, 3rd, and 4th best CNN predictions:
- Handles cases where second-best prediction is also wrong
- Improves Sudoku 16×16 from 36% → 40%

**4. Cage Re-detection (KenKen)**
When cage detection fails validation (too many cages, too many single-cells), retry with stricter thresholds:
- Uses threshold multipliers (1.5x, 2.0x, 2.5x) to filter thin grid lines misdetected as cage walls
- Validates cages: cell count, single-cell count, total cage count
- Fixes 7×7 puzzles where integer division rounding causes false cage walls

**5. Operator Inference (KenKen)**
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

### KenKen Error Correction Breakdown (V2)

Out of 700 handwritten KenKen puzzles (sizes 3-9), the correction methods contributed as follows:

| Correction Type | Puzzles | Description |
|-----------------|---------|-------------|
| **None (direct solve)** | 288 | CNN predictions correct, no correction needed |
| **Simple single** | 104 | Fixed 1 OCR error using top-K predictions |
| **Constraint single** | 10 | Fixed 1 error via Z3 unsat core analysis |
| **Simple two** | 15 | Fixed 2 OCR errors using top-K predictions |
| **Constraint two** | 1 | Fixed 2 errors via unsat core |
| **Constraint three** | 1 | Fixed 3 errors via unsat core |
| **Still uncorrectable** | 281 | Too many errors (4+) to correct |

**Key insights**:
- 41% of puzzles (288/700) solved directly without correction
- 19% of puzzles (131/700) recovered via error correction
- Simple single-error correction was most effective (104 puzzles)
- 40% of puzzles remain unsolvable due to 4+ OCR errors

### Sudoku Error Correction Breakdown (V2)

Out of 100 puzzles per variant, the correction methods contributed as follows:

| Puzzle Type | Direct Solve | Single Correction | Double Correction | Uncorrectable |
|-------------|--------------|-------------------|-------------------|---------------|
| Sudoku 4×4 | 100 | 0 | 0 | 0 |
| Sudoku 9×9 | 96 | 2 | 0 | 2 |
| Sudoku 16×16 (Hex) | 77 | 13 | 1 | 9 |
| Sudoku 16×16 (Numeric) | 60 | 11 | 1 | 28 |

**Key insights**:
- Sudoku 4×4 achieves perfect accuracy without any corrections needed
- Sudoku 9×9 near-perfect with only 2 single-error corrections
- Sudoku 16×16 Hex notation benefits significantly from correction (77% → 91%)
- Sudoku 16×16 Numeric more challenging with 28 uncorrectable puzzles (many have 3+ OCR errors)

### V1 Error Correction Effectiveness

V1's weaker CNN baseline (2 conv layers, ~400K params, ~95% accuracy) meant most puzzles had too many OCR errors to correct:

| Outcome | Puzzles | % |
|---------|---------|---|
| Direct solve | 133 | 19% |
| Corrected | 52 | 7% |
| Uncorrectable | 515 | 74% |

The 1↔7 confusion alone caused 37% of V1 errors. V2's improved CNN architecture eliminates this confusion and dramatically increases both baseline accuracy and correction effectiveness.

## V2 Model Improvements

### KenKen Handwritten V2

The V2 approach dramatically improves handwritten KenKen solving through better CNN architecture and training methodology.

**CNN Architecture (ImprovedCNN)**:
- 4 conv layers (vs 2 in V1)
- BatchNorm after every layer
- 872K parameters (vs ~400K)
- 99.9% validation accuracy (vs ~95%)

**Training Data**:
- Characters extracted from actual board images using the same pipeline as inference
- Eliminates domain gap between training and inference
- 70% handwritten / 30% computer-generated mix
- Balanced to equal representation per class

**Data Augmentation (10x)**:
- Rotation: ±20° (vs ±5°)
- Scale: 0.5-1.2x (vs 0.9-1.1x)
- Elastic deformation (new)
- Morphological erosion/dilation (new)

**Training Process**:
- Focal Loss (focuses on hard examples)
- Early stopping with patience=15
- Lower learning rate (0.0003 vs 0.001)

**Key Result**: The 1↔7 confusion that caused 37% of V1 errors was **completely eliminated** in V2.

### Why Larger Puzzles Remain Challenging

Even with 99.9% per-character accuracy, larger puzzles have more characters:
- 3×3: ~15 characters → 98.5% chance of zero errors
- 9×9: ~80 characters → 92% chance of zero errors

The error correction can fix 1-3 errors per puzzle, but some 9×9 puzzles have 4+ OCR errors.

See [`archive/KenKen-handwritten-v2/V1_V2_COMPARISON_REPORT.md`](archive/KenKen-handwritten-v2/V1_V2_COMPARISON_REPORT.md) for detailed analysis.

### Sudoku Handwritten V2

A unified solver handling all Sudoku variants (4×4, 9×9, 16×16) with a single 17-class CNN model.

**Unified Model Architecture**:
- Single ImprovedCNN (17 classes: digits 0-9 + letters A-G)
- 850K parameters with BatchNorm and deeper architecture
- 99.3% validation accuracy

**Sudoku 16×16 Numeric Handling**:
- **Tens digit forced to 1**: Eliminates 1↔7 confusion for values 10-16
- **Ones digit constrained to 0-6**: Values 17-19 don't exist in 16×16 Sudoku
- **Ink density ratio** for single/double digit detection (threshold 1.7)
- **Fixed spatial split** for two-digit extraction (left/right halves)

See [`archive/Sudoku-handwritten-v2/REPORT.md`](archive/Sudoku-handwritten-v2/REPORT.md) for detailed analysis.

## Pipeline Architecture

The system uses different CNN models for computer-generated vs handwritten puzzles:

| Pipeline | Computer Models | Handwritten Models |
|----------|-----------------|-------------------|
| KenKen Grid Size | `kenken_grid_cnn.pth` | `kenken_grid_cnn.pth` |
| KenKen Digits | `kenken_sudoku_character_cnn.pth` | V1: `kenken_cnn.pth`, V2: `improved_cnn.pth` |
| Sudoku Digits | `kenken_sudoku_character_cnn.pth` | V1: `sudoku_cnn.pth`, V2: `improved_cnn.pth` |
| Sudoku 16×16 (Hex) | `hexasudoku_character_cnn.pth` | V1: `hexasudoku_cnn.pth`, V2: `improved_cnn.pth` |
| Sudoku 16×16 (Numeric) | `hexasudoku_numeric_cnn.pth` | V1: `hexasudoku_numeric_cnn.pth`, V2: `improved_cnn.pth` |

### KenKen Pipeline (300px fixed cells, variable board size)

```
Image Input (size varies: 900×900 to 2700×2700 based on grid size)
        ↓
   ┌──────────────────────┐
   │  Size Detection CNN  │  ← 128×128 input, 7 classes (3,4,5,6,7,8,9)
   │  (kenken_grid_cnn)   │     Shared between computer and handwritten
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │   Cage Detection     │  ← OpenCV morphology to find cage borders
   │   (OpenCV)           │     Each cell is fixed 300px
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │  KenKen Digit CNN    │  ← 28×28 input, 14 classes
   │                      │     (digits 0-9 + operators +-×÷)
   │                      │     Computer: kenken_sudoku_character_cnn
   │                      │     Handwritten: improved_cnn (V2)
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │   Z3 Solver          │  ← Latin square + cage arithmetic constraints
   │   + Error Correction │     (handwritten only)
   └──────────┬───────────┘
              ↓
        Solution Output
```

### Sudoku Pipeline (fixed board size, variable cell size)

```
Image Input (900×900 for 4×4/9×9, 1600×1600 for 16×16)
        ↓
   ┌──────────────────────┐
   │   Grid Extraction    │  ← OpenCV to find grid lines and boxes
   │   (OpenCV)           │     Cell size = board_size / grid_size
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │  Sudoku Digit CNN    │  ← 28×28 input
   │                      │     Sudoku: 10 classes (0-9)
   │                      │     16×16: 17 classes (0-9 + A-G)
   │                      │     Computer: kenken_sudoku_character_cnn
   │                      │              or hexasudoku_character_cnn
   │                      │     Handwritten: improved_cnn (V2)
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │   Z3 Solver          │  ← Latin square + box constraints
   │   + Error Correction │     (handwritten only)
   └──────────┬───────────┘
              ↓
        Solution Output
```

## Neural Network Architectures

### Size Detection CNN (KenKen only)
- **Purpose**: Determine grid size from board image
- **Input**: 128×128 grayscale board image
- **Output**: 7 classes (sizes 3, 4, 5, 6, 7, 8, 9)
- **Architecture**: 2 conv layers + 2 FC layers
- **Validation accuracy**: 99%+

### KenKen Digit CNN (V2 - ImprovedCNN)
- **Purpose**: Recognize digits and arithmetic operators
- **Input**: 28×28 grayscale cell (from 300px extraction)
- **Output**: 14 classes (digits 0-9, operators +, -, ×, ÷)
- **Architecture**: 4 conv layers with BatchNorm, ~850K params
- **Validation accuracy**: 99.9%

### Sudoku Digit CNN (V2 - Unified ImprovedCNN)
- **Purpose**: Recognize digits and hex letters
- **Input**: 28×28 grayscale cell
- **Output**: 17 classes (digits 0-9, letters A-G for values 10-16)
- **Architecture**: 4 conv layers with BatchNorm, ~850K params
- **Validation accuracy**: 99.3%

### Baseline Digit CNN (V1 - CNN_v2)
- **Purpose**: Used in handwritten V1 experiments
- **Input**: 28×28 grayscale cell
- **Architecture**: 2 conv layers, ~400K params
- **Note**: Less accurate than V2 models (~95% vs 99.9% validation accuracy)

## KenKen Cage Detection

KenKen puzzles have bold borders separating "cages" (groups of cells with arithmetic constraints). The cage detection algorithm uses OpenCV to identify these borders:

### Algorithm

1. **Preprocessing**: Image is scaled 2× and filtered with `pyrMeanShiftFiltering` to reduce noise while preserving edges.

2. **Edge Detection**: Canny edge detection finds all edges in the image.

3. **Line Detection**: Probabilistic Hough Transform (`HoughLinesP`) identifies line segments. Lines are classified as horizontal (Δy < 2) or vertical (Δx < 2).

4. **Border Detection**: For each cell boundary position, the algorithm checks if detected lines cross that boundary with sufficient thickness. A thickness threshold (default 22px at 2× scale) distinguishes thick cage borders from thin grid lines.

5. **Cage Construction**: Starting from each unvisited cell, flood-fill groups adjacent cells that share no border between them into a single cage.

### Border Arrays

The algorithm produces two binary arrays:
- `h_borders[i][j]` = 1 if there's a horizontal border below row `i` at column `j`
- `v_borders[i][j]` = 1 if there's a vertical border to the right of row `i` at column `j`

### Validation

Detected cages are validated against heuristics:
- Total cells must equal grid area (size²)
- Single-cell cages should not exceed grid size
- Total cages should not exceed `(size² / 2) + size`

If validation fails (often due to thin grid lines being misdetected as cage walls), the algorithm retries with stricter thickness thresholds (1.5×, 2.0×, 2.5× the base threshold).

## Key Files

| Resource | Location | Description |
|----------|----------|-------------|
| Benchmark Images | `benchmarks/` | 2,200 puzzle images |
| Pre-trained Models | `models/` | CNN weights (.pth files) |
| Training Scripts | `models/training/` | Scripts to train CNN models |
| Evaluation Results | `results/` | CSV files with accuracy data |
| LLM Benchmark | `evaluation/llm/llm_benchmark.py` | Evaluate LLMs on puzzles |
| KenKen Solver | `evaluation/neurosymbolic/kenken/` | Computer and handwritten solvers |
| Sudoku Solver | `evaluation/neurosymbolic/sudoku/` | Unified solver for all variants |

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

# Install dependencies
pip install -r requirements.txt

# For LLM evaluations (optional)
pip install -r requirements-llm.txt
```

## Quick Start

### KenKen Solver (Computer-Generated)
```bash
cd archive/KenKen
python solve_all_sizes.py --sizes 3,4,5,6,7,8,9 --num 100
```

### KenKen Solver (Handwritten V2)
```bash
cd archive/KenKen-handwritten-v2/solver
python3 solve_all_sizes.py --sizes 3,4,5,6,7,8,9 --num 100
```

### KenKen Solver (Jupyter Notebook)
```bash
cd archive/KenKen
jupyter notebook NeuroSymbolicSolver.ipynb
```

### Sudoku Solver
```bash
cd archive/Sudoku
jupyter notebook NeuroSymbolicSolver.ipynb
```

### LLM Benchmark
```bash
cd evaluation
python llm_benchmark.py --llm claude --puzzle kenken --sizes 3,4,5 --num 30
python llm_benchmark.py --llm all --puzzle kenken --sizes 3,4,5,6,7 --num 30
```

## Project Structure

```
KenKenSolver/
├── README.md                    # This file
│
├── benchmarks/                  # 2,200 puzzle images
│   ├── KenKen/                 # Computer & Handwritten (3×3 to 9×9)
│   ├── Sudoku/                 # Computer & Handwritten (4×4, 9×9)
│   └── HexaSudoku_16x16/       # Hex & Numeric notation
│
├── models/                      # Pre-trained CNN weights
│   ├── computer/               # CNNs for computer-generated puzzles
│   ├── handwritten_v1/         # 90-10 split models
│   ├── handwritten_v2/         # ImprovedCNN models
│   ├── grid_detection/         # Grid size detection CNN
│   └── training/               # Scripts to train models
│
├── puzzles/                     # Puzzle definitions (JSON)
│
├── results/                     # Evaluation CSVs
│   ├── neurosymbolic/          # Solver accuracy results
│   └── llm/                    # LLM comparison results
│
├── evaluation/                  # Evaluation scripts
│   ├── llm/                    # LLM benchmark
│   └── neurosymbolic/          # KenKen/Sudoku solvers
│
└── archive/                     # Complete experimental history
    ├── KenKen/                # KenKen solver (computer-generated)
    ├── KenKen handwritten/    # V1 handwritten solver
    ├── KenKen-handwritten-v2/ # V2 improved solver
    ├── KenKen-300px/          # 300px computer-generated variant
    ├── KenKen-300px-handwritten/ # 300px handwritten variant
    ├── Sudoku/                # Sudoku solver
    ├── Sudoku-handwritten-v2/ # Unified Sudoku solver
    ├── HexaSudoku/            # 16×16 solver
    ├── 83-17 split handwritten/
    ├── 90-10 split handwritten/
    ├── 90-10 split handwritten digits only/
    ├── 90-10 split detect handwritten digit errors/
    ├── 95-5 split detect handwritten digit errors/
    └── 100-0 split detect handwritten digit errors/
```

## License

MIT License
