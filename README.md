# Neuro-Symbolic Puzzle Solvers

AI systems that solve constraint-based puzzles by combining **computer vision (CNNs)** with **symbolic reasoning (Z3 SMT Solver)**. This hybrid approach achieves near-perfect accuracy on problems where LLMs fail completely.

## Why Neuro-Symbolic?

Large Language Models (GPT-4, Claude, Gemini) struggle with constraint satisfaction puzzles beyond trivial sizes. Our approach separates the problem into:

1. **Perception (Neural)**: CNNs extract puzzle structure from images
2. **Reasoning (Symbolic)**: Z3 solver computes valid solutions using formal constraints

This division of labor achieves near-perfect accuracy where LLMs fail completely.

## KenKen Results (Computer-Generated)

| Size | NeuroSymbolic | Gemini 2.5 Pro | Claude Sonnet 4 | Qwen 2.5-VL | GPT-4o | GPT-4o Mini |
|------|---------------|----------------|-----------------|-------------|--------|-------------|
| 3×3 | **100%** | 69% | 41% | 24% | 21% | 5% |
| 4×4 | **100%** | 35% | 6% | 0% | 0% | 0% |
| 5×5 | **100%** | 0% | 0% | - | - | - |
| 6×6 | **100%** | - | - | - | - | - |
| 7×7 | **100%** | - | - | - | - | - |
| 9×9 | **100%** | - | - | - | - | - |

*All LLMs fail completely on KenKen puzzles 5×5 and larger. The neuro-symbolic approach achieves 100% accuracy without error correction due to perfect digit recognition.*

## Sudoku & HexaSudoku Results (Computer-Generated)

| Puzzle | NeuroSymbolic | Gemini 2.5 Pro | Claude Sonnet 4 | Qwen 2.5-VL | GPT-4o | GPT-4o Mini |
|--------|---------------|----------------|-----------------|-------------|--------|-------------|
| Sudoku 4×4 | **100%** | 99% | 73% | 51% | 75% | 65% |
| Sudoku 9×9 | **100%** | 0% | 0% | 0% | 8% | 1% |
| HexaSudoku 16×16 (Hex) | **100%** | - | - | - | 0% | 0% |
| HexaSudoku 16×16 (Numeric) | **100%** | - | - | - | 0% | 0% |

*LLMs struggle with 9×9 Sudoku (GPT-4o: 8%, others: ≤1%) and fail completely on 16×16 HexaSudoku. The neuro-symbolic approach achieves 100% accuracy without error correction due to perfect digit recognition.*

## Handwritten Results (Baseline vs Error-Corrected)

| Puzzle | Size | V1 Baseline | V1 Corrected | V2 Baseline | V2 Corrected |
|--------|------|-------------|--------------|-------------|--------------|
| KenKen | 3×3 | 69% | 89% | 98% | **100%** |
| KenKen | 4×4 | 36% | 58% | 86% | **94%** |
| KenKen | 5×5 | 18% | 26% | 36% | **74%** |
| KenKen | 6×6 | 7% | 15% | 32% | **66%** |
| KenKen | 7×7 | 1% | 2% | 10% | **41%** |
| KenKen | 9×9 | 0% | 1% | 13% | **26%** |
| Sudoku | 4×4 | 90% | 92% | 100% | **100%** |
| Sudoku | 9×9 | 75% | 85% | 96% | **98%** |
| HexaSudoku | 16×16 (Hex) | 10% | 10% | 77% | **91%** |
| HexaSudoku | 16×16 (Numeric) | 30% | 32% | 60% | **72%** |

*V1: 90-10 train/test split with MNIST digits. V2: ImprovedCNN trained on board-extracted characters with augmentation. V1 Sudoku/HexaSudoku baselines derived from extraction accuracy.*

## Pipeline Architecture

The system uses different CNN models for computer-generated vs handwritten puzzles:

| Pipeline | Computer Models | Handwritten Models |
|----------|-----------------|-------------------|
| KenKen Grid Size | `kenken_grid_cnn.pth` | `kenken_grid_cnn.pth` |
| KenKen Digits | `kenken_sudoku_character_cnn.pth` | V1: `kenken_cnn.pth`, V2: `improved_cnn.pth` |
| Sudoku Digits | `kenken_sudoku_character_cnn.pth` | V1: `sudoku_cnn.pth`, V2: `improved_cnn.pth` |
| HexaSudoku (Hex) | `hexasudoku_character_cnn.pth` | V1: `hexasudoku_cnn.pth`, V2: `improved_cnn.pth` |
| HexaSudoku (Numeric) | `hexasudoku_numeric_cnn.pth` | V1: `hexasudoku_numeric_cnn.pth`, V2: `improved_cnn.pth` |

### KenKen Pipeline (300px fixed cells, variable board size)

```
Image Input (size varies: 900×900 to 2700×2700 based on grid size)
        ↓
   ┌──────────────────────┐
   │  Size Detection CNN  │  ← 128×128 input, 6 classes (3,4,5,6,7,9)
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

### Sudoku/HexaSudoku Pipeline (fixed board size, variable cell size)

```
Image Input (900×900 for Sudoku, 1600×1600 for HexaSudoku)
        ↓
   ┌──────────────────────┐
   │   Grid Extraction    │  ← OpenCV to find grid lines and boxes
   │   (OpenCV)           │     Cell size = board_size / grid_size
   └──────────┬───────────┘
              ↓
   ┌──────────────────────┐
   │  Sudoku Digit CNN    │  ← 28×28 input
   │                      │     Sudoku: 10 classes (0-9)
   │                      │     HexaSudoku: 17 classes (0-9 + A-G)
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
- **Output**: 6 classes (sizes 3, 4, 5, 6, 7, 9)
- **Architecture**: 2 conv layers + 2 FC layers
- **Validation accuracy**: 99%+

### KenKen Digit CNN (V2 - ImprovedCNN)
- **Purpose**: Recognize digits and arithmetic operators
- **Input**: 28×28 grayscale cell (from 300px extraction)
- **Output**: 14 classes (digits 0-9, operators +, -, ×, ÷)
- **Architecture**: 4 conv layers with BatchNorm, ~850K params
- **Validation accuracy**: 99.9%

### Sudoku/HexaSudoku Digit CNN (V2 - Unified ImprovedCNN)
- **Purpose**: Recognize digits and hex letters
- **Input**: 28×28 grayscale cell
- **Output**: 17 classes (digits 0-9, letters A-G for values 10-16)
- **Architecture**: 4 conv layers with BatchNorm, ~850K params
- **Validation accuracy**: 99.3%

### Baseline Digit CNN (V1 - CNN_v2)
- **Purpose**: Used in handwritten V1 experiments
- **Input**: 28×28 grayscale cell
- **Architecture**: 2 conv layers, ~400K params
- **Note**: Less accurate than V2 models

## Benchmark Dataset

| Category | Puzzle Type | Sizes | Images/Size | Total | Location |
|----------|-------------|-------|-------------|-------|----------|
| Computer | KenKen | 3,4,5,6,7,9 | 100 | 600 | `final/benchmarks/KenKen/computer/` |
| Computer | Sudoku | 4,9 | 100 | 200 | `final/benchmarks/Sudoku/computer/` |
| Computer | HexaSudoku (Hex) | 16 | 100 | 100 | `final/benchmarks/HexaSudoku_16x16/computer_hex/` |
| Computer | HexaSudoku (Numeric) | 16 | 100 | 100 | `final/benchmarks/HexaSudoku_16x16/computer_numeric/` |
| Handwritten | KenKen | 3,4,5,6,7,9 | 100 | 600 | `final/benchmarks/KenKen/handwritten/` |
| Handwritten | Sudoku | 4,9 | 100 | 200 | `final/benchmarks/Sudoku/handwritten/` |
| Handwritten | HexaSudoku (Hex) | 16 | 100 | 100 | `final/benchmarks/HexaSudoku_16x16/handwritten_hex/` |
| Handwritten | HexaSudoku (Numeric) | 16 | 100 | 100 | `final/benchmarks/HexaSudoku_16x16/handwritten_numeric/` |
| **Total** | | | | **2,000** | |

## Key Files

| Resource | Location | Description |
|----------|----------|-------------|
| Benchmark Images | `final/benchmarks/` | 2,000 puzzle images |
| Pre-trained Models | `final/models/` | CNN weights (.pth files) |
| Evaluation Results | `final/results/` | CSV files with accuracy data |
| LLM Benchmark | `final/evaluation/llm/llm_benchmark.py` | Evaluate LLMs on puzzles |
| KenKen Solver (V2) | `archive/KenKen-handwritten-v2/solver/solve_all_sizes.py` | Production KenKen solver |
| Sudoku Solver (V2) | `archive/Sudoku-handwritten-v2/solver/solve_all_sizes.py` | Unified Sudoku/HexaSudoku solver |
| CNN Architecture | `archive/KenKen-handwritten-v2/models/improved_cnn.py` | ImprovedCNN definition |
| Technical Report | `final/REPORT.md` | Detailed technical documentation |

## Error Correction Methods

For handwritten puzzles, CNN misclassifications can cause constraint conflicts. Five correction approaches were developed:

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

## KenKen Handwritten V2

The V2 approach dramatically improves handwritten KenKen solving through better CNN architecture and training methodology.

### V1 vs V2 Comparison

| Size | V1 Corrected | V2 Corrected | Improvement |
|------|--------------|--------------|-------------|
| 3×3  | 89%          | **100%**     | +11%        |
| 4×4  | 58%          | **94%**      | +36%        |
| 5×5  | 26%          | **74%**      | +48%        |
| 6×6  | 15%          | **66%**      | +51%        |
| 7×7  | 2%           | **41%**      | +39%        |
| 9×9  | 1%           | **26%**      | +25%        |

### Error Correction Breakdown (V2)

Out of 600 handwritten KenKen puzzles, the correction methods contributed as follows:

| Correction Type | Puzzles | Description |
|-----------------|---------|-------------|
| **None (direct solve)** | 275 | CNN predictions correct, no correction needed |
| **Simple single** | 76 | Fixed 1 OCR error using top-K predictions |
| **Constraint single** | 29 | Fixed 1 error via Z3 unsat core analysis |
| **Simple two** | 14 | Fixed 2 OCR errors using top-K predictions |
| **Confidence swap** | 2 | Fixed via lowest-confidence character swaps |
| **Constraint two** | 2 | Fixed 2 errors via unsat core |
| **Constraint auto** | 2 | Auto-inferred missing operators |
| **Constraint four** | 1 | Fixed 4 errors via unsat core |
| **Still uncorrectable** | 199 | Too many errors (4+) to correct |

**Key insights**:
- 46% of puzzles (275/600) solved directly without correction
- 21% of puzzles (126/600) recovered via error correction
- Simple single-error correction was most effective (76 puzzles)
- 33% of puzzles remain unsolvable due to 4+ OCR errors

### What Changed in V2

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

### Key Result

The 1↔7 confusion that caused 37% of V1 errors was **completely eliminated** in V2. This single improvement accounts for most of the accuracy gains.

### Why Larger Puzzles Remain Challenging

Even with 99.9% per-character accuracy, larger puzzles have more characters:
- 3×3: ~15 characters → 98.5% chance of zero errors
- 9×9: ~80 characters → 92% chance of zero errors

The error correction can fix 1-3 errors per puzzle, but some 9×9 puzzles have 4+ OCR errors.

See [`archive/KenKen-handwritten-v2/V1_V2_COMPARISON_REPORT.md`](archive/KenKen-handwritten-v2/V1_V2_COMPARISON_REPORT.md) for detailed analysis.

## Sudoku/HexaSudoku Handwritten V2

A unified solver handling all Sudoku and HexaSudoku variants with a single 17-class CNN model.

### V1 vs V2 Comparison

| Puzzle Type | V1 | V2 | Improvement |
|-------------|-----|-----|-------------|
| Sudoku 4×4 | 92% | **100%** | +8% |
| Sudoku 9×9 | 85% | **98%** | +13% |
| HexaSudoku (A-G) | 10% | **91%** | +81% |
| HexaSudoku (Numeric) | 32% | **72%** | +40% |

### What Changed in V2

**Unified Model Architecture**:
- Single ImprovedCNN (17 classes: digits 0-9 + letters A-G)
- 850K parameters with BatchNorm and deeper architecture
- 99.3% validation accuracy

**HexaSudoku Numeric Handling**:
- **Tens digit forced to 1**: Eliminates 1↔7 confusion for values 10-16
- **Ones digit constrained to 0-6**: Values 17-19 don't exist in HexaSudoku
- **Ink density ratio** for single/double digit detection (threshold 1.7)
- **Fixed spatial split** for two-digit extraction (left/right halves)

**Error Correction**:
- Z3 unsat core analysis identifies conflicting cells
- Single and double-error correction with top-K CNN predictions
- Domain-aware alternatives (only valid values tried)

### Baseline vs Error-Corrected

| Puzzle Type | Baseline | Corrected | Improvement |
|-------------|----------|-----------|-------------|
| Sudoku 4×4 | 100% | 100% | +0% |
| Sudoku 9×9 | 96% | 98% | +2% |
| HexaSudoku (A-G) | 77% | 91% | +14% |
| HexaSudoku (Numeric) | 60% | 72% | +12% |

See [`archive/Sudoku-handwritten-v2/REPORT.md`](archive/Sudoku-handwritten-v2/REPORT.md) for detailed analysis.

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

### KenKen Solver (Computer-Generated)
```bash
cd archive/KenKen
python solve_all_sizes.py --sizes 3,4,5,6,7,9 --num 100
```

### KenKen Solver (Handwritten V2)
```bash
cd archive/KenKen-handwritten-v2/solver
python3 solve_all_sizes.py --sizes 3,4,5,6,7,9 --num 100
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
cd final/evaluation
python llm_benchmark.py --llm claude --puzzle kenken --sizes 3,4,5 --num 30
python llm_benchmark.py --llm all --puzzle kenken --sizes 3,4,5,6,7 --num 30
```

## Project Structure

```
KenKenSolver/
├── README.md                    # This file
│
├── final/                       # Curated release package
│   ├── benchmarks/             # 2,000 puzzle images
│   │   ├── KenKen/            # Computer & Handwritten (3×3 to 9×9)
│   │   ├── Sudoku/            # Computer & Handwritten (4×4, 9×9)
│   │   └── HexaSudoku_16x16/  # Hex & Numeric notation
│   ├── models/                 # Pre-trained CNN weights
│   │   ├── computer/          # CNNs for computer-generated puzzles
│   │   ├── handwritten_v1/    # 90-10 split models
│   │   └── handwritten_v2/    # ImprovedCNN models
│   ├── puzzles/               # Puzzle definitions (JSON)
│   ├── results/               # Evaluation CSVs
│   │   ├── neurosymbolic/     # Solver accuracy results
│   │   └── llm/               # LLM comparison results
│   ├── evaluation/            # LLM benchmark script
│   │   └── llm_benchmark.py   # Unified evaluation tool
│   └── REPORT.md              # Technical report
│
└── archive/                    # Complete experimental history
    ├── KenKen/                # KenKen solver (computer-generated)
    ├── KenKen handwritten/    # V1 handwritten solver
    ├── KenKen-handwritten-v2/ # V2 improved solver
    ├── KenKen-300px/          # 300px computer-generated variant
    ├── KenKen-300px-handwritten/ # 300px handwritten variant
    ├── Sudoku/                # Sudoku solver
    ├── Sudoku-handwritten-v2/ # Unified Sudoku/HexaSudoku
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

## Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
- MNIST dataset (LeCun et al., 1998) for handwritten digits
- EMNIST dataset (Cohen et al., 2017) for handwritten letters
