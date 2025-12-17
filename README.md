# KenKen Puzzle Solver

A neuro-symbolic AI system that solves KenKen puzzles by combining computer vision (CNNs) with symbolic constraint solving (Z3 SMT Solver). This project also benchmarks leading LLMs (Claude, GPT-4, Gemini, Qwen) on the same puzzles, demonstrating the superiority of hybrid neuro-symbolic approaches for structured logical problems.

## Key Results

| Solver | 3×3 | 4×4 | 5×5 | 6×6 | 7×7 | Avg Time |
|--------|-----|-----|-----|-----|-----|----------|
| **NeuroSymbolic** | 100% | 100% | 100% | 100% | 93% | ~5s |
| Gemini 2.5 Pro | 74% | 30% | 0% | 0% | 0% | ~238s |
| Claude Sonnet 4 | 39% | 7% | 0% | 0% | 0% | ~24s |
| Qwen 2.5 VL | 10% | 0% | 0% | 0% | 0% | ~46s |
| GPT-4o Mini | 8% | 0% | 0% | 0% | 0% | ~4s |

**Finding:** All LLMs fail on puzzles 5×5 and larger. The neuro-symbolic approach maintains near-perfect accuracy across all sizes.

## Project Structure

```
KenKenSolver/
├── NeuroSymbolicSolver.ipynb      # Main solver pipeline (CNN + Z3)
├── SymbolicPuzzleGenerator.ipynb  # Generate valid KenKen puzzles
├── BoardImageGeneration.ipynb     # Create puzzle images from JSON
├── AnalyzingResults.ipynb         # Compare solver performance
├── ClaudeEvaluation.ipynb         # Claude Sonnet 4 benchmark
├── GPTEvaluation.ipynb            # GPT-4o Mini benchmark
├── GeminiEvaluation.ipynb         # Gemini 2.5 Pro/Flash benchmark
├── QwenEvaluation.ipynb           # Qwen 2.5 VL benchmark
├── models/                        # Pre-trained CNN weights (Git LFS)
│   ├── character_recognition_v2_model_weights.pth
│   └── grid_detection_model_weights.pth
├── puzzles/
│   └── puzzles_dict.json          # 290 puzzles (3×3 to 7×7)
├── board_images/                  # 430+ generated puzzle PNGs
├── symbols/
│   ├── TMNIST_NotoSans.csv        # Character training data
│   └── operators/                 # +, -, ×, ÷ symbol images
└── results/                       # Evaluation CSV files
```

## Installation

### Prerequisites

- Python 3.10+
- Git LFS (for model weights)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kiguli/KenKenSolver.git
   cd KenKenSolver
   ```

2. **Install Git LFS and pull model weights:**
   ```bash
   # macOS
   brew install git-lfs

   # Ubuntu/Debian
   sudo apt install git-lfs

   # Then pull the actual model files
   git lfs install
   git lfs pull
   ```

3. **Install dependencies:**
   ```bash
   pip install z3-solver torch torchvision opencv-python pillow pandas numpy matplotlib
   ```

4. **For LLM evaluations (optional):**
   ```bash
   pip install anthropic openai google-generativeai transformers qwen-vl-utils python-dotenv
   ```

## Usage

### Running the NeuroSymbolic Solver

The main solver pipeline processes puzzle images and returns solutions:

```bash
jupyter notebook NeuroSymbolicSolver.ipynb
```

**Pipeline overview:**
1. **Grid Detection (CNN)** → Determines puzzle size (3-7)
2. **Border Detection (OpenCV)** → Identifies cage boundaries
3. **Character Recognition (CNN)** → Reads targets and operators
4. **Constraint Solving (Z3)** → Computes valid solution

### Generating New Puzzles

Create new puzzle datasets:

```bash
jupyter notebook SymbolicPuzzleGenerator.ipynb
```

This generates `puzzles_dict.json` with valid, solvable puzzles.

### Creating Puzzle Images

Convert puzzle JSON to 900×900px PNG images:

```bash
jupyter notebook BoardImageGeneration.ipynb
```

### Running LLM Evaluations

Each evaluation notebook requires API credentials:

#### Claude
```python
# Set environment variable
export ANTHROPIC_API_KEY="your-key-here"

# Or edit ClaudeEvaluation.ipynb cell-4:
client = anthropic.Anthropic(api_key="your-key-here")
```

#### GPT
```bash
export OPENAI_API_KEY="your-key-here"
```

#### Gemini
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-key-here
```

#### Qwen
No API key needed - runs locally via Hugging Face Transformers.

### Analyzing Results

Compare all solver performances:

```bash
jupyter notebook AnalyzingResults.ipynb
```

## Architecture

### NeuroSymbolic Pipeline

```
Image (900×900)
    ↓
Grid_CNN → Size (3-7)
    ↓
OpenCV (Canny + HoughLines) → Cage boundaries
    ↓
CNN_v2 → Target numbers + operators
    ↓
Z3 Solver → Valid solution
```

### Neural Networks

**Grid Detection CNN:**
- Input: 128×128 grayscale
- Output: 5 classes (sizes 3-7)
- Architecture: Conv(1→32→64) → FC(262144→128→5)

**Character Recognition CNN:**
- Input: 28×28 grayscale
- Output: 14 classes (0-9, +, -, ×, ÷)
- Architecture: Conv(1→32→64) → FC(3136→128→14)

### Z3 Constraints

```python
# Cell values in range
1 ≤ X[i][j] ≤ size

# Latin square rules
Distinct(row), Distinct(column)

# Cage operations
Sum(cage) == target      # Addition
Product(cage) == target  # Multiplication
|a - b| == target        # Subtraction
max(a/b, b/a) == target  # Division
```

## Dataset

| Size | Count | Description |
|------|-------|-------------|
| 3×3 | 100 | Simple puzzles |
| 4×4 | 100 | Standard puzzles |
| 5×5 | 100 | Medium puzzles |
| 6×6 | 100 | Hard puzzles |
| 7×7 | 30 | Expert puzzles |

**Total:** 430 puzzles with validated solutions

## Troubleshooting

### Model files show as text/pointers
```bash
git lfs pull
```

### PyTorch 2.6+ UnpicklingError
The notebooks already include `weights_only=False` for compatibility.

### File not found errors
Ensure you're running notebooks from the repository root directory.

## License

MIT License

## Acknowledgments

- Z3 Theorem Prover by Microsoft Research
- TMNIST dataset for character recognition training
