#!/usr/bin/env python3
"""
Unified LLM Benchmark Script for Puzzle Solving

This script evaluates multiple LLMs (Claude, GPT, Gemini, Qwen) on puzzle solving tasks
including KenKen, Sudoku, and HexaSudoku.

Usage:
    python llm_benchmark.py --llm claude --puzzle kenken --sizes 3,4,5 --num 30
    python llm_benchmark.py --llm all --puzzle all --num 100
    python llm_benchmark.py --llm gemini --puzzle sudoku --sizes 4,9 --num 50

Requirements:
    pip install z3-solver anthropic openai google-generativeai python-dotenv pandas

Environment Variables (or .env file):
    ANTHROPIC_API_KEY - For Claude
    OPENAI_API_KEY - For GPT
    GOOGLE_API_KEY - For Gemini
    OPENROUTER_API_KEY - For Qwen (via OpenRouter)
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from z3 import And, Distinct, Int, Or, Solver, Sum, sat

# Load environment variables
load_dotenv()


# ============================================================================
# Prompts for different puzzle types
# ============================================================================

KENKEN_PROMPT = """
You will be provided an empty KenKen puzzle board, which is a puzzle similar to Sudoku but with mathematical operations. Like Sudoku,
every row and column must contain the numbers 1 through n, where n is the size of the grid. The thick border lines represent cages,
which contain a target number and arithmetic operator (+-/*) in the top left cell of each cage. For a given cage, all of the numbers
that will make up that cage must arrive at the target number through the arithmetic operator. For example in a cage with two cells
and the symbol 5+, it could be filled in with a 2 and a 3 because 2 + 3 = 5. If there is only one cell in the cage, then it can be
automatically filled in with the target number.

Your task is to provide a correct solution to the puzzle provided. The puzzle could have size 3, 4, 5, 6, 7, 8, or 9. All puzzles have at least
one solution. Format your response as a 2 dimensional list representing the solution for the puzzle. An example response for a 3x3 KenKen puzzle is:
[[1, 2, 3],[3, 1, 2],[2, 3, 1]]
"""

SUDOKU_PROMPT = """
You will be provided a Sudoku puzzle board. Like standard Sudoku, every row, column, and box must contain unique numbers.
For a 4x4 puzzle, use numbers 1-4 with 2x2 boxes.
For a 9x9 puzzle, use numbers 1-9 with 3x3 boxes.

The puzzle shows some pre-filled numbers. Your task is to complete the puzzle by filling in the empty cells.

Format your response as a 2 dimensional list representing the complete solution. An example response for a 4x4 Sudoku is:
[[1, 2, 3, 4],[3, 4, 1, 2],[2, 3, 4, 1],[4, 1, 2, 3]]
"""

HEXASUDOKU_HEX_PROMPT = """
You will be provided a 16x16 HexaSudoku puzzle board. Each row, column, and 4x4 box must contain unique values.
Values are: digits 1-9 and letters A-G (where A=10, B=11, C=12, D=13, E=14, F=15, G=16).

Your task is to complete the puzzle by filling in the empty cells.

Format your response as a 2 dimensional list representing the complete solution using digits 1-9 and letters A-G.
Example cell values: 1, 2, 3, ..., 9, A, B, C, D, E, F, G
"""

HEXASUDOKU_NUMERIC_PROMPT = """
You will be provided a 16x16 HexaSudoku puzzle board. Each row, column, and 4x4 box must contain unique values.
Values are the numbers 1-16 (two-digit numbers 10-16 appear in cells).

Your task is to complete the puzzle by filling in the empty cells.

Format your response as a 2 dimensional list representing the complete solution using numbers 1-16.
Example: [[1, 2, 10, 16, ...], [...], ...]
"""


# ============================================================================
# Solution extraction and validation
# ============================================================================

def extract_solution(response, notation="numeric"):
    """Extract 2D list solution from LLM response text.

    Args:
        response: LLM response text
        notation: "numeric" for numbers 1-16, "hex" for 1-9 and A-G

    Returns:
        2D list of integers (1-16) or None if extraction fails
    """
    if not response:
        return None

    solution = [[]]
    row = 0

    start = response.rfind("[[")
    end = response.find("]]", start)

    if start == -1 or end == -1:
        return None

    # Extract the array portion
    array_str = response[start:end+2]

    if notation == "hex":
        # Parse hex notation (1-9, A-G)
        hex_map = {'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16}
        i = 0
        while i < len(array_str):
            char = array_str[i]
            if char.isdigit():
                solution[row].append(int(char))
            elif char.upper() in hex_map:
                solution[row].append(hex_map[char.upper()])
            elif char == ']':
                solution.append([])
                row += 1
            i += 1
    else:
        # Parse numeric notation (1-16, handles two-digit numbers)
        i = 0
        while i < len(array_str):
            char = array_str[i]
            if char.isdigit():
                # Check for two-digit number
                num_str = char
                while i + 1 < len(array_str) and array_str[i + 1].isdigit():
                    i += 1
                    num_str += array_str[i]
                solution[row].append(int(num_str))
            elif char == ']':
                solution.append([])
                row += 1
            i += 1

    # Remove ALL empty trailing lists
    while solution and solution[-1] == []:
        solution.pop()

    return solution


def parse_kenken_constraints(puzzle, cells):
    """Parse KenKen cage constraints for Z3."""
    constraints = []
    for block in puzzle:
        op = block["op"]
        target = block["target"]
        vars_in_block = [cells[i][j] for i, j in block["cells"]]

        if op == "":
            constraints.append(vars_in_block[0] == target)
        elif op == "add":
            constraints.append(Sum(vars_in_block) == target)
        elif op == "mul":
            product = vars_in_block[0]
            for v in vars_in_block[1:]:
                product *= v
            constraints.append(product == target)
        elif op == "sub" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a - b == target, b - a == target))
        elif op == "div" and len(vars_in_block) == 2:
            a, b = vars_in_block
            constraints.append(Or(a / b == target, b / a == target))

    return constraints


def validate_kenken_solution(puzzle, size, solution):
    """Validate KenKen solution using Z3."""
    if not solution or len(solution) != size:
        return False
    if not all(len(row) == size for row in solution):
        return False

    X = [[Int(f"x_{i+1}_{j+1}") for j in range(size)] for i in range(size)]

    cells_c = [And(1 <= X[i][j], X[i][j] <= size) for i in range(size) for j in range(size)]
    rows_c = [Distinct(X[i]) for i in range(size)]
    cols_c = [Distinct([X[i][j] for i in range(size)]) for j in range(size)]
    cage_c = parse_kenken_constraints(puzzle, X)

    instance = [X[i][j] == solution[i][j] for i in range(size) for j in range(size)]

    s = Solver()
    s.add(cells_c + rows_c + cols_c + cage_c + instance)
    return s.check() == sat


def validate_sudoku_solution(puzzle, size, solution):
    """Validate Sudoku solution using Z3."""
    if not solution or len(solution) != size:
        return False
    if not all(len(row) == size for row in solution):
        return False

    X = [[Int(f"x_{i+1}_{j+1}") for j in range(size)] for i in range(size)]

    # Cell range constraints
    cells_c = [And(1 <= X[i][j], X[i][j] <= size) for i in range(size) for j in range(size)]

    # Row and column constraints
    rows_c = [Distinct(X[i]) for i in range(size)]
    cols_c = [Distinct([X[i][j] for i in range(size)]) for j in range(size)]

    # Box constraints
    box_size = int(size ** 0.5)
    boxes_c = []
    for box_row in range(box_size):
        for box_col in range(box_size):
            box_cells = [
                X[box_row * box_size + i][box_col * box_size + j]
                for i in range(box_size)
                for j in range(box_size)
            ]
            boxes_c.append(Distinct(box_cells))

    instance = [X[i][j] == solution[i][j] for i in range(size) for j in range(size)]

    s = Solver()
    s.add(cells_c + rows_c + cols_c + boxes_c + instance)
    return s.check() == sat


# ============================================================================
# LLM Client Classes
# ============================================================================

class ClaudeClient:
    """Anthropic Claude API client."""

    def __init__(self, model="claude-sonnet-4-20250514"):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def get_response(self, prompt, image_path):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        start = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    }
                ]
            }]
        )
        elapsed = time.time() - start

        tokens = response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text, tokens, elapsed


class GPTClient:
    """OpenAI GPT API client."""

    def __init__(self, model="gpt-4o"):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_response(self, prompt, image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        data_uri = f"data:image/png;base64,{encoded}"

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }],
            max_tokens=2048
        )
        elapsed = time.time() - start

        tokens = response.usage.total_tokens
        return response.choices[0].message.content, tokens, elapsed


class GeminiClient:
    """Google Gemini API client."""

    def __init__(self, model="gemini-2.5-pro"):
        import google.generativeai as genai
        import PIL.Image
        self.PIL = PIL

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model)

    def get_response(self, prompt, image_path):
        image = self.PIL.Image.open(image_path)

        start = time.time()
        response = self.model.generate_content([prompt, image])
        elapsed = time.time() - start

        # Gemini doesn't easily expose token counts in same way
        return response.text, 0, elapsed


class QwenClient:
    """Qwen API client via OpenRouter."""

    def __init__(self, model="qwen/qwen3-vl-235b-a22b-instruct"):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model

    def get_response(self, prompt, image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        data_uri = f"data:image/png;base64,{encoded}"

        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }],
            max_tokens=2048
        )
        elapsed = time.time() - start

        tokens = response.usage.total_tokens if response.usage else 0
        return response.choices[0].message.content, tokens, elapsed


def get_llm_client(llm_name):
    """Factory function to create LLM client."""
    clients = {
        "claude": ClaudeClient,
        "gpt": GPTClient,
        "gemini": GeminiClient,
        "qwen": QwenClient,
    }

    if llm_name not in clients:
        raise ValueError(f"Unknown LLM: {llm_name}. Available: {list(clients.keys())}")

    return clients[llm_name]()


# ============================================================================
# Main evaluation logic
# ============================================================================

def get_prompt_for_puzzle(puzzle_type, notation="hex"):
    """Get the appropriate prompt for puzzle type.

    Args:
        puzzle_type: "kenken", "sudoku", "hexasudoku_hex", or "hexasudoku_numeric"
        notation: For hexasudoku, "hex" for A-G notation, "numeric" for 10-16 notation

    Returns:
        Prompt string
    """
    if puzzle_type == "kenken":
        return KENKEN_PROMPT
    elif puzzle_type == "sudoku":
        return SUDOKU_PROMPT
    elif puzzle_type == "hexasudoku_hex" or (puzzle_type == "hexasudoku" and notation == "hex"):
        return HEXASUDOKU_HEX_PROMPT
    elif puzzle_type == "hexasudoku_numeric" or (puzzle_type == "hexasudoku" and notation == "numeric"):
        return HEXASUDOKU_NUMERIC_PROMPT
    else:
        return KENKEN_PROMPT


def get_validator_for_puzzle(puzzle_type):
    """Get the appropriate validator for puzzle type."""
    if puzzle_type == "kenken":
        return validate_kenken_solution
    else:
        return validate_sudoku_solution


def run_evaluation(
    llm_name,
    puzzle_type,
    sizes,
    num_puzzles,
    benchmark_dir,
    puzzles_dir,
    output_dir,
    variant="Computer",
    notation="hex",
    delay=5
):
    """Run evaluation for a single LLM on a puzzle type.

    Args:
        llm_name: Name of LLM to use
        puzzle_type: "kenken", "sudoku", "hexasudoku_hex", or "hexasudoku_numeric"
        sizes: List of puzzle sizes to evaluate
        num_puzzles: Number of puzzles per size
        benchmark_dir: Path to benchmark images
        puzzles_dir: Path to puzzle JSON files
        output_dir: Path to save results
        variant: "Computer" or "Handwritten"
        notation: For hexasudoku, "hex" for A-G or "numeric" for 10-16
        delay: Seconds between API calls
    """

    print(f"\n{'='*60}")
    print(f"Evaluating {llm_name.upper()} on {puzzle_type.upper()} ({variant})")
    print(f"Sizes: {sizes}, Puzzles per size: {num_puzzles}")
    print(f"{'='*60}\n")

    # Initialize LLM client
    try:
        client = get_llm_client(llm_name)
    except Exception as e:
        print(f"Error initializing {llm_name}: {e}")
        return None

    # Determine base puzzle type for loading puzzle file
    base_puzzle_type = puzzle_type.replace("_hex", "").replace("_numeric", "")

    # Load puzzles
    puzzle_file = puzzles_dir / f"{base_puzzle_type}_puzzles.json"
    if not puzzle_file.exists():
        print(f"Puzzle file not found: {puzzle_file}")
        return None

    with open(puzzle_file) as f:
        puzzles_data = json.load(f)

    # Determine notation for hexasudoku
    if puzzle_type == "hexasudoku_numeric":
        notation = "numeric"
    elif puzzle_type == "hexasudoku_hex":
        notation = "hex"

    prompt = get_prompt_for_puzzle(puzzle_type, notation)
    validator = get_validator_for_puzzle(base_puzzle_type)

    results = {
        "size": [],
        "puzzle_idx": [],
        "correct": [],
        "response_time": [],
        "tokens": [],
    }

    accuracy_by_size = {}
    time_by_size = {}

    for size in sizes:
        size_str = str(size)
        if size_str not in puzzles_data:
            print(f"No puzzles found for size {size}")
            continue

        puzzles = puzzles_data[size_str]
        count = min(num_puzzles, len(puzzles))

        correct_count = 0
        total_time = 0

        print(f"\nSize {size}x{size}:")

        for i in range(count):
            # Construct image path based on puzzle type
            if puzzle_type == "kenken":
                image_path = benchmark_dir / "KenKen" / variant / f"{size}x{size}" / f"board{size}_{i}.png"
            elif puzzle_type == "sudoku":
                image_path = benchmark_dir / "Sudoku" / variant / f"{size}x{size}" / f"board{size}_{i}.png"
            elif puzzle_type == "hexasudoku_hex":
                image_path = benchmark_dir / "HexaSudoku_16x16" / f"{variant}_Hex_Notation" / f"board{size}_{i}.png"
            elif puzzle_type == "hexasudoku_numeric":
                image_path = benchmark_dir / "HexaSudoku_16x16" / f"{variant}_Numeric" / f"board{size}_{i}.png"
            else:  # generic hexasudoku (default to hex notation)
                image_path = benchmark_dir / "HexaSudoku_16x16" / f"{variant}_Hex_Notation" / f"board{size}_{i}.png"

            if not image_path.exists():
                print(f"  Image not found: {image_path}")
                continue

            try:
                response, tokens, elapsed = client.get_response(prompt, str(image_path))
                solution = extract_solution(response, notation)

                is_correct = False
                if solution:
                    puzzle = puzzles[i]
                    is_correct = validator(puzzle, size, solution)

                if is_correct:
                    correct_count += 1

                total_time += elapsed

                results["size"].append(size)
                results["puzzle_idx"].append(i)
                results["correct"].append(is_correct)
                results["response_time"].append(elapsed)
                results["tokens"].append(tokens)

                print(f"  Puzzle {i+1}/{count}: {'CORRECT' if is_correct else 'WRONG'} ({elapsed:.1f}s)")

            except Exception as e:
                print(f"  Puzzle {i+1}/{count}: ERROR - {e}")
                results["size"].append(size)
                results["puzzle_idx"].append(i)
                results["correct"].append(False)
                results["response_time"].append(0)
                results["tokens"].append(0)

            if delay > 0 and i < count - 1:
                time.sleep(delay)

        accuracy_by_size[size] = correct_count / count * 100 if count > 0 else 0
        time_by_size[size] = total_time / count if count > 0 else 0

        print(f"  Accuracy: {correct_count}/{count} ({accuracy_by_size[size]:.1f}%)")
        print(f"  Avg time: {time_by_size[size]:.2f}s")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{llm_name}_{puzzle_type}_{variant.lower()}_{timestamp}.csv"

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print(f"\n{'='*40}")
    print(f"SUMMARY: {llm_name.upper()} on {puzzle_type.upper()}")
    print(f"{'='*40}")
    for size in sizes:
        if size in accuracy_by_size:
            print(f"  {size}x{size}: {accuracy_by_size[size]:.1f}% accuracy, {time_by_size[size]:.2f}s avg")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Unified LLM Benchmark for Puzzle Solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python llm_benchmark.py --llm claude --puzzle kenken --sizes 3,4,5 --num 30
    python llm_benchmark.py --llm gemini --puzzle sudoku --sizes 4,9 --num 100
    python llm_benchmark.py --llm gpt --puzzle hexasudoku_hex --sizes 16 --num 50
    python llm_benchmark.py --llm gemini --puzzle hexasudoku_numeric --sizes 16 --num 50
    python llm_benchmark.py --llm all --puzzle kenken --sizes 3,4,5,6,7 --num 30

Puzzle types:
    kenken            - KenKen puzzles (sizes 3-9)
    sudoku            - Sudoku puzzles (sizes 4, 9)
    hexasudoku_hex    - 16x16 HexaSudoku with A-G notation
    hexasudoku_numeric - 16x16 HexaSudoku with 10-16 notation
    all               - Run all puzzle types
        """
    )

    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="LLM to evaluate: claude, gpt, gemini, qwen, or 'all'"
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        required=True,
        help="Puzzle type: kenken, sudoku, hexasudoku_hex, hexasudoku_numeric, or 'all'"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="3,4,5,6,7",
        help="Comma-separated puzzle sizes (default: 3,4,5,6,7 for kenken)"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=30,
        help="Number of puzzles per size (default: 30)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="Computer",
        help="Benchmark variant: Computer or Handwritten (default: Computer)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay between API calls in seconds (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ../results/llm)"
    )

    args = parser.parse_args()

    # Setup paths - go up two levels from llm/ to final/
    script_dir = Path(__file__).parent
    final_dir = script_dir.parent.parent  # Go from llm/ -> evaluation/ -> final/
    benchmark_dir = final_dir / "benchmarks"
    puzzles_dir = final_dir / "puzzles"
    output_dir = Path(args.output_dir) if args.output_dir else final_dir / "results" / "llm"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # Determine LLMs to evaluate
    llms = ["claude", "gpt", "gemini", "qwen"] if args.llm == "all" else [args.llm]

    # Determine puzzles to evaluate
    # Include both HexaSudoku variants when 'all' is specified
    if args.puzzle == "all":
        puzzles = ["kenken", "sudoku", "hexasudoku_hex", "hexasudoku_numeric"]
    else:
        puzzles = [args.puzzle]

    # Run evaluations
    for llm in llms:
        for puzzle in puzzles:
            try:
                run_evaluation(
                    llm_name=llm,
                    puzzle_type=puzzle,
                    sizes=sizes,
                    num_puzzles=args.num,
                    benchmark_dir=benchmark_dir,
                    puzzles_dir=puzzles_dir,
                    output_dir=output_dir,
                    variant=args.variant,
                    delay=args.delay
                )
            except Exception as e:
                print(f"Error evaluating {llm} on {puzzle}: {e}")
                continue


if __name__ == "__main__":
    main()
