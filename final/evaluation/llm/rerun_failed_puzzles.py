#!/usr/bin/env python3
"""
Rerun Failed LLM Puzzle Evaluations

This script re-runs specific puzzle indices that had API errors (0 response_time).
It updates the CSV files in-place with the new results.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_benchmark import (
    GeminiClient,
    extract_solution,
    validate_kenken_solution,
    KENKEN_PROMPT,
)


def load_puzzles(puzzle_type, puzzles_dir):
    """Load puzzles from JSON file."""
    puzzle_file = puzzles_dir / f"{puzzle_type}_puzzles.json"
    if not puzzle_file.exists():
        raise FileNotFoundError(f"Puzzle file not found: {puzzle_file}")

    with open(puzzle_file) as f:
        return json.load(f)


def update_csv_row(csv_path, puzzle_idx, new_data):
    """Update a specific row in the CSV file."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row['puzzle_idx']) == puzzle_idx:
                row['correct'] = str(new_data['correct'])
                row['response_time'] = str(new_data['response_time'])
                row['tokens'] = str(new_data['tokens'])
            rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rerun_puzzle(client, puzzle_type, size, puzzle_idx, puzzles_data, benchmark_dir, prompt):
    """Re-run a single puzzle and return results."""
    # Get puzzle data - structure is {size: [puzzle0_cages, puzzle1_cages, ...]}
    size_key = str(size)
    if size_key not in puzzles_data or puzzle_idx >= len(puzzles_data[size_key]):
        raise ValueError(f"Puzzle index {puzzle_idx} not found for size {size}")

    puzzle_cages = puzzles_data[size_key][puzzle_idx]

    # Find the image - use same format as llm_benchmark.py
    if puzzle_type == "kenken":
        image_path = benchmark_dir / "KenKen" / "Computer" / f"{size}x{size}" / f"board{size}_{puzzle_idx}.png"
    else:
        raise ValueError(f"Unsupported puzzle type: {puzzle_type}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"  Evaluating {puzzle_type} {size}x{size} puzzle {puzzle_idx}...")

    try:
        response, tokens, elapsed = client.get_response(prompt, str(image_path))

        extracted = extract_solution(response, notation="numeric")

        if puzzle_type == "kenken":
            # validate_kenken_solution(puzzle_cages, size, solution)
            is_correct = validate_kenken_solution(puzzle_cages, size, extracted)
        else:
            is_correct = False  # Not implemented for other puzzle types

        result = {
            'correct': is_correct,
            'response_time': elapsed,
            'tokens': tokens
        }

        status = "CORRECT" if is_correct else "WRONG"
        print(f"    Result: {status} ({elapsed:.1f}s)")

        return result

    except Exception as e:
        print(f"    ERROR: {e}")
        return {
            'correct': False,
            'response_time': 0,
            'tokens': 0
        }


def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    results_dir = base_dir / "final/results/llm"
    puzzles_dir = base_dir / "final/puzzles"
    benchmark_dir = base_dir / "final/benchmarks"

    # Failed puzzles to rerun (from analysis)
    # Updated to only include remaining failures
    failed_puzzles = {
        'gemini_kenken_4x4.csv': {
            'puzzle_type': 'kenken',
            'size': 4,
            'indices': [40]  # Still failing with 504 timeout
        },
    }

    # Initialize Gemini client
    print("Initializing Gemini client...")
    client = GeminiClient()

    # Load puzzles
    kenken_puzzles = load_puzzles('kenken', puzzles_dir)

    # Process each file
    for csv_filename, config in failed_puzzles.items():
        csv_path = results_dir / csv_filename

        if not csv_path.exists():
            print(f"WARNING: {csv_filename} not found, skipping")
            continue

        print(f"\nProcessing {csv_filename}...")

        puzzle_type = config['puzzle_type']
        size = config['size']

        for puzzle_idx in config['indices']:
            # Rate limiting
            time.sleep(5)

            # Retry up to 3 times
            max_retries = 3
            for attempt in range(max_retries):
                result = rerun_puzzle(
                    client=client,
                    puzzle_type=puzzle_type,
                    size=size,
                    puzzle_idx=puzzle_idx,
                    puzzles_data=kenken_puzzles,
                    benchmark_dir=benchmark_dir,
                    prompt=KENKEN_PROMPT
                )

                # Only update if we got a valid response (non-zero time)
                if result['response_time'] > 0:
                    update_csv_row(csv_path, puzzle_idx, result)
                    print(f"    Updated CSV row for puzzle {puzzle_idx}")
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"    Retrying puzzle {puzzle_idx} (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(10)  # Wait longer before retry
                    else:
                        print(f"    Skipped updating puzzle {puzzle_idx} (still got error after {max_retries} attempts)")

    print("\nDone!")


if __name__ == "__main__":
    main()
