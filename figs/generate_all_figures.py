#!/usr/bin/env python3
"""
Generate publication-quality figures for IJCAI paper on Neuro-Symbolic Puzzle Solving.

Figures:
1. KenKen: NeuroSymbolic vs VLMs (hero figure)
2. Multi-Puzzle Type Comparison
3. Error Correction Impact (V2)
4. V1 vs V2 Comparison
5a. KenKen Error Correction Breakdown
5b. Sudoku/HexaSudoku Error Correction Breakdown
5c. Efficiency Comparison (KenKen only)
5d. Efficiency Comparison (All Puzzles - 3 panels)
6. Handwritten: NeuroSymbolic vs GPT-4o
7. Pipeline Architecture Diagram

Usage:
    python generate_all_figures.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import Counter

# Set up paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / 'results'
NS_RESULTS = RESULTS_DIR / 'neurosymbolic'
LLM_RESULTS = RESULTS_DIR / 'llm'

# =============================================================================
# Color Scheme & Style
# =============================================================================

COLORS = {
    'neurosymbolic': '#4682B4',  # steelblue
    'gpt4o': '#FF6347',          # tomato
    'gemini': '#3CB371',         # mediumseagreen
    'claude': '#9370DB',         # mediumpurple
    'qwen': '#FFD700',           # gold
    'gpt4o_mini': '#FFA07A',     # lightsalmon
    'baseline': '#87CEEB',       # lightsteelblue
    'corrected': '#FF7F50',      # coral
    'v1_base': '#FFA07A',        # lightsalmon (V1 = orange/red family)
    'v1_corr': '#CD5C5C',        # indianred (V1 corrected)
    'v2_base': '#87CEEB',        # lightskyblue (V2 = blue family)
    'v2_corr': '#4169E1',        # royalblue (V2 corrected)
    # Error correction types
    'none': '#2E8B57',           # seagreen (solved directly)
    'simple': '#FFD700',         # gold (simple correction)
    'constraint': '#FF8C00',     # darkorange (constraint-based)
    'uncorrectable': '#DC143C',  # crimson (failed)
}

def setup_style():
    """Configure matplotlib for IJCAI publication quality."""
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'serif',
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    })

# =============================================================================
# Data Loading
# =============================================================================

def load_llm_results(puzzle_type, variant='Computer'):
    """Load LLM benchmark results for a puzzle type."""
    results = {}
    llm_dir = LLM_RESULTS / variant

    if not llm_dir.exists():
        return results

    for f in llm_dir.glob(f'*_{puzzle_type}_*.csv'):
        # Parse filename: model_puzzle_size.csv
        name = f.stem
        parts = name.split('_')
        model = parts[0]

        try:
            df = pd.read_csv(f)
            # Handle different column names
            if 'correct' in df.columns:
                accuracy = df['correct'].mean() * 100
            elif 'is_correct' in df.columns:
                accuracy = df['is_correct'].mean() * 100
            else:
                continue

            # Extract size from filename or data
            if 'size' in df.columns:
                size = df['size'].iloc[0]
            elif 'puzzle_size' in df.columns:
                size = df['puzzle_size'].iloc[0]
            else:
                # Try to extract from filename
                for part in parts:
                    if 'x' in part:
                        size = int(part.split('x')[0])
                        break
                else:
                    continue

            key = (model, size)
            results[key] = accuracy
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    return results

def load_handwritten_results():
    """Load V1 and V2 handwritten results."""
    v1_path = NS_RESULTS / 'kenken_handwritten_v1.csv'
    v2_path = NS_RESULTS / 'kenken_handwritten_v2.csv'

    v1_data = {}
    v2_data = {}

    if v1_path.exists():
        df = pd.read_csv(v1_path)
        for size in df['size'].unique():
            size_df = df[df['size'] == size]
            v1_data[size] = {
                'baseline': size_df['baseline_solved'].mean() * 100,
                'corrected': size_df['corrected_solved'].mean() * 100,
                'correction_types': Counter(size_df['correction_type'])
            }

    if v2_path.exists():
        df = pd.read_csv(v2_path)
        for size in df['size'].unique():
            size_df = df[df['size'] == size]
            v2_data[size] = {
                'baseline': size_df['base_solved'].mean() * 100,
                'corrected': size_df['corrected_solved'].mean() * 100,
                'correction_types': Counter(size_df['correction_type'])
            }

    return v1_data, v2_data

# =============================================================================
# Figure 1: KenKen - NeuroSymbolic vs VLMs (Hero Figure)
# =============================================================================

def figure1_kenken_vlm_comparison():
    """Generate hero figure comparing NeuroSymbolic vs VLMs on KenKen."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # Data from README/results
    sizes = [3, 4, 5, 6, 7, 8, 9]

    # NeuroSymbolic: 100% on all sizes
    ns_acc = [100, 100, 100, 100, 100, 100, 100]

    # LLM accuracies (from benchmark results)
    gemini_acc = [73, 35, 0, 0, 0, 0, 0]
    claude_acc = [57, 6, 0, 0, 0, 0, 0]
    gpt4o_acc = [33, 0, 0, 0, 0, 0, 0]
    qwen_acc = [17, 0, 0, 0, 0, 0, 0]

    x = np.arange(len(sizes))
    width = 0.15

    # Plot bars
    bars_ns = ax.bar(x - 2*width, ns_acc, width, label='Neuro-Symbolic',
                     color=COLORS['neurosymbolic'], edgecolor='black', linewidth=0.5)
    bars_gemini = ax.bar(x - width, gemini_acc, width, label='Gemini 2.5 Pro',
                         color=COLORS['gemini'], edgecolor='black', linewidth=0.5)
    bars_claude = ax.bar(x, claude_acc, width, label='Claude Sonnet 4',
                         color=COLORS['claude'], edgecolor='black', linewidth=0.5)
    bars_gpt = ax.bar(x + width, gpt4o_acc, width, label='GPT-4o',
                      color=COLORS['gpt4o'], edgecolor='black', linewidth=0.5)
    bars_qwen = ax.bar(x + 2*width, qwen_acc, width, label='Qwen 2.5-VL',
                       color=COLORS['qwen'], edgecolor='black', linewidth=0.5)

    # Add value labels on bars (only for non-zero values)
    for bars in [bars_ns, bars_gemini, bars_claude, bars_gpt, bars_qwen]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 2), textcoords='offset points',
                           ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Formatting
    ax.set_xlabel('KenKen Grid Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)

    # Add annotation for VLM failure region
    ax.annotate('VLMs fail completely', xy=(4, 5), fontsize=8, style='italic',
                color='gray', ha='center')

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig1_kenken_vlm_comparison.pdf')
    plt.savefig(SCRIPT_DIR / 'fig1_kenken_vlm_comparison.png')
    plt.close()
    print("Generated: fig1_kenken_vlm_comparison.pdf")

# =============================================================================
# Figure 2: Multi-Puzzle Type Comparison
# =============================================================================

def figure2_puzzle_types():
    """Compare accuracy across different puzzle types."""
    fig, ax = plt.subplots(figsize=(3.5, 4))

    # Data: (puzzle, NeuroSymbolic, GPT-4o, Gemini)
    puzzles = ['KenKen 3×3', 'KenKen 4×4', 'Sudoku 4×4', 'Sudoku 9×9',
               'HexaSudoku\n16×16 (Hex)', 'HexaSudoku\n16×16 (Num)']

    ns_acc = [100, 100, 100, 100, 100, 100]
    gpt_acc = [33, 0, 75, 8, 0, 0]
    gemini_acc = [73, 35, 99, 0, 0, 0]

    y = np.arange(len(puzzles))
    height = 0.25

    ax.barh(y - height, ns_acc, height, label='Neuro-Symbolic',
            color=COLORS['neurosymbolic'], edgecolor='black', linewidth=0.5)
    ax.barh(y, gemini_acc, height, label='Gemini 2.5 Pro',
            color=COLORS['gemini'], edgecolor='black', linewidth=0.5)
    ax.barh(y + height, gpt_acc, height, label='GPT-4o',
            color=COLORS['gpt4o'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Accuracy (%)')
    ax.set_yticks(y)
    ax.set_yticklabels(puzzles)
    ax.set_xlim(0, 110)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig2_puzzle_types.pdf')
    plt.savefig(SCRIPT_DIR / 'fig2_puzzle_types.png')
    plt.close()
    print("Generated: fig2_puzzle_types.pdf")

# =============================================================================
# Figure 3: Error Correction Impact (V2)
# =============================================================================

def figure3_error_correction():
    """Show impact of error correction on V2 handwritten results."""
    fig, ax = plt.subplots(figsize=(3.5, 3))

    sizes = [3, 4, 5, 6, 7, 8, 9]
    baseline = [98, 86, 36, 32, 10, 13, 13]
    corrected = [100, 94, 74, 66, 41, 46, 26]
    improvement = [c - b for b, c in zip(baseline, corrected)]

    x = np.arange(len(sizes))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline',
                   color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, corrected, width, label='With Error Correction',
                   color=COLORS['corrected'], edgecolor='black', linewidth=0.5)

    # Add improvement annotations
    for i, (b, c, imp) in enumerate(zip(baseline, corrected, improvement)):
        if imp > 0:
            ax.annotate(f'+{imp}%', xy=(x[i] + width/2, c + 2),
                       ha='center', va='bottom', fontsize=7, fontweight='bold',
                       color='darkgreen')

    ax.set_xlabel('KenKen Grid Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig3_error_correction.pdf')
    plt.savefig(SCRIPT_DIR / 'fig3_error_correction.png')
    plt.close()
    print("Generated: fig3_error_correction.pdf")

# =============================================================================
# Figure 4: V1 vs V2 Comparison
# =============================================================================

def figure4_v1_vs_v2():
    """Compare V1 and V2 models with and without error correction."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    sizes = [3, 4, 5, 6, 7, 8, 9]
    x = np.arange(len(sizes))

    # V1 data
    v1_base = [69, 36, 18, 7, 1, 1, 0]
    v1_corr = [89, 58, 26, 15, 2, 1, 1]

    # V2 data
    v2_base = [98, 86, 36, 32, 10, 13, 13]
    v2_corr = [100, 94, 74, 66, 41, 46, 26]

    # Plot lines with markers
    ax.plot(x, v1_base, 'o--', color=COLORS['v1_base'], label='V1 Baseline',
            markersize=6, linewidth=1.5, alpha=0.7)
    ax.plot(x, v1_corr, 's-', color=COLORS['v1_corr'], label='V1 Corrected',
            markersize=6, linewidth=1.5)
    ax.plot(x, v2_base, '^--', color=COLORS['v2_base'], label='V2 Baseline',
            markersize=6, linewidth=1.5, alpha=0.7)
    ax.plot(x, v2_corr, 'D-', color=COLORS['v2_corr'], label='V2 Corrected',
            markersize=6, linewidth=2)

    # Fill between baseline and corrected for V1
    ax.fill_between(x, v1_base, v1_corr, alpha=0.2, color=COLORS['v1_corr'],
                    label='V1 Correction Gain')

    # Fill between baseline and corrected for V2
    ax.fill_between(x, v2_base, v2_corr, alpha=0.2, color=COLORS['v2_corr'],
                    label='V2 Correction Gain')

    ax.set_xlabel('KenKen Grid Size')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes])
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', ncol=2, fontsize=7)
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig4_v1_vs_v2.pdf')
    plt.savefig(SCRIPT_DIR / 'fig4_v1_vs_v2.png')
    plt.close()
    print("Generated: fig4_v1_vs_v2.pdf")

# =============================================================================
# Figure 5a: KenKen Error Correction Breakdown
# =============================================================================

def figure5a_kenken_correction_breakdown():
    """Show breakdown of error correction types for KenKen puzzles."""
    fig, ax = plt.subplots(figsize=(3.5, 3))

    sizes = [3, 4, 5, 6, 7, 8, 9]
    labels = [f'{s}×{s}' for s in sizes]

    # KenKen V2 data: none, simple, constraint, uncorrectable
    none = [98, 86, 36, 32, 10, 13, 13]
    simple = [0, 1, 30, 22, 26, 33, 11]
    constraint = [2, 5, 8, 11, 5, 0, 2]
    uncorrectable = [0, 6, 26, 34, 59, 54, 74]

    # Normalize to 100%
    totals = [n + s + c + u for n, s, c, u in zip(none, simple, constraint, uncorrectable)]
    none_pct = [100 * n / t if t > 0 else 0 for n, t in zip(none, totals)]
    simple_pct = [100 * s / t if t > 0 else 0 for s, t in zip(simple, totals)]
    constraint_pct = [100 * c / t if t > 0 else 0 for c, t in zip(constraint, totals)]
    uncorrectable_pct = [100 * u / t if t > 0 else 0 for u, t in zip(uncorrectable, totals)]

    x = np.arange(len(sizes))
    width = 0.7

    ax.bar(x, none_pct, width, label='Solved Directly', color=COLORS['none'],
           edgecolor='white', linewidth=0.5)
    ax.bar(x, simple_pct, width, bottom=none_pct, label='Simple Correction',
           color=COLORS['simple'], edgecolor='white', linewidth=0.5)
    bottom2 = [n + s for n, s in zip(none_pct, simple_pct)]
    ax.bar(x, constraint_pct, width, bottom=bottom2, label='Constraint Correction',
           color=COLORS['constraint'], edgecolor='white', linewidth=0.5)
    bottom3 = [b + c for b, c in zip(bottom2, constraint_pct)]
    ax.bar(x, uncorrectable_pct, width, bottom=bottom3, label='Uncorrectable',
           color=COLORS['uncorrectable'], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('KenKen Grid Size')
    ax.set_ylabel('Proportion (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig5a_kenken_correction_breakdown.pdf')
    plt.savefig(SCRIPT_DIR / 'fig5a_kenken_correction_breakdown.png')
    plt.close()
    print("Generated: fig5a_kenken_correction_breakdown.pdf")

# =============================================================================
# Figure 5b: Sudoku/HexaSudoku Error Correction Breakdown
# =============================================================================

def figure5b_sudoku_correction_breakdown():
    """Show breakdown of error correction types for Sudoku and HexaSudoku puzzles."""
    fig, ax = plt.subplots(figsize=(3.5, 3))

    puzzles = ['Sudoku\n4×4', 'Sudoku\n9×9', 'HexaSudoku\n16×16 (Hex)', 'HexaSudoku\n16×16 (Num)']

    # Sudoku/HexaSudoku V2 data: none, single, double, uncorrectable
    none = [100, 96, 77, 60]
    simple = [0, 4, 18, 16]       # single correction
    constraint = [0, 0, 1, 1]    # double correction
    uncorrectable = [0, 0, 4, 23]

    # Normalize to 100%
    totals = [n + s + c + u for n, s, c, u in zip(none, simple, constraint, uncorrectable)]
    none_pct = [100 * n / t if t > 0 else 0 for n, t in zip(none, totals)]
    simple_pct = [100 * s / t if t > 0 else 0 for s, t in zip(simple, totals)]
    constraint_pct = [100 * c / t if t > 0 else 0 for c, t in zip(constraint, totals)]
    uncorrectable_pct = [100 * u / t if t > 0 else 0 for u, t in zip(uncorrectable, totals)]

    x = np.arange(len(puzzles))
    width = 0.7

    ax.bar(x, none_pct, width, label='Solved Directly', color=COLORS['none'],
           edgecolor='white', linewidth=0.5)
    ax.bar(x, simple_pct, width, bottom=none_pct, label='Single Correction',
           color=COLORS['simple'], edgecolor='white', linewidth=0.5)
    bottom2 = [n + s for n, s in zip(none_pct, simple_pct)]
    ax.bar(x, constraint_pct, width, bottom=bottom2, label='Double Correction',
           color=COLORS['constraint'], edgecolor='white', linewidth=0.5)
    bottom3 = [b + c for b, c in zip(bottom2, constraint_pct)]
    ax.bar(x, uncorrectable_pct, width, bottom=bottom3, label='Uncorrectable',
           color=COLORS['uncorrectable'], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Proportion (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(puzzles, fontsize=7)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig5b_sudoku_correction_breakdown.pdf')
    plt.savefig(SCRIPT_DIR / 'fig5b_sudoku_correction_breakdown.png')
    plt.close()
    print("Generated: fig5b_sudoku_correction_breakdown.pdf")

# =============================================================================
# Figure 5c: Efficiency Comparison (Solve Time)
# =============================================================================

def figure5c_efficiency_comparison():
    """Compare solve time efficiency between Neuro-Symbolic and VLMs across all sizes."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    # All KenKen sizes
    sizes = ['3×3', '4×4', '5×5', '6×6', '7×7', '8×8', '9×9']

    # Neuro-Symbolic times in seconds (from results)
    ns_times = [0.132, 0.229, 0.363, 0.580, 0.893, 1.525, 3.070]

    # LLM times (seconds) - None where no data or 0% accuracy
    # GPT-4o-mini (fastest LLM)
    gpt_mini_times = [2.7, 2.2, None, None, None, None, None]
    # GPT-4o
    gpt_times = [9.8, 6.2, None, None, None, None, None]
    # Claude Sonnet 4
    claude_times = [23.6, 23.0, 20.7, None, None, None, None]
    # Gemini 2.5 Pro (thinking model)
    gemini_times = [98.3, 178.3, 189.8, None, None, None, None]

    x = np.arange(len(sizes))

    # Plot Neuro-Symbolic as a line (continuous across all sizes)
    ax.plot(x, ns_times, 'o-', color=COLORS['neurosymbolic'], label='Neuro-Symbolic (100% acc)',
            markersize=8, linewidth=2, zorder=10)

    # Plot LLM times as scatter points (only where they have data)
    for i, t in enumerate(gpt_mini_times):
        if t is not None:
            ax.scatter(i, t, color=COLORS['gpt4o_mini'], s=80, marker='s',
                      edgecolor='black', linewidth=0.5, zorder=5,
                      label='GPT-4o-mini' if i == 0 else '')
    for i, t in enumerate(gpt_times):
        if t is not None:
            ax.scatter(i, t, color=COLORS['gpt4o'], s=80, marker='^',
                      edgecolor='black', linewidth=0.5, zorder=5,
                      label='GPT-4o' if i == 0 else '')
    for i, t in enumerate(claude_times):
        if t is not None:
            ax.scatter(i, t, color=COLORS['claude'], s=80, marker='D',
                      edgecolor='black', linewidth=0.5, zorder=5,
                      label='Claude Sonnet 4' if i == 0 else '')
    for i, t in enumerate(gemini_times):
        if t is not None:
            ax.scatter(i, t, color=COLORS['gemini'], s=80, marker='p',
                      edgecolor='black', linewidth=0.5, zorder=5,
                      label='Gemini 2.5 Pro' if i == 0 else '')

    # Add "VLMs fail" region annotation
    ax.axvspan(2.5, 6.5, alpha=0.1, color='red')
    ax.text(4.5, 150, 'VLMs: 0% accuracy', ha='center', fontsize=9,
            style='italic', color='darkred')

    # Add horizontal reference line at fastest LLM time
    fastest_llm = min([t for t in gpt_mini_times if t is not None])
    ax.axhline(y=fastest_llm, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(6.7, fastest_llm * 0.8, f'Fastest VLM\n({fastest_llm:.1f}s)', fontsize=7,
            va='top', ha='right', color='gray')

    # Use log scale for y-axis
    ax.set_yscale('log')
    ax.set_ylim(0.05, 500)

    ax.set_xlabel('KenKen Grid Size')
    ax.set_ylabel('Solve Time (seconds, log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(loc='upper left', fontsize=7, ncol=2)

    # Add horizontal grid lines
    ax.grid(axis='y', alpha=0.3, linestyle=':', which='both')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig5c_efficiency_comparison.pdf')
    plt.savefig(SCRIPT_DIR / 'fig5c_efficiency_comparison.png')
    plt.close()
    print("Generated: fig5c_efficiency_comparison.pdf")

# =============================================================================
# Figure 5d: Comprehensive Efficiency Comparison (All Puzzle Types)
# =============================================================================

def figure5d_efficiency_all_puzzles():
    """Compare solve time efficiency across all puzzle types with legend outside."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 6))

    # Common marker styles
    markers = {
        'gpt_mini': ('s', COLORS['gpt4o_mini']),
        'gpt': ('^', COLORS['gpt4o']),
        'claude': ('D', COLORS['claude']),
        'gemini': ('p', COLORS['gemini']),
        'qwen': ('*', COLORS['qwen']),
    }

    # ===== Panel 1: KenKen =====
    ax = axes[0]
    sizes = ['3×3', '4×4', '5×5', '6×6', '7×7', '8×8', '9×9']
    ns_times = [0.132, 0.229, 0.363, 0.580, 0.893, 1.525, 3.070]
    x = np.arange(len(sizes))

    ax.plot(x, ns_times, 'o-', color=COLORS['neurosymbolic'], markersize=6, linewidth=2, zorder=10)

    # LLM data points - collect all for finding fastest per x position
    llm_data = {
        'gpt_mini': [(0, 2.7), (1, 2.2)],
        'gpt': [(0, 9.8), (1, 6.2)],
        'claude': [(0, 23.6), (1, 23.0), (2, 20.7)],
        'gemini': [(0, 98.3), (1, 178.3), (2, 189.8)],
        'qwen': [(0, 5.5), (1, 8.5)],
    }

    # Find fastest VLM at each x position
    fastest_vlm_kenken = {}
    for model, pts in llm_data.items():
        for xi, t in pts:
            if xi not in fastest_vlm_kenken or t < fastest_vlm_kenken[xi]:
                fastest_vlm_kenken[xi] = t

    for model, pts in llm_data.items():
        marker, color = markers[model]
        for xi, t in pts:
            ax.scatter(xi, t, color=color, s=60, marker=marker, edgecolor='black', linewidth=0.5, zorder=5)

    # Add speedup labels above NS points (only where VLMs have data)
    for i, ns_t in enumerate(ns_times):
        if i in fastest_vlm_kenken:
            speedup = fastest_vlm_kenken[i] / ns_t
            ax.annotate(f'{speedup:.0f}×', xy=(i, ns_t), xytext=(0, 8),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=7, fontweight='bold', color=COLORS['neurosymbolic'])

    ax.axvspan(2.5, 6.5, alpha=0.1, color='red')
    ax.text(4.5, 100, 'VLMs: 0%', ha='center', fontsize=8, style='italic', color='darkred')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 500)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_xlabel('KenKen Size')
    ax.set_ylabel('Solve Time (seconds, log)')
    ax.set_title('KenKen', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':', which='both')

    # ===== Panel 2: Sudoku (including HexaSudoku) =====
    ax = axes[1]
    # Combined labels: Sudoku 4x4, 9x9, 16x16 (Hex), 16x16 (Num)
    sizes = ['4×4', '9×9', '16×16\n(Hex)', '16×16\n(Numeric)']
    # HexaSudoku times are in seconds from CSV (solve_time column is actually in seconds)
    ns_times = [0.027, 0.275, 1.65, 1.61]
    x = np.arange(len(sizes))

    ax.plot(x, ns_times, 'o-', color=COLORS['neurosymbolic'], markersize=6, linewidth=2, zorder=10)

    # Sudoku LLM data (indices 0, 1)
    llm_data_sudoku = {
        'gpt_mini': [(0, 2.3), (1, 6.7)],
        'gpt': [(0, 4.5), (1, 10.8)],
        'claude': [(0, 21.8), (1, 22.2)],
        'gemini': [(0, 26.9), (1, 245.1)],
        'qwen': [(0, 2.4), (1, 8.7)],
    }

    # Find fastest VLM at each Sudoku position
    fastest_vlm_sudoku = {}
    for model, pts in llm_data_sudoku.items():
        for xi, t in pts:
            if xi not in fastest_vlm_sudoku or t < fastest_vlm_sudoku[xi]:
                fastest_vlm_sudoku[xi] = t

    for model, pts in llm_data_sudoku.items():
        marker, color = markers[model]
        for xi, t in pts:
            ax.scatter(xi, t, color=color, s=60, marker=marker, edgecolor='black', linewidth=0.5, zorder=5)

    # HexaSudoku LLM data (indices 2, 3)
    llm_data_hexa = {
        'gpt_mini': [(2, 15.7), (3, 17.5)],
        'gpt': [(2, 37.7), (3, 27.0)],
    }

    # Find fastest VLM at each HexaSudoku position
    fastest_vlm_hexa = {}
    for model, pts in llm_data_hexa.items():
        for xi, t in pts:
            if xi not in fastest_vlm_hexa or t < fastest_vlm_hexa[xi]:
                fastest_vlm_hexa[xi] = t

    for model, pts in llm_data_hexa.items():
        marker, color = markers[model]
        for xi, t in pts:
            ax.scatter(xi, t, color=color, s=60, marker=marker, edgecolor='black', linewidth=0.5, zorder=5)

    # Add speedup labels above NS points
    fastest_vlm_panel2 = {**fastest_vlm_sudoku, **fastest_vlm_hexa}
    for i, ns_t in enumerate(ns_times):
        if i in fastest_vlm_panel2:
            speedup = fastest_vlm_panel2[i] / ns_t
            ax.annotate(f'{speedup:.0f}×', xy=(i, ns_t), xytext=(0, 8),
                       textcoords='offset points', ha='center', va='bottom',
                       fontsize=7, fontweight='bold', color=COLORS['neurosymbolic'])

    ax.set_yscale('log')
    ax.set_ylim(0.01, 500)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=8)
    ax.set_xlabel('Sudoku Size')
    ax.set_ylabel('Solve Time (seconds, log)')
    ax.set_title('Sudoku', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle=':', which='both')

    # Create legend outside the figure
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color=COLORS['neurosymbolic'], label='Neuro-Symbolic (100%)',
               markersize=6, linewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['gpt4o_mini'],
               label='GPT-4o-mini', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['gpt4o'],
               label='GPT-4o', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['claude'],
               label='Claude Sonnet 4', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='p', color='w', markerfacecolor=COLORS['gemini'],
               label='Gemini 2.5 Pro', markersize=8, markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['qwen'],
               label='Qwen 2.5-VL', markersize=10, markeredgecolor='black'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=8, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(SCRIPT_DIR / 'fig5d_efficiency_all_puzzles.pdf')
    plt.savefig(SCRIPT_DIR / 'fig5d_efficiency_all_puzzles.png')
    plt.close()
    print("Generated: fig5d_efficiency_all_puzzles.pdf")

# =============================================================================
# Figure 6: Handwritten - NeuroSymbolic vs GPT-4o
# =============================================================================

def figure6_handwritten_comparison():
    """Compare NeuroSymbolic vs GPT-4o on handwritten puzzles."""
    fig, ax = plt.subplots(figsize=(4.5, 3))

    puzzles = ['KenKen\n3×3', 'KenKen\n4×4', 'Sudoku\n4×4', 'Sudoku\n9×9',
               'HexaSudoku\n16×16 (Hex)', 'HexaSudoku\n16×16 (Num)']

    ns_acc = [100, 94, 100, 98, 91, 72]
    gpt_acc = [20, 0, 63, 9, 0, 0]

    x = np.arange(len(puzzles))
    width = 0.35

    ax.bar(x - width/2, ns_acc, width, label='Neuro-Symbolic (V2)',
           color=COLORS['neurosymbolic'], edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, gpt_acc, width, label='GPT-4o',
           color=COLORS['gpt4o'], edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (ns, gpt) in enumerate(zip(ns_acc, gpt_acc)):
        ax.annotate(f'{ns}', xy=(x[i] - width/2, ns + 2),
                   ha='center', va='bottom', fontsize=6, fontweight='bold')
        ax.annotate(f'{gpt}', xy=(x[i] + width/2, gpt + 2),
                   ha='center', va='bottom', fontsize=6)

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(puzzles, fontsize=6)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig6_handwritten_comparison.pdf')
    plt.savefig(SCRIPT_DIR / 'fig6_handwritten_comparison.png')
    plt.close()
    print("Generated: fig6_handwritten_comparison.pdf")

# =============================================================================
# Figure 7: Pipeline Architecture Diagram
# =============================================================================

def figure7_architecture():
    """Create pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(3.5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Box style
    box_style = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5)
    neural_style = dict(boxstyle='round,pad=0.3', facecolor=COLORS['neurosymbolic'],
                       edgecolor='black', linewidth=1.5, alpha=0.3)
    cv_style = dict(boxstyle='round,pad=0.3', facecolor='lightgray',
                   edgecolor='black', linewidth=1.5, alpha=0.5)
    symbolic_style = dict(boxstyle='round,pad=0.3', facecolor=COLORS['corrected'],
                         edgecolor='black', linewidth=1.5, alpha=0.3)

    # Components (y positions from top to bottom)
    components = [
        (5, 13, 'Input Image\n(Puzzle Board)', box_style),
        (5, 11, 'Size Detection CNN\n(7 classes: 3-9)', neural_style),
        (5, 9, 'Cage Detection\n(OpenCV Morphology)', cv_style),
        (5, 7, 'Character Recognition\nCNN (14 classes)', neural_style),
        (5, 4.5, 'Z3 SMT Solver\n+ Error Correction', symbolic_style),
        (5, 2, 'Solution', box_style),
    ]

    for x, y, text, style in components:
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
               bbox=style, fontweight='bold' if 'Input' in text or 'Solution' in text else 'normal')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', linewidth=1.5)
    for i in range(len(components) - 1):
        y1 = components[i][1] - 0.5
        y2 = components[i+1][1] + 0.5
        ax.annotate('', xy=(5, y2), xytext=(5, y1),
                   arrowprops=arrow_style)

    # Legend for component types
    legend_y = 0.5
    ax.add_patch(plt.Rectangle((0.5, legend_y), 0.4, 0.3,
                               facecolor=COLORS['neurosymbolic'], alpha=0.3, edgecolor='black'))
    ax.text(1.1, legend_y + 0.15, 'Neural', fontsize=7, va='center')

    ax.add_patch(plt.Rectangle((3, legend_y), 0.4, 0.3,
                               facecolor='lightgray', alpha=0.5, edgecolor='black'))
    ax.text(3.6, legend_y + 0.15, 'Classical CV', fontsize=7, va='center')

    ax.add_patch(plt.Rectangle((6, legend_y), 0.4, 0.3,
                               facecolor=COLORS['corrected'], alpha=0.3, edgecolor='black'))
    ax.text(6.6, legend_y + 0.15, 'Symbolic', fontsize=7, va='center')

    # Error correction feedback arrow
    ax.annotate('', xy=(7.5, 7), xytext=(7.5, 4.5),
               arrowprops=dict(arrowstyle='->', color='darkred', linewidth=1,
                              connectionstyle='arc3,rad=0.3'))
    ax.text(8.5, 5.75, 'Error\nFeedback', fontsize=6, color='darkred', ha='center')

    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'fig7_architecture.pdf')
    plt.savefig(SCRIPT_DIR / 'fig7_architecture.png')
    plt.close()
    print("Generated: fig7_architecture.pdf")

# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating IJCAI Paper Figures")
    print("=" * 60)
    print()

    setup_style()

    print("Generating figures...")
    figure1_kenken_vlm_comparison()
    figure2_puzzle_types()
    figure3_error_correction()
    figure4_v1_vs_v2()
    figure5a_kenken_correction_breakdown()
    figure5b_sudoku_correction_breakdown()
    figure5c_efficiency_comparison()
    figure5d_efficiency_all_puzzles()
    figure6_handwritten_comparison()
    figure7_architecture()

    print()
    print("=" * 60)
    print(f"All figures saved to: {SCRIPT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
