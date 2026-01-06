"""
Compare neuro-symbolic solver accuracy between TMNIST and MNIST handwritten digits.

This script generates a fair comparison report showing:
1. Base accuracy (no error correction) for both pipelines
2. Corrected accuracy (with error correction) for both pipelines
3. The delta showing the impact of using handwritten digits

Fair comparison methodology:
- Same model architecture (14 classes: 0-9 + 4 operators)
- Same board layout (900x900 pixels)
- Same evaluation pipeline (cage detection, character segmentation, Z3 solving)
- Only digit source differs (TMNIST rendered fonts vs MNIST handwritten)
"""

import pandas as pd
import os


def load_original_results():
    """Load results from original KenKen (TMNIST) pipeline."""
    # Try to load from the original KenKen folder
    results_path = "../KenKen/results/optimized_summary.csv"

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        return df
    else:
        # Fallback to known values from the README
        return pd.DataFrame({
            'size': [3, 4, 5, 6, 7, 9],
            'accuracy': [100.0, 100.0, 100.0, 100.0, 95.0, 62.0],
            'avg_time_ms': [143.6, 167.6, 204.4, 293.8, 510.9, 500.0]
        })


def load_handwritten_results():
    """Load results from handwritten (MNIST) pipeline."""
    base_path = "./results/summary.csv"
    corrected_path = "./results/error_correction_summary.csv"

    results = {}

    if os.path.exists(base_path):
        base_df = pd.read_csv(base_path)
        results['base'] = base_df
    else:
        results['base'] = None

    if os.path.exists(corrected_path):
        corrected_df = pd.read_csv(corrected_path)
        results['corrected'] = corrected_df
    else:
        results['corrected'] = None

    return results


def generate_comparison_report():
    """Generate side-by-side comparison report."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tmnist = load_original_results()
    handwritten = load_handwritten_results()

    print("=" * 80)
    print("KenKen Neuro-Symbolic Solver: TMNIST vs MNIST Handwritten Comparison")
    print("=" * 80)
    print()

    print("Methodology:")
    print("  - Same model architecture: CNN_v2 with 14 classes (0-9 + 4 operators)")
    print("  - Same board layout: 900x900 pixels")
    print("  - Same evaluation pipeline: Grid CNN → OpenCV → Character CNN → Z3")
    print("  - Only difference: Digit source (TMNIST rendered fonts vs MNIST handwritten)")
    print()

    # Build comparison table
    print("-" * 80)
    print(f"{'Size':<8} {'TMNIST Base':<15} {'MNIST Base':<15} {'Δ Base':<12} {'MNIST Corr.':<15} {'Δ Corr.':<12}")
    print("-" * 80)

    sizes = [3, 4, 5, 6, 7, 9]

    comparison_data = []

    for size in sizes:
        # Get TMNIST baseline
        tmnist_row = tmnist[tmnist['size'] == size]
        if len(tmnist_row) > 0:
            tmnist_acc = tmnist_row['accuracy'].values[0]
        else:
            tmnist_acc = None

        # Get handwritten base
        if handwritten['base'] is not None:
            hw_base_row = handwritten['base'][handwritten['base']['size'] == size]
            if len(hw_base_row) > 0:
                hw_base_acc = hw_base_row['accuracy'].values[0]
            else:
                hw_base_acc = None
        else:
            hw_base_acc = None

        # Get handwritten corrected
        if handwritten['corrected'] is not None:
            hw_corr_row = handwritten['corrected'][handwritten['corrected']['size'] == size]
            if len(hw_corr_row) > 0:
                hw_corr_acc = hw_corr_row['corrected_accuracy'].values[0]
            else:
                hw_corr_acc = None
        else:
            hw_corr_acc = None

        # Calculate deltas
        delta_base = None
        delta_corr = None
        if tmnist_acc is not None and hw_base_acc is not None:
            delta_base = hw_base_acc - tmnist_acc
        if tmnist_acc is not None and hw_corr_acc is not None:
            delta_corr = hw_corr_acc - tmnist_acc

        # Format strings
        tmnist_str = f"{tmnist_acc:.1f}%" if tmnist_acc is not None else "N/A"
        hw_base_str = f"{hw_base_acc:.1f}%" if hw_base_acc is not None else "N/A"
        delta_base_str = f"{delta_base:+.1f}%" if delta_base is not None else "N/A"
        hw_corr_str = f"{hw_corr_acc:.1f}%" if hw_corr_acc is not None else "N/A"
        delta_corr_str = f"{delta_corr:+.1f}%" if delta_corr is not None else "N/A"

        print(f"{size}x{size:<5} {tmnist_str:<15} {hw_base_str:<15} {delta_base_str:<12} {hw_corr_str:<15} {delta_corr_str:<12}")

        comparison_data.append({
            'size': size,
            'tmnist_accuracy': tmnist_acc,
            'mnist_base_accuracy': hw_base_acc,
            'delta_base': delta_base,
            'mnist_corrected_accuracy': hw_corr_acc,
            'delta_corrected': delta_corr
        })

    print("-" * 80)
    print()

    # Summary statistics
    if handwritten['base'] is not None:
        valid_deltas = [d['delta_base'] for d in comparison_data if d['delta_base'] is not None]
        if valid_deltas:
            avg_delta = sum(valid_deltas) / len(valid_deltas)
            print(f"Average accuracy drop (handwritten vs printed): {avg_delta:+.1f}%")

    if handwritten['corrected'] is not None:
        valid_corr_deltas = [d['delta_corrected'] for d in comparison_data if d['delta_corrected'] is not None]
        if valid_corr_deltas:
            avg_corr_delta = sum(valid_corr_deltas) / len(valid_corr_deltas)
            print(f"Average accuracy drop with error correction: {avg_corr_delta:+.1f}%")

    print()
    print("Key Findings:")
    print("  - The accuracy drop measures the impact of using real handwritten digits")
    print("  - Error correction can recover some failures due to OCR errors")
    print("  - Remaining delta is due to inherent handwriting variability")

    # Save comparison report
    os.makedirs('./results', exist_ok=True)
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('./results/comparison_report.csv', index=False)
    print()
    print(f"Comparison report saved to ./results/comparison_report.csv")
    print()


def main():
    generate_comparison_report()


if __name__ == '__main__':
    main()
