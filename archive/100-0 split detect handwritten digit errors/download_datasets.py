"""
Download and prepare handwritten character datasets for HexaSudoku.

100-0 Split Configuration:
- Training: ALL available samples (MNIST train+test, EMNIST train+test combined)
- Test: Same as training (board images use training data)
- This tests CNN performance when recognizing digits it has memorized

The key insight: If the CNN achieves near-perfect recognition on its training data,
we can measure the upper bound of the neuro-symbolic system's performance.

Datasets used:
- MNIST: Handwritten digits 0-9 (60,000 train + 10,000 test = 70,000 total)
  Source: http://yann.lecun.com/exdb/mnist/
  Citation: LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.

- EMNIST-Letters: Handwritten letters A-Z (uppercase)
  Source: https://www.nist.gov/itl/products-and-services/emnist-dataset
  Citation: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
            EMNIST: Extending MNIST to handwritten letters.
  We filter for letters A-G only (for hexadecimal values 10-16).

Class mapping for HexaSudoku (17 classes):
- Class 0: Empty cell
- Classes 1-9: Digits 1-9 (from MNIST)
- Classes 10-16: Letters A-G (from EMNIST-Letters)
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import os
import ssl
import random

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Set random seeds for reproducibility
RANDOM_SEED = 1000
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Output directory for prepared data
OUTPUT_DIR = "./data"

# 100-0 split: Use ALL available data for training
# No held-out test set - board images will use training data
MAX_SAMPLES_PER_CLASS = None  # Use all available


def download_mnist():
    """
    Download MNIST dataset.

    Returns:
        train_dataset, test_dataset
    """
    print("Downloading MNIST dataset...")
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root='./datasets',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./datasets',
        train=False,
        download=True,
        transform=transform
    )

    print(f"  MNIST train: {len(train_dataset)} samples")
    print(f"  MNIST test: {len(test_dataset)} samples")

    return train_dataset, test_dataset


def download_emnist_letters():
    """
    Download EMNIST-Letters dataset.

    Note: EMNIST images are stored transposed and need rotation correction.
    The 'letters' split has 26 classes (1-26 for A-Z).

    Returns:
        train_dataset, test_dataset
    """
    print("Downloading EMNIST-Letters dataset...")

    # EMNIST images need to be transposed for correct orientation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2))  # Fix EMNIST orientation
    ])

    train_dataset = datasets.EMNIST(
        root='./datasets',
        split='letters',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.EMNIST(
        root='./datasets',
        split='letters',
        train=False,
        download=True,
        transform=transform
    )

    print(f"  EMNIST-Letters train: {len(train_dataset)} samples")
    print(f"  EMNIST-Letters test: {len(test_dataset)} samples")

    return train_dataset, test_dataset


def extract_all_samples(dataset, source_class):
    """
    Extract ALL samples of a specific class from a dataset.

    Args:
        dataset: PyTorch dataset
        source_class: The class label to extract

    Returns:
        numpy array of shape (N, 28, 28) with values in [0, 1]
    """
    all_samples = []
    for img, label in dataset:
        if label == source_class:
            img_np = img.squeeze().numpy()
            all_samples.append(img_np)

    return np.array(all_samples)


def prepare_hexasudoku_data():
    """
    Prepare training data for HexaSudoku character recognition.

    100-0 split: Combines ALL MNIST train+test and EMNIST train+test data.
    No held-out test set - board images will use the same training data.

    Class mapping:
    - Class 0: Empty (will be generated separately during training)
    - Classes 1-9: Digits 1-9 from MNIST
    - Classes 10-16: Letters A-G from EMNIST (EMNIST classes 1-7)
    """
    # Download datasets
    mnist_train, mnist_test = download_mnist()
    emnist_train, emnist_test = download_emnist_letters()

    train_images = []
    train_labels = []

    print("\nExtracting ALL samples for HexaSudoku classes...")
    print("Combining train + test sets for each class")

    # Extract digits 1-9 from MNIST (CNN classes 1-9)
    # Combine both train and test sets
    for digit in range(1, 10):
        print(f"  Digit {digit} (class {digit})...", end=" ")

        # Get samples from training set
        train_samples = extract_all_samples(mnist_train, source_class=digit)
        # Get samples from test set
        test_samples = extract_all_samples(mnist_test, source_class=digit)

        # Combine both
        all_samples = np.concatenate([train_samples, test_samples], axis=0)

        train_images.append(all_samples)
        train_labels.append(np.full(len(all_samples), digit, dtype=np.int64))

        print(f"train: {len(train_samples)}, test: {len(test_samples)}, total: {len(all_samples)}")

    # Extract letters A-G from EMNIST (CNN classes 10-16)
    # EMNIST-Letters: class 1=A, 2=B, ..., 7=G
    for letter_idx, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G']):
        target_class = 10 + letter_idx  # 10, 11, 12, 13, 14, 15, 16
        emnist_class = letter_idx + 1    # 1, 2, 3, 4, 5, 6, 7

        print(f"  Letter {letter} (class {target_class})...", end=" ")

        # Get samples from training set
        train_samples = extract_all_samples(emnist_train, source_class=emnist_class)
        # Get samples from test set
        test_samples = extract_all_samples(emnist_test, source_class=emnist_class)

        # Combine both
        all_samples = np.concatenate([train_samples, test_samples], axis=0)

        train_images.append(all_samples)
        train_labels.append(np.full(len(all_samples), target_class, dtype=np.int64))

        print(f"train: {len(train_samples)}, test: {len(test_samples)}, total: {len(all_samples)}")

    # Concatenate all classes
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    return train_images, train_labels


def save_data(train_images, train_labels):
    """Save prepared data to numpy files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save training data
    np.save(f"{OUTPUT_DIR}/train_images.npy", train_images)
    np.save(f"{OUTPUT_DIR}/train_labels.npy", train_labels)

    # For 100-0 split, "test" data is the SAME as training data
    # This is the key difference - board images will use training data
    np.save(f"{OUTPUT_DIR}/test_images.npy", train_images)
    np.save(f"{OUTPUT_DIR}/test_labels.npy", train_labels)

    print(f"\nData saved to {OUTPUT_DIR}/")
    print(f"  train_images.npy: {train_images.shape}")
    print(f"  train_labels.npy: {train_labels.shape}")
    print(f"  test_images.npy: {train_images.shape} (SAME as training)")
    print(f"  test_labels.npy: {train_labels.shape} (SAME as training)")


def print_dataset_stats(train_labels):
    """Print class distribution statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics (100-0 Split)")
    print("=" * 60)

    print("\nTraining set class distribution (ALL available data):")
    for cls in sorted(np.unique(train_labels)):
        count = np.sum(train_labels == cls)
        if cls == 0:
            name = "Empty"
        elif cls <= 9:
            name = str(cls)
        else:
            name = chr(ord('A') + cls - 10)
        print(f"  Class {cls:2d} ({name}): {count:5d} samples")

    print(f"\n  Total training samples: {len(train_labels)}")
    print(f"\n  Test samples: {len(train_labels)} (identical to training)")


def main():
    """Main function to download and prepare all data."""
    print("=" * 60)
    print("Handwritten HexaSudoku Dataset Preparation")
    print("100-0 Train/Test Split (ALL data for training)")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Training: ALL available samples")
    print(f"  - Test: SAME as training (for board image generation)")
    print("\nDatasets:")
    print("  - MNIST: Digits 1-9 (classes 1-9)")
    print("  - EMNIST-Letters: Letters A-G (classes 10-16)")
    print("\nPurpose:")
    print("  Test upper bound of system performance when CNN")
    print("  recognizes digits it has memorized (training data)")
    print("=" * 60)

    # Prepare data
    train_images, train_labels = prepare_hexasudoku_data()

    # Print statistics
    print_dataset_stats(train_labels)

    # Save to disk
    save_data(train_images, train_labels)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print("\nNote: Class 0 (empty cells) will be generated during training.")
    print("Run train_cnn.py next to train the character recognition model.")


if __name__ == '__main__':
    main()
