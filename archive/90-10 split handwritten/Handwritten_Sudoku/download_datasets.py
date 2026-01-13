"""
Download and prepare handwritten digit dataset for Sudoku.

Dataset used:
- MNIST: Handwritten digits 0-9 (60,000 train, 10,000 test)
  Source: http://yann.lecun.com/exdb/mnist/
  Citation: LeCun, Y., Cortes, C., & Burges, C. (1998). The MNIST database of handwritten digits.

Data split strategy:
- Training data: Used to train the CNN
- Test data: Used ONLY for board image generation (never seen by CNN during training)

Class mapping for Sudoku (10 classes):
- Class 0: Empty cell (generated separately during training)
- Classes 1-9: Digits 1-9 (from MNIST)
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import os
import ssl

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context


# Output directory for prepared data
OUTPUT_DIR = "./handwritten_data"

# Samples per class limits (to balance dataset)
# 90/10 split: 5400 train, 600 test per class
MAX_TRAIN_SAMPLES_PER_CLASS = 5400
MAX_TEST_SAMPLES_PER_CLASS = 600


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


def extract_class_samples(dataset, source_class, max_samples=None):
    """
    Extract all samples of a specific class from a dataset.

    Args:
        dataset: PyTorch dataset
        source_class: The class label to extract
        max_samples: Maximum samples to return (None = all)

    Returns:
        numpy array of shape (N, 28, 28) with values in [0, 1]
    """
    samples = []
    for img, label in dataset:
        if label == source_class:
            # Convert tensor to numpy, squeeze channel dimension
            img_np = img.squeeze().numpy()
            samples.append(img_np)

            if max_samples and len(samples) >= max_samples:
                break

    return np.array(samples)


def prepare_sudoku_data():
    """
    Prepare training and test data for Sudoku digit recognition.

    Class mapping:
    - Class 0: Empty (will be generated separately during training)
    - Classes 1-9: Digits 1-9 from MNIST
    """
    # Download dataset
    mnist_train, mnist_test = download_mnist()

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    print("\nExtracting samples for Sudoku classes...")

    # Extract digits 1-9 from MNIST (CNN classes 1-9)
    for digit in range(1, 10):
        print(f"  Digit {digit} (class {digit})...", end=" ")

        # Training samples
        samples = extract_class_samples(
            mnist_train,
            source_class=digit,
            max_samples=MAX_TRAIN_SAMPLES_PER_CLASS
        )
        train_images.append(samples)
        train_labels.append(np.full(len(samples), digit, dtype=np.int64))

        # Test samples
        test_samples = extract_class_samples(
            mnist_test,
            source_class=digit,
            max_samples=MAX_TEST_SAMPLES_PER_CLASS
        )
        test_images.append(test_samples)
        test_labels.append(np.full(len(test_samples), digit, dtype=np.int64))

        print(f"train: {len(samples)}, test: {len(test_samples)}")

    # Concatenate all classes
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_images = np.concatenate(test_images, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    return train_images, train_labels, test_images, test_labels


def save_data(train_images, train_labels, test_images, test_labels):
    """Save prepared data to numpy files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/train_images.npy", train_images)
    np.save(f"{OUTPUT_DIR}/train_labels.npy", train_labels)
    np.save(f"{OUTPUT_DIR}/test_images.npy", test_images)
    np.save(f"{OUTPUT_DIR}/test_labels.npy", test_labels)

    print(f"\nData saved to {OUTPUT_DIR}/")
    print(f"  train_images.npy: {train_images.shape}")
    print(f"  train_labels.npy: {train_labels.shape}")
    print(f"  test_images.npy: {test_images.shape}")
    print(f"  test_labels.npy: {test_labels.shape}")


def print_dataset_stats(train_labels, test_labels):
    """Print class distribution statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    print("\nTraining set class distribution:")
    for cls in sorted(np.unique(train_labels)):
        count = np.sum(train_labels == cls)
        print(f"  Class {cls} ({cls}): {count:5d} samples")

    print(f"\n  Total training samples: {len(train_labels)}")

    print("\nTest set class distribution:")
    for cls in sorted(np.unique(test_labels)):
        count = np.sum(test_labels == cls)
        print(f"  Class {cls} ({cls}): {count:5d} samples")

    print(f"\n  Total test samples: {len(test_labels)}")


def main():
    """Main function to download and prepare all data."""
    print("=" * 60)
    print("Handwritten Sudoku Dataset Preparation")
    print("=" * 60)
    print("\nDataset:")
    print("  - MNIST: Digits 1-9 (classes 1-9)")
    print("\nData split strategy:")
    print("  - Training data: For CNN training")
    print("  - Test data: For board image generation (unseen by CNN)")
    print("=" * 60)

    # Prepare data
    train_images, train_labels, test_images, test_labels = prepare_sudoku_data()

    # Print statistics
    print_dataset_stats(train_labels, test_labels)

    # Save to disk
    save_data(train_images, train_labels, test_images, test_labels)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print("\nNote: Class 0 (empty cells) will be generated during training.")
    print("Run train_cnn.py next to train the digit recognition model.")


if __name__ == '__main__':
    main()
