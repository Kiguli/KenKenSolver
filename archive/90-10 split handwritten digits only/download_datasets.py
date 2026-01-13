"""
Download and prepare MNIST-only dataset for HexaSudoku (Digits Only).

This version uses ONLY MNIST digits (0-9) - no EMNIST letters.
Values 10-16 will be rendered as two-digit numbers (e.g., "16").

90-10 Train/Test Split:
- Training: 5400 samples per class (digits 0-9)
- Testing: 600 samples per class

Note: Digit 0 is included (needed for values 10, 20, etc.)
"""

import os
import numpy as np
from torchvision import datasets, transforms

# Configuration
RANDOM_SEED = 9010  # Same as 90-10 split for consistency
MAX_TRAIN_PER_CLASS = 5400
MAX_TEST_PER_CLASS = 600
OUTPUT_DIR = "./data"

np.random.seed(RANDOM_SEED)


def download_mnist():
    """Download MNIST dataset."""
    print("Downloading MNIST dataset...")

    transform = transforms.Compose([transforms.ToTensor()])

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


def extract_mnist_samples(train_dataset, test_dataset):
    """
    Extract MNIST samples for digits 0-9.

    Returns:
        dict: {class_label: {'train': [images], 'test': [images]}}
    """
    samples = {i: {'train': [], 'test': []} for i in range(10)}

    # Extract training samples
    print("\nExtracting training samples...")
    for img, label in train_dataset:
        img_np = img.squeeze().numpy()
        samples[label]['train'].append(img_np)

    # Extract test samples
    print("Extracting test samples...")
    for img, label in test_dataset:
        img_np = img.squeeze().numpy()
        samples[label]['test'].append(img_np)

    # Print statistics
    print("\nSamples per class:")
    for cls in range(10):
        print(f"  Digit {cls}: train={len(samples[cls]['train'])}, "
              f"test={len(samples[cls]['test'])}")

    return samples


def apply_90_10_split(samples):
    """
    Apply 90-10 split to get training and test sets.

    Args:
        samples: {class: {'train': [...], 'test': [...]}}

    Returns:
        train_images, train_labels, test_images, test_labels
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    print(f"\nApplying 90-10 split (max {MAX_TRAIN_PER_CLASS} train, "
          f"{MAX_TEST_PER_CLASS} test per class)...")

    for cls in range(10):
        # Combine original train and test for resampling
        all_samples = samples[cls]['train'] + samples[cls]['test']
        np.random.shuffle(all_samples)

        # Split 90-10
        n_train = min(MAX_TRAIN_PER_CLASS, int(len(all_samples) * 0.9))
        n_test = min(MAX_TEST_PER_CLASS, len(all_samples) - n_train)

        cls_train = all_samples[:n_train]
        cls_test = all_samples[n_train:n_train + n_test]

        train_images.extend(cls_train)
        train_labels.extend([cls] * len(cls_train))

        test_images.extend(cls_test)
        test_labels.extend([cls] * len(cls_test))

        print(f"  Digit {cls}: {len(cls_train)} train, {len(cls_test)} test")

    return (np.array(train_images), np.array(train_labels),
            np.array(test_images), np.array(test_labels))


def save_data(train_images, train_labels, test_images, test_labels):
    """Save processed data to numpy files."""
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


def main():
    print("=" * 60)
    print("MNIST-Only Dataset Preparation (Digits 0-9)")
    print("90-10 Train/Test Split for HexaSudoku")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Max training samples per class: {MAX_TRAIN_PER_CLASS}")
    print(f"  - Max test samples per class: {MAX_TEST_PER_CLASS}")
    print(f"  - Classes: Digits 0-9 (10 classes)")

    # Download MNIST
    train_dataset, test_dataset = download_mnist()

    # Extract samples
    samples = extract_mnist_samples(train_dataset, test_dataset)

    # Apply 90-10 split
    train_images, train_labels, test_images, test_labels = apply_90_10_split(samples)

    # Save data
    save_data(train_images, train_labels, test_images, test_labels)

    # Summary
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nTotal samples:")
    print(f"  Training: {len(train_images)} ({len(train_images)//10} per class)")
    print(f"  Testing: {len(test_images)} ({len(test_images)//10} per class)")
    print(f"\nClasses: 0-9 (digits only, no letters)")
    print("\nNote: Empty cells will be generated during training.")
    print("Run train_cnn.py next to train the digit recognition model.")


if __name__ == '__main__':
    main()
