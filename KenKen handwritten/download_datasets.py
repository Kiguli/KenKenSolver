# -*- coding: utf-8 -*-
"""
Download MNIST dataset and create 90-10 train/test split.
Creates 5,400 training samples and 600 test samples per digit class (0-9).
"""

import os
import numpy as np
import random
import ssl
import urllib.request

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context

from torchvision import datasets, transforms

def download_and_split_mnist(output_dir='./handwritten_data', train_per_class=5400, test_per_class=600, seed=42):
    """
    Download MNIST and create stratified 90-10 split.

    Args:
        output_dir: Directory to save the split data
        train_per_class: Number of training samples per class (default: 5400)
        test_per_class: Number of test samples per class (default: 600)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading MNIST dataset...")

    # Download both train and test sets
    mnist_train = datasets.MNIST(root='./mnist_raw', train=True, download=True)
    mnist_test = datasets.MNIST(root='./mnist_raw', train=False, download=True)

    # Combine all data
    all_images = np.concatenate([
        mnist_train.data.numpy(),
        mnist_test.data.numpy()
    ])
    all_labels = np.concatenate([
        mnist_train.targets.numpy(),
        mnist_test.targets.numpy()
    ])

    print(f"Total MNIST samples: {len(all_images)}")

    # Group by class
    class_indices = {i: [] for i in range(10)}
    for idx, label in enumerate(all_labels):
        class_indices[label].append(idx)

    # Print class distribution
    print("\nClass distribution in full MNIST:")
    for digit in range(10):
        print(f"  Digit {digit}: {len(class_indices[digit])} samples")

    # Shuffle and split each class
    train_indices = []
    test_indices = []

    for digit in range(10):
        indices = class_indices[digit].copy()
        random.shuffle(indices)

        if len(indices) < train_per_class + test_per_class:
            print(f"Warning: Digit {digit} has only {len(indices)} samples, "
                  f"need {train_per_class + test_per_class}")
            # Adjust proportionally
            total_needed = train_per_class + test_per_class
            train_count = int(len(indices) * train_per_class / total_needed)
            test_count = len(indices) - train_count
        else:
            train_count = train_per_class
            test_count = test_per_class

        train_indices.extend(indices[:train_count])
        test_indices.extend(indices[train_count:train_count + test_count])

    # Create final arrays
    train_images = all_images[train_indices]
    train_labels = all_labels[train_indices]
    test_images = all_images[test_indices]
    test_labels = all_labels[test_indices]

    print(f"\n90-10 Split created:")
    print(f"  Training: {len(train_images)} samples ({len(train_images)//10} per class)")
    print(f"  Test: {len(test_images)} samples ({len(test_images)//10} per class)")

    # Verify class balance
    print("\nTraining set class distribution:")
    for digit in range(10):
        count = np.sum(train_labels == digit)
        print(f"  Digit {digit}: {count} samples")

    print("\nTest set class distribution:")
    for digit in range(10):
        count = np.sum(test_labels == digit)
        print(f"  Digit {digit}: {count} samples")

    # Save to numpy files
    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    print(f"\nData saved to {output_dir}/")
    print(f"  train_images.npy: {train_images.shape}")
    print(f"  train_labels.npy: {train_labels.shape}")
    print(f"  test_images.npy: {test_images.shape}")
    print(f"  test_labels.npy: {test_labels.shape}")

    # Clean up raw download
    import shutil
    if os.path.exists('./mnist_raw'):
        shutil.rmtree('./mnist_raw')
        print("\nCleaned up raw MNIST download.")

    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    download_and_split_mnist()
