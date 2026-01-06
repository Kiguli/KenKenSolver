"""
Train unified 14-class CNN for handwritten KenKen.

Matches the original KenKen model architecture exactly:
- Classes 0-9: Digits (from MNIST, inverted to match TMNIST convention)
- Class 10: add (+)
- Class 11: div (÷)
- Class 12: mul (×)
- Class 13: sub (−)

Key difference from original: digits are from real handwritten MNIST
instead of rendered TMNIST fonts. Operators remain from TMNIST/PNG files.

This creates a fair comparison where only the digit source differs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image
import os
from scipy import ndimage


# Constants - match original KenKen training
IMG_SIZE = 28
NUM_CLASSES = 14  # 0-9 digits + 4 operators
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SAMPLES_PER_OPERATOR = 5000  # Balance with digit classes


class CNN_v2(nn.Module):
    """CNN for character recognition (same architecture as original KenKen)."""

    def __init__(self, output_dim):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)  # 64 * 7 * 7 = 3136
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)             # Flatten to 3136
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_mnist_digits():
    """
    Load MNIST digits and INVERT to match TMNIST convention.

    MNIST: ink=HIGH (white on black), values near 1.0 for ink
    TMNIST: ink=LOW (black on white), values near 0.0 for ink

    After inversion: 0 = ink (dark), 1 = background (light)
    This matches how characters appear on the board images.
    """
    data_dir = "./handwritten_data"

    if not os.path.exists(f"{data_dir}/train_images.npy"):
        raise FileNotFoundError(
            f"Training data not found in {data_dir}/. "
            "Run download_datasets.py first."
        )

    images = np.load(f"{data_dir}/train_images.npy")
    labels = np.load(f"{data_dir}/train_labels.npy")

    # CRITICAL: Invert MNIST images to match TMNIST convention
    # MNIST has ink=HIGH (near 1.0), we need ink=LOW (near 0.0)
    images = 1.0 - images

    print(f"Loaded {len(images)} MNIST digit samples")
    print(f"  Classes: {sorted(np.unique(labels))}")
    print(f"  Image range after inversion: [{images.min():.3f}, {images.max():.3f}]")

    return images, labels


def load_operator_image(filepath):
    """
    Load operator image from PNG and convert to 28x28 grayscale.

    Operators are stored as RGBA PNGs - composite against white background
    to get grayscale with ink=LOW convention (matching TMNIST).
    """
    if not os.path.exists(filepath):
        print(f"Warning: Operator image not found: {filepath}")
        return None

    img = Image.open(filepath)

    # Handle RGBA by compositing against white
    if img.mode == 'RGBA':
        rgba = np.array(img)
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
        # Convert to grayscale
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        # Alpha compositing against white background (255)
        alpha = a / 255.0
        composited = gray * alpha + 255 * (1 - alpha)
        img_array = composited.astype(np.float32) / 255.0
    else:
        img = img.convert('L')
        img_array = np.array(img).astype(np.float32) / 255.0

    # Resize to 28x28 if needed
    if img_array.shape != (IMG_SIZE, IMG_SIZE):
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil).astype(np.float32) / 255.0

    return img_array


def augment_image(img, max_rotation=5, max_shift=2, max_scale=0.1):
    """
    Apply random augmentation to create variety.

    Args:
        img: 28x28 numpy array with ink=LOW convention
        max_rotation: Maximum rotation in degrees
        max_shift: Maximum translation in pixels
        max_scale: Maximum scale variation (0.1 = +/-10%)

    Returns:
        Augmented 28x28 numpy array
    """
    # Random rotation
    angle = np.random.uniform(-max_rotation, max_rotation)
    img = ndimage.rotate(img, angle, reshape=False, mode='constant', cval=1.0)  # cval=1.0 for white background

    # Random shift
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    img = ndimage.shift(img, (shift_y, shift_x), mode='constant', cval=1.0)

    # Random scale (zoom)
    scale = np.random.uniform(1 - max_scale, 1 + max_scale)
    if scale != 1.0:
        img = ndimage.zoom(img, scale, mode='constant', cval=1.0)
        # Crop or pad to 28x28
        if img.shape[0] > IMG_SIZE:
            start = (img.shape[0] - IMG_SIZE) // 2
            img = img[start:start+IMG_SIZE, start:start+IMG_SIZE]
        elif img.shape[0] < IMG_SIZE:
            pad = (IMG_SIZE - img.shape[0]) // 2
            img = np.pad(img, ((pad, IMG_SIZE-img.shape[0]-pad),
                               (pad, IMG_SIZE-img.shape[1]-pad)),
                         mode='constant', constant_values=1.0)

    # Ensure valid range and size
    img = np.clip(img, 0, 1)
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img_pil) / 255.0

    return img.astype(np.float32)


def generate_operator_samples(base_image, num_samples):
    """
    Generate augmented operator samples from a single base image.

    Creates variety through rotation, shift, and scale augmentation.
    """
    samples = []

    # First sample is the original
    samples.append(base_image.copy())

    # Rest are augmented
    for _ in range(num_samples - 1):
        aug_img = augment_image(base_image)
        samples.append(aug_img)

    return np.array(samples, dtype=np.float32)


def load_operators():
    """
    Load operator images and generate augmented samples.

    Operators:
    - Class 10: add (+)
    - Class 11: div (÷)
    - Class 12: mul (×)
    - Class 13: sub (−)

    Returns:
        images: numpy array of shape (N, 28, 28)
        labels: numpy array of shape (N,)
    """
    operator_dir = "../KenKen/symbols/operators"

    operator_map = {
        'add': 10,
        'div': 11,
        'mul': 12,
        'sub': 13
    }

    all_images = []
    all_labels = []

    print(f"\nGenerating operator samples ({SAMPLES_PER_OPERATOR} per class)...")

    for op_name, class_id in operator_map.items():
        filepath = os.path.join(operator_dir, f"{op_name}.png")
        base_image = load_operator_image(filepath)

        if base_image is None:
            print(f"  Warning: Could not load {op_name}, skipping")
            continue

        samples = generate_operator_samples(base_image, SAMPLES_PER_OPERATOR)
        labels = np.full(len(samples), class_id, dtype=np.int64)

        all_images.append(samples)
        all_labels.append(labels)

        print(f"  Class {class_id} ({op_name}): {len(samples)} samples")

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return images, labels


def train_model(images, labels):
    """
    Train CNN on the combined digit + operator dataset.

    Args:
        images: numpy array of shape (N, 28, 28) with ink=LOW convention
        labels: numpy array of shape (N,) with labels 0-13

    Returns:
        Trained model
    """
    # Shuffle and split into train/validation
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    train_size = int(0.85 * n_samples)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train = images[train_idx]
    y_train = labels[train_idx]
    X_val = images[val_idx]
    y_val = labels[val_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = CNN_v2(output_dim=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    print(f"\nTraining Configuration:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "-" * 60)

    # Training loop
    best_val_acc = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        train_acc = train_correct / len(X_train)

        # Validation phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_acc = val_correct / len(X_val)

        # Update scheduler
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch+1:2d}/{EPOCHS}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}"
              f"{' *' if val_acc == best_val_acc else ''}")

    print("-" * 60)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_state)

    return model


def evaluate_per_class(model, images, labels):
    """Evaluate model accuracy per class."""
    model.eval()

    # Convert to tensors
    X = torch.tensor(images, dtype=torch.float32).unsqueeze(1)

    # Get predictions
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(1).numpy()

    class_names = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'add', 11: 'div', 12: 'mul', 13: 'sub'
    }

    print("\nPer-class accuracy:")
    for cls in sorted(np.unique(labels)):
        mask = labels == cls
        cls_correct = np.sum(predictions[mask] == labels[mask])
        cls_total = np.sum(mask)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0

        name = class_names.get(cls, str(cls))
        print(f"  Class {cls:2d} ({name:>4s}): {cls_acc:.4f} ({cls_correct}/{cls_total})")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Unified Handwritten KenKen CNN Training")
    print("=" * 60)
    print("\nThis trains a 14-class model matching the original KenKen:")
    print("  - Classes 0-9: MNIST digits (inverted to ink=LOW)")
    print("  - Classes 10-13: Operators (+, ÷, ×, −)")

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load MNIST digits (inverted)
    print("\n1. Loading MNIST digits...")
    digit_images, digit_labels = load_mnist_digits()

    # Load operators
    print("\n2. Loading operator samples...")
    op_images, op_labels = load_operators()

    # Combine datasets
    print("\n3. Combining datasets...")
    images = np.concatenate([digit_images, op_images], axis=0)
    labels = np.concatenate([digit_labels, op_labels], axis=0)
    print(f"   Total samples: {len(images)}")
    print(f"   Classes present: {sorted(np.unique(labels))}")

    # Shuffle
    shuffle_idx = np.random.permutation(len(images))
    images = images[shuffle_idx]
    labels = labels[shuffle_idx]

    # Train model
    print("\n4. Training CNN model...")
    model = train_model(images, labels)

    # Evaluate per-class accuracy
    print("\n5. Evaluating per-class accuracy...")
    eval_idx = np.random.choice(len(images), min(10000, len(images)), replace=False)
    evaluate_per_class(model, images[eval_idx], labels[eval_idx])

    # Save model
    print("\n6. Saving model...")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/unified_kenken_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"   Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run generate_images.py to create board images (900x900)")
    print("  2. Run evaluate.py to test the unified pipeline")


if __name__ == '__main__':
    main()
