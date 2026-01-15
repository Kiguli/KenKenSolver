"""
Train CNN for HexaSudoku character recognition.

Recognizes 17 classes:
- 0: empty cell
- 1-9: digits
- 10-16: letters A-G (values 10-16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import csv


# Constants
IMG_SIZE = 28
NUM_CLASSES = 17  # 0-9 digits + A-G letters (0=empty, 1-16 values)
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SAMPLES_PER_CLASS = 500  # Number of augmented samples per character


class CNN_v2(nn.Module):
    """CNN for character recognition (same architecture as KenKen/Sudoku)."""
    def __init__(self, output_dim):
        super(CNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_character(label):
    """Convert label (0-16) to display character."""
    if label == 0:
        return '0'  # We'll use '0' to represent empty, but detect as class 0
    if label <= 9:
        return str(label)
    return chr(ord('A') + label - 10)  # 10→A, 11→B, ..., 16→G


def render_character(char, font, size=IMG_SIZE, offset=(0, 0)):
    """
    Render a character to a grayscale image.

    Args:
        char: Character to render
        font: PIL ImageFont
        size: Output image size
        offset: (x, y) offset for augmentation

    Returns:
        28x28 numpy array (0.0 = black/ink, 1.0 = white/background)
    """
    # Create larger canvas for rendering
    canvas_size = size * 3
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text on canvas with offset
    x = (canvas_size - text_width) // 2 - bbox[0] + offset[0]
    y = (canvas_size - text_height) // 2 - bbox[1] + offset[1]

    draw.text((x, y), char, fill=0, font=font)

    # Crop to center and resize
    margin = (canvas_size - size) // 2
    img = img.crop((margin, margin, margin + size, margin + size))

    return np.array(img) / 255.0


def generate_training_data():
    """
    Generate training data for all 16 character classes.
    Uses multiple font sizes and slight position variations for augmentation.

    Returns:
        (images, labels) - numpy arrays
    """
    images = []
    labels = []

    # Try to load fonts
    fonts = []
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]

    font_sizes = [18, 20, 22, 24]  # Various sizes for 28x28 target

    for path in font_paths:
        if os.path.exists(path):
            for size in font_sizes:
                try:
                    fonts.append(ImageFont.truetype(path, size))
                except:
                    pass

    if not fonts:
        print("Warning: No TrueType fonts found, using default font")
        fonts = [ImageFont.load_default()]

    print(f"Using {len(fonts)} font variations")

    # Generate samples for each class
    for label in range(NUM_CLASSES):
        char = get_character(label)

        samples_generated = 0
        sample_idx = 0

        while samples_generated < SAMPLES_PER_CLASS:
            # Cycle through fonts
            font = fonts[sample_idx % len(fonts)]

            # Add position variation for augmentation
            offset_x = np.random.randint(-2, 3)
            offset_y = np.random.randint(-2, 3)

            try:
                img = render_character(char, font, offset=(offset_x, offset_y))

                # Invert so ink is high value (like MNIST convention)
                img = 1.0 - img

                images.append(img)
                labels.append(label)
                samples_generated += 1
            except Exception as e:
                pass

            sample_idx += 1

            # Safety check
            if sample_idx > SAMPLES_PER_CLASS * 10:
                break

        print(f"  Class {label} ({char}): {samples_generated} samples")

    return np.array(images), np.array(labels)


def save_training_data(images, labels, output_path):
    """Save training data to CSV in TMNIST format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header: names, labels, 1, 2, ..., 784
        header = ['names', 'labels'] + [str(i) for i in range(1, 785)]
        writer.writerow(header)

        # Data rows
        for i, (img, label) in enumerate(zip(images, labels)):
            char = get_character(label)
            # Flatten image and convert to 0-255
            pixels = (img.flatten() * 255).astype(int).tolist()
            row = [f'HexaSudoku-{char}', label] + pixels
            writer.writerow(row)

    print(f"Saved {len(images)} samples to {output_path}")


def train_model(images, labels):
    """
    Train CNN on the generated data.

    Returns:
        Trained model
    """
    # Split into train/val
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)

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

    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_acc = 0
    for epoch in range(EPOCHS):
        # Training
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

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        val_acc = val_correct / len(X_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    return model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("HexaSudoku CNN Training")
    print("=" * 60)

    # Generate training data
    print("\n1. Generating training data...")
    images, labels = generate_training_data()
    print(f"   Total samples: {len(images)}")

    # Save training data
    print("\n2. Saving training data...")
    csv_path = "./symbols/hex_characters.csv"
    save_training_data(images, labels, csv_path)

    # Train model
    print("\n3. Training CNN model...")
    model = train_model(images, labels)

    # Save model
    print("\n4. Saving model...")
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/hex_character_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"   Model saved to {model_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
