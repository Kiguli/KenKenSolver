# -*- coding: utf-8 -*-
"""
Advanced data augmentation for handwritten KenKen character recognition.

Key additions over basic augmentation:
1. Elastic deformation (handwriting variation)
2. Erosion/dilation (stroke width variation)
3. Higher rotation range (-20 to +20 degrees)
4. Aspect ratio distortion
5. Variable scale (0.5-1.2)
"""

import numpy as np
from PIL import Image, ImageFilter
import random
from scipy.ndimage import map_coordinates, gaussian_filter


def elastic_transform(img_array, alpha=34, sigma=4):
    """
    Apply elastic deformation to simulate handwriting variation.

    Args:
        img_array: Input image (28x28, 0-1 range)
        alpha: Intensity of displacement (higher = more deformation)
        sigma: Smoothness of displacement field (higher = smoother)

    Returns:
        Deformed image
    """
    shape = img_array.shape

    # Generate random displacement fields
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Apply displacement
    indices = [
        np.clip(y + dy, 0, shape[0] - 1).astype(np.float32),
        np.clip(x + dx, 0, shape[1] - 1).astype(np.float32)
    ]

    return map_coordinates(img_array, indices, order=1, mode='constant', cval=1.0)


def rotate_image(img_array, angle):
    """
    Rotate image by given angle (degrees).

    Args:
        img_array: Input image (0-1 range, white background)
        angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    rotated = img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
    return np.array(rotated).astype(np.float32) / 255.0


def scale_and_center(img_array, scale_factor, target_size=28):
    """
    Scale image and center in target_size canvas.

    Args:
        img_array: Input image
        scale_factor: Scale factor (0.5-1.2)
        target_size: Output size

    Returns:
        Scaled and centered image
    """
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')

    new_size = max(8, int(target_size * scale_factor))
    new_size = min(new_size, target_size * 2)  # Cap at 2x

    scaled = img.resize((new_size, new_size), Image.Resampling.LANCZOS)

    # Center on canvas
    canvas = Image.new('L', (target_size, target_size), color=255)

    if new_size <= target_size:
        offset = (target_size - new_size) // 2
        canvas.paste(scaled, (offset, offset))
    else:
        # Crop center
        offset = (new_size - target_size) // 2
        cropped = scaled.crop((offset, offset, offset + target_size, offset + target_size))
        canvas = cropped

    return np.array(canvas).astype(np.float32) / 255.0


def distort_aspect_ratio(img_array, x_scale=1.0, y_scale=1.0):
    """
    Distort aspect ratio to simulate writing variation.

    Args:
        img_array: Input image
        x_scale: Horizontal scale factor
        y_scale: Vertical scale factor

    Returns:
        Distorted image
    """
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    h, w = img_array.shape

    new_w = max(8, int(w * x_scale))
    new_h = max(8, int(h * y_scale))

    stretched = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on original size canvas
    canvas = Image.new('L', (w, h), color=255)

    if new_w <= w and new_h <= h:
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas.paste(stretched, (offset_x, offset_y))
    else:
        # Crop to fit
        offset_x = max(0, (new_w - w) // 2)
        offset_y = max(0, (new_h - h) // 2)
        cropped = stretched.crop((offset_x, offset_y, offset_x + w, offset_y + h))
        canvas.paste(cropped, (0, 0))

    return np.array(canvas).astype(np.float32) / 255.0


def erode_dilate(img_array, operation='dilate', kernel_size=2):
    """
    Apply morphological operation to vary stroke width.

    Args:
        img_array: Input image (0-1, white bg, dark strokes)
        operation: 'erode' (thinner) or 'dilate' (thicker)
        kernel_size: Size of morphological kernel

    Returns:
        Transformed image
    """
    import cv2 as cv

    # Convert to uint8 with inverted colors (strokes = white for morphology)
    img_uint8 = (255 - img_array * 255).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == 'erode':
        result = cv.erode(img_uint8, kernel, iterations=1)
    else:  # dilate
        result = cv.dilate(img_uint8, kernel, iterations=1)

    # Convert back to original format
    return 1.0 - result.astype(np.float32) / 255.0


def translate_image(img_array, dx=0, dy=0):
    """
    Translate image by (dx, dy) pixels.

    Args:
        img_array: Input image
        dx: Horizontal offset
        dy: Vertical offset

    Returns:
        Translated image
    """
    result = np.ones_like(img_array)  # White background
    h, w = img_array.shape

    # Calculate source and destination regions
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(w, w - dx)
    src_y2 = min(h, h - dy)

    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = min(w, w + dx)
    dst_y2 = min(h, h + dy)

    result[dst_y1:dst_y2, dst_x1:dst_x2] = img_array[src_y1:src_y2, src_x1:src_x2]
    return result


def add_gaussian_noise(img_array, sigma=0.05):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, img_array.shape)
    noisy = img_array + noise
    return np.clip(noisy, 0, 1).astype(np.float32)


def add_salt_pepper_noise(img_array, amount=0.02):
    """Add salt and pepper noise."""
    noisy = img_array.copy()

    # Salt (white pixels)
    num_salt = int(amount * img_array.size * 0.5)
    salt_coords = [np.random.randint(0, i, num_salt) for i in img_array.shape]
    noisy[salt_coords[0], salt_coords[1]] = 1.0

    # Pepper (black pixels)
    num_pepper = int(amount * img_array.size * 0.5)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in img_array.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0.0

    return noisy.astype(np.float32)


def apply_blur(img_array, radius=0.5):
    """Apply Gaussian blur."""
    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(blurred).astype(np.float32) / 255.0


def augment_handwritten(img_array, level='medium'):
    """
    Apply comprehensive augmentation for handwritten digits.

    Args:
        img_array: Input image (28x28, 0-1 range, white bg)
        level: 'none', 'light', 'medium', 'heavy'

    Returns:
        Augmented image
    """
    if level == 'none':
        return img_array

    img = img_array.copy()

    # Define probabilities and ranges by level
    config = {
        'light': {
            'scale_range': (0.8, 1.0),
            'rotation_range': (-10, 10),
            'translate_range': (-2, 2),
            'elastic_prob': 0.2,
            'elastic_alpha': 20,
            'morph_prob': 0.1,
            'aspect_prob': 0.1,
            'blur_prob': 0.2,
            'noise_prob': 0.1,
        },
        'medium': {
            'scale_range': (0.6, 1.1),
            'rotation_range': (-15, 15),
            'translate_range': (-3, 3),
            'elastic_prob': 0.4,
            'elastic_alpha': 30,
            'morph_prob': 0.3,
            'aspect_prob': 0.3,
            'blur_prob': 0.3,
            'noise_prob': 0.2,
        },
        'heavy': {
            'scale_range': (0.5, 1.2),
            'rotation_range': (-20, 20),
            'translate_range': (-4, 4),
            'elastic_prob': 0.6,
            'elastic_alpha': 40,
            'morph_prob': 0.4,
            'aspect_prob': 0.4,
            'blur_prob': 0.4,
            'noise_prob': 0.3,
        },
    }

    cfg = config.get(level, config['medium'])

    # 1. Scale variation (simulate different puzzle sizes)
    if random.random() < 0.7:
        scale = random.uniform(*cfg['scale_range'])
        img = scale_and_center(img, scale)

    # 2. Rotation (simulate tilted writing)
    if random.random() < 0.5:
        angle = random.uniform(*cfg['rotation_range'])
        img = rotate_image(img, angle)

    # 3. Translation (simulate off-center extraction)
    if random.random() < 0.5:
        dx = random.randint(*cfg['translate_range'])
        dy = random.randint(*cfg['translate_range'])
        img = translate_image(img, dx, dy)

    # 4. Elastic deformation (simulate handwriting variation)
    if random.random() < cfg['elastic_prob']:
        alpha = random.uniform(cfg['elastic_alpha'] * 0.5, cfg['elastic_alpha'] * 1.5)
        sigma = random.uniform(3, 5)
        img = elastic_transform(img, alpha=alpha, sigma=sigma)

    # 5. Morphological operations (stroke width variation)
    if random.random() < cfg['morph_prob']:
        op = random.choice(['erode', 'dilate'])
        img = erode_dilate(img, operation=op, kernel_size=2)

    # 6. Aspect ratio distortion
    if random.random() < cfg['aspect_prob']:
        x_scale = random.uniform(0.9, 1.1)
        y_scale = random.uniform(0.9, 1.1)
        img = distort_aspect_ratio(img, x_scale, y_scale)

    # 7. Blur (simulate downsampling artifacts)
    if random.random() < cfg['blur_prob']:
        radius = random.uniform(0.3, 0.8)
        img = apply_blur(img, radius)

    # 8. Noise
    if random.random() < cfg['noise_prob']:
        if random.random() < 0.5:
            img = add_gaussian_noise(img, sigma=random.uniform(0.02, 0.08))
        else:
            img = add_salt_pepper_noise(img, amount=random.uniform(0.01, 0.03))

    return img


def augment_for_confusion_pairs(img_array, label, level='medium'):
    """
    Extra augmentation for commonly confused digit pairs.

    Adds targeted transformations for:
    - 1 <-> 7: Vary stroke angle, add/remove serif
    - 6 <-> 8: Vary loop clarity
    - 3 <-> 8: Vary curve sharpness
    """
    img = augment_handwritten(img_array, level)

    # Extra augmentation for confusing pairs
    if label in [1, 7]:
        # Extra rotation to simulate 1/7 confusion
        if random.random() < 0.3:
            angle = random.uniform(-8, 8)
            img = rotate_image(img, angle)

    elif label in [6, 8]:
        # Extra morphological ops for loop clarity
        if random.random() < 0.3:
            op = random.choice(['erode', 'dilate'])
            img = erode_dilate(img, operation=op, kernel_size=2)

    elif label in [3, 8]:
        # Extra elastic deformation for curve variation
        if random.random() < 0.3:
            img = elastic_transform(img, alpha=25, sigma=4)

    return img


if __name__ == '__main__':
    # Test augmentation
    import matplotlib.pyplot as plt

    # Create a simple test image (digit-like pattern)
    test_img = np.ones((28, 28), dtype=np.float32)
    test_img[5:23, 10:18] = 0.2  # Vertical stroke
    test_img[10:15, 8:20] = 0.2  # Horizontal stroke

    fig, axes = plt.subplots(3, 5, figsize=(12, 8))

    axes[0, 0].imshow(test_img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Apply different augmentations
    for i, level in enumerate(['light', 'medium', 'heavy']):
        for j in range(4):
            aug_img = augment_handwritten(test_img, level)
            axes[i, j + 1].imshow(aug_img, cmap='gray')
            axes[i, j + 1].set_title(f'{level} {j + 1}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_test.png')
    print("Saved augmentation_test.png")
