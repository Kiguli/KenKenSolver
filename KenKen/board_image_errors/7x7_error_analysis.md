# 7x7 KenKen Board Image Error Analysis

## Summary

9 out of 100 7x7 puzzles failed (91% accuracy). The failures fall into two distinct categories:

| Error Type | Count | Affected Puzzles |
|------------|-------|------------------|
| Cage Detection (extra walls) | 5 | 13, 14, 31, 69, 85 |
| Character Recognition (missing operator) | 4 | 35, 45, 49, 87 |

---

## Error Type 1: Extra Cage Walls Detected

**Affected puzzles:** 13, 14, 31, 69, 85

### Symptoms
- OpenCV detects more cages than expected (e.g., 23-24 detected vs 15-19 expected)
- Extra walls appear between columns 0-1 or columns 5-6 (near board edges)

### Root Cause: Integer Division Rounding + Edge Overlap

The border detection code in `run_full_evaluation.py` (lines 133-154):

```python
cell_size = ((BOARD_SIZE * SCALE_FACTOR) // size)  # 1800 // 7 = 257
horizontal_window = (cell_size - delta, cell_size + delta)
```

**Problem 1: Rounding Error Accumulation**
- For 7x7: `cell_size = 1800 // 7 = 257` pixels
- But `257 * 7 = 1799`, not 1800
- This 1-pixel error accumulates across columns, causing detection windows to drift out of alignment with actual grid lines

**Problem 2: Thick Border Overlap**
- The outer board border is 16 pixels thick (`add_border(grid, 16)`)
- Cage walls are also 16 pixels thick (`range(-8, 8)` in `add_side_wall`)
- When a cage wall is near column 0 or column 6, the thick cage wall may overlap with or be adjacent to the thick outer border
- The detection algorithm uses `if max_val - min_val > 11` to identify "thick" lines
- Near edges, this can trigger false positives where the algorithm sees two separate thick line segments

### Visual Pattern
In the error images, you'll see red walls appearing at:
- Between column 0 and column 1 (leftmost internal boundary)
- Between column 5 and column 6 (rightmost internal boundary)

---

## Error Type 2: Multiplication Operator Not Detected

**Affected puzzles:** 35, 45, 49, 87

### Symptoms
- Cage count matches expected (e.g., 17 detected vs 17 expected)
- But `op` is read as `""` instead of `"mul"`
- Only affects cages with 4-digit target numbers (e.g., "1234×")

### Root Cause: Character Positioning Exceeds Cell Width

The character segmentation code in `run_full_evaluation.py` (lines 302-311):

```python
def segment_cell(grid, size, border_thickness, row, col):
    cell_size = len(grid) // size  # 900 // 7 = 128 pixels
    cell = grid[row * cell_size + border_thickness: row * cell_size + cell_size // 2,
                col * cell_size + border_thickness: (col + 1) * cell_size - border_thickness]
```

The scanned region width is: `cell_size - 2 * border_thickness ≈ 128 - 2*21 = 86 pixels`

The character positioning in `BoardImageGeneration.ipynb`:

```python
number_size = square_size // 4  # 128 // 4 = 32 pixels per character
col_position = 20 + position * width  # width ≈ 19 pixels after cropping
```

**Position calculation for "1234×":**
| Position | Character | Start Pixel |
|----------|-----------|-------------|
| 0 | 1 | 20 |
| 1 | 2 | ~39 |
| 2 | 3 | ~58 |
| 3 | 4 | ~77 |
| 4 | × | ~96 |

**The × symbol starts at pixel 96, but the scanned region is only 86 pixels wide.**

The multiplication symbol is either:
1. Completely outside the detection region, or
2. Partially clipped, making it unrecognizable

### Visual Pattern
In the error images, you'll see:
- Cage walls in black (correctly detected)
- The target number in red (because the detected cage has `op: ""` instead of `op: "mul"`)

---

## Potential Fixes

### For Cage Detection Errors
1. Use floating-point cell sizes instead of integer division
2. Increase tolerance near board edges
3. Filter out detected lines that are too close to the outer border

### For Character Recognition Errors
1. Increase the scanned region width
2. Use dynamic character positioning based on available cell width
3. Scale character sizes based on the number of characters to display

---

## Files Referenced

- `KenKen/run_full_evaluation.py` - Border detection and character segmentation
- `KenKen/BoardImageGeneration.ipynb` - Board image generation with character positioning
- `KenKen/board_image_errors/board7_*.png` - Visual diff images (black = correct, red = error)
