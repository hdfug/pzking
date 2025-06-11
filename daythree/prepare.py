import os
import random
from typing import List, Dict, Tuple


def generate_txt_files(
        root_dir: str = 'images',
        train_txt: str = 'train.txt',
        val_txt: str = 'val.txt',
        test_txt: str = 'test.txt',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        shuffle: bool = True,
        seed: int = 42
) -> None:
    """
    Generate train, validation, and test txt files from image directory structure.

    Args:
        root_dir: Root directory containing class folders
        train_txt: Output file for training set
        val_txt: Output file for validation set
        test_txt: Output file for test set
        train_ratio: Ratio of samples for training
        val_ratio: Ratio of samples for validation
        shuffle: Whether to shuffle the samples
        seed: Random seed for reproducibility
    """
    # Validate ratios
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1")
    if train_ratio + val_ratio > 1:
        raise ValueError("Sum of train and val ratios cannot exceed 1")

    # Set random seed for reproducibility
    if shuffle:
        random.seed(seed)

    # Get class information
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not classes:
        raise ValueError(f"No class folders found in {root_dir}")

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Initialize lists to store file paths
    train_lines: List[str] = []
    val_lines: List[str] = []
    test_lines: List[str] = []

    # Process each class
    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        imgs = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
#
        if not imgs:
            print(f"⚠️ Warning: No images found in class {cls}")
            continue

        if shuffle:
            random.shuffle(imgs)

        # Calculate split indices
        n_total = len(imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split the data
        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        # Create lines for each split
        for img in train_imgs:
            train_lines.append(f"{cls}/{img} {class_to_idx[cls]}\n")
        for img in val_imgs:
            val_lines.append(f"{cls}/{img} {class_to_idx[cls]}\n")
        for img in test_imgs:
            test_lines.append(f"{cls}/{img} {class_to_idx[cls]}\n")

    # Write to files
    for path, lines in [
        (train_txt, train_lines),
        (val_txt, val_lines),
        (test_txt, test_lines)
    ]:
        with open(path, 'w') as f:
            f.writelines(lines)

    # Print summary
    print("✅ Dataset split complete:")
    print(f"  - Training samples: {len(train_lines)}")
    print(f"  - Validation samples: {len(val_lines)}")
    print(f"  - Test samples: {len(test_lines)}")
    print(f"  - Total classes: {len(classes)}")


if __name__ == "__main__":
    # Example usage
    generate_txt_files(
        root_dir='images',
        train_txt='train.txt',
        val_txt='val.txt',
        test_txt='test.txt',
        train_ratio=0.7,
        val_ratio=0.15
    )