"""
Visualization Script with CLI Arguments
Creates colored and grayscale visualizations of stitched WSI masks
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from PIL import Image

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None


def visualize_masks(mask_path, class_mapping, grayscale_mapping, output_size=(1024, 1024)):
    """
    Create colored and grayscale visualizations of WSI mask.
    
    Parameters:
    -----------
    mask_path : str
        Path to stitched WSI mask
    class_mapping : dict
        Mapping of class indices to names and colors
    grayscale_mapping : dict
        Mapping of class indices to grayscale values
    output_size : tuple
        Output size for visualization (width, height)
    
    Returns:
    --------
    tuple : (colored_mask, grayscale_mask) as numpy arrays
    """
    print(f"\n✅ Starting visualization for mask: {mask_path}")

    if not os.path.exists(mask_path):
        print(f"❌ ERROR: Mask file not found at {mask_path}")
        return None, None

    # Load and resize mask
    try:
        mask = Image.open(mask_path)
        print(f"✅ Mask loaded successfully (mode={mask.mode}, size={mask.size})")
    except Exception as e:
        print(f"❌ ERROR while loading mask: {e}")
        return None, None

    mask = mask.resize(output_size, Image.Resampling.NEAREST)
    print(f"✅ Resized mask to {output_size}")
    mask_array = np.array(mask)

    # Print unique values in original mask for verification
    unique_vals = np.unique(mask_array)
    print(f"✅ Unique values in original mask: {unique_vals}")

    # Create colored mask
    colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)

    print("✅ Applying color mapping...")
    for class_idx, class_info in class_mapping.items():
        count = np.sum(mask_array == class_idx)
        colored_mask[mask_array == class_idx] = class_info['color']
        print(f"   → Mapped class {class_idx} ({class_info['name']}), "
              f"count={count}, color={class_info['color']}")

    # Create grayscale mask with new mapping
    grayscale_mask = np.zeros_like(mask_array, dtype=np.uint8)
    print("✅ Applying grayscale mapping...")
    for orig_val, new_val in grayscale_mapping.items():
        count = np.sum(mask_array == orig_val)
        grayscale_mask[mask_array == orig_val] = new_val
        print(f"   → Grayscale map {orig_val} → {new_val} (count={count})")

    print(f"✅ Unique values in remapped grayscale mask: {np.unique(grayscale_mask)}")
    print("✅ Visualization finished successfully")
    
    return colored_mask, grayscale_mask


def main():
    parser = argparse.ArgumentParser(description='Visualize stitched WSI mask')
    parser.add_argument('--wsi_mask_path', type=str, required=True,
                        help='Path to stitched WSI mask')
    parser.add_argument('--colored_save_path', type=str, required=True,
                        help='Path to save colored visualization')
    parser.add_argument('--grayscale_save_path', type=str, required=True,
                        help='Path to save grayscale visualization')
    parser.add_argument('--output_size', type=int, nargs=2, default=[1024, 1024],
                        help='Output size (width height), default: 1024 1024')
    
    args = parser.parse_args()
    
    # Class mapping
    predicted_class_mapping = {
        0: {'name': 'Exclude', 'color': [0, 255, 255]},      # Cyan 
        1: {'name': 'Gray Matter', 'color': [255, 0, 0]},    # Red
        2: {'name': 'White Matter', 'color': [0, 255, 0]},   # Green
        3: {'name': 'Leptomeninges', 'color': [0, 0, 255]},  # Blue
        4: {'name': 'Superficial', 'color': [255, 255, 0]},  # Yellow
        5: {'name': 'Background', 'color': [0, 0, 0]}        # Black
    }

    grayscale_mapping = {
        0: 197,  # Exclude
        1: 31,   # Gray Matter
        2: 63,   # White Matter
        3: 95,   # Leptomeninges
        4: 127,  # Superficial
        5: 0     # Background
    }
    
    print(f"\n=== Running visualization for {args.wsi_mask_path} ===")
    
    colored_mask, grayscale_mask = visualize_masks(
        args.wsi_mask_path, 
        predicted_class_mapping, 
        grayscale_mapping,
        tuple(args.output_size)
    )

    if colored_mask is not None and grayscale_mask is not None:
        # Create output directories
        os.makedirs(os.path.dirname(args.colored_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.grayscale_save_path), exist_ok=True)

        print(f"✅ Saving colored image → {args.colored_save_path}")
        Image.fromarray(colored_mask).save(args.colored_save_path)

        print(f"✅ Saving grayscale image → {args.grayscale_save_path}")
        Image.fromarray(grayscale_mask).save(args.grayscale_save_path)

        # Confirm the save worked
        print(f"✅ Files created:")
        for f in [args.colored_save_path, args.grayscale_save_path]:
            if os.path.exists(f):
                print(f"   ✔ {f} ({os.path.getsize(f)/1024:.1f} KB)")
            else:
                print(f"   ❌ Missing: {f}")
    else:
        print("❌ Visualization failed")
        return 1

    print("\n=== Script completed ===")
    return 0


if __name__ == "__main__":
    exit(main())