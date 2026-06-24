"""
FIXED Patch Stitching Script
Correctly handles edge patches by recalculating their actual positions
"""

import os
import argparse
import numpy as np
from PIL import Image
import pyvips

def get_wsi_dimensions_at_level(wsi_path, level=1):
    """
    Get WSI dimensions at specified pyramid level.
    """
    try:
        pyvips.cache_set_max(100)
        img = pyvips.Image.new_from_file(str(wsi_path), access='random', level=level)
        width = img.width
        height = img.height
        return width, height
    except Exception as e:
        print(f"Error reading WSI dimensions: {e}")
        raise


def calculate_patch_positions(target_width, target_height, patch_size):
    """
    Calculate patch positions using the SAME logic as extraction.
    This ensures stitching positions match extraction positions.
    
    Returns:
    --------
    dict : Mapping from (row_idx, col_idx) to (y, x) position
    """
    position_map = {}
    
    # Generate Y coordinates (same as extraction)
    y_positions = []
    row_idx = 0
    y = 0
    while y + patch_size <= target_height:
        y_positions.append((row_idx, y))
        y += patch_size
        row_idx += 1
    
    # Add edge patch at bottom if needed
    if y < target_height:
        y_edge = target_height - patch_size
        if not y_positions or y_positions[-1][1] != y_edge:
            y_positions.append((row_idx, y_edge))
    
    # Generate X coordinates (same as extraction)
    x_positions = []
    col_idx = 0
    x = 0
    while x + patch_size <= target_width:
        x_positions.append((col_idx, x))
        x += patch_size
        col_idx += 1
    
    # Add edge patch at right if needed
    if x < target_width:
        x_edge = target_width - patch_size
        if not x_positions or x_positions[-1][1] != x_edge:
            x_positions.append((col_idx, x_edge))
    
    # Create position map for all combinations
    for row_idx, y in y_positions:
        for col_idx, x in x_positions:
            position_map[(row_idx, col_idx)] = (y, x)
    
    return position_map


def stitch_wsi_patches_complete(patch_folder, target_width, target_height, patch_size=1024):
    """
    Stitch patches back into WSI with correct edge handling.
    
    Parameters:
    -----------
    patch_folder : str
        Path to folder with patch masks (patch_RRRR_CCCC_mask.png)
    target_width : int
        Target WSI width (at 10×)
    target_height : int
        Target WSI height (at 10×)
    patch_size : int
        Size of each patch (1024)
    
    Returns:
    --------
    PIL.Image : Stitched WSI mask
    """
    
    print(f"\n{'='*80}")
    print(f"STITCHING PATCHES")
    print(f"{'='*80}")
    print(f"Target dimensions: {target_width} × {target_height}")
    print(f"Patch size: {patch_size}")
    
    # Initialize canvas with class 5 (Background)
    canvas = np.full((target_height, target_width), 5, dtype=np.uint8)
    print(f"Canvas initialized with class 5 (Background)")
    
    # Calculate correct positions for all patches
    print(f"Calculating patch positions...")
    position_map = calculate_patch_positions(target_width, target_height, patch_size)
    print(f"Position map created for {len(position_map)} patches")
    
    # Track coverage
    coverage_mask = np.zeros((target_height, target_width), dtype=bool)
    
    # Get all patch files
    patch_files = sorted([f for f in os.listdir(patch_folder) 
                         if f.endswith('_mask.png') or f.endswith('.png')])
    
    if not patch_files:
        print(f"ERROR: No patches found in {patch_folder}")
        return Image.fromarray(canvas)
    
    print(f"Found {len(patch_files)} patch files")
    
    # Process each patch
    placed = 0
    errors = 0
    position_mismatches = 0
    
    for filename in patch_files:
        # Parse filename: patch_RRRR_CCCC_mask.png or patch_RRRR_CCCC.png
        parts = filename.replace('_mask.png', '.png').replace('.png', '').split('_')
        
        if len(parts) >= 3 and parts[0] == 'patch':
            try:
                row_idx = int(parts[1])
                col_idx = int(parts[2])
            except ValueError:
                print(f"Skipping {filename}: cannot parse indices")
                errors += 1
                continue
        else:
            print(f"Skipping {filename}: unexpected format")
            errors += 1
            continue
        
        # Load patch
        patch_path = os.path.join(patch_folder, filename)
        try:
            patch = Image.open(patch_path)
            
            # Convert to grayscale if needed
            if patch.mode != 'L':
                if patch.mode == 'RGB' or patch.mode == 'RGBA':
                    patch = patch.split()[0]
                else:
                    patch = patch.convert('L')
            
            patch_array = np.array(patch)
            
            # Verify patch size
            if patch_array.shape[0] != patch_size or patch_array.shape[1] != patch_size:
                print(f"Warning: {filename} has size {patch_array.shape}, expected {patch_size}×{patch_size}")
                patch = patch.resize((patch_size, patch_size), Image.Resampling.NEAREST)
                patch_array = np.array(patch)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            errors += 1
            continue
        
        # Get correct position from position map
        if (row_idx, col_idx) not in position_map:
            print(f"Warning: {filename} has indices ({row_idx}, {col_idx}) not in position map")
            position_mismatches += 1
            continue
        
        y, x = position_map[(row_idx, col_idx)]
        
        # DEBUG: Check if this differs from simple multiplication
        simple_y = row_idx * patch_size
        simple_x = col_idx * patch_size
        if y != simple_y or x != simple_x:
            print(f"  Edge patch {filename}: actual pos ({y},{x}) != simple pos ({simple_y},{simple_x})")
        
        # Place patch on canvas
        y_end = min(y + patch_size, target_height)
        x_end = min(x + patch_size, target_width)
        
        patch_h = y_end - y
        patch_w = x_end - x
        
        if y < target_height and x < target_width:
            try:
                canvas[y:y_end, x:x_end] = patch_array[:patch_h, :patch_w]
                coverage_mask[y:y_end, x:x_end] = True
                placed += 1
            except Exception as e:
                print(f"Error placing {filename} at ({y},{x}): {e}")
                errors += 1
        else:
            print(f"Skipping {filename}: position ({y},{x}) outside canvas")
            errors += 1
    
    # Coverage analysis
    covered_pixels = np.sum(coverage_mask)
    total_pixels = coverage_mask.size
    coverage_pct = (covered_pixels / total_pixels) * 100
    
    print(f"\n{'='*80}")
    print(f"STITCHING RESULTS")
    print(f"{'='*80}")
    print(f"Patches placed: {placed}/{len(patch_files)}")
    print(f"Errors: {errors}")
    print(f"Position mismatches: {position_mismatches}")
    print(f"\nCoverage:")
    print(f"  Covered pixels: {covered_pixels:,} / {total_pixels:,}")
    print(f"  Coverage: {coverage_pct:.4f}%")
    
    if coverage_pct < 99.99:
        uncovered = np.sum(~coverage_mask)
        print(f"  ⚠️  Uncovered pixels: {uncovered:,} ({(uncovered/total_pixels)*100:.4f}%)")
        print(f"  These areas will show as Background (class 5)")
    else:
        print(f"  ✓ FULL COVERAGE ACHIEVED")
    
    # Check unique values
    unique_vals = np.unique(canvas)
    print(f"\nUnique class values in stitched mask: {unique_vals}")
    
    if 0 in unique_vals:
        count_0 = np.sum(canvas == 0)
        pct_0 = (count_0 / total_pixels) * 100
        print(f"  ⚠️  Class 0 (Exclude): {count_0:,} pixels ({pct_0:.4f}%)")
    else:
        print(f"  ✓ No class 0 (Exclude)")
    
    if 5 in unique_vals:
        count_5 = np.sum(canvas == 5)
        pct_5 = (count_5 / total_pixels) * 100
        if count_5 > 0.01 * total_pixels:  # More than 1% background
            print(f"  ⚠️  Class 5 (Background): {count_5:,} pixels ({pct_5:.4f}%)")
            print(f"      This might indicate gaps in coverage!")
    
    print(f"{'='*80}\n")
    
    return Image.fromarray(canvas)


def main():
    parser = argparse.ArgumentParser(description='Stitch predicted patches back to WSI')
    parser.add_argument('--original_wsi_path', type=str, required=True,
                        help='Path to original WSI (to get dimensions)')
    parser.add_argument('--patches_folder', type=str, required=True,
                        help='Path to folder containing predicted patch masks')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save stitched WSI mask')
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Patch size (default: 1024)')
    parser.add_argument('--level', type=int, default=1,
                        help='Pyramid level (default: 1 for 10x)')
    
    args = parser.parse_args()
    
    # Get dimensions at specified level
    print("Reading WSI dimensions...")
    width, height = get_wsi_dimensions_at_level(args.original_wsi_path, level=args.level)
    print(f"Dimensions at level {args.level}: {width} × {height}\n")
    
    # Stitch patches
    stitched_wsi = stitch_wsi_patches_complete(
        patch_folder=args.patches_folder,
        target_width=width,
        target_height=height,
        patch_size=args.patch_size
    )
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save result
    print(f"Saving stitched WSI to: {args.output_path}")
    stitched_wsi.save(args.output_path)
    
    print(f"Final image size: {stitched_wsi.size}")
    
    # Create debug thumbnail
    debug_size = (1024, 1024)
    debug_image = stitched_wsi.resize(debug_size, Image.Resampling.NEAREST)
    debug_path = args.output_path.replace('.png', '_debug_1024.png')
    debug_image.save(debug_path)
    print(f"Debug thumbnail saved to: {debug_path}")
    
    print("\n✓ Stitching complete!")


if __name__ == "__main__":
    main()
