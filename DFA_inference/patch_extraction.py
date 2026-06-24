"""
COMPLETE Patch Extraction Script with CLI Arguments
Guarantees 100% coverage with NO gaps at edges
"""

import os
import argparse
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import pyvips

class WSIPatchExtractor:
    def __init__(self, wsi_path, save_dir, 
                 patch_size=1024, 
                 level=1,           # level=1 ≈ 10x, level=2 for 2.5X
                 jpg_quality=90):
        self.wsi_path = Path(wsi_path)
        self.save_dir = Path(save_dir)
        self.patch_size = patch_size
        self.level = level
        self.jpg_quality = jpg_quality

        # Output
        self.image_dir = self.save_dir / 'patches'
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # Initialize
        self._init_image()

    def _init_image(self):
        """Initialize vips image at requested level."""
        try:
            pyvips.cache_set_max(2000)
            pyvips.cache_set_max_mem(768 * 1024 * 1024)

            # Open at target level
            self.wsi = pyvips.Image.new_from_file(
                str(self.wsi_path),
                access='random',
                level=self.level
            )

            # Dimensions at this level
            self.width = self.wsi.width
            self.height = self.wsi.height

            print(f"\n{'='*80}")
            print(f"LOADED WSI AT LEVEL {self.level}")
            print(f"{'='*80}")
            print(f"WSI: {self.wsi_path.name}")
            print(f"Dimensions: {self.width} × {self.height}")
            print(f"Patch size: {self.patch_size}")
            
            # Tilecache for performance
            self.wsi = self.wsi.tilecache(
                tile_width=self.patch_size,
                tile_height=self.patch_size,
                max_tiles=1024, 
                access='random'
            )

        except Exception as e:
            print(f"Error loading image: {e}")
            raise

    def _generate_complete_coverage_coordinates(self):
        """
        Generate coordinates ensuring 100% coverage.
        
        Strategy:
        1. Start at (0,0)
        2. Regular grid with stride = patch_size
        3. Add final patches at edges if remaining space > 0
        
        Returns: List of (row_idx, col_idx, y, x) tuples
        """
        coords = []
        
        # Generate Y coordinates
        y_positions = []
        row_idx = 0
        y = 0
        while y + self.patch_size <= self.height:
            y_positions.append((row_idx, y))
            y += self.patch_size
            row_idx += 1
        
        # Add edge patch at bottom if needed
        if y < self.height:
            y_edge = self.height - self.patch_size
            # Only add if it doesn't duplicate the last patch
            if not y_positions or y_positions[-1][1] != y_edge:
                y_positions.append((row_idx, y_edge))
                row_idx += 1
        
        # Generate X coordinates
        x_positions = []
        col_idx = 0
        x = 0
        while x + self.patch_size <= self.width:
            x_positions.append((col_idx, x))
            x += self.patch_size
            col_idx += 1
        
        # Add edge patch at right if needed
        if x < self.width:
            x_edge = self.width - self.patch_size
            # Only add if it doesn't duplicate the last patch
            if not x_positions or x_positions[-1][1] != x_edge:
                x_positions.append((col_idx, x_edge))
                col_idx += 1
        
        # Create all combinations
        for row_idx, y in y_positions:
            for col_idx, x in x_positions:
                coords.append((row_idx, col_idx, y, x))
        
        # Print coverage info
        print(f"\n{'='*80}")
        print(f"COVERAGE ANALYSIS")
        print(f"{'='*80}")
        print(f"Y coverage:")
        print(f"  First patch: y={y_positions[0][1]}")
        print(f"  Last patch:  y={y_positions[-1][1]} to {y_positions[-1][1] + self.patch_size}")
        print(f"  Image height: {self.height}")
        print(f"  Total Y positions: {len(y_positions)}")
        
        print(f"\nX coverage:")
        print(f"  First patch: x={x_positions[0][1]}")
        print(f"  Last patch:  x={x_positions[-1][1]} to {x_positions[-1][1] + self.patch_size}")
        print(f"  Image width: {self.width}")
        print(f"  Total X positions: {len(x_positions)}")
        
        # Verify coverage
        y_covered = y_positions[-1][1] + self.patch_size >= self.height
        x_covered = x_positions[-1][1] + self.patch_size >= self.width
        
        print(f"\nCoverage verification:")
        print(f"  Y axis: {'✓ FULL COVERAGE' if y_covered else '✗ GAPS EXIST'}")
        print(f"  X axis: {'✓ FULL COVERAGE' if x_covered else '✗ GAPS EXIST'}")
        print(f"  Total patches: {len(coords)}")
        print(f"{'='*80}\n")
        
        return coords

    def extract_single_patch(self, coord):
        """Extract a single patch."""
        row_idx, col_idx, y, x = coord
        
        try:
            # Extract patch
            patch = self.wsi.extract_area(x, y, self.patch_size, self.patch_size)
            
            # Filename based on indices
            name = f"patch_{row_idx:04d}_{col_idx:04d}.jpg"
            out = self.image_dir / name

            # Save as JPEG
            patch.jpegsave(
                str(out), 
                Q=self.jpg_quality,
                optimize_coding=True, 
                strip=True
            )

            return True
        except Exception as e:
            print(f"\nError at row={row_idx}, col={col_idx}, y={y}, x={x}: {e}")
            return False

    def extract_patches(self):
        """Extract all patches with progress bar."""
        # Generate coordinates
        coords = self._generate_complete_coverage_coordinates()
        
        total = len(coords)
        print(f"Extracting {total} patches...")
        
        start = time.time()
        saved = 0

        # Extract with progress bar
        pbar = tqdm(total=total, desc="Extracting patches")
        
        for coord in coords:
            if self.extract_single_patch(coord):
                saved += 1
            pbar.update(1)
        
        pbar.close()
        
        # Summary
        elapsed = time.time() - start
        rate = saved / elapsed if elapsed > 0 else 0.0
        
        print(f"\n{'='*80}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*80}")
        print(f"Patches saved: {saved}/{total}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Rate: {rate:.2f} patches/s")
        
        # Calculate total size
        size_mb = sum(f.stat().st_size for f in self.image_dir.glob('*.jpg')) / (1024*1024)
        print(f"Total size: {size_mb:.2f} MB")
        print(f"{'='*80}\n")
        
        return saved


def main():
    parser = argparse.ArgumentParser(description='Extract patches from WSI')
    parser.add_argument('--wsi_path', type=str, required=True,
                        help='Path to WSI file')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save patches')
    parser.add_argument('--patch_size', type=int, default=1024,
                        help='Patch size (default: 1024)')
    parser.add_argument('--level', type=int, default=1,
                        help='Pyramid level (default: 1 for 10x)')
    parser.add_argument('--jpg_quality', type=int, default=90,
                        help='JPEG quality (default: 90)')
    
    args = parser.parse_args()
    
    extractor = WSIPatchExtractor(
        wsi_path=args.wsi_path,
        save_dir=args.save_dir,
        patch_size=args.patch_size,
        level=args.level,
        jpg_quality=args.jpg_quality
    )
    
    extractor.extract_patches()


if __name__ == '__main__':
    main()