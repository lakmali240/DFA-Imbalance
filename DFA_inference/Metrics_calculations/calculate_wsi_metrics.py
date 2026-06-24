#!/usr/bin/env python3
"""
Calculate class-wise and overall segmentation metrics (Dice and IoU) 
for WSI predictions vs ground truth masks.

IMPORTANT: Exclude class is reported as 0.0 but NOT included in mean calculation!
Mean is calculated ONLY from the 5 real tissue classes.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Disable PIL size limit
Image.MAX_IMAGE_PIXELS = None


def resize_ground_truth_to_10x(gt_path, target_size, save_path=None):
    """
    Resize ground truth from 40× to 10× (0.25 scale).
    """
    print(f"  Loading GT from: {gt_path}")
    gt = Image.open(gt_path)
    original_size = gt.size
    print(f"  Original GT size (40×): {original_size}")
    print(f"  Original GT mode: {gt.mode}")
    
    # Convert to grayscale if RGB
    if gt.mode == 'RGB' or gt.mode == 'RGBA':
        print(f"  Converting from {gt.mode} to grayscale (L mode)")
        gt = gt.convert('L')
    
    # Resize to 10× using NEAREST to preserve label values
    gt_10x = gt.resize(target_size, Image.Resampling.NEAREST)
    print(f"  Resized GT to 10×: {target_size}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gt_10x.save(save_path)
        print(f"  Saved resized GT to: {save_path}")
    
    return np.array(gt_10x)


def calculate_dice_iou(gt_mask, pred_mask, class_value, exclude_value=None):
    """
    Calculate Dice and IoU for a specific class.
    GT pixels with exclude_value are ignored (not counted in evaluation).
    """
    # Create binary masks for the class
    gt_binary = (gt_mask == class_value).astype(bool)
    pred_binary = (pred_mask == class_value).astype(bool)
    
    # Create mask for valid GT pixels (ignore GT Exclude pixels)
    if exclude_value is not None:
        valid_mask = (gt_mask != exclude_value).astype(bool)
        # Apply valid mask: only evaluate where GT is not Exclude
        gt_binary = gt_binary & valid_mask
        pred_binary = pred_binary & valid_mask
    
    # Calculate intersection and union
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    
    # Calculate total area for Dice
    gt_area = gt_binary.sum()
    pred_area = pred_binary.sum()
    total_area = gt_area + pred_area
    
    # Calculate metrics
    if total_area > 0:
        dice = (2.0 * intersection) / total_area
    else:
        dice = 1.0 if intersection == 0 else 0.0
    
    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0 if intersection == 0 else 0.0
    
    return dice, iou


def calculate_metrics_for_wsi(gt_mask, pred_mask, label2idx, classes_to_evaluate):
    """
    Calculate all metrics for a single WSI.
    
    CRITICAL: Mean is calculated ONLY from classes_to_evaluate.
              Exclude class is set to 0.0 but NOT included in mean!
    
    Args:
        gt_mask: Ground truth mask (numpy array)
        pred_mask: Predicted mask (numpy array)
        label2idx: Dictionary mapping class names to grayscale values
        classes_to_evaluate: List of class names to include in mean calculation
    
    Returns:
        dict: Metrics for each class and overall mean
    """
    metrics = {}
    
    # Lists to collect scores ONLY for classes in classes_to_evaluate
    evaluated_dice_scores = []
    evaluated_iou_scores = []
    
    # Get GT Exclude value to ignore those pixels
    gt_exclude_value = label2idx['Exclude']
    
    print(f"  Classes to evaluate (for mean): {classes_to_evaluate}")
    print(f"  Total classes: {len(label2idx.keys())}")
    
    # Calculate per-class metrics
    for class_name in label2idx.keys():
        class_value = label2idx[class_name]
        
        if class_name in classes_to_evaluate:
            # EVALUATE THIS CLASS - calculate real metrics
            dice, iou = calculate_dice_iou(gt_mask, pred_mask, class_value, exclude_value=gt_exclude_value)
            
            metrics[f'{class_name}_dice'] = dice
            metrics[f'{class_name}_iou'] = iou
            
            # ADD TO LISTS FOR MEAN CALCULATION
            evaluated_dice_scores.append(dice)
            evaluated_iou_scores.append(iou)
            
            print(f"    ✓ {class_name}: Dice={dice:.4f}, IoU={iou:.4f} [INCLUDED in mean]")
        else:
            # DON'T EVALUATE THIS CLASS - set to 0.0 and DON'T include in mean
            metrics[f'{class_name}_dice'] = 0.0
            metrics[f'{class_name}_iou'] = 0.0
            
            print(f"    ✗ {class_name}: Set to 0.0 [EXCLUDED from mean]")
    
    # Calculate mean ONLY from evaluated classes
    # CRITICAL: This mean does NOT include Exclude class!
    if evaluated_dice_scores:
        metrics['mean_dice'] = np.mean(evaluated_dice_scores)
        metrics['mean_iou'] = np.mean(evaluated_iou_scores)
        print(f"  Mean calculated from {len(evaluated_dice_scores)} classes: {classes_to_evaluate}")
    else:
        metrics['mean_dice'] = 0.0
        metrics['mean_iou'] = 0.0
        print(f"  WARNING: No classes evaluated!")
    
    return metrics


def remap_prediction_mask(pred_mask, pred_label2idx, label2idx):
    """
    Remap prediction mask values to match ground truth values.
    """
    remapped = np.zeros_like(pred_mask, dtype=np.uint8)
    
    # Map from prediction values to ground truth values
    for class_name in pred_label2idx.keys():
        pred_val = pred_label2idx[class_name]
        gt_val = label2idx[class_name]
        remapped[pred_mask == pred_val] = gt_val
    
    return remapped


def merge_exclude_to_background(mask, label2idx, is_prediction=False):
    """
    Merge Exclude class into Background class.
    For predictions: Treats all Exclude pixels as Background pixels.
    For ground truth: Does nothing (GT Exclude pixels will be ignored in evaluation).
    """
    if not is_prediction:
        # For GT, don't merge - we'll ignore Exclude pixels during evaluation
        return mask.copy()
    
    # For prediction, merge Exclude into Background
    mask_copy = mask.copy()
    bg_val = label2idx['Background']
    exclude_val = label2idx['Exclude']
    
    # Replace all Exclude pixels with Background value
    mask_copy[mask_copy == exclude_val] = bg_val
    
    return mask_copy


def main():
    # ======================== CONFIGURATION ========================
    
    # IMPORTANT: These are the actual GRAYSCALE VALUES in the masks
    label2idx = {
        'Exclude': 191,        # GT value
        'Gray_Matter': 31,
        'White_Matter': 63,
        'Leptomeninges': 95,
        'Superficial': 127,
        'Background': 0
    }
    
    # Prediction uses slightly different value for Exclude
    pred_label2idx = {
        'Exclude': 197,        # Pred value (different from GT)
        'Gray_Matter': 31,
        'White_Matter': 63,
        'Leptomeninges': 95,
        'Superficial': 127,
        'Background': 0
    }
    
    # CRITICAL: Classes to include in mean calculation
    # Exclude is NOT in this list, so it won't be included in mean!
    classes_to_evaluate = ['Background', 'Gray_Matter', 'White_Matter', 'Leptomeninges', 'Superficial']
    
    print("\n" + "="*80)
    print("EVALUATION CONFIGURATION")
    print("="*80)
    print(f"Total classes defined: {len(label2idx)}")
    print(f"Classes to evaluate: {len(classes_to_evaluate)}")
    print(f"Classes INCLUDED in mean: {classes_to_evaluate}")
    print(f"Classes EXCLUDED from mean: {[c for c in label2idx.keys() if c not in classes_to_evaluate]}")
    print("="*80 + "\n")
    

    BATCH_RESULTS_DIR = "/path/to/output/batch/results"


    # Define your image paths using batch processing structure
    image_info = [
        {
            'name': 'Image_1',
            'wsi_id': '1',
            'gt_path': '/path/to/input/mask1.png',
            'wsi_name': '<WSI name 1>',
            'original_dims_40x': (95615, 84127)
        },
        {
            'name': 'Image_2',
            'wsi_id': '2',
            'gt_path': '/path/to/input/mask2.png',
            'wsi_name': '<WSI name 2>',
            'original_dims_40x': (109559, 71469)
        },
    ]
    
    # Add prediction paths from batch processing structure
    for img_info in image_info:
        wsi_dir = f"WSI_{img_info['wsi_id']}_{img_info['wsi_name']}"
        img_info['pred_path'] = os.path.join(BATCH_RESULTS_DIR, wsi_dir, 'results', 'grayscale_wsi.png')
    
    # Output directory
    output_dir = os.path.join(BATCH_RESULTS_DIR, 'metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    # Directory to save resized GT masks
    resized_gt_dir = os.path.join(BATCH_RESULTS_DIR, 'gt_masks_10x')
    os.makedirs(resized_gt_dir, exist_ok=True)
    
    # ======================== PROCESS EACH IMAGE ========================
    
    all_results = []
    
    print("\n" + "="*80)
    print("CALCULATING WSI SEGMENTATION METRICS")
    print("="*80 + "\n")
    print(f"Predictions directory: {BATCH_RESULTS_DIR}")
    print(f"Output directory: {output_dir}")
    print("")
    
    for img_info in tqdm(image_info, desc="Processing images"):
        print(f"\n{'='*80}")
        print(f"Processing: {img_info['name']}")
        print(f"{'='*80}")
        
        # Check if files exist
        if not os.path.exists(img_info['gt_path']):
            print(f"  ⚠️  WARNING: GT not found: {img_info['gt_path']}")
            continue
        
        if not os.path.exists(img_info['pred_path']):
            print(f"  ⚠️  WARNING: Prediction not found: {img_info['pred_path']}")
            print(f"  Expected at: {img_info['pred_path']}")
            continue
        
        # Load prediction mask
        print(f"  Loading prediction from: {img_info['pred_path']}")
        pred_img = Image.open(img_info['pred_path'])
        print(f"  Prediction mode: {pred_img.mode}, size: {pred_img.size}")
        
        # Convert to grayscale if needed
        if pred_img.mode != 'L':
            print(f"  Converting prediction from {pred_img.mode} to grayscale")
            pred_img = pred_img.convert('L')
        
        pred_mask = np.array(pred_img)
        print(f"  Prediction shape: {pred_mask.shape}")
        print(f"  Prediction unique values: {np.unique(pred_mask)}")
        
        # Calculate 10× dimensions
        dims_40x = img_info['original_dims_40x']
        dims_10x = (int(dims_40x[0] * 0.25), int(dims_40x[1] * 0.25))
        
        # Resize ground truth to match prediction (40x -> 10x)
        gt_save_path = os.path.join(resized_gt_dir, f"{img_info['name']}_gt_10x.png")
        gt_mask = resize_ground_truth_to_10x(
            img_info['gt_path'],
            dims_10x,
            save_path=gt_save_path
        )
        print(f"  GT size after resize: {gt_mask.shape}")
        print(f"  GT unique values: {np.unique(gt_mask)}")
        
        # Ensure masks are the same size
        if gt_mask.shape != pred_mask.shape:
            print(f"  ⚠️  Size mismatch! GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            print(f"  Resizing prediction to match GT...")
            pred_img = Image.fromarray(pred_mask)
            pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), 
                                      Image.Resampling.NEAREST)
            pred_mask = np.array(pred_img)
            print(f"  New prediction size: {pred_mask.shape}")
        
        # Remap prediction to match ground truth encoding
        print("  Remapping prediction mask to match GT encoding...")
        pred_mask_remapped = remap_prediction_mask(pred_mask, pred_label2idx, label2idx)
        
        # Merge Exclude into Background ONLY for Prediction
        print("  Merging Prediction's Exclude into Background...")
        print("  GT's Exclude pixels will be ignored in evaluation...")
        gt_mask_processed = merge_exclude_to_background(gt_mask, label2idx, is_prediction=False)
        pred_mask_merged = merge_exclude_to_background(pred_mask_remapped, label2idx, is_prediction=True)
        
        print(f"  GT unique values (unchanged): {np.unique(gt_mask_processed)}")
        print(f"  Pred unique values after merging: {np.unique(pred_mask_merged)}")
        
        # Calculate metrics
        print("  Calculating metrics...")
        print(f"  GT Exclude pixels (value 191) are IGNORED in all class evaluations")
        print(f"  Exclude class will be set to 0.0 but NOT included in mean calculation")
        metrics = calculate_metrics_for_wsi(gt_mask_processed, pred_mask_merged, label2idx, classes_to_evaluate)
        metrics['image_name'] = img_info['name']
        metrics['wsi_id'] = img_info['wsi_id']
        
        # Print per-class results
        print(f"\n  Results for {img_info['name']}:")
        print(f"  {'Class':<20} {'Dice':>10} {'IoU':>10} {'In Mean?'}")
        print(f"  {'-'*52}")
        for class_name in label2idx.keys():
            dice_key = f'{class_name}_dice'
            iou_key = f'{class_name}_iou'
            if dice_key in metrics:
                in_mean = "✓ YES" if class_name in classes_to_evaluate else "✗ NO"
                print(f"  {class_name:<20} {metrics[dice_key]:>10.4f} {metrics[iou_key]:>10.4f} {in_mean}")
        print(f"  {'-'*52}")
        print(f"  {'Mean (5 classes)':<20} {metrics['mean_dice']:>10.4f} {metrics['mean_iou']:>10.4f}")
        print(f"  Calculated from: {', '.join(classes_to_evaluate)}")
        
        all_results.append(metrics)
    
    # ======================== SUMMARY STATISTICS ========================
    
    if not all_results:
        print("\n❌ No results to summarize!")
        return
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    print("📝 EVALUATION STRATEGY:")
    print("   ✓ GT Exclude pixels (191): IGNORED in all evaluations")
    print("   ✓ Prediction Exclude pixels (197): Merged into Background (0)")
    print("   ✓ Evaluated classes (5): Background, Gray_Matter, White_Matter, Leptomeninges, Superficial")
    print("   ✓ Exclude class: Reported as 0.0 but NOT included in mean")
    print(f"   ✓ Mean: Average of {len(classes_to_evaluate)} evaluated classes ONLY\n")
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns
    cols = ['image_name', 'wsi_id']
    for class_name in label2idx.keys():
        cols.append(f"{class_name}_dice")
        cols.append(f"{class_name}_iou")
    cols.extend(['mean_dice', 'mean_iou'])
    df_results = df_results[cols]
    
    # Save individual results
    csv_path = os.path.join(output_dir, 'wsi_metrics_individual.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Individual results saved to: {csv_path}\n")
    
    # Calculate overall statistics
    metric_cols = [c for c in df_results.columns if c not in ['image_name', 'wsi_id']]
    
    summary_stats = pd.DataFrame({
        'Metric': metric_cols,
        'Mean': df_results[metric_cols].mean().values,
        'Std': df_results[metric_cols].std().values,
        'Min': df_results[metric_cols].min().values,
        'Max': df_results[metric_cols].max().values
    })
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, 'wsi_metrics_summary.csv')
    summary_stats.to_csv(summary_path, index=False)
    print(f"✅ Summary statistics saved to: {summary_path}\n")
    
    # Print summary table
    print("\nOVERALL SUMMARY (Mean ± Std):")
    print("="*80)
    print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*80)
    for _, row in summary_stats.iterrows():
        metric_name = row['Metric']
        # Mark which metrics are included in mean calculation
        if 'mean_' in metric_name:
            marker = " ← CALCULATED FROM 5 CLASSES"
        elif 'Exclude' in metric_name:
            marker = " [NOT in mean]"
        else:
            marker = " [IN mean]" if any(c in metric_name for c in classes_to_evaluate) else ""
        
        print(f"{metric_name:<30} {row['Mean']:>10.4f} {row['Std']:>10.4f} "
              f"{row['Min']:>10.4f} {row['Max']:>10.4f}{marker}")
    
    print("\n" + "="*80)
    print("METRICS CALCULATION COMPLETE!")
    print("="*80 + "\n")
    
    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    print(f"   Mean Dice Score:  {summary_stats[summary_stats['Metric']=='mean_dice']['Mean'].values[0]:.4f} "
          f"(averaged from {len(classes_to_evaluate)} classes)")
    print(f"   Mean IoU Score:   {summary_stats[summary_stats['Metric']=='mean_iou']['Mean'].values[0]:.4f} "
          f"(averaged from {len(classes_to_evaluate)} classes)")
    
    # Find best and worst performing classes (excluding Exclude)
    class_dice_metrics = [m for m in metric_cols if m.endswith('_dice') and m not in ['mean_dice', 'Exclude_dice']]
    best_class = summary_stats[summary_stats['Metric'].isin(class_dice_metrics)].nlargest(1, 'Mean')
    worst_class = summary_stats[summary_stats['Metric'].isin(class_dice_metrics)].nsmallest(1, 'Mean')
    
    print(f"\n   Best performing class:  {best_class['Metric'].values[0]} ({best_class['Mean'].values[0]:.4f})")
    print(f"   Worst performing class: {worst_class['Metric'].values[0]} ({worst_class['Mean'].values[0]:.4f})")
    
    print("\n✅ All outputs saved to:", output_dir)
    print(f"\nOutput files:")
    print(f"  - Individual results: {csv_path}")
    print(f"  - Summary statistics: {summary_path}")
    print(f"  - Resized GT masks (10x): {resized_gt_dir}")
    
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    print(f"✓ Mean calculated from {len(classes_to_evaluate)} classes:")
    for i, cls in enumerate(classes_to_evaluate, 1):
        print(f"  {i}. {cls}")
    print(f"✗ Exclude class: Set to 0.0 but EXCLUDED from mean")
    print("="*80)


if __name__ == '__main__':
    main()


