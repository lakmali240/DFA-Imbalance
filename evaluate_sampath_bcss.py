"""
Quick Evaluation Script for SAMPath BCSS Model
==============================================

This script is pre-configured for your BCSS setup.
Just run it from your SAMPath directory!

Usage:
    python evaluate_bcss_classwise.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# ============================================================================
# CONFIGURATION - MODIFY THESE PATHS AS NEEDED
# ============================================================================

# Your trained model checkpoint
CHECKPOINT_PATH = "/pscratch/scheu2_dgxbdsa24/BDSA_ZoomLDM/SAMPath_BCSS_Orginal_data/sampath/x5g08464/checkpoints/epoch=31-step=2304.ckpt"

# Dataset paths (from your BCSS.py config)
DATASET_ROOT = "/project/scheu2_dgxbdsa24/Lakmali_data/data/BCSS/merged_dataset"
DATASET_CSV = "/home/lra240/SAMpath_model_test/SAMPath/dataset_cfg/BCSS_cv.csv"

# SAM checkpoints
SAM_CHECKPOINT = "/home/lra240/SAMpath_model_test/SAMPath/sam_vit_b_01ec64.pth"
HIPT_CHECKPOINT = "/home/lra240/SAMpath_model_test/SAMPath/vit256_small_dino.pth"

# Evaluation settings
DATA_SPLIT = "val"  # "val" or "test"
VAL_FOLD_ID = 0
BATCH_SIZE = 4  # Reduce if OOM
NUM_WORKERS = 4
GPU_ID = 0

# Output
OUTPUT_DIR = "./bcss_evaluation_results"

# BCSS class names
BCSS_CLASS_NAMES = {
    0: "Background",
    1: "Tumor",
    2: "Stroma",
    3: "Inflammatory",
    4: "Necrosis",
    5: "Other"
}

NUM_CLASSES = 6  # Not including background in count, but 0 exists

# ============================================================================


def calculate_classwise_metrics(pred_masks, gt_masks, num_classes, ignored_class=0):
    """Calculate class-wise IoU, Dice, Precision, Recall"""
    metrics = {}
    
    for cls in range(num_classes + 1):  # +1 because we have 0-5
        if cls == ignored_class:
            continue
            
        pred_binary = (pred_masks == cls)
        gt_binary = (gt_masks == cls)
        
        tp = np.sum(pred_binary & gt_binary)
        fp = np.sum(pred_binary & ~gt_binary)
        fn = np.sum(~pred_binary & gt_binary)
        tn = np.sum(~pred_binary & ~gt_binary)
        
        iou = tp / (tp + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        metrics[cls] = {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return metrics


def calculate_pixel_distribution(gt_masks, num_classes):
    """Calculate pixel distribution from masks"""
    distribution = {}
    total_pixels = gt_masks.size
    
    for cls in range(num_classes + 1):
        count = np.sum(gt_masks == cls)
        percentage = (count / total_pixels) * 100
        distribution[cls] = {
            'pixel_count': int(count),
            'percentage': percentage
        }
    
    return distribution, total_pixels


def print_comparison_table(metrics, pixel_dist, class_names):
    """Print formatted comparison table"""
    
    # Calculate foreground total
    total_fg = sum(pixel_dist[c]['pixel_count'] for c in range(1, 6) if c in pixel_dist)
    
    print("\n" + "=" * 110)
    print("CLASS-WISE PERFORMANCE vs PIXEL DISTRIBUTION COMPARISON")
    print("=" * 110)
    print(f"\n{'Class':<5} {'Name':<15} {'Pixels':<12} {'Total%':<8} {'FG%':<8} {'IoU':<10} {'Dice':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 110)
    
    all_ious = []
    all_dices = []
    
    for cls in range(1, 6):
        if cls not in metrics:
            continue
            
        m = metrics[cls]
        pd_info = pixel_dist.get(cls, {'pixel_count': 0, 'percentage': 0})
        fg_pct = (pd_info['pixel_count'] / total_fg) * 100 if total_fg > 0 else 0
        
        all_ious.append(m['iou'])
        all_dices.append(m['dice'])
        
        name = class_names.get(cls, f'Class {cls}')
        print(f"{cls:<5} {name:<15} {pd_info['pixel_count']:>10,}  {pd_info['percentage']:>6.2f}%  {fg_pct:>6.2f}%  "
              f"{m['iou']*100:>7.2f}%   {m['dice']*100:>7.2f}%   {m['precision']*100:>7.2f}%    {m['recall']*100:>7.2f}%")
    
    print("-" * 110)
    print(f"\n{'MEAN (mIoU):':<40} {np.mean(all_ious)*100:.2f}%")
    print(f"{'MEAN Dice:':<40} {np.mean(all_dices)*100:.2f}%")
    
    # Analysis
    print("\n" + "=" * 110)
    print("IMBALANCE ANALYSIS")
    print("=" * 110)
    
    for cls in range(1, 6):
        if cls not in metrics:
            continue
        m = metrics[cls]
        pd_info = pixel_dist.get(cls, {'pixel_count': 0, 'percentage': 0})
        fg_pct = (pd_info['pixel_count'] / total_fg) * 100 if total_fg > 0 else 0
        name = class_names.get(cls, f'Class {cls}')
        
        # Analysis based on distribution vs performance
        if fg_pct < 10 and m['iou'] < 0.5:
            print(f"⚠️  {name}: MINORITY class ({fg_pct:.1f}% pixels) with LOW IoU ({m['iou']*100:.1f}%)")
            print(f"    → Suggestion: Increase class weight or use focal loss")
        elif fg_pct > 30 and m['iou'] > 0.6:
            print(f"✓  {name}: MAJORITY class ({fg_pct:.1f}% pixels) with GOOD IoU ({m['iou']*100:.1f}%)")
        elif fg_pct < 10 and m['iou'] > 0.5:
            print(f"✓  {name}: MINORITY class ({fg_pct:.1f}% pixels) but DECENT IoU ({m['iou']*100:.1f}%)")


def main():
    import albumentations
    from albumentations.pytorch import ToTensorV2
    
    print("=" * 60)
    print("SAMPath BCSS Model Evaluation")
    print("=" * 60)
    
    # Setup device
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Import SAMPath modules (make sure you're in the SAMPath directory)
    try:
        from network.sam_network import PromptSAMLateFusion
        from network.get_network import get_hipt
        from image_mask_dataset import ImageMaskDataset
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure you run this script from the SAMPath directory!")
        sys.exit(1)
    
    # Build model
    print("\nLoading model...")
    extra_encoder = get_hipt(HIPT_CHECKPOINT, neck=False)
    
    model = PromptSAMLateFusion(
        model_type='vit_b',
        checkpoint=SAM_CHECKPOINT,
        prompt_dim=256,
        num_classes=NUM_CLASSES,
        extra_encoder=extra_encoder,
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        freeze_mask_decoder=False,
        mask_HW=(1024, 1024),
        feature_input=False,
        prompt_decoder=False,
        dense_prompt_decoder=False,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')['state_dict']
    state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Setup dataset
    print(f"\nLoading {DATA_SPLIT} dataset...")
    transform_fn = albumentations.Compose([
        albumentations.Resize(1024, 1024),
    ])
    
    dataset = ImageMaskDataset(
        dataset_root=DATASET_ROOT,
        dataset_csv_path=DATASET_CSV,
        data_type=DATA_SPLIT,
        val_fold_id=VAL_FOLD_ID,
        augmentation=transform_fn,
        data_ext=".jpg",
        dataset_mean=(0.485, 0.456, 0.406),
        dataset_std=(0.229, 0.224, 0.225),
        ignored_classes=(0),
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Dataset size: {len(dataset)} images")
    
    # Run inference
    print("\nRunning inference...")
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for images, gt_masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            
            pred_masks, _ = model(images)
            pred_masks = torch.stack(pred_masks, dim=0)
            
            # Get predictions (skip class 0 in argmax)
            pred_cls = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
            
            # Handle ignored regions
            ignored = (gt_masks == 0)
            pred_cls = pred_cls * (~ignored).long()
            
            all_preds.append(pred_cls.cpu().numpy())
            all_gts.append(gt_masks.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_classwise_metrics(all_preds, all_gts, NUM_CLASSES, ignored_class=0)
    pixel_dist, total_pixels = calculate_pixel_distribution(all_gts, NUM_CLASSES)
    
    # Print results
    print_comparison_table(metrics, pixel_dist, BCSS_CLASS_NAMES)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save to CSV
    results_data = []
    total_fg = sum(pixel_dist[c]['pixel_count'] for c in range(1, 6) if c in pixel_dist)
    
    for cls in range(1, 6):
        if cls not in metrics:
            continue
        m = metrics[cls]
        pd_info = pixel_dist.get(cls, {'pixel_count': 0, 'percentage': 0})
        fg_pct = (pd_info['pixel_count'] / total_fg) * 100 if total_fg > 0 else 0
        
        results_data.append({
            'class_id': cls,
            'class_name': BCSS_CLASS_NAMES.get(cls, f'Class {cls}'),
            'pixel_count': pd_info['pixel_count'],
            'total_percentage': pd_info['percentage'],
            'foreground_percentage': fg_pct,
            'iou': m['iou'] * 100,
            'dice': m['dice'] * 100,
            'precision': m['precision'] * 100,
            'recall': m['recall'] * 100
        })
    
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(OUTPUT_DIR, f'bcss_classwise_results_{DATA_SPLIT}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Create simple bar plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        classes = [BCSS_CLASS_NAMES[i] for i in range(1, 6)]
        fg_pcts = [results_data[i]['foreground_percentage'] for i in range(5)]
        ious = [results_data[i]['iou'] for i in range(5)]
        dices = [results_data[i]['dice'] for i in range(5)]
        
        # Pixel distribution
        axes[0].bar(classes, fg_pcts, color='steelblue')
        axes[0].set_title('Pixel Distribution (FG%)')
        axes[0].set_ylabel('Percentage (%)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # IoU
        axes[1].bar(classes, ious, color='forestgreen')
        axes[1].set_title('Class-wise IoU')
        axes[1].set_ylabel('IoU (%)')
        axes[1].set_ylim(0, 100)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Dice
        axes[2].bar(classes, dices, color='darkorange')
        axes[2].set_title('Class-wise Dice')
        axes[2].set_ylabel('Dice (%)')
        axes[2].set_ylim(0, 100)
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f'bcss_comparison_{DATA_SPLIT}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"✅ Plot saved to: {plot_path}")
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

