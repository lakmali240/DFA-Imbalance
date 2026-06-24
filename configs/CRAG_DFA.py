"""
CRAG Dataset --- Dynamic Focal Attention (DFA) Configuration

Formula: b_c = δ_c   (fully learned)
         └── δ_c      [learned end-to-end; warm-start from log(π_c)]

NOTE: log(π_c) = γ·log(1−f_c) is used ONLY to initialise δ_c.

"""

from box import Box

config = {

    "batch_size": 12,
    "accumulate_grad_batches": 2,
    "num_workers": 4,
    "out_dir": "/path/to/output",

    "opt": {
        "num_epochs": 60,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "precision": "32",
        "steps": [23 * 50, 23 * 55],
        "warmup_steps": 46,
    },

    "model": {
        "type": 'vit_b',
        "checkpoint": "/path/to/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,
        "extra_encoder": 'hipt',
        "extra_type": "focal_fusion",
        "extra_checkpoint": "/path/to/vit256_small_dino.pth",
    },

    # =========================================================================
    # DFA FOCAL ATTENTION --- b_c = δ_c
    # =========================================================================

    "focal_attention": {
        "enabled": True,
        "gamma": 2.0,
        "learnable_bias": True,
        "delta_init_scale": 0.1,
        # ── Pixel frequencies f_c ─────────────────────────────────────────
        "class_pixel_frequencies": {
            0: 0.000,   # Background / Ignored
            1: 0.460,   # Benign gland  
            2: 0.390,   # Malignant gland
        },
        # Deprecated --- always False for DFA
        "warm_start": False,
    },

    # =========================================================================
    # LOSS + OPTIMIZER
    # =========================================================================

    "loss": {
        "focal_cof": 0.125,
        "dice_cof":  0.875,
        "ce_cof":    0.0,
        "iou_cof":   0.0,
        # ── Dedicated high lr for δ_c ──────────────────────────────────────
        # δ_c learning rate = learning_rate × delta_lr_multiplier
        # = 1e-4 × 100 = 1e-2
        "delta_lr_multiplier": 100.0,
        # ── Tiny L2 regularization on δ_c ─────────────────────────────────
        "bias_l2_lambda": 0.0001,
        # Logging
        "log_bias_every_n_epochs": 1,
        "save_bias_plots": True,
    },

    "dataset": {
        "dataset_root": "/path/to/data/CRAG/dataset",
        "dataset_csv_path": "/dataset_cfg/CRAG_cv.csv",
        "data_ext": ".png",
        "val_fold_id": 0,
        "num_classes": 3,
        "ignored_classes": None,
        "ignored_classes_metric": 1,
        "image_hw": (1536, 1536),
        "feature_input": False,
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std":  (0.229, 0.224, 0.225),
    },

}

cfg = Box(config)