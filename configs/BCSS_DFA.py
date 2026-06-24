"""
BCSS Dataset --- Dynamic Focal Attention (DFA) Configuration

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
        "num_epochs": 35,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "precision": "32",
        "steps": [72 * 25, 72 * 29],
        "warmup_steps": 72,
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
        # ── Warm-start δ_c initialization ─────────────────────────────────

        "delta_init_scale": 0.5,

        "class_pixel_frequencies": {
            0: 0.0000,   # Outside ROI / Ignored  ← 0.0, excluded from centering
            1: 0.078,   # Inflammatory
            2: 0.398,   # Tumor             
            3: 0.347,   # Stroma
            4: 0.097,   # Other
            5: 0.080,   # Necrosis
        },
        # ── Forward formula switch ────────────────────────────────────────
        # True  → b_c = log(π_c) + δ_c 
        # False → b_c = δ_c              
        "use_prior_in_forward": False,
        # Deprecated --- always False for DFA
        "warm_start": False,
    },

    # =========================================================================
    # LOSS + OPTIMIZER
    # =========================================================================

    "loss": {
        "focal_cof": 0.25,
        "dice_cof":  0.75,
        "ce_cof":    0.0,
        "iou_cof":   0.0625,
        "delta_lr_multiplier": 20.0,
        "bias_l2_lambda": 0.0,
        # Logging
        "log_bias_every_n_epochs": 1,
        "save_bias_plots": True,
    },

    "dataset": {
        "dataset_root": "/path/to/data/BCSS/dataset",
        "dataset_csv_path": "/dataset_cfg/BCSS_cv.csv",
        "val_fold_id": 0,
        "num_classes": 6,
        "ignored_classes": (0),
        "ignored_classes_metric": None,
        "image_hw": (1024, 1024),
        "feature_input": False,
        "dataset_mean": (0.485, 0.456, 0.406),
        "dataset_std":  (0.229, 0.224, 0.225),
    },

}

cfg = Box(config)