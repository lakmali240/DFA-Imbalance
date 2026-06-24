"""
BDSA_DFA.py — SAMPath Dynamic Focal Attention Config for BDSA Dataset
======================================================================

    "use_prior_in_forward": True   →  b_c = log(π_c) + δ_c 
    "use_prior_in_forward": False  →  b_c = δ_c              

BDSA class index mapping:
    0: Ignored / Outside ROI
    1: Gray Matter        
    2: White Matter      
    3: Leptomeninges     
    4: Superficial      
    5: Background         

"""

from box import Box

config = {
    "batch_size": 12,
    "accumulate_grad_batches": 2,
    "num_workers": 8,

    # ── Output directory ──────────────────────────────────────────────────────
    "out_dir": "/path/to/output",

    # ── Optimiser ────────────────────────────────────────────────────────────
    "opt": {
        "num_epochs": 30,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "precision": "32",
        "steps": [72 * 25, 72 * 29],
        "warmup_steps": 72,
    },

    # ── Model ─────────────────────────────────────────────────────────────────
    "model": {
        "type": "vit_b",
        "checkpoint": "/path/to/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": False,
        "extra_encoder": "hipt",
        "extra_type": "focal_fusion",   # late-fusion + DFA
        "extra_checkpoint": "/path/to/vit256_small_dino.pth",
    },

    # =========================================================================
    # FOCAL ATTENTION — DFA CONFIGURATION
    # =========================================================================
    "focal_attention": {
        "enabled": True,

        "gamma": 2.0,

        "learnable_bias": True,

        "delta_init_scale": 0.1,

        # ── Forward formula switch ────────────────────────────────────────────
        # True  → b_c = log(π_c) + δ_c
        # False → b_c = δ_c
        "use_prior_in_forward": True,  

        # ── Pixel frequencies f_c ─────────────────────────────────────────────
        # Used for warm-start init
        "class_pixel_frequencies": {
            0: 0.000,   # Ignored / Outside ROI  ← 0.0: excluded from init mean
            1: 0.334,   # Gray Matter             
            2: 0.106,   # White Matter
            3: 0.064,   # Leptomeninges           
            4: 0.030,   # Superficial             
            5: 0.466,   # Background              
        },

        # Deprecated — always False for DFA (warm-start handled by delta_init_scale)
        "warm_start": False,
    },

    # =========================================================================
    # LOSS CONFIGURATION
    # =========================================================================
    "loss": {
        "focal_cof": 0.25,
        "dice_cof":  0.75,
        "ce_cof":    0.0,
        "iou_cof":   0.0625,

        "delta_lr_multiplier": 20.0,    

        "bias_l2_lambda": 0.0,

        # Plotting frequency (epochs)
        "log_bias_every_n_epochs": 1,
        "save_bias_plots": True,
        "bias_plot_dir": "bias_plots",   # relative to out_dir
    },

    # =========================================================================
    # DATASET
    # =========================================================================
    "dataset": {

        "dataset_name": "bdsa",

        "dataset_root": (
            "/path/to/training/data"
        ),
        "dataset_csv_path": (
            "/dataset_cfg/BDSA_cv.csv"
        ),
        "val_fold_id": 0,

        # 6 = classes 0–5 (0 is Ignored)
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
