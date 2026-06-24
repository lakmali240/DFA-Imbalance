import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from network.sam_network import PromptSAM, PromptSAMLateFusion
from network.sam_network_focal import (PromptSAMFocalAttention,
                                       PromptSAMLateFusionFocalAttention)
from pl_module_sam_seg import SamSeg
import albumentations
from torch.utils.data import DataLoader


# =============================================================================
# AUGMENTATION
# =============================================================================

def get_augmentation(cfg):
    W, H = cfg.dataset.image_hw if cfg.dataset.image_hw is not None else (1024, 1024)
    transform_test_fn = albumentations.Compose([
        albumentations.Resize(H, W),
    ])
    return transform_test_fn


# =============================================================================
# MODEL — mirrors get_model() from main_save_best.py exactly
# =============================================================================

def get_model(cfg, pretrained=None):
    # ── Extra encoder ─────────────────────────────────────────────────────────
    if cfg.model.extra_encoder is not None:
        print(f"Using {cfg.model.extra_encoder} as extra encoder")
        neck = (cfg.model.extra_type == 'plus')

        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)

        elif cfg.model.extra_encoder == 'virchow2':
            import timm
            from timm.layers import SwiGLUPacked
            print("Loading Virchow2 encoder from:", cfg.model.extra_checkpoint)
            extra_encoder = timm.create_model(
                cfg.model.extra_checkpoint,
                pretrained=True,
                mlp_layer=SwiGLUPacked,
                act_layer=torch.nn.SiLU,
            )
            extra_encoder.eval()
            extra_encoder.requires_grad_(False)

        elif cfg.model.extra_encoder == 'uni':
            import timm
            print("Loading UNI encoder from:", cfg.model.extra_checkpoint)
            extra_encoder = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
                pretrained=False,
            )
            state_dict = torch.load(cfg.model.extra_checkpoint, map_location="cpu")
            extra_encoder.load_state_dict(state_dict, strict=True)
            extra_encoder.eval()
            extra_encoder.requires_grad_(False)

        else:
            raise NotImplementedError(f"Unknown extra_encoder: {cfg.model.extra_encoder}")
    else:
        extra_encoder = None

    # ── Model class selection — includes DFA variants ─────────────────────────
    MODEL_MAP = {
        'plus':         PromptSAM,
        'fusion':       PromptSAMLateFusion,
        'focal':        PromptSAMFocalAttention,
        'focal_fusion': PromptSAMLateFusionFocalAttention,
    }
    if cfg.model.extra_type not in MODEL_MAP:
        raise NotImplementedError(f"Unknown extra_type: {cfg.model.extra_type}")

    MODEL = MODEL_MAP[cfg.model.extra_type]

    # ── DFA focal kwargs — only passed for focal/focal_fusion models ──────────
    focal_kwargs = {}
    if "focal_attention" in cfg and cfg.focal_attention.enabled:
        focal_kwargs["focal_gamma"] = cfg.focal_attention.gamma

        if "class_pixel_frequencies" in cfg.focal_attention:
            focal_kwargs["class_pixel_frequencies"] = \
                dict(cfg.focal_attention.class_pixel_frequencies)
        elif "class_dice_scores" in cfg.focal_attention:
            print("\n[WARNING] Config uses 'class_dice_scores' --- "
                  "rename to 'class_pixel_frequencies'.")
            focal_kwargs["class_pixel_frequencies"] = \
                dict(cfg.focal_attention.class_dice_scores)

        focal_kwargs["learnable_bias"]       = cfg.focal_attention.get("learnable_bias", True)
        focal_kwargs["delta_init_scale"]     = cfg.focal_attention.get("delta_init_scale", 0.1)
        focal_kwargs["use_prior_in_forward"] = cfg.focal_attention.get(
            "use_prior_in_forward", True)
        focal_kwargs["warm_start"] = False   # deprecated — always False

    # ── Instantiate model ─────────────────────────────────────────────────────
    model = MODEL(
        model_type=cfg.model.type,
        checkpoint=cfg.model.checkpoint,
        prompt_dim=cfg.model.prompt_dim,
        num_classes=cfg.dataset.num_classes,
        extra_encoder=extra_encoder,
        freeze_image_encoder=cfg.model.freeze.image_encoder,
        freeze_prompt_encoder=cfg.model.freeze.prompt_encoder,
        freeze_mask_decoder=cfg.model.freeze.mask_decoder,
        mask_HW=cfg.dataset.image_hw,
        feature_input=cfg.dataset.feature_input,
        prompt_decoder=cfg.model.prompt_decoder,
        dense_prompt_decoder=cfg.model.dense_prompt_decoder,
        no_sam=cfg.model.no_sam if "no_sam" in cfg.model else None,
        **focal_kwargs,
    )

    # ── Load pretrained weights ───────────────────────────────────────────────
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location='cpu')['state_dict']
        state_dict = {
            k[len('model.'):]: v
            for k, v in state_dict.items()
            if k.startswith('model.')
        }
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loading weights from {pretrained} got msg: {msg}")

    return model


# =============================================================================
# DATA MODULE
# =============================================================================

def get_data_module(cfg):
    from image_mask_dataset import GeneralDataModule, ImageMaskDataset, FtMaskDataset
    augs = get_augmentation(cfg)
    common_cfg_dic = {
        "dataset_root":     cfg.dataset.dataset_root,
        "dataset_csv_path": cfg.dataset.dataset_csv_path,
        "val_fold_id":      cfg.dataset.val_fold_id,
        "data_ext":         cfg.dataset.get("data_ext", ".jpg"),
        "dataset_mean":     cfg.dataset.dataset_mean,
        "dataset_std":      cfg.dataset.dataset_std,
        "ignored_classes":  cfg.dataset.ignored_classes,
    }
    dataset_cls = FtMaskDataset if cfg.dataset.feature_input else ImageMaskDataset
    return GeneralDataModule(common_cfg_dic, dataset_cls, cus_transforms=augs,
                             batch_size=cfg.batch_size, num_workers=cfg.num_workers)


# =============================================================================
# PL MODULE — mirrors get_pl_module() from main_save_best.py exactly
# =============================================================================

def get_pl_module(cfg, model, metrics):
    loss_cfg = cfg.get("loss", {})

    pl_module = SamSeg(
        cfg=cfg,
        sam_model=model,
        metrics=metrics,
        num_classes=cfg.dataset.num_classes,
        focal_cof=loss_cfg.get("focal_cof", 20.),
        dice_cof=loss_cfg.get("dice_cof",   1.),
        ce_cof=loss_cfg.get("ce_cof",       0.),
        iou_cof=loss_cfg.get("iou_cof",     1.),
        lr=cfg.opt.learning_rate,
        weight_decay=cfg.opt.weight_decay,
        lr_steps=cfg.opt.steps,
        warmup_steps=cfg.opt.warmup_steps,
        ignored_index=cfg.dataset.ignored_classes_metric,
        delta_lr_multiplier=loss_cfg.get("delta_lr_multiplier", 100.0),
        bias_l2_lambda=loss_cfg.get("bias_l2_lambda", 0.0001),
    )
    return pl_module


# =============================================================================
# MAIN — prediction loop
# =============================================================================

def main(cfg, args):
    from image_mask_dataset import PredictionDataset

    dataset = PredictionDataset(
        args.input_dir,
        data_ext=args.data_ext,
        augmentation=get_augmentation(cfg),
        dataset_mean=cfg.dataset.dataset_mean,
        dataset_std=cfg.dataset.dataset_std,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    sam_model = get_model(cfg, pretrained=args.pretrained)
    pl_module = get_pl_module(cfg, model=sam_model, metrics=None)

    accumulate_grad_batches = cfg.get("accumulate_grad_batches", 1)

    trainer = Trainer(
        default_root_dir=os.path.join(args.output_dir, "log"),
        devices=cfg.devices,
        max_epochs=cfg.opt.num_epochs,
        accelerator="gpu",
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        precision=cfg.opt.precision,
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=False,
    )

    pred_masks = trainer.predict(pl_module, dataloaders=dataloader)
    pred_masks = torch.cat(pred_masks, dim=0).cpu()
    print(pred_masks.shape)

    os.makedirs(args.output_dir, exist_ok=True)
    for f, pmask in zip(dataset.img_list, pred_masks):
        pmask = pmask.numpy().astype(np.uint8)
        out_f = os.path.join(
            args.output_dir,
            f[:-len(args.data_ext)] + "_mask.png",
        )
        cv2.imwrite(out_f, pmask)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config",     type=str, default=None,
                        help="Python config module (e.g. configs.BDSA_DFA)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained .ckpt checkpoint")
    parser.add_argument("--input_dir",  type=str, default=None,
                        help="Directory containing input images")
    parser.add_argument("--data_ext",   type=str, default=".jpg",
                        help="Image file extension (e.g. .jpg, .png)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predicted mask PNGs")
    parser.add_argument("--devices",
                        type=lambda s: [int(x) for x in s.split(',')],
                        default=[0],
                        help="GPU device IDs (e.g. 0 or 0,1)")
    args = parser.parse_args()

    module = __import__(args.config, globals(), locals(), ['cfg'])
    cfg = module.cfg
    cfg["devices"] = args.devices

    print(cfg)
    main(cfg, args)








