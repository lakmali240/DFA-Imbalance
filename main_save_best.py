"""
main_save_best.py --- SAMPath DFA Training Entry Point

BCSS class index mapping :
  0: Ignored   1: Inflammatory   2: Tumor   3: Stroma   4: Other   5: Necrosis
"""

from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import MetricCollection, JaccardIndex, F1Score, Dice

from network.sam_network import PromptSAM, PromptSAMLateFusion
from network.sam_network_focal import (PromptSAMFocalAttention,
                                       PromptSAMLateFusionFocalAttention)
from pl_module_sam_seg import SamSeg

import albumentations


def get_augmentation(cfg):
    W, H = cfg.dataset.image_hw if cfg.dataset.image_hw is not None else (1024, 1024)
    transform_train_fn = albumentations.Compose([
        albumentations.RandomResizedCrop(H, W, scale=(0.08, 1.0), p=1.0),
        albumentations.Flip(p=0.75),
        albumentations.RandomRotate90(),
        albumentations.ColorJitter(0.1, 0.1, 0.1, 0.1),
    ])
    transform_test_fn = albumentations.Compose([
        albumentations.Resize(H, W),
    ])
    return transform_train_fn, transform_test_fn


def get_metrics(cfg):
    num_classes  = cfg.dataset.num_classes + 1
    # num_classes  = cfg.dataset.num_classes
    ignore_index = 0
    metrics = MetricCollection({
        "IOU_Jaccard_Bal": JaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index, task='multiclass'),
        "IOU_Jaccard": JaccardIndex(
            num_classes=num_classes, ignore_index=ignore_index,
            task='multiclass', average="micro"),
        "F1": F1Score(
            num_classes=num_classes, ignore_index=ignore_index,
            task='multiclass', average="micro"),
        "Dice": Dice(
            num_classes=num_classes, ignore_index=ignore_index, average="micro"),
        "Dice_Bal": Dice(
            num_classes=num_classes, ignore_index=ignore_index, average="macro"),
    })
    return metrics


def get_model(cfg):
    # ── Extra encoder ────────────────────────────────────────────────────────
    if cfg.model.extra_encoder is not None:
        print(f"Using {cfg.model.extra_encoder} as extra encoder")
        neck = (cfg.model.extra_type == 'plus')

        if cfg.model.extra_encoder == 'hipt':
            from network.get_network import get_hipt
            extra_encoder = get_hipt(cfg.model.extra_checkpoint, neck=neck)

        elif cfg.model.extra_encoder == 'virchow2':
            import timm, torch
            from timm.layers import SwiGLUPacked
            extra_encoder = timm.create_model(
                cfg.model.extra_checkpoint, pretrained=True,
                mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
            extra_encoder.eval()
            extra_encoder.requires_grad_(False)

        elif cfg.model.extra_encoder == 'uni':
            import timm, torch
            extra_encoder = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16,
                init_values=1e-5, num_classes=0,
                dynamic_img_size=True, pretrained=False)
            state_dict = torch.load(cfg.model.extra_checkpoint, map_location="cpu")
            extra_encoder.load_state_dict(state_dict, strict=True)
            extra_encoder.eval()
            extra_encoder.requires_grad_(False)

        else:
            raise NotImplementedError(f"Unknown extra_encoder: {cfg.model.extra_encoder}")
    else:
        extra_encoder = None

    # ── Model class selection ─────────────────────────────────────────────────
    MODEL_MAP = {
        'plus':         PromptSAM,
        'fusion':       PromptSAMLateFusion,
        'focal':        PromptSAMFocalAttention,
        'focal_fusion': PromptSAMLateFusionFocalAttention,
    }
    if cfg.model.extra_type not in MODEL_MAP:
        raise NotImplementedError(f"Unknown extra_type: {cfg.model.extra_type}")

    MODEL = MODEL_MAP[cfg.model.extra_type]

    # ── DFA focal kwargs ──────────────────────────────────────────────────────
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

        focal_kwargs["learnable_bias"] = cfg.focal_attention.get("learnable_bias", True)
        focal_kwargs["delta_init_scale"] = cfg.focal_attention.get("delta_init_scale", 0.1)
        focal_kwargs["use_prior_in_forward"] = cfg.focal_attention.get(
            "use_prior_in_forward", True)
        focal_kwargs["warm_start"] = False

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
    return model


def get_data_module(cfg):
    from image_mask_dataset import GeneralDataModule, ImageMaskDataset, FtMaskDataset
    augs = get_augmentation(cfg)
    common_cfg = {
        "dataset_root":      cfg.dataset.dataset_root,
        "dataset_csv_path":  cfg.dataset.dataset_csv_path,
        "val_fold_id":       cfg.dataset.val_fold_id,
        "data_ext":          cfg.dataset.get("data_ext", ".jpg"),
        "dataset_mean":      cfg.dataset.dataset_mean,
        "dataset_std":       cfg.dataset.dataset_std,
        "ignored_classes":   cfg.dataset.ignored_classes,
    }
    dataset_cls = FtMaskDataset if cfg.dataset.feature_input else ImageMaskDataset
    return GeneralDataModule(common_cfg, dataset_cls, cus_transforms=augs,
                             batch_size=cfg.batch_size, num_workers=cfg.num_workers)


# def get_pl_module(cfg, model, metrics):
#     loss_cfg = cfg.get("loss", {})

#     pl_module = SamSeg(
#         cfg=cfg,
#         sam_model=model,
#         metrics=metrics,
#         num_classes=cfg.dataset.num_classes,
#         focal_cof=loss_cfg.get("focal_cof", 20.),
#         dice_cof=loss_cfg.get("dice_cof",   1.),
#         ce_cof=loss_cfg.get("ce_cof",       0.),
#         iou_cof=loss_cfg.get("iou_cof",     1.),
#         lr=cfg.opt.learning_rate,
#         weight_decay=cfg.opt.weight_decay,
#         lr_steps=cfg.opt.steps,
#         warmup_steps=cfg.opt.warmup_steps,
#         ignored_index=cfg.dataset.ignored_classes_metric,
#         delta_lr_multiplier=loss_cfg.get("delta_lr_multiplier", 100.0),
#         bias_l2_lambda=loss_cfg.get("bias_l2_lambda", 0.0001),
#     )
#     return pl_module

def get_pl_module(cfg, model, metrics):
    loss_cfg = cfg.get("loss", {})

    delta_lr_multiplier = loss_cfg.get("delta_lr_multiplier", 100.0)
    bias_l2_lambda      = loss_cfg.get("bias_l2_lambda",      0.0001)

    # ── Class names (index → label) ───────────────────────────────────────────
    num_classes  = cfg.dataset.num_classes
    dataset_name = cfg.dataset.get("dataset_name", "").lower()

    if dataset_name == "bdsa":
        # BDSA class mapping:
        # 0=Ignored, 1=Gray Matter, 2=White Matter,
        # 3=Leptomeninges, 4=Superficial, 5=Background
        class_names = {0: "Ignored",  1: "GrayMat.",  2: "WhiteMat.",
                       3: "Lepto.",   4: "Superfic.", 5: "Backgnd."}
    elif num_classes == 6:
        # BCSS correct mapping:
        # 0=Ignored, 1=Inflammatory, 2=Tumor, 3=Stroma, 4=Other, 5=Necrosis
        class_names = {0: "Ignored",  1: "Inflam.", 2: "Tumor",
                       3: "Stroma",   4: "Other",   5: "Necrosis"}
    elif num_classes == 3:
        # CRAG mapping:
        # 0=Background, 1=Benign, 2=Malignant
        class_names = {0: "Background", 1: "Benign", 2: "Malignant"}
    else:
        class_names = {i: f"Class{i}" for i in range(num_classes)}

    pl_module = SamSeg(
        cfg=cfg,
        sam_model=model,
        metrics=metrics,
        num_classes=num_classes,
        focal_cof=loss_cfg.get("focal_cof", 20.),
        dice_cof=loss_cfg.get("dice_cof",   1.),
        ce_cof=loss_cfg.get("ce_cof",       0.),
        iou_cof=loss_cfg.get("iou_cof",     1.),
        lr=cfg.opt.learning_rate,
        weight_decay=cfg.opt.weight_decay,
        lr_steps=cfg.opt.steps,
        warmup_steps=cfg.opt.warmup_steps,
        ignored_index=cfg.dataset.ignored_classes_metric,
        delta_lr_multiplier=delta_lr_multiplier,
        bias_l2_lambda=bias_l2_lambda,
    )
    return pl_module


def main(cfg):
    data_module = get_data_module(cfg)
    sam_model   = get_model(cfg)
    metrics     = get_metrics(cfg=cfg)
    pl_module   = get_pl_module(cfg, model=sam_model, metrics=metrics)

    logger     = WandbLogger(project=cfg.project, name=cfg.name,
                              save_dir=cfg.out_dir, log_model=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    accum      = cfg.get("accumulate_grad_batches", 1)

    trainer = Trainer(
        default_root_dir=cfg.out_dir,
        logger=logger,
        devices=cfg.devices,
        max_epochs=cfg.opt.num_epochs,
        accelerator="gpu",
        log_every_n_steps=20,
        num_sanity_val_steps=0,
        precision=cfg.opt.precision,
        callbacks=[lr_monitor],
        accumulate_grad_batches=accum,
        fast_dev_run=False,
    )
    trainer.fit(pl_module, data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config",  default=None)
    parser.add_argument('--devices', type=lambda s: [int(x) for x in s.split(',')],
                        default=[0])
    parser.add_argument('--project', type=str, default="test")
    parser.add_argument('--name',    type=str, default="test_sam_prompt")
    args = parser.parse_args()

    module = __import__(args.config, globals(), locals(), ['cfg'])
    cfg = module.cfg
    cfg["project"] = args.project
    cfg["devices"] = args.devices
    cfg["name"]    = args.name

    print(cfg)
    main(cfg)
