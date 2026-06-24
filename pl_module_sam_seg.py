"""
pl_module_sam_seg.py --- SAMPath Lightning Module with DFA (b_c = δ_c)

Formula change: b_c = δ_c only (log(π_c) used for warm-start init only, NOT added).

Retained:
configure_optimizers: δ_c in dedicated high-lr param group (×100)
  compute_bias_regularization: bias_l2_lambda 0.0001

Fix: process_masks now remaps out-of-range labels (e.g. 255) to 0 (ignored).
     This prevents torchmetrics ValueError on BDSA and similar datasets.
"""

import os
import time

try:
    import torch
    from pytorch_lightning import LightningModule
    from torch import nn
    import torch.nn.functional as F
    from torchmetrics import MetricCollection
except ImportError as e:
    raise ImportError(
        f"[pl_module_sam_seg] Missing core dependency: {e}\n"
        "  → Make sure your conda/venv environment is activated."
    ) from e

try:
    from losses import SAMLoss
except ImportError as e:
    raise ImportError(
        f"[pl_module_sam_seg] Cannot import SAMLoss from 'losses': {e}\n"
        "  → losses.py must be in the same directory or on PYTHONPATH.\n"
        "  → Quick fix:  cp ../SAMPath_FA_Dynamic_FINAL_Corrected/losses.py ."
    ) from e


class SamSeg(LightningModule):

    def __init__(
        self,
        cfg,
        sam_model: nn.Module,
        metrics: MetricCollection,
        num_classes: int,
        focal_cof: float = 20.,
        dice_cof: float = 1.,
        iou_cof: float = 1.,
        ce_cof: float = 0.,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lr_steps: list = (10, 20),
        warmup_steps: int = 0,
        ignored_index=None,
        # ── Opt Fix 1 ─────────────────────────────────────────────────────
        delta_lr_multiplier: float = 100.0,
        # ── Opt Fix 2 ─────────────────────────────────────────────────────
        bias_l2_lambda: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sam_model", "metrics"])

        self.model       = sam_model
        self.num_classes = num_classes
        self.loss        = SAMLoss(focal_cof, dice_cof, ce_cof, iou_cof)

        if metrics is not None:
            self.train_metrics = metrics.clone(postfix='/train')
            self.valid_metrics = nn.ModuleList([
                metrics.clone(postfix='/val'),
                metrics.clone(postfix='/test'),
            ])
            self.test_metrics = metrics.clone(prefix='final_test/')

        self.lr                  = lr
        self.ignored_index       = ignored_index
        self.delta_lr_multiplier = delta_lr_multiplier
        self.bias_l2_lambda      = bias_l2_lambda
        self.time_and_cnt        = [0., 0]

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _get_decoder(self):
        """
        Robustly find FocalMaskDecoder regardless of PL/DDP model wrapping.
        Tries multiple known attribute paths then falls back to a recursive search.
        """
        # Path 1: standard  SamSeg.model → PromptSAMFocalAttention.model → SAM.mask_decoder
        try:
            d = self.model.model.mask_decoder
            if hasattr(d, 'get_bias_decomposition'):
                return d
        except AttributeError:
            pass
        # Path 2: DDP wraps the inner model  (.module)
        try:
            d = self.model.module.model.mask_decoder
            if hasattr(d, 'get_bias_decomposition'):
                return d
        except AttributeError:
            pass
        # Path 3: direct .mask_decoder on model (some SAM variants)
        try:
            d = self.model.mask_decoder
            if hasattr(d, 'get_bias_decomposition'):
                return d
        except AttributeError:
            pass
        # Path 4: recursive search through all named sub-modules (last resort)
        for _, module in self.model.named_modules():
            if hasattr(module, 'get_bias_decomposition'):
                return module
        return None

    def _get_bias_decomposition(self):
        d = self._get_decoder()
        if d is None:
            return None
        return d.get_bias_decomposition()

    # =========================================================================
    # LOSS
    # =========================================================================

    def forward(self, images):
        pred_masks, iou_pred = self.model(images)
        return torch.stack(pred_masks, dim=0), torch.stack(iou_pred, dim=0)

    def calc_loss(self, pred_masks, gt_masks, iou_pred, ignored_masks):
        ld = self.loss(pred_masks, gt_masks, iou_pred, ignored_masks=ignored_masks)
        assert "loss" in ld
        return ld

    def compute_bias_regularization(self) -> torch.Tensor:
        """L_reg = λ·Σ_c δ_c²  with λ = 0.0001."""
        if hasattr(self.model, 'get_bias_param'):
            bp = self.model.get_bias_param()
            if bp is not None:
                return self.bias_l2_lambda * torch.sum(bp ** 2)
        return torch.tensor(0.0, device=self.device)

    # =========================================================================
    # WANDB LOGGING
    # =========================================================================

    def log_dfa_bias(self):
        """Log δ_c (= b_c) per class to WandB."""
        decomp = self._get_bias_decomposition()
        if decomp is None:
            return

        dc = decomp.get('delta_c')    # δ_c = b_c
        lp = decomp.get('log_prior')  # reference only

        if dc is None:
            return

        n = min(len(dc), self.num_classes)
        for i in range(n):
            self.log(f"DFA/delta_c_{i}",  dc[i].item(),
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"DFA/total_b_{i}",  dc[i].item(),
                     on_epoch=True, on_step=False, sync_dist=True)
            if lp is not None:
                self.log(f"DFA/log_prior_{i}", lp[i].item(),
                         on_epoch=True, on_step=False, sync_dist=True)

        self.log("DFA/delta_mean",  dc.mean().item(),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log("DFA/delta_std",   dc.std().item(),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log("DFA/delta_range", (dc.max() - dc.min()).item(),
                 on_epoch=True, on_step=False, sync_dist=True)

    # =========================================================================
    # TRAINING
    # =========================================================================

    @torch.no_grad()
    def process_masks(self, gt_masks):
        """
        Prepare GT masks for training/metric computation.

        Remaps any pixel label outside the valid range [0, num_classes-1]
        to 0 (treated as ignored).  This handles datasets like BDSA that
        use label 255 for boundary/unlabeled regions — without this remap
        torchmetrics raises:
            ValueError: The highest label in `target` should be smaller
                        than `num_classes`.

        Valid label range:
            0                   → ignored (always)
            1 ... num_classes-1 → foreground classes
            >= num_classes      → out-of-range → remapped to 0
        """
        gt_masks = gt_masks.clone()
        out_of_range = gt_masks >= self.num_classes   # catches 255, 7, etc.
        if out_of_range.any():
            gt_masks[out_of_range] = 0   # treat as ignored
        ignored_masks = (gt_masks == 0).unsqueeze(1).long()
        return gt_masks, ignored_masks

    def predict_mask(self, pred_masks, gt_masks, ignored_masks):
        pred_masks = torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1
        pred_masks = pred_masks * (1 - ignored_masks.squeeze(1))
        if self.ignored_index is not None:
            pred_masks[pred_masks == self.ignored_index] = 0
            gt_masks[gt_masks == self.ignored_index]     = 0
        return pred_masks, gt_masks

    def training_step(self, batch, batch_idx):
        images, gt_masks     = batch
        gt_masks, ign_masks  = self.process_masks(gt_masks)
        pred_masks, iou_pred = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_pred,
                                ignored_masks=ign_masks)
        bias_l2           = self.compute_bias_regularization()
        losses["bias_l2"] = bias_l2
        losses["loss"]    = losses["loss"] + bias_l2

        self.log_losses(losses, "train")
        mp, gt = self.predict_mask(pred_masks, gt_masks, ignored_masks=ign_masks)
        self.train_metrics.update(mp, gt)
        self.log_dict(self.train_metrics.compute(), on_step=False, on_epoch=True)
        return losses["loss"]

    def on_before_optimizer_step(self, optimizer):
        """
        Clip δ_c gradients to max_norm=0.5 before the optimizer step.

        Why 0.5 (not 0.1):
          clip=0.1 for a 6-element vector → max per-element grad = 0.041.
          Adam over-normalises these tiny clipped values early in training,
          making second moments artificially small and producing erratic updates.
          0.5 still prevents true spikes while allowing the genuine class-difficulty
          gradient to propagate fully.

        Regular params are NOT clipped here — PyTorch Lightning's
        gradient_clip_val in the Trainer handles those if needed.
        """
        if hasattr(self.model, 'get_bias_param'):
            bp = self.model.get_bias_param()
            if bp is not None and bp.grad is not None:
                torch.nn.utils.clip_grad_norm_([bp], max_norm=0.5)

    def on_train_epoch_end(self):
        self.train_metrics.reset()
        self.log_dfa_bias()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        images, gt_masks    = batch
        gt_masks, ign_masks = self.process_masks(gt_masks)
        prefix = get_prefix_from_val_id(dataloader_idx)
        midx   = dataloader_idx if dataloader_idx is not None else 0

        pred_masks, iou_pred = self(images)
        losses = self.calc_loss(pred_masks, gt_masks, iou_pred,
                                ignored_masks=ign_masks)
        mp, gt = self.predict_mask(pred_masks, gt_masks, ignored_masks=ign_masks)

        if not self.trainer.sanity_checking:
            self.log_losses(losses, prefix)
            self.valid_metrics[midx].update(mp, gt)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            for vm in self.valid_metrics:
                self.log_dict(vm.compute(), add_dataloader_idx=False)
                vm.reset()

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images = batch
        with torch.no_grad():
            t0 = time.perf_counter()
            pred_masks, _ = self.model(images)
            self.time_and_cnt[0] += time.perf_counter() - t0
            self.time_and_cnt[1] += 1
        print(f"Avg predict time: {self.time_and_cnt[0]/self.time_and_cnt[1]:.4f}s")
        pred_masks = torch.stack(pred_masks, dim=0)
        return torch.argmax(pred_masks[:, 1:, ...], dim=1) + 1

    # =========================================================================
    # LOGGING HELPERS
    # =========================================================================

    def log_losses(self, losses, prefix):
        step_flag = (prefix == "train")
        for t in losses:
            self.log(f"Loss/{prefix}_{t}", losses[t],
                     on_epoch=True, on_step=step_flag, sync_dist=True,
                     add_dataloader_idx=not step_flag)

    # =========================================================================
    # OPT FIX 1 --- TWO PARAMETER GROUPS
    # =========================================================================

    def configure_optimizers(self):
        """
        δ_c in its own dedicated parameter group with a CONSTANT lr schedule.

        ┌─────────────┬────────────────────┬───────────────────────────────────────────┐
        │  Group      │  Parameters        │  Learning Rate Schedule                   │
        ├─────────────┼────────────────────┼───────────────────────────────────────────┤
        │  regular    │  all except δ_c    │  warmup(72) → const → ×0.1 → ×0.01       │
        │  delta_c    │  bias_param only   │  CONSTANT  base_lr × mult  (no decay)     │
        └─────────────┴────────────────────┴───────────────────────────────────────────┘

        Why separate schedules?
          - Regular params benefit from warmup (avoid early instability) and lr decay
            (fine-tune at the end of training).
          - δ_c already has a warm-start init so it doesn't need warmup.
            It also shouldn't be decayed — it needs a steady gradient signal
            throughout all epochs to track class difficulty.  Applying the
            same decay causes δ_c to effectively freeze after epoch 25, and
            the warmup phase starts it too cold, both of which produce the
            zigzag oscillation pattern.

        weight_decay = 0 for δ_c; L2 regularisation applied manually via
        bias_l2_lambda in compute_bias_regularization().

        Gradient clipping: δ_c gradients are clipped to max_norm=0.5 inside
        on_before_optimizer_step to prevent focal-loss spikes from causing large jumps.
        """
        bias_param_ids: set = set()
        if hasattr(self.model, 'get_bias_param'):
            bp = self.model.get_bias_param()
            if bp is not None:
                bias_param_ids.add(id(bp))

        regular_params: list = []
        bias_params:    list = []
        for _, p in self.model.named_parameters():
            if id(p) in bias_param_ids:
                bias_params.append(p)
            else:
                regular_params.append(p)

        delta_lr = self.lr * self.delta_lr_multiplier
        param_groups = [
            {'params': regular_params, 'lr': self.lr,  'name': 'regular'},
            {'params': bias_params,    'lr': delta_lr, 'name': 'delta_c',
             'weight_decay': 0.0},
        ]

        print(f"\n[DFA Optimizer]  Regular params lr   : {self.lr:.2e}  "
              f"(warmup={self.hparams.warmup_steps} steps, "
              f"decay at steps {self.hparams.lr_steps})")
        print(f"[DFA Optimizer]  δ_c params lr       : {delta_lr:.2e}"
              f"  (×{self.delta_lr_multiplier:.0f}, CONSTANT — no warmup/decay)")
        print(f"[DFA Optimizer]  bias_l2_lambda       : {self.bias_l2_lambda:.2e}")
        print(f"[DFA Optimizer]  δ_c grad clip norm   : 0.5\n")

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.hparams.weight_decay)

        def lr_lambda_regular(step: int) -> float:
            """Warmup → constant → ×0.1 → ×0.01 step decay for regular params."""
            ws     = self.hparams.warmup_steps
            s0, s1 = self.hparams.lr_steps
            if step < ws:
                return step / max(ws, 1)
            elif step < s0:
                return 1.0
            elif step < s1:
                return 0.1
            return 0.01

        def lr_lambda_delta(step: int) -> float:
            """Constant lr for δ_c throughout all training.
            No warmup needed (warm-start init handles epoch 0),
            no decay needed (δ_c must keep tracking class difficulty until the end).
            """
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_regular, lr_lambda_delta],
            verbose=False)

        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


# =============================================================================
# HELPER
# =============================================================================

def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    raise NotImplementedError



