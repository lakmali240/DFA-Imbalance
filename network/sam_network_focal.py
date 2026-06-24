"""
SAMPath with Dynamic Focal Attention (DFA)

b_c = δ_c   (fully learned; log(π_c) is used for warm-start ONLY, not added at inference)

Changes vs previous version:
  • Passes delta_init_scale to FocalMaskDecoder
  • warm_start / bias_init_value removed from interface --- use delta_init_scale
  • Both PromptSAMFocalAttention and PromptSAMLateFusionFocalAttention updated
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.modeling.common import LayerNorm2d
from segment_anything.modeling.focal_transformer import FocalTwoWayTransformer
from segment_anything.modeling.focal_mask_decoder import FocalMaskDecoder

# =============================================================================
# Shared helper --- builds and installs the focal decoder
# =============================================================================

def _build_focal_decoder(
    num_classes: int,
    prompt_dim: int,
    focal_gamma: float,
    class_pixel_frequencies: dict,
    learnable_bias: bool,
    delta_init_scale: float,
    use_prior_in_forward: bool = True,
) -> FocalMaskDecoder:
    """
    Construct FocalMaskDecoder and resize token embeddings to num_classes.
    Called by both Standard and Late-Fusion models.

    use_prior_in_forward=True  → b_c = log(π_c) + δ_c   (recommended)
    use_prior_in_forward=False → b_c = δ_c               (ablation)
    """
    focal_transformer = FocalTwoWayTransformer(
        depth=2, embedding_dim=prompt_dim, num_heads=8, mlp_dim=2048,
    )
    decoder = FocalMaskDecoder(
        transformer_dim=prompt_dim,
        transformer=focal_transformer,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        num_classes=num_classes,
        focal_gamma=focal_gamma,
        class_pixel_frequencies=class_pixel_frequencies,
        learnable_bias=learnable_bias,
        delta_init_scale=delta_init_scale,
        use_prior_in_forward=use_prior_in_forward,
        warm_start=False,                    # deprecated --- always False
    )

    # Resize token heads to actual number of classes
    decoder.mask_tokens = nn.Embedding(num_classes + 1, prompt_dim)
    decoder.num_mask_tokens = num_classes + 1
    decoder.output_hypernetworks_mlps = nn.ModuleList([
        copy.deepcopy(decoder.output_hypernetworks_mlps[0])
        for _ in range(num_classes + 1)
    ])
    decoder.iou_prediction_head.layers[-1] = nn.Linear(prompt_dim, num_classes + 1)

    return decoder

# =============================================================================
# Standard Fusion
# =============================================================================

class PromptSAMFocalAttention(nn.Module):
    """SAMPath with Dynamic Focal Attention (DFA) --- Standard (early) fusion."""

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str = "",
        prompt_dim: int = 256,
        num_classes: int = 6,
        extra_encoder=None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        mask_HW=(1024, 1024),
        feature_input: bool = False,
        prompt_decoder: bool = False,
        dense_prompt_decoder: bool = False,
        no_sam=None,
        # ── DFA ───────────────────────────────────────────────────────────
        focal_gamma: float = 2.0,
        class_pixel_frequencies: dict = None,
        learnable_bias: bool = True,
        # warm-start scale for δ_c  (0.0 = cold start)
        delta_init_scale: float = 0.1,
        # Forward formula switch:
        #   True  → b_c = log(π_c) + δ_c  (prior + learned residual)
        #   False → b_c = δ_c             (fully learned, no prior at inference)
        use_prior_in_forward: bool = True,
        # Legacy / deprecated
        class_dice_scores: dict = None,
        warm_start: bool = False,
        bias_init_value: float = 0.0,
    ):
        super().__init__()

        self.model         = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mask_HW       = mask_HW
        self.feature_input = feature_input
        self.extra_encoder = extra_encoder
        self.no_sam        = no_sam

        # Backward compat
        if class_pixel_frequencies is None and class_dice_scores is not None:
            print("\n[WARNING] 'class_dice_scores' is deprecated. "
                  "Use 'class_pixel_frequencies'.")
            class_pixel_frequencies = class_dice_scores

        if class_pixel_frequencies is None:
            class_pixel_frequencies = {i: 1.0 / num_classes for i in range(num_classes)}

        # Install focal decoder
        formula_str = ("b_c = log(π_c) + δ_c  [prior + residual]"
                       if use_prior_in_forward else
                       "b_c = δ_c             [fully learned]")
        print("\n" + "=" * 70)
        print(f"  Installing DFA Focal Decoder  [{'LEARNABLE' if learnable_bias else 'FIXED'}]")
        print(f"  Formula: {formula_str}")
        print("=" * 70)
        self.model.mask_decoder = _build_focal_decoder(
            num_classes, prompt_dim, focal_gamma,
            class_pixel_frequencies, learnable_bias, delta_init_scale,
            use_prior_in_forward=use_prior_in_forward,
        )
        print("  ✓ Done\n" + "=" * 70 + "\n")

        if freeze_image_encoder:
            for p in self.model.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.model.prompt_encoder.parameters():
                p.requires_grad = False
        if freeze_mask_decoder:
            for p in self.model.mask_decoder.parameters():
                p.requires_grad = False

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            dl = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(dl, num_layers=1)

    # ── DFA accessors (delegated to decoder) ──────────────────────────────
    def get_bias_param(self):
        """δ_c parameter --- for optimizer param group + L2 reg."""
        return self.model.mask_decoder.get_bias_param()

    def get_focal_bias(self):
        """Total bias b_c = δ_c."""
        return self.model.mask_decoder.get_focal_bias()

    def get_bias_decomposition(self):
        """Detached CPU dict for plotting."""
        return self.model.mask_decoder.get_bias_decomposition()

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, images):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(
                    images, (1024, 1024), mode="bilinear", align_corners=False)

        if not self.no_sam:
            with torch.no_grad():
                image_embeddings = self.model.image_encoder(images)

        if self.extra_encoder is not None:
            extra_emb = self.extra_encoder(images)
            image_embeddings = (extra_emb if self.no_sam
                                else image_embeddings + extra_emb)

        pred_masks, ious = [], []
        for emb in image_embeddings:
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None, boxes=None, masks=None)

            if self.dense_prompt_decoder is not None:
                emb_img      = emb.flatten(1).permute(1, 0)
                sparse_v     = self.model.mask_decoder.mask_tokens.weight.clone()
                org_shape    = dense_emb.shape
                dense_gen    = self.dense_prompt_decoder(emb_img, sparse_v)
                dense_gen    = dense_gen.permute(1, 0).reshape(*org_shape)
                dense_emb    = dense_emb + dense_gen

            low_res, iou_pred = self.model.mask_decoder(
                image_embeddings=emb.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            pred_masks.append(
                F.interpolate(low_res, (H, W),
                              mode="bilinear", align_corners=False).squeeze(0))
            ious.append(iou_pred.reshape(-1, 1))

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)

# =============================================================================
# Late Fusion
# =============================================================================

class PromptSAMLateFusionFocalAttention(nn.Module):
    """SAMPath Late Fusion with Dynamic Focal Attention (DFA)."""

    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint: str = "",
        prompt_dim: int = 256,
        num_classes: int = 6,
        extra_encoder=None,
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        mask_HW=(1024, 1024),
        feature_input: bool = False,
        prompt_decoder: bool = False,
        dense_prompt_decoder: bool = False,
        no_sam=None,
        # ── DFA ───────────────────────────────────────────────────────────
        focal_gamma: float = 2.0,
        class_pixel_frequencies: dict = None,
        learnable_bias: bool = True,
        # warm-start scale for δ_c  (0.0 = cold start)
        delta_init_scale: float = 0.1,
        # Forward formula switch:
        #   True  → b_c = log(π_c) + δ_c  (prior + learned residual)
        #   False → b_c = δ_c             (fully learned, no prior at inference)
        use_prior_in_forward: bool = True,
        # Legacy / deprecated
        class_dice_scores: dict = None,
        warm_start: bool = False,
        bias_init_value: float = 0.0,
    ):
        super().__init__()

        self.model         = sam_model_registry[model_type](checkpoint=checkpoint)
        self.mask_HW       = mask_HW
        self.feature_input = feature_input
        self.extra_encoder = extra_encoder

        # Backward compat
        if class_pixel_frequencies is None and class_dice_scores is not None:
            print("\n[WARNING] 'class_dice_scores' is deprecated. "
                  "Use 'class_pixel_frequencies'.")
            class_pixel_frequencies = class_dice_scores

        if class_pixel_frequencies is None:
            class_pixel_frequencies = {i: 1.0 / num_classes for i in range(num_classes)}

        # Install focal decoder
        formula_str = ("b_c = log(π_c) + δ_c  [prior + residual]"
                       if use_prior_in_forward else
                       "b_c = δ_c             [fully learned]")
        print("\n" + "=" * 70)
        print(f"  Installing DFA Focal Decoder "
              f"[{'LEARNABLE' if learnable_bias else 'FIXED'}] --- Late Fusion")
        print(f"  Formula: {formula_str}")
        print("=" * 70)
        self.model.mask_decoder = _build_focal_decoder(
            num_classes, prompt_dim, focal_gamma,
            class_pixel_frequencies, learnable_bias, delta_init_scale,
            use_prior_in_forward=use_prior_in_forward,
        )
        print("  ✓ Done\n" + "=" * 70 + "\n")

        if freeze_image_encoder:
            for p in self.model.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.model.prompt_encoder.parameters():
                p.requires_grad = False
        if freeze_mask_decoder:
            for p in self.model.mask_decoder.parameters():
                p.requires_grad = False

        # Late-fusion neck: concatenates SAM + HIPT embeddings
        self.fusion_neck = nn.Sequential(
            nn.Conv2d(768 + 384, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

        self.dense_prompt_decoder = None
        if dense_prompt_decoder:
            dl = nn.TransformerDecoderLayer(d_model=prompt_dim, nhead=8)
            self.dense_prompt_decoder = nn.TransformerDecoder(dl, num_layers=1)

    # ── DFA accessors ──────────────────────────────────────────────────────
    def get_bias_param(self):
        return self.model.mask_decoder.get_bias_param()

    def get_focal_bias(self):
        """Total bias b_c = δ_c."""
        return self.model.mask_decoder.get_focal_bias()

    def get_bias_decomposition(self):
        return self.model.mask_decoder.get_bias_decomposition()

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, images):
        H, W = self.mask_HW

        if not self.feature_input:
            if images.shape[-2] != 1024 or images.shape[-1] != 1024:
                images = F.interpolate(
                    images, (1024, 1024), mode="bilinear", align_corners=False)

        with torch.no_grad():
            image_embeddings = self.model.image_encoder(images, no_neck=True)

        if self.extra_encoder is not None:
            ex = self.extra_encoder(images)
            ex = ex.reshape(ex.shape[0], 64, 64, -1)
            image_embeddings = torch.cat((image_embeddings, ex), dim=-1)

        image_embeddings = self.fusion_neck(image_embeddings.permute(0, 3, 1, 2))

        pred_masks, ious = [], []
        for emb in image_embeddings:
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=None, boxes=None, masks=None)

            if self.dense_prompt_decoder is not None:
                emb_img   = emb.flatten(1).permute(1, 0)
                sparse_v  = self.model.mask_decoder.mask_tokens.weight.clone()
                org_shape = dense_emb.shape
                dense_gen = self.dense_prompt_decoder(emb_img, sparse_v)
                dense_gen = dense_gen.permute(1, 0).reshape(*org_shape)
                dense_emb = dense_emb + dense_gen

            low_res, iou_pred = self.model.mask_decoder(
                image_embeddings=emb.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            pred_masks.append(
                F.interpolate(low_res, (H, W),
                              mode="bilinear", align_corners=False).squeeze(0))
            ious.append(iou_pred.reshape(-1, 1))

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)