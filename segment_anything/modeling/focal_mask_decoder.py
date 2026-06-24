# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by: Lakmali Nadeesha --- Dynamic Focal Attention (DFA)

"""
Focal Mask Decoder --- Dynamic Focal Attention (DFA)

Formula:   b_c  =  δ_c
                   ────
                   Fully Learned Bias (nn.Parameter)

   (log(π_c) is used ONLY to warm-start δ_c; it is NOT added at inference.)

═══════════════════════════════════════════════════════════════════════════
Warm-Start δ_c Initialization  (applied in _setup_dfa_bias)
═══════════════════════════════════════════════════════════════════════════

PROBLEM:
  Cold-start (δ_c = 0) means all gradient signals start at zero through
  the softmax Jacobian path.  Since ∂ã/∂δ_c = ã·(1−ã), and ã is small
  for rare/hard classes at init, those classes receive essentially no
  gradient --- they can never escape the near-zero regime.

SOLUTION:
  δ_c_init  =  alpha · ( log(π_c) − mean(log(π_c)) )

  where alpha = delta_init_scale (default 0.1).

  This warm-start uses frequency information only as an INITIALIZER.
  After init, b_c = δ_c alone --- log(π_c) is NOT added at forward time.

  Set delta_init_scale=0.0 to revert to cold start.

═══════════════════════════════════════════════════════════════════════════
"""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Optional, Dict

from .common import LayerNorm2d
from .focal_transformer import FocalTwoWayTransformer


class FocalMaskDecoder(nn.Module):
    """
    SAM Mask Decoder with Dynamic Focal Attention (DFA).

    b_c = δ_c

    └── δ_c  --- nn.Parameter, warm-start initialised from log(π_c)
                 (log(π_c) is used for init only, NOT added at forward time)
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        # ── DFA core ──────────────────────────────────────────────────────────
        num_classes: int = 6,
        focal_gamma: float = 2.0,
        class_pixel_frequencies: Optional[Dict] = None,
        learnable_bias: bool = True,
        # Warm-start scale for δ_c
        #   0.0  → cold start, all zeros  (old behaviour)
        #   0.1  → recommended warm start (10 % of prior spread)
        delta_init_scale: float = 0.1,
        # ── Forward-pass formula switch ────────────────────────────────────
        # True  → b_c = log(π_c) + δ_c   (prior anchors; δ_c learns residual)
        # False → b_c = δ_c              (δ_c carries full bias; ablation mode)
        # Both modes use identical warm-start init for δ_c.
        use_prior_in_forward: bool = True,
        # ── Legacy params (kept for backward compat) ──────────────────────
        class_dice_scores: Optional[Dict] = None,
        warm_start: bool = False,
        bias_init_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer     = transformer

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token   = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4,
                               kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8,
                               kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # ── DFA state ─────────────────────────────────────────────────────
        self.num_classes          = num_classes
        self.focal_gamma          = focal_gamma
        self.learnable_bias       = learnable_bias
        self.delta_init_scale     = delta_init_scale
        self.use_prior_in_forward = use_prior_in_forward

        # Backward compat: class_dice_scores → class_pixel_frequencies
        if class_pixel_frequencies is None:
            if class_dice_scores is not None:
                print("\n  [WARNING] 'class_dice_scores' is DEPRECATED for DFA.")
                print("  [WARNING] Use 'class_pixel_frequencies' with actual pixel frequencies.")
                class_pixel_frequencies = class_dice_scores
            else:
                class_pixel_frequencies = {i: 1.0 / num_classes for i in range(num_classes)}
                print("\n  [WARNING] No pixel frequencies provided --- using uniform distribution.")

        if learnable_bias:
            self._setup_dfa_bias(class_pixel_frequencies, num_classes,
                                 focal_gamma, delta_init_scale, bias_init_value)
        else:
            self._setup_fixed_bias(class_pixel_frequencies, num_classes, focal_gamma)

        self._print_config_table(class_pixel_frequencies)

    # =========================================================================
    # INITIALISATION HELPERS
    # =========================================================================

    def _compute_frequency_prior(
        self, class_pixel_freqs: Dict, num_classes: int, gamma: float
    ) -> torch.Tensor:
        """
        Compute log(π_c) = γ · log(1 − f_c).
        Used ONLY for warm-start initialisation of δ_c; never added at inference.

        Frequencies are used AS-IS — no normalization.
        Class 0 (Ignored/Outside ROI) must be 0.0 in config so it gets
        log(1 − 0) = 0 and is excluded from the centering mean in _setup_dfa_bias.
        Classes 1–N use their true pixel proportions directly from the dataset.
        """
        f_c = torch.zeros(num_classes)
        for i in range(num_classes):
            f_c[i] = float(class_pixel_freqs.get(i, 0.0))
        print(f"\n  [DFA] Using raw pixel frequencies (no normalization):")
        for i in range(num_classes):
            flag = " ← ignored (excluded from centering)" if f_c[i].item() == 0.0 and i == 0 else ""
            print(f"        class {i}: f_c = {f_c[i].item():.4f}{flag}")
        return gamma * torch.log(torch.clamp(1.0 - f_c, min=1e-8))

    def _setup_dfa_bias(
        self,
        class_pixel_freqs: Dict,
        num_classes: int,
        gamma: float,
        delta_init_scale: float,
        bias_init_value: float,
    ) -> None:
        """
        Initialise bias_param (δ_c) with warm-start formula.
        log(π_c) is stored only as a reference buffer; it is NOT used in forward.

        Warm-start:
          prior_centred = log(π_c) − mean(log(π_c))
          δ_c_init      = alpha × prior_centred

        This gives δ_c a non-zero start that captures relative frequency
        difficulty without locking in the prior as a fixed offset.
        """
        # Compute log_prior for init reference only
        log_prior = self._compute_frequency_prior(class_pixel_freqs, num_classes, gamma)
        # Store as a non-trainable reference buffer (for logging/plotting only)
        self.register_buffer('log_prior', log_prior)

        if delta_init_scale > 0.0:
            # Centre using ONLY active classes (index > 0, i.e. f_c > 0).
            # Class 0 (Ignored) has log(π_0) = 0 and is never updated by
            # gradients — including it in the mean would shift every active
            # class's warm-start in the wrong direction.
            active_mask   = torch.tensor(
                [class_pixel_freqs.get(i, 0.0) > 0.0 for i in range(num_classes)],
                dtype=torch.bool)
            active_mean   = log_prior[active_mask].mean()
            prior_centred = log_prior - active_mean   # centred over active classes only
            delta_init    = delta_init_scale * prior_centred

            # Force class 0 (Ignored) bias to exactly 0 — it is masked during
            # training so its gradient is always zero; a non-zero init is misleading.
            delta_init[0] = 0.0

            print(f"\n  [DFA] Warm-start δ_c: α·(log(π_c) − mean_active)  α={delta_init_scale}")
            print(f"  [DFA] Active class mean log(π_c): {active_mean.item():.4f}  "
                  f"(class 0 excluded from mean)")
            print(f"  [DFA] δ_c init: {delta_init.numpy().round(5).tolist()}")
            print(f"  [DFA] NOTE: b_c = δ_c only — log(π_c) is NOT added at inference.")
        else:
            delta_init = torch.full((num_classes,), float(bias_init_value))
            delta_init[0] = 0.0   # always zero for ignored class
            print(f"\n  [DFA] Cold-start δ_c: all {bias_init_value} (class 0 forced to 0)")

        self.bias_param = nn.Parameter(delta_init.clone())
        # focal_bias snapshot: just δ_c (no prior added)
        self.register_buffer('focal_bias', delta_init.clone())

        print(f"  [DFA] log(π_c) range (init ref only): "
              f"[{log_prior.min().item():.4f}, {log_prior.max().item():.4f}]")

    def _setup_fixed_bias(
        self, class_pixel_freqs: Dict, num_classes: int, gamma: float
    ) -> None:
        """CFFA ablation: b_c = γ·log(1−f_c), no learning."""
        log_prior = self._compute_frequency_prior(class_pixel_freqs, num_classes, gamma)
        self.register_buffer('focal_bias', log_prior)
        self.register_buffer('log_prior',  log_prior.clone())
        self.bias_param = None
        print(f"\n  [CFFA] Fixed b_c = γ·log(1−f_c), γ={gamma}. No δ_c learning.")

    # =========================================================================
    # GETTERS --- used by pl_module and optimizer
    # =========================================================================

    def get_focal_bias(self) -> torch.Tensor:
        """
        Total bias used in every forward pass.

        use_prior_in_forward=True  → b_c = log(π_c) + δ_c
            Prior anchors the bias at a frequency-based value.
            δ_c only learns the small residual correction.
            More stable; less sensitive to lr tuning.

        use_prior_in_forward=False → b_c = δ_c
            δ_c carries the full bias from scratch.
            log(π_c) used only at warm-start init, never at inference.
        """
        if self.learnable_bias and self.bias_param is not None:
            if self.use_prior_in_forward:
                return self.log_prior + self.bias_param   # b_c = log(π_c) + δ_c
            else:
                return self.bias_param                    # b_c = δ_c only
        return self.focal_bias

    def get_bias_param(self) -> Optional[torch.Tensor]:
        """δ_c parameter --- used by optimizer for separate lr group + L2 reg."""
        return self.bias_param

    def get_delta_c(self) -> Optional[torch.Tensor]:
        """δ_c (live, attached to graph)."""
        return self.bias_param if self.learnable_bias else None

    def get_log_prior(self) -> Optional[torch.Tensor]:
        """Frozen log(π_c) buffer (init reference only, not used in forward)."""
        return self.log_prior if hasattr(self, 'log_prior') else None

    def get_bias_decomposition(self) -> Dict:
        """Detached CPU dict for plotting --- called every epoch end."""
        dc = self.get_delta_c()
        tb = self.get_focal_bias()          # = δ_c (total bias is δ_c alone)
        lp = self.get_log_prior()           # reference only, not part of forward
        return {
            'log_prior':  lp.detach().cpu() if lp is not None else None,  # ref only
            'delta_c':    dc.detach().cpu() if dc is not None else None,
            'total_bias': tb.detach().cpu() if tb is not None else None,  # = δ_c
        }

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _print_config_table(self, class_pixel_freqs: Dict) -> None:
        print("\n" + "═" * 78)
        if self.learnable_bias:
            print("  DYNAMIC FOCAL ATTENTION (DFA)")
            print("  Formula : b_c = δ_c   (fully learned; log(π_c) used for warm-start ONLY)")
            if self.delta_init_scale > 0:
                print(f"  δ_c init: warm-start  α = {self.delta_init_scale}")
            else:
                print("  δ_c init: cold-start (zeros)")
        else:
            print("  CLASS FREQUENCY FOCAL ATTENTION (CFFA)  [fixed, no learning]")
            print("  Formula : b_c = γ·log(1−f_c)")
        print(f"  γ = {self.focal_gamma}   |   num_classes = {self.num_classes}   "
              f"|   learnable = {self.learnable_bias}")
        print("─" * 78)

        f_c = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            f_c[i] = float(class_pixel_freqs.get(i, 0.0))
        lp  = self.log_prior
        dc  = self.bias_param if self.bias_param is not None \
              else torch.zeros(self.num_classes)
        if self.learnable_bias and self.use_prior_in_forward:
            tb = lp + dc   # b_c(init) = log(π_c) + δ_c(init)
            bc_col_label = "b_c(init)=lp+δ_c"
        elif self.learnable_bias:
            tb = dc        # b_c(init) = δ_c(init)
            bc_col_label = "b_c(init)=δ_c"
        else:
            tb = lp
            bc_col_label = "b_c(fixed)=log(π_c)"

        print(f"  {'Cls':>4}  {'f_c':>8}  {'log(π_c)':>10}  "
              f"{'δ_c(init)':>12}  {bc_col_label:>18}")
        print("─" * 78)
        for i in range(self.num_classes):
            print(f"  {i:>4}  {f_c[i].item():>8.4f}  "
                  f"{lp[i].item():>+10.4f}  "
                  f"{dc[i].item():>+12.5f}  "
                  f"{tb[i].item():>+18.5f}")
        print("═" * 78 + "\n")

    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ):
        masks, iou_pred = self.predict_masks(
            image_embeddings, image_pe,
            sparse_prompt_embeddings, dense_prompt_embeddings,
        )
        sl = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, sl, :, :], iou_pred[:, sl]

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src     = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src     = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # b_c = δ_c  --- computed fresh each forward (δ_c is live)
        current_bias = self.get_focal_bias()

        hs, src = self.transformer(src, pos_src, tokens, focal_bias=current_bias)
        iou_token_out   = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        src      = src.transpose(1, 2).view(b, c, h, w)
        upscaled = self.output_upscaling(src)

        hyper_in = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ], dim=1)

        b, c, h, w = upscaled.shape
        masks    = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# =============================================================================
# MLP helper (unchanged from SAM)
# =============================================================================

class MLP(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        num_layers: int, sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x








