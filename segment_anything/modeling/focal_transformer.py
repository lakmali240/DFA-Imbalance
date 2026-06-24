# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by: Lakmali Nadeesha - Added Focal Class Attention
# All rights reserved.

"""
Focal Class Attention Transformer

This module modifies SAM's TwoWayTransformer to apply focal attention bias,
which amplifies attention toward difficult classes.

Key Innovation:
- Standard attention: attn = softmax(Q·K^T / √d)
- Focal attention:    attn = softmax(Q·K^T / √d + focal_bias)

The focal_bias is computed as log((1 - Dice_c)^γ) for each class c.
Adding log(weight) before softmax is mathematically equivalent to
multiplying attention probabilities by weight after softmax.
"""

import torch
from torch import Tensor, nn
import math
from typing import Tuple, Type, Optional
from .common import MLPBlock


class FocalTwoWayTransformer(nn.Module):
    """
    Two-Way Transformer with Focal Class Attention.
    
    This is a drop-in replacement for the standard TwoWayTransformer
    that applies focal attention bias to boost hard classes.
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Args:
            depth: Number of transformer layers
            embedding_dim: Channel dimension for embeddings
            num_heads: Number of attention heads
            mlp_dim: Hidden dimension for MLP blocks
            activation: Activation function for MLP
            attention_downsample_rate: Downsample rate for attention
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                FocalTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = FocalAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        focal_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with optional focal attention bias.
        
        Args:
            image_embedding: [B, C, H, W] image features
            image_pe: [B, C, H, W] positional encoding
            point_embedding: [B, N, C] point/prompt embeddings
            focal_bias: [num_classes] log-space focal bias for each class
            
        Returns:
            queries: Updated point embeddings
            keys: Updated image embeddings
        """
        # Flatten: [B, C, H, W] -> [B, HW, C]
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries and keys
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks with focal attention
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                focal_bias=focal_bias,
            )

        # Final attention from tokens to image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(
            q=q, k=k, v=keys,
            focal_bias=focal_bias,
        )
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class FocalTwoWayAttentionBlock(nn.Module):
    """
    Transformer block with focal attention for cross-attention layers.
    
    Structure:
    1. Self-attention on sparse inputs (class prompts)
    2. Cross-attention: tokens -> image (WITH FOCAL BIAS)
    3. MLP block
    4. Cross-attention: image -> tokens (WITH FOCAL BIAS)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        
        # Self attention (no focal bias needed - operates on tokens only)
        self.self_attn = FocalAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Cross attention: tokens attending to image
        self.cross_attn_token_to_image = FocalAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # MLP
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        # Cross attention: image attending to tokens
        self.cross_attn_image_to_token = FocalAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
        focal_bias: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass with focal attention.
        
        Args:
            queries: [B, N_tokens, C] class prompt tokens
            keys: [B, N_image, C] image features
            query_pe: [B, N_tokens, C] positional encoding for tokens
            key_pe: [B, N_image, C] positional encoding for image
            focal_bias: [num_classes] focal attention bias
        """
        # 1. Self attention on queries (class prompts)
        # No focal bias here - this is token self-attention
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # 2. Cross attention: tokens attending to image
        # APPLY FOCAL BIAS HERE - boost attention FROM hard class tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(
            q=q, k=k, v=keys,
            focal_bias=focal_bias,
            apply_to_query=True,  # Bias based on which token (class) is querying
        )
        queries = queries + attn_out
        queries = self.norm2(queries)

        # 3. MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # 4. Cross attention: image attending to tokens
        # APPLY FOCAL BIAS HERE - boost attention TO hard class tokens
        q = keys + key_pe
        k = queries + query_pe
        attn_out = self.cross_attn_image_to_token(
            q=q, k=k, v=queries,
            focal_bias=focal_bias,
            apply_to_query=False,  # Bias based on which token (class) is being attended to
        )
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class FocalAttention(nn.Module):
    """
    Multi-head attention with optional focal class bias.
    
    This is the core innovation: we add a learned bias to attention scores
    that amplifies attention for difficult classes.
    
    Mathematical formulation:
    - Standard: attn = softmax(Q·K^T / √d)
    - Focal:    attn = softmax(Q·K^T / √d + bias)
    
    Where bias[i] = log((1 - Dice_i)^γ)
    
    This is equivalent to multiplying attention weights by (1-Dice)^γ after softmax,
    but numerically more stable.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """Reshape: [B, N, C] -> [B, num_heads, N, C/num_heads]"""
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """Reshape: [B, num_heads, N, C/num_heads] -> [B, N, C]"""
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        focal_bias: Optional[Tensor] = None,
        apply_to_query: bool = True,
    ) -> Tensor:
        """
        Forward pass with optional focal attention bias.
        
        Args:
            q: [B, N_q, C] queries
            k: [B, N_k, C] keys
            v: [B, N_k, C] values
            focal_bias: [num_classes] log-space bias for each class
            apply_to_query: If True, bias query dimension (token->image attention)
                           If False, bias key dimension (image->token attention)
                           
        Returns:
            out: [B, N_q, C] attention output
        """
        # Project inputs
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Compute attention scores: [B, num_heads, N_q, N_k]
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)

        # ================================================================
        # FOCAL ATTENTION BIAS - This is the key innovation!
        # ================================================================
        if focal_bias is not None:
            B, num_heads, N_q, N_k = attn.shape
            num_classes = focal_bias.shape[0]
            
            if apply_to_query:
                # Token -> Image attention
                # Bias based on which CLASS TOKEN is doing the querying
                # focal_bias: [num_classes] -> [1, 1, num_classes, 1]
                if N_q <= num_classes + 2:  # +1 for iou_token, +1 for safety
                    # Queries are likely class tokens
                    # Skip the first token (iou_token) when applying bias
                    bias = torch.zeros(N_q, device=focal_bias.device)
                    tokens_to_bias = min(num_classes, N_q - 1)  # -1 for iou token
                    if tokens_to_bias > 0:
                        bias[1:1+tokens_to_bias] = focal_bias[:tokens_to_bias]
                    bias = bias.view(1, 1, -1, 1)
                    attn = attn + bias
            else:
                # Image -> Token attention
                # Bias based on which CLASS TOKEN is being attended TO
                # focal_bias: [num_classes] -> [1, 1, 1, num_classes]
                if N_k <= num_classes + 2:
                    # Keys are likely class tokens
                    bias = torch.zeros(N_k, device=focal_bias.device)
                    tokens_to_bias = min(num_classes, N_k - 1)
                    if tokens_to_bias > 0:
                        bias[1:1+tokens_to_bias] = focal_bias[:tokens_to_bias]
                    bias = bias.view(1, 1, 1, -1)
                    attn = attn + bias

        # Softmax and apply to values
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        
        # Recombine heads and project
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
