"""Altered from -- Cross-Image Attention for Zero-Shot Appearance Transfer
"""
from typing import Optional

import timm
import torch 
import kornia
import torch.nn as nn
import numpy as np 
import torchvision
import torch.nn.functional as F

from diffusers.models.attention import Attention
import math
import torch

OUT_INDEX = 0


def should_mix_keys_and_values(model, hidden_states: torch.Tensor) -> bool:
    """ Verify whether we should perform the mixing in the current timestep. """
    is_in_32_timestep_range = (
            model.config.cross_attn_32_range.start <= model.step < model.config.cross_attn_32_range.end
    )
    is_in_64_timestep_range = (
            model.config.cross_attn_64_range.start <= model.step < model.config.cross_attn_64_range.end
    )
    is_hidden_states_32_square = (hidden_states.shape[1] == 32 ** 2)
    is_hidden_states_64_square = (hidden_states.shape[1] == 64 ** 2)
    should_mix = (is_in_32_timestep_range and is_hidden_states_32_square) or \
                 (is_in_64_timestep_range and is_hidden_states_64_square)
    return should_mix


def compute_scaled_dot_product_attention(Q, K, V, edit_map=False, is_cross=False, contrast_strength=1.0):
    """ Compute the scale dot product attention, potentially with our contrasting operation. """
    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
    if edit_map and not is_cross:
        attn_weight[OUT_INDEX] = torch.stack([
            torch.clip(enhance_tensor(attn_weight[OUT_INDEX][head_idx], contrast_factor=contrast_strength),
                       min=0.0, max=1.0)
            for head_idx in range(attn_weight.shape[1])
        ])
    return attn_weight @ V, attn_weight


def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    adjusted_tensor = (tensor - tensor.mean(dim=-1)) * contrast_factor + tensor.mean(dim=-1)
    return adjusted_tensor


class AttentionGuidance(nn.Module):
    def __init__(self, query_dim, cross_attention_dim, heads=8, dim_head=64,):
        super().__init__()
        self.attn = Attention(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            heads=heads,
            dim_head=dim_head,
            # only_cross_attention=True,
            processor=None,
            # out_dim=64
            residual_connection=False,
        ).to('cuda')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask=None,
    ):
        residual = hidden_states

        if self.attn.spatial_norm is not None:
            hidden_states = self.attn.spatial_norm(hidden_states, temb)

        input_ndim = len(hidden_states.shape)
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.attn.heads, -1, attention_mask.shape[-1])

        if self.attn.group_norm is not None:
            hidden_states = self.attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if not is_cross:
            encoder_hidden_states = hidden_states
        elif self.attn.norm_cross:
            encoder_hidden_states = self.attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.attn.to_k(encoder_hidden_states)
        value = self.attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.attn.heads
        should_mix = False

        # Potentially apply our cross image attention operation
        # To do so, we need to be in a self-attention alyer in the decoder part of the denoising network
        # if perform_swap and not is_cross and "up" in self.place_in_unet and model_self.enable_edit:
        #     if should_mix_keys_and_values(model_self, hidden_states):
        #         should_mix = True
        #         if model_self.step % 5 == 0 and model_self.step < 40:
        #             # Inject the structure's keys and values
        #             key[OUT_INDEX] = key[STRUCT_INDEX]
        #             value[OUT_INDEX] = value[STRUCT_INDEX]
        #         else:
        #             # Inject the appearance's keys and values
        #             key[OUT_INDEX] = key[STYLE_INDEX]
        #             value[OUT_INDEX] = value[STYLE_INDEX]

        query = query.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.attn.heads, head_dim).transpose(1, 2)

        # Compute the cross attention and apply our contrasting operation
        hidden_states, attn_weight = compute_scaled_dot_product_attention(
            query, key, value,
            edit_map=False,
            is_cross=is_cross,
            # contrast_strength=model_self.config.contrast_strength,
        )

        # Update attention map for segmentation
        # if model_self.config.use_masked_adain and model_self.step == model_self.config.adain_range.start - 1:
        #     model_self.segmentor.update_attention(attn_weight, is_cross)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.attn.heads * head_dim)
        hidden_states = hidden_states.to(query[OUT_INDEX].dtype)

        # linear proj
        hidden_states = self.attn.to_out[0](hidden_states)
        # dropout
        hidden_states = self.attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.attn.rescale_output_factor

        return hidden_states


