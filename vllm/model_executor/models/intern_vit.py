# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# adapted from https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/modeling_intern_vit.py
# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from collections.abc import Iterable
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.attention.layer import MultiHeadAttention
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .vision import run_dp_sharded_vision_model

NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': nn.LayerNorm,
}
_MM_CLS2PATCH_ANYRES_PRUNE_ENV = (
    os.getenv("VLLM_MM_CLS2PATCH_ANYRES_PRUNE", "0") == "1")


class InternVisionEmbeddings(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size,
            self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed,
                                  size=(H, W),
                                  mode='bicubic',
                                  align_corners=False)
        return pos_embed.reshape(1, -1, H * W).permute(0, 2,
                                                       1).to(target_dtype)

    def _get_position_embedding(self, H: int, W: int) -> torch.Tensor:
        position_embedding = self.position_embedding
        if self.num_patches == H * W:
            return position_embedding

        return torch.cat(
            [
                position_embedding[:, :1, :],
                self._get_pos_embed(position_embedding[:, 1:, :], H, W),
            ],
            dim=1,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(
            target_dtype))  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1,
                                                   -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = self._get_position_embedding(height, width)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternVisionPatchModel(nn.Module):

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embeddings = InternVisionEmbeddings(config)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                'You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f'wrong pixel_values size: {pixel_values.shape}')

        return hidden_states


class InternParallelAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads '
                f'(got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        self.tp_size = (1 if use_data_parallel else
                        get_tensor_model_parallel_world_size())
        self.tp_rank = (0 if use_data_parallel else
                        get_tensor_model_parallel_rank())

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + self.num_heads) * self.head_dim
        self.num_heads_per_partition = divide(num_dummy_heads + self.num_heads,
                                              self.tp_size)

        self.scale = self.head_dim**-0.5
        self.qkv = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            num_dummy_heads + self.num_heads,
            bias=config.qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)
            self.k_norm = RMSNorm(self.dummy_dim,
                                  eps=config.layer_norm_eps,
                                  var_hidden_size=self.embed_dim)

        self.proj = RowParallelLinear(
            self.dummy_dim,
            self.embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        self.attn = MultiHeadAttention(self.num_heads_per_partition,
                                       self.head_dim, self.scale)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim,
                               num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv, _ = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if self.qk_normalization:
            q, k = self._apply_qk_norm(q, k)

        out = self.attn(q, k, v)
        out, _ = self.proj(out)
        return out


class InternMLP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = ColumnParallelLinear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.fc1",
                                        disable_tp=use_data_parallel)
        self.fc2 = RowParallelLinear(config.intermediate_size,
                                     config.hidden_size,
                                     bias=True,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.fc2",
                                     disable_tp=use_data_parallel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)

        return hidden_states


class InternVisionEncoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = self._init_attn(config,
                                    quant_config,
                                    num_dummy_heads=num_dummy_heads,
                                    prefix=f"{prefix}.attn",
                                    use_data_parallel=use_data_parallel)

        self.mlp = InternMLP(config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.mlp",
                             use_data_parallel=use_data_parallel)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim,
                                             eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor *
                                torch.ones(self.embed_dim))

    def _init_attn(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig],
        *,
        num_dummy_heads: int,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        # fallback to sdpa attention if tp unavailable
        tp_size = (1 if use_data_parallel else
                   get_tensor_model_parallel_world_size())
        num_heads = config.num_attention_heads

        # if the number of heads is not divisible by tp_size,
        # we also disable Attention's TP
        use_data_parallel = (use_data_parallel
                             or (num_heads + num_dummy_heads) % tp_size != 0)
        return InternParallelAttention(config,
                                       quant_config=quant_config,
                                       num_dummy_heads=num_dummy_heads,
                                       prefix=prefix,
                                       use_data_parallel=use_data_parallel)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states)) * self.ls1

        hidden_states = hidden_states + self.mlp(
            self.norm2(hidden_states)) * self.ls2

        return hidden_states


class InternVisionEncoder(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config,
                                     quant_config,
                                     num_dummy_heads=num_dummy_heads,
                                     prefix=f"{prefix}.layers.{layer_idx}",
                                     use_data_parallel=use_data_parallel)
            for layer_idx in range(num_hidden_layers)
        ])
        self._mm_last_cls2patch = None

    def _compute_cls2patch(
        self,
        encoder_layer: InternVisionEncoderLayer,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        normed_states = encoder_layer.norm1(hidden_states)
        qkv_states, _ = encoder_layer.attn.qkv(normed_states)
        query_states, key_states, _ = qkv_states.chunk(3, dim=-1)

        if encoder_layer.attn.qk_normalization:
            query_states, key_states = encoder_layer.attn._apply_qk_norm(
                query_states,
                key_states,
            )

        bsz, seq_len, _ = query_states.shape
        num_heads = encoder_layer.attn.num_heads_per_partition
        head_dim = encoder_layer.attn.head_dim
        query_states = query_states.view(
            bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, seq_len, num_heads, head_dim).transpose(1, 2)
        q_cls = query_states[:, :, 0:1, :]
        attn_logits = torch.matmul(q_cls, key_states.transpose(-2, -1)).squeeze(2)
        attn_logits = attn_logits * encoder_layer.attn.scale
        attn_probs = torch.softmax(attn_logits.float(), dim=-1)
        cls2patch = attn_probs[:, :, 1:]
        return cls2patch.mean(dim=1)

    def forward(self, inputs_embeds: torch.Tensor):

        hidden_states = inputs_embeds
        self._mm_last_cls2patch = None
        target_layer_idx = len(self.layers) - 1
        for layer_idx, encoder_layer in enumerate(self.layers):
            if _MM_CLS2PATCH_ANYRES_PRUNE_ENV and layer_idx == target_layer_idx:
                self._mm_last_cls2patch = self._compute_cls2patch(
                    encoder_layer,
                    hidden_states,
                )
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class InternVisionModel(nn.Module):

    packed_modules_mapping = {
        "qkv": ["qkv"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        num_hidden_layers_override: Optional[int] = None,
        num_dummy_heads: int = 0,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.use_data_parallel = use_data_parallel

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(
            config=config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            num_dummy_heads=num_dummy_heads,
            prefix=f"{prefix}.encoder",
            use_data_parallel=use_data_parallel,
        )
        self._mm_last_cls2patch = None

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_embeds: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if pixel_values is None and pixel_embeds is None:
            raise ValueError(
                'You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        elif pixel_values is not None:
            if pixel_values.ndim == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(
                    f'wrong pixel_values size: {pixel_values.shape}')

        if self.use_data_parallel:
            encoder_outputs = run_dp_sharded_vision_model(
                hidden_states, self.encoder)
        else:
            encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        self._mm_last_cls2patch = getattr(
            self.encoder,
            "_mm_last_cls2patch",
            None,
        )

        return encoder_outputs

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
