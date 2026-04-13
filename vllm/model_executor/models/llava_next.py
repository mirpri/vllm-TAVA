# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import (Annotated, Final, Literal, Optional, Protocol, TypeVar,
                    Union)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, LlavaNextConfig, LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape, unpad_image)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.processing import PromptReplacement, PromptUpdate
from vllm.mm_trace import get_mm_trace_state
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.logger import init_logger

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .llava import (BaseLlavaMultiModalProcessor, BaseLlavaProcessingInfo,
                    LlavaDummyInputsBuilder, LlavaLikeConfig,
                    LlavaMultiModalProjector, init_vision_tower_for_llava)
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, WeightsMapper, embed_multimodal,
                    flatten_bn, init_vllm_registered_model, maybe_prefix)


logger = init_logger(__name__)
DEBUG = os.getenv("LLAVA_NEXT_DEBUG") == "1"
_BASE_ONLY_ENV = os.getenv("VLLM_LLAVA_NEXT_BASE_ONLY", "0") == "1"
_MM_CLS2PATCH_ANYRES_TRACE_ENV = (
    os.getenv("VLLM_MM_CLS2PATCH_ANYRES_TRACE", "0") == "1")
#保留tiles + global，但对每个patches-image各自topk
_MM_CLS2PATCH_ANYRES_PRUNE_ENV = (
    os.getenv("VLLM_MM_CLS2PATCH_ANYRES_PRUNE", "0") == "1")
_MM_CLS2PATCH_ANYRES_PRUNE_LOG_TILES_ENV = (
    os.getenv("VLLM_MM_CLS2PATCH_ANYRES_PRUNE_LOG_TILES", "0") == "1")


def _get_mm_cls2patch_topk() -> int:
    text = os.getenv("VLLM_MM_CLS2PATCH_TOPK", "0")
    try:
        value = int(text)
    except ValueError:
        return 0
    return max(0, value)


def _get_mm_cls2patch_prune_log_topk() -> int:
    text = os.getenv("VLLM_MM_CLS2PATCH_ANYRES_PRUNE_LOG_TOPK")
    if not text:
        return 8
    try:
        value = int(text)
    except ValueError:
        return 8
    return max(1, value)


def _get_mm_cls2patch_tiles_topp() -> float:
    text = os.getenv("VLLM_MM_CLS2PATCH_TILES_TOPP", "")
    if not text:
        return 1.0
    try:
        value = float(text)
    except ValueError:
        return 1.0
    if value <= 0:
        return 0.0
    return min(value, 1.0)


def _get_mm_prune_score_mode() -> str:
    text = os.getenv("VLLM_MM_PRUNE_SCORE_MODE", "cls2patch")
    mode = text.strip().lower() if text else "cls2patch"
    if mode not in ("cls2patch", "relevance", "fusion"):
        print("没有选中正确的剪枝模式！！，默认使用 cls2patch")
        return "cls2patch"
    return mode


def _get_mm_prune_fusion_alpha() -> float:
    text = os.getenv("VLLM_MM_PRUNE_FUSION_ALPHA", "0.5")
    try:
        value = float(text)
    except ValueError:
        return 0.5
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _get_mm_anyres_budget_mode() -> str:
    text = os.getenv("VLLM_MM_ANYRES_BUDGET_MODE", "default")
    mode = text.strip().lower() if text else "default"
    if mode not in ("default", "relevance"):
        logger.warning(
            "[MMAnyresBudget] unknown mode=%s, fallback to default", mode)
        return "default"
    return mode


def _get_mm_anyres_image_prune_ratio() -> Optional[float]:
    text = os.getenv("VLLM_MM_ANYRES_IMAGE_PRUNE_RATIO", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _get_request_prune_ratio(
    hf_processor_mm_kwargs: Optional[Mapping[str, object]],
) -> Optional[float]:
    if not hf_processor_mm_kwargs or "prune_ratio" not in hf_processor_mm_kwargs:
        return None
    raw_value = hf_processor_mm_kwargs["prune_ratio"]
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid request prune_ratio in mm_processor_kwargs: {raw_value!r}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Request prune_ratio out of range in mm_processor_kwargs: {value}"
        )
    return value


def _get_mm_anyres_zone_budget_tau() -> float:
    text = os.getenv("VLLM_MM_ANYRES_ZONE_BUDGET_TAU", "1.0")
    try:
        value = float(text)
    except ValueError:
        return 1.0
    if value <= 0.0:
        return 1.0
    return value


def _get_mm_anyres_budget_log_enabled() -> bool:
    return os.getenv("VLLM_MM_ANYRES_BUDGET_LOG", "0") == "1"


def _compute_anyres_unpad_window( #
    *,
    orig_height: int,
    orig_width: int,
    num_patch_height: int,
    num_patch_width: int,
    grid_len: int,
) -> tuple[int, int, int, int, int, int, int, int, str, int]:
    base_height = grid_len * num_patch_height #24*2
    base_width = grid_len * num_patch_width #24*2
    current_height = base_height
    current_width = base_width
    #根据原图宽高比计算padding
    aspect_ratio = orig_width / orig_height
    current_aspect_ratio = current_width / current_height
    branch = "width"
    new_height = None
    new_width = None
    padding = 0
    if aspect_ratio > current_aspect_ratio:
         # 原图更宽 → 高度方向有 padding
        new_height = int(
            round(orig_height * (current_width / orig_width), 7))
        padding = (current_height - new_height) // 2
        current_height = current_height - (2 * padding)
        branch = "height"
    else:
        # 原图更高 → 宽度方向有 padding
        new_width = int(
            round(orig_width * (current_height / orig_height), 7))
        padding = (current_width - new_width) // 2
        current_width = current_width - (2 * padding)

    if branch == "height":
        row_min = padding
        row_max = padding + current_height
        col_min = 0
        col_max = base_width
    else:
        row_min = 0
        row_max = base_height
        col_min = padding
        col_max = padding + current_width

    return (base_height, base_width, current_height, current_width, row_min,
            row_max, col_min, col_max, branch, padding)


class LlavaNextImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - np: Number of patches + 1
        - c: Number of channels (3)
        - h: Height
        - w: Width
    
    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """
    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("bn", "np", 3, "h", "w", dynamic_dims={"np"})]

    image_sizes: Annotated[Optional[torch.Tensor], TensorShape("bn", 2)]
    # This should be in `(height, width)` format.


class LlavaNextImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - ifs: Image feature size
        - hs: Hidden size (must match language model backbone)
    """
    type: Literal["image_embeds"] = "image_embeds"
    data: Annotated[torch.Tensor, TensorShape("bn", "ifs", "hs")]


LlavaNextImageInputs = Union[LlavaNextImagePixelInputs,
                             LlavaNextImageEmbeddingInputs]


class LlavaNextLikeConfig(LlavaLikeConfig, Protocol):
    image_grid_pinpoints: Final[list[list[int]]]


class LlavaNextProcessingInfo(BaseLlavaProcessingInfo):

    def _get_num_image_tokens_unpruned(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        base_feature_size = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )
        if _BASE_ONLY_ENV:
            return base_feature_size

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_encoder_info.get_image_size(),
        )
        grid_length = vision_encoder_info.get_patch_grid_length()
        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=grid_length,
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        return (base_feature_size + unpadded_feature_size +
                newline_feature_size)

    def get_hf_config(self) -> LlavaNextLikeConfig:
        return self.ctx.get_hf_config(LlavaNextConfig)

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(LlavaNextProcessor, **kwargs)

        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size

        return hf_processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        if mm_counts.get("image", 0) == 0:
            return {"image": 0}

        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()
        base_feature_size = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=vision_encoder_info.get_image_size(),
                image_height=vision_encoder_info.get_image_size(),
            ),
        )
        if _BASE_ONLY_ENV:
            return {"image": min(base_feature_size, seq_len)}

        max_tokens = 0
        for (height, width) in hf_config.image_grid_pinpoints:
            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                image_size=(height, width),
                grid_pinpoints=hf_config.image_grid_pinpoints,
                patch_size=vision_encoder_info.get_image_size(),
            )
            grid_length = vision_encoder_info.get_patch_grid_length()
            (
                unpadded_feature_size,
                newline_feature_size,
            ) = self._get_num_unpadded_features(
                original_height=height,
                original_width=width,
                npatches=grid_length,
                num_patch_height=num_patch_height,
                num_patch_width=num_patch_width,
            )
            total_tokens = (base_feature_size + unpadded_feature_size +
                            newline_feature_size)
            if total_tokens > max_tokens:
                max_tokens = total_tokens

        return {"image": min(max_tokens, seq_len)}

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L113
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        hf_processor_mm_kwargs: Optional[Mapping[str, object]] = None,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        base_feature_size = self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )
        if _BASE_ONLY_ENV:
            topk = _get_mm_cls2patch_topk()
            if topk > 0:
                returned = min(topk, base_feature_size)
            else:
                returned = base_feature_size
            if DEBUG:
                logger.info(
                    ("llava_next base_only num_tokens: orig=(%s,%s) "
                     "base=%s topk=%s return=%s"),
                    image_height,
                    image_width,
                    base_feature_size,
                    topk,
                    returned,
                )
            return returned

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_encoder_info.get_image_size(),
        )
        grid_length = vision_encoder_info.get_patch_grid_length()
        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=grid_length,
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        total_tokens = (unpadded_feature_size + newline_feature_size +
                        base_feature_size)
        trace = get_mm_trace_state()
        if trace is not None:
            logger.info(
                ("[MMTrace] llava_next_num_tokens request_id=%s orig=(%s,%s) "
                 "g=%s grid=(%s,%s) base=%s unpadded=%s newline=%s "
                 "total=%s total_patches=%s"),
                trace.request_id,
                image_height,
                image_width,
                grid_length,
                num_patch_height,
                num_patch_width,
                base_feature_size,
                unpadded_feature_size,
                newline_feature_size,
                total_tokens,
                (num_patch_height * num_patch_width),
            )
        if DEBUG:
            logger.info(
                ("llava_next vllm_num_tokens: orig=(%s,%s) g=%s grid=(%s,%s) "
                 "base=%s unpadded=%s newline=%s N=%s"),
                image_height,
                image_width,
                grid_length,
                num_patch_height,
                num_patch_width,
                base_feature_size,
                unpadded_feature_size,
                newline_feature_size,
                total_tokens,
            )

        if _MM_CLS2PATCH_ANYRES_TRACE_ENV:
            topk = _get_mm_cls2patch_topk()
            if topk > 0:
                (
                    base_height,
                    base_width,
                    current_height,
                    current_width,
                    row_min,
                    row_max,
                    col_min,
                    col_max,
                    branch,
                    padding,
                ) = _compute_anyres_unpad_window(
                    orig_height=image_height,
                    orig_width=image_width,
                    num_patch_height=num_patch_height,
                    num_patch_width=num_patch_width,
                    grid_len=grid_length,
                )
                valid_sum = 0
                valid_min = None
                valid_max = None
                tiles_keep_sum = 0
                tiles_topp = _get_mm_cls2patch_tiles_topp()
                for tile_row in range(num_patch_height):
                    tile_row_start = tile_row * grid_length
                    tile_row_end = tile_row_start + grid_length
                    row_overlap = max(
                        0,
                        min(tile_row_end, row_max) -
                        max(tile_row_start, row_min),
                    )
                    for tile_col in range(num_patch_width):
                        tile_col_start = tile_col * grid_length
                        tile_col_end = tile_col_start + grid_length
                        col_overlap = max(
                            0,
                            min(tile_col_end, col_max) -
                            max(tile_col_start, col_min),
                        )
                        valid = row_overlap * col_overlap
                        valid_sum += valid
                        valid_min = valid if valid_min is None else min(
                            valid_min, valid)
                        valid_max = valid if valid_max is None else max(
                            valid_max, valid)
                        keep = int(math.ceil(valid * tiles_topp))
                        keep = max(0, min(valid, keep))
                        tiles_keep_sum += keep
                if valid_min is None:
                    valid_min = 0
                    valid_max = 0
                global_keep = min(topk, base_feature_size)
                newline_keep = newline_feature_size
                predicted_keep_total = global_keep + tiles_keep_sum + newline_keep
                trace_id = trace.request_id if trace is not None else "n/a"
                logger.info(
                    ("[MMAnyresTopK] request_id=%s orig=(%s,%s) grid=(%s,%s) "
                     "grid_len=%s H0W0=(%s,%s) window=%s:%s,%s:%s "
                     "H1W1=(%s,%s) branch=%s pad=%s base=%s unpadded=%s "
                     "newline=%s total=%s topk=%s tiles_topp=%s global_keep=%s "
                     "tiles_keep_sum=%s newline_keep=%s "
                     "predicted_keep_total=%s valid_sum=%s valid_min=%s "
                     "valid_max=%s"),
                    trace_id,
                    image_height,
                    image_width,
                    num_patch_height,
                    num_patch_width,
                    grid_length,
                    base_height,
                    base_width,
                    row_min,
                    row_max,
                    col_min,
                    col_max,
                    current_height,
                    current_width,
                    branch,
                    padding,
                    base_feature_size,
                    unpadded_feature_size,
                    newline_feature_size,
                    total_tokens,
                    topk,
                    tiles_topp,
                    global_keep,
                    tiles_keep_sum,
                    newline_keep,
                    predicted_keep_total,
                    valid_sum,
                    valid_min,
                    valid_max,
                )
                if valid_sum != unpadded_feature_size:
                    logger.warning(
                        ("[MMAnyresTopK] valid_sum_mismatch request_id=%s "
                         "valid_sum=%s unpadded=%s"),
                        trace_id,
                        valid_sum,
                        unpadded_feature_size,
                    )

        if _MM_CLS2PATCH_ANYRES_PRUNE_ENV: #根据attention score 剪枝数的计算
            budget_mode = _get_mm_anyres_budget_mode()
            if budget_mode == "relevance":
                prune_ratio = _get_request_prune_ratio(hf_processor_mm_kwargs)
                if prune_ratio is None:
                    prune_ratio = _get_mm_anyres_image_prune_ratio()
                if prune_ratio is None:
                    msg = (
                        "[MMAnyresBudget] get_num_image_tokens missing "
                        "request prune_ratio and "
                        "VLLM_MM_ANYRES_IMAGE_PRUNE_RATIO "
                        f"image=({image_height},{image_width})")
                    logger.error(msg)
                    raise ValueError(msg)
                total_image_tokens = base_feature_size + unpadded_feature_size
#根据剪枝比例VLLM_MM_ANYRES_IMAGE_PRUNE_RATIO计算保留的image tokens数量
                keep_image_tokens = int(
                    math.floor(total_image_tokens * (1.0 - prune_ratio)))
                keep_image_tokens = max(
                    0, min(total_image_tokens, keep_image_tokens))
                pruned_len = keep_image_tokens + newline_feature_size
                if DEBUG:
                    trace_id = trace.request_id if trace is not None else "n/a"
                    logger.info(
                        ("[MMAnyresBudgetLen] request_id=%s image=(%s,%s) "
                         "mode=%s prune_ratio=%s keep_image=%s newline=%s "
                         "pruned_len=%s"),
                        trace_id,
                        image_height,
                        image_width,
                        budget_mode,
                        prune_ratio,
                        keep_image_tokens,
                        newline_feature_size,
                        pruned_len,
                    )
                return pruned_len
            topk = _get_mm_cls2patch_topk()
            if topk > 0:
                (
                    _,
                    _,
                    current_height,
                    _,
                    row_min,
                    row_max,
                    col_min,
                    col_max,
                    _,
                    _,
                ) = _compute_anyres_unpad_window(
                    orig_height=image_height,
                    orig_width=image_width,
                    num_patch_height=num_patch_height,
                    num_patch_width=num_patch_width,
                    grid_len=grid_length,
                )
                tiles_keep_sum = 0
                tiles_topp = _get_mm_cls2patch_tiles_topp()
                for tile_row in range(num_patch_height):#遍历每一行 tiles
                #当前 tile 行在全局 patch 坐标系中的范围
                    tile_row_start = tile_row * grid_length#tile_row=0 → 0
                    tile_row_end = tile_row_start + grid_length#tile_row=0 → 24
                    row_overlap = max(# 计算该 tile 行与有效窗口在行方向的重叠
                        0,
                        min(tile_row_end, row_max) -
                        max(tile_row_start, row_min),
                    )
                    for tile_col in range(num_patch_width):#遍历每一列 tiles
                        tile_col_start = tile_col * grid_length
                        tile_col_end = tile_col_start + grid_length
                        col_overlap = max(
                            0,
                            min(tile_col_end, col_max) -
                            max(tile_col_start, col_min),
                        )
                        valid = row_overlap * col_overlap
                        keep = int(math.ceil(valid * tiles_topp))
                        keep = max(0, min(valid, keep))
                        tiles_keep_sum += keep
                global_keep = min(topk, base_feature_size)
                newline_keep = newline_feature_size
                pruned_len = global_keep + tiles_keep_sum + newline_keep
                if DEBUG:
                    trace_id = trace.request_id if trace is not None else "n/a"
                    logger.info(
                        ("[MMAnyresTopKPrunedLen] request_id=%s orig=(%s,%s) "
                         "grid=(%s,%s) topk=%s tiles_topp=%s global_keep=%s "
                         "tiles_keep_sum=%s newline_keep=%s pruned_len=%s"),
                        trace_id,
                        image_height,
                        image_width,
                        num_patch_height,
                        num_patch_width,
                        topk,
                        tiles_topp,
                        global_keep,
                        tiles_keep_sum,
                        newline_keep,
                        pruned_len,
                    )
                return pruned_len

        return total_tokens

    # Based on: https://github.com/huggingface/text-generation-inference/blob/v3.0.1/server/text_generation_server/models/vlm_causal_lm.py#L86
    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        base_height = npatches * num_patch_height
        base_width = npatches * num_patch_width
        current_height = base_height
        current_width = base_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        branch = "width"
        new_height = None
        new_width = None
        padding = 0

        if aspect_ratio > current_aspect_ratio:
            new_height = int(
                round(original_height * (current_width / original_width), 7))
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
            branch = "height"
        else:
            new_width = int(
                round(original_width * (current_height / original_height), 7))
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height

        trace = get_mm_trace_state()
        if trace is not None:
            logger.info(
                ("[MMTrace] llava_next_unpadded request_id=%s orig=(%s,%s) "
                 "g=%s grid=(%s,%s) H0W0=(%s,%s) branch=%s new_h=%s new_w=%s "
                 "pad=%s H1W1=(%s,%s) unpadded=%s newline=%s"),
                trace.request_id,
                original_height,
                original_width,
                npatches,
                num_patch_height,
                num_patch_width,
                base_height,
                base_width,
                branch,
                new_height,
                new_width,
                padding,
                current_height,
                current_width,
                unpadded_features,
                newline_features,
            )
        if DEBUG:
            logger.info(
                ("llava_next vllm_unpadded: orig=(%s,%s) g=%s grid=(%s,%s) "
                 "H0W0=(%s,%s) branch=%s new_h=%s new_w=%s pad=%s "
                 "H1W1=(%s,%s) unpadded=%s newline=%s"),
                original_height,
                original_width,
                npatches,
                num_patch_height,
                num_patch_width,
                base_height,
                base_width,
                branch,
                new_height,
                new_width,
                padding,
                current_height,
                current_width,
                unpadded_features,
                newline_features,
            )

        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for (height, width) in hf_config.image_grid_pinpoints:
            feat_size = self._get_num_image_tokens_unpruned(
                image_width=width,
                image_height=height,
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width,
                                                     height=height)

        if largest_feature_size == 0 or largest_feature_pinpoint is None:
            raise ValueError("Cannot have a largest feature size of 0!")

        if DEBUG:
            logger.info(
                "[MMProfileBound] base_only=%s topk=%s chosen_pinpoint=(%s,%s) "
                "unpruned_tokens=%s",
                _BASE_ONLY_ENV,
                _get_mm_cls2patch_topk(),
                largest_feature_pinpoint.height,
                largest_feature_pinpoint.width,
                largest_feature_size,
            )

        return largest_feature_pinpoint

    def get_max_image_tokens(self) -> int:
        target = self.get_image_size_with_most_features()
        return self._get_num_image_tokens_unpruned(image_width=target.width,
                                                   image_height=target.height)


_I = TypeVar("_I", bound=LlavaNextProcessingInfo)


class BaseLlavaNextMultiModalProcessor(BaseLlavaMultiModalProcessor[_I]):

    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError


class LlavaNextMultiModalProcessor(
        BaseLlavaNextMultiModalProcessor[LlavaNextProcessingInfo]):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        if _BASE_ONLY_ENV:
            pixel_values = processed_outputs.get("pixel_values")
            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.ndim == 5 and pixel_values.shape[1] > 1:
                    processed_outputs["pixel_values"] = pixel_values[:, :1]
            elif isinstance(pixel_values, list):
                processed_outputs["pixel_values"] = [
                    p[:1] for p in pixel_values
                ]
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


@MULTIMODAL_REGISTRY.register_processor(LlavaNextMultiModalProcessor,
                                        info=LlavaNextProcessingInfo,
                                        dummy_inputs=LlavaDummyInputsBuilder)
class LlavaNextForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        })

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        vision_feature_layer = config.vision_feature_layer
        # Determine the layer up to which we will initialize the vision tower
        if isinstance(vision_feature_layer, int):
            vision_hidden_size = config.vision_config.hidden_size
            self.feature_sample_layers = None
        # Used for multimodal granite models to control encoder outputs
        elif isinstance(vision_feature_layer, (list, tuple)):
            vision_hidden_size = config.vision_config.hidden_size * len(
                vision_feature_layer)
            self.feature_sample_layers = vision_feature_layer
        else:
            raise TypeError(
                f"vision_layer_feature type: {type(vision_feature_layer)}"
                " is not supported")

        self.config = config
        self.multimodal_config = multimodal_config

        # TODO: Optionally initializes this for supporting embeddings.
        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"))
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size))
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias)

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[LlavaNextImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            expected_h = expected_w = self.config.vision_config.image_size
            return LlavaNextImagePixelInputs(
                type="pixel_values",
                pixel_values=flatten_bn(pixel_values),
                image_sizes=flatten_bn(image_sizes, concat=True),
                resolve_bindings={
                    "h": expected_h,
                    "w": expected_w,
                })

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return LlavaNextImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/llava/modeling_llava.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(
            pixel_values, feature_sample_layers=self.feature_sample_layers)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(self, image_size: torch.Tensor,
                                      patch_embeddings: torch.Tensor, *,
                                      strategy: str) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size \
                // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the "
                    "image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                # image_aspect_ratio == "anyres"
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches] \
                    .view(num_patch_height, num_patch_width, height, width, -1)

                if "unpad" in strategy:
                    other_patch_embeds = other_patch_embeds \
                        .permute(4, 0, 2, 1, 3).contiguous() \
                        .flatten(1, 2).flatten(2, 3)
                    other_patch_embeds = unpad_image(other_patch_embeds,
                                                     (orig_height, orig_width))
                    other_patch_embeds = torch.cat((
                        other_patch_embeds,
                        self.image_newline[:, None, None] \
                            .expand(*other_patch_embeds.shape[:-1], 1) \
                            .to(other_patch_embeds.device),
                    ), dim=-1)
                    other_patch_embeds = other_patch_embeds \
                        .flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds \
                        .permute(0, 2, 1, 3, 4).contiguous() \
                        .flatten(0, 3)

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds,
                         self.image_newline[None] \
                            .to(base_patch_embeds.device)
                    ), dim=0)
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaNextImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, torch.Tensor): #1
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values)
            # Visual embeddings: multi_modal_projector outputs in LM space.
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features)

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:])

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values)
        # Visual embeddings: multi_modal_projector outputs in LM space.
        return torch.split(self.multi_modal_projector(stacked_image_features),
                           num_patches_per_batch)

    def _process_image_input(
        self,
        image_input: LlavaNextImageInputs,
        relevance_text_embed: Optional[torch.Tensor] = None,
        relevance_apply: Optional[torch.Tensor] = None,
        budget_prune_ratios_override: Optional[list[Optional[float]]] = None,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor([[default_height, default_width]
                                           for _ in range(batch_size)])

        base_only = _BASE_ONLY_ENV
        cls2patch_topk = _get_mm_cls2patch_topk()
        tiles_topp = _get_mm_cls2patch_tiles_topp()
        cls2patch = None
        anyres_prune_enabled = _MM_CLS2PATCH_ANYRES_PRUNE_ENV
        budget_mode = _get_mm_anyres_budget_mode()
        default_budget_prune_ratio = _get_mm_anyres_image_prune_ratio()
        budget_tau = _get_mm_anyres_zone_budget_tau()
        budget_log_enabled = _get_mm_anyres_budget_log_enabled()
        #这里的topk数字只是一个开关，后面需要优化
        if (base_only or anyres_prune_enabled) \
                and cls2patch_topk > 0 \
                and isinstance(self.vision_tower, CLIPVisionModel):
            cls2patch = getattr(
                self.vision_tower.vision_model.encoder,
                "_mm_last_cls2patch",
                None,
            )
        score_mode = _get_mm_prune_score_mode()
        fusion_alpha = _get_mm_prune_fusion_alpha()
        if budget_prune_ratios_override is not None \
                and len(budget_prune_ratios_override) != len(patch_embeddings):
            raise RuntimeError(
                "Per-request prune ratio count must match multimodal batch size: "
                f"ratios={len(budget_prune_ratios_override)} "
                f"items={len(patch_embeddings)}"
            )

        def _minmax_norm(scores: torch.Tensor) -> torch.Tensor:
            min_val = scores.min(dim=-1, keepdim=True).values
            max_val = scores.max(dim=-1, keepdim=True).values
            denom = (max_val - min_val).clamp_min(1e-6)
            return (scores - min_val) / denom
        merge_strategy = "spatial" if base_only else "spatial_unpad"
        merged_embeddings = []
        for i, patch_features_batch in enumerate(patch_embeddings):
            budget_prune_ratio = default_budget_prune_ratio
            if budget_prune_ratios_override is not None:
                request_ratio = budget_prune_ratios_override[i]
                if request_ratio is not None:
                    budget_prune_ratio = float(request_ratio)
            merged = self._merge_image_patch_embeddings(
                image_sizes[i],
                patch_features_batch,
                strategy=merge_strategy,
            )
            relevance_img = None
            if (budget_mode == "relevance"
                    or score_mode in ("relevance", "fusion")) \
                    and isinstance(relevance_text_embed, torch.Tensor):
                apply_relevance = True
                if relevance_apply is not None:
                    if isinstance(relevance_apply, torch.Tensor):
                        apply_relevance = bool(
                            relevance_apply[i].item()) \
                            if relevance_apply.numel() > i else False
                    else:
                        apply_relevance = bool(relevance_apply)
                if apply_relevance and relevance_text_embed.shape[0] > i:
                    text_vec = F.normalize(relevance_text_embed[i], dim=-1)
                    # Cosine relevance: text vector vs visual token embeddings.
                    patch_norm = F.normalize(patch_features_batch, dim=-1)
                    relevance_img = torch.matmul(patch_norm, text_vec)
            if (not base_only and anyres_prune_enabled
                    and (budget_mode == "relevance" or cls2patch_topk > 0)):
                orig_height, orig_width = image_sizes[i].tolist()
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches_used = 1 + (num_patch_height * num_patch_width)
                num_patches_all = patch_features_batch.shape[0]
                # 始终使用累加方式计算 offset，避免 batch 内不同图像 patch 数不一致导致切错
                offset = sum(patch_embeddings[j].shape[0] for j in range(i))
                cls2patch_img = None
                if cls2patch is not None and cls2patch.dim() == 2:
                    cls2patch_img = cls2patch[offset:offset + num_patches_all]
                if budget_mode == "relevance" and relevance_img is None:
                    msg = (
                        f"[MMAnyresBudget] _process_image_input img_idx={i} "
                        "missing relevance_text_embed or apply flag")
                    logger.error(msg)
                    raise ValueError(msg)
                score_img = None
                if score_mode == "cls2patch":
                    if cls2patch_img is None:
                        msg = (
                            f"[MMAnyresTopK] _process_image_input img_idx={i} "
                            "missing cls2patch for score_mode=cls2patch")
                        logger.error(msg)
                        raise ValueError(msg)
                    score_img = cls2patch_img
                elif score_mode == "relevance":
                    if relevance_img is None:
                        msg = (
                            f"[MMAnyresTopK] _process_image_input img_idx={i} "
                            "missing relevance scores for score_mode=relevance")
                        logger.error(msg)
                        raise ValueError(msg)
                    score_img = relevance_img
                elif score_mode == "fusion":
                    if relevance_img is None or cls2patch_img is None:
                        msg = (
                            f"[MMAnyresTopK] _process_image_input img_idx={i} "
                            "missing cls2patch/relevance for score_mode=fusion")
                        logger.error(msg)
                        raise ValueError(msg)
                    score_img = fusion_alpha * _minmax_norm(cls2patch_img) \
                        + (1.0 - fusion_alpha) * _minmax_norm(relevance_img)
                score_rows = 0 if score_img is None else score_img.shape[0]
                if score_img is None \
                        or score_rows != num_patches_all \
                        or num_patches_all < num_patches_used:
                    msg = (
                        f"[MMAnyresTopK] _process_image_input img_idx={i} "
                        f"score_rows={score_rows} patches_all={num_patches_all} "
                        f"used={num_patches_used} score_mode={score_mode}")
                    logger.error(msg)
                    #会导致vllm warmup没有relevance embedding时报错
                    raise ValueError(msg)
                grid_len = self.config.vision_config.image_size // \
                    self.config.vision_config.patch_size
                base_offset = grid_len * grid_len
                (
                    _,
                    _,
                    current_height,
                    current_width,
                    row_min,
                    _,
                    col_min,
                    _,
                    _,
                    _,
                ) = _compute_anyres_unpad_window(
                    orig_height=orig_height,
                    orig_width=orig_width,
                    num_patch_height=num_patch_height,
                    num_patch_width=num_patch_width,
                    grid_len=grid_len,
                )
                token_idx = torch.arange(
                    grid_len * grid_len,
                    device=score_img.device,
                )
                token_rows = token_idx // grid_len
                token_cols = token_idx % grid_len
                global_scores = score_img[0]
                tile_valid_idx = []
                tile_valid_count = []
                tile_rows = []
                tile_cols = []
                tile_means = []
                for patch_idx in range(1, num_patches_used):
                    tile_idx = patch_idx - 1
                    tile_row = tile_idx // num_patch_width
                    tile_col = tile_idx % num_patch_width
                    row_canvas = tile_row * grid_len + token_rows
                    col_canvas = tile_col * grid_len + token_cols
                    valid_mask = (
                        (row_canvas >= row_min)
                        & (row_canvas < row_min + current_height)
                        & (col_canvas >= col_min)
                        & (col_canvas < col_min + current_width)
                    )
                    valid_idx = torch.where(valid_mask)[0]
                    valid_count = int(valid_idx.numel())
                    if budget_mode == "relevance" and valid_count == 0:
                        msg = (
                            f"[MMAnyresBudget] _process_image_input img_idx={i} "
                            f"tile_idx={tile_idx} valid_count=0")
                        logger.error(msg)
                        raise ValueError(msg)
                    tile_rows.append(tile_row)
                    tile_cols.append(tile_col)
                    tile_valid_idx.append(valid_idx)
                    tile_valid_count.append(valid_count)
                    if budget_mode == "relevance":
                        tile_scores = relevance_img[patch_idx][valid_idx]
                        tile_means.append(tile_scores.mean())
                    scores = score_img[patch_idx]
                    if _MM_CLS2PATCH_ANYRES_PRUNE_LOG_TILES_ENV:
                        k_raw = min(_get_mm_cls2patch_prune_log_topk(),
                                    scores.shape[0])
                        if k_raw > 0:
                            raw_topk_idx = torch.topk(
                                scores, k=k_raw).indices
                            raw_in_valid = int(
                                valid_mask[raw_topk_idx].sum().item())
                            raw_in_padding = k_raw - raw_in_valid
                            raw_sample = raw_topk_idx[:k_raw] \
                                .detach().cpu().tolist()
                            logger.info(
                                "[MMAnyresTopKPruneTileRaw] img_idx=%s "
                                "tile_idx=%s k_raw=%s raw_in_valid=%s "
                                "raw_in_padding=%s raw_topk_idx=%s",
                                i,
                                tile_idx,
                                k_raw,
                                raw_in_valid,
                                raw_in_padding,
                                raw_sample,
                            )
                keep_indices = []
                k_global = 0
                tiles_keep_sum = 0
                keep_image_tokens = None
                tile_keep_budget = []
                if budget_mode == "relevance":
                    if budget_prune_ratio is None:
                        msg = (
                            f"[MMAnyresBudget] _process_image_input img_idx={i} "
                            "missing VLLM_MM_ANYRES_IMAGE_PRUNE_RATIO")
                        logger.error(msg)
                        raise ValueError(msg)
                    # Relevance budget: mean cosine per zone -> softmax -> integer budget.
                    base_mean = relevance_img[0].mean()
                    zone_means = torch.stack([base_mean] + tile_means)
                    scaled = zone_means / budget_tau
                    scaled = scaled - scaled.max()
                    zone_probs = torch.softmax(scaled, dim=0)
#按相关性得分进行每个分区的预算分配
                    total_image_tokens = int(global_scores.shape[0]) + \
                        sum(tile_valid_count)
                    keep_image_tokens = int(
                        math.floor(total_image_tokens *
                                   (1.0 - budget_prune_ratio)))
                    keep_image_tokens = max(
                        0, min(total_image_tokens, keep_image_tokens))
                #每个zone期待保留的 token 数，是浮点数例如12.7
                    expected = zone_probs * keep_image_tokens
                #给每个 zone 分配预算的整数 12.7 → 12
                    floors = torch.floor(expected).to(torch.int64)
                    caps = torch.tensor(
                        [int(global_scores.shape[0])] + tile_valid_count,
                        device=floors.device,
                        dtype=floors.dtype,
                    )
                    budgets = torch.minimum(floors, caps)
                    #还需要补多少 token 才能达到总保留数
                    remaining = int(keep_image_tokens - budgets.sum().item())
                    budgets_list = budgets.detach().cpu().tolist()
                    caps_list = caps.detach().cpu().tolist()
                    if remaining > 0:
                        # 按相关性优先补齐剩余预算，且不超过每个 zone 的上限。
                        zone_probs_list = zone_probs.detach().cpu().tolist()
                        order = sorted(
                            range(len(zone_probs_list)),
                            key=lambda idx: (-zone_probs_list[idx], idx),
                        )
                        for idx in order:
                            if remaining <= 0:
                                break
                            capacity = caps_list[idx] - budgets_list[idx]
                            if capacity <= 0:
                                continue
                            add = remaining if remaining < capacity else capacity
                            budgets_list[idx] += add
                            remaining -= add
                    if remaining != 0:
                        msg = (
                            f"[MMAnyresBudget] _process_image_input img_idx={i} "
                            f"budget_allocation_failed remaining={remaining} "
                            f"keep_image={keep_image_tokens} "
                            f"cap_sum={sum(caps_list)}")
                        logger.error(msg)
                        raise ValueError(msg)
                    k_global = int(budgets_list[0])
                    tile_keep_budget = [int(v) for v in budgets_list[1:]]
                    tiles_keep_sum = int(sum(tile_keep_budget))
                    if budget_log_enabled:
                        logger.info(
                            "[MMAnyresBudget] img_idx=%s mode=%s "
                            "total_image=%s keep_image=%s tau=%s",
                            i,
                            budget_mode,
                            total_image_tokens,
                            keep_image_tokens,
                            budget_tau,
                        )
                        logger.info(
                            "[MMAnyresBudget] img_idx=%s zone=base "
                            "mean=%s prob=%s budget=%s cap=%s",
                            i,
                            float(base_mean.item()),
                            float(zone_probs[0].item()),
                            k_global,
                            int(global_scores.shape[0]),
                        )
                        for tile_idx, (mean, prob, budget, cap) in enumerate(
                                zip(tile_means, zone_probs[1:], tile_keep_budget,
                                    tile_valid_count)):
                            logger.info(
                                "[MMAnyresBudget] img_idx=%s zone=tile_%s "
                                "mean=%s prob=%s budget=%s cap=%s",
                                i,
                                tile_idx,
                                float(mean.item()),
                                float(prob.item()),
                                budget,
                                cap,
                            )
                else:
                    k_global = min(cls2patch_topk, global_scores.shape[0])
                    for valid_count in tile_valid_count:
                        k_tile = int(math.ceil(valid_count * tiles_topp))
                        k_tile = max(0, min(valid_count, k_tile))
                        tile_keep_budget.append(k_tile)
                    tiles_keep_sum = int(sum(tile_keep_budget))
                if k_global > 0:
                    by_score_idx = torch.topk(
                        global_scores, k=k_global).indices
                    sorted_idx, _ = torch.sort(by_score_idx)
                    keep_indices.append(sorted_idx)
                for tile_idx, valid_idx in enumerate(tile_valid_idx):
                    k_tile = tile_keep_budget[tile_idx]
                    if k_tile <= 0:
                        continue
                    patch_idx = tile_idx + 1
                    scores = score_img[patch_idx]
                    by_score_local = torch.topk(
                        scores[valid_idx], k=k_tile).indices
                    local_idx = valid_idx[by_score_local]
                    sorted_local, _ = torch.sort(local_idx)
                    r_local = sorted_local // grid_len
                    c_local = sorted_local % grid_len
                    r_canvas = tile_rows[tile_idx] * grid_len + r_local
                    c_canvas = tile_cols[tile_idx] * grid_len + c_local
                    r_u = r_canvas - row_min
                    c_u = c_canvas - col_min
                    merged_idx = base_offset + \
                        r_u * (current_width + 1) + c_u
                    keep_indices.append(merged_idx)
                    if _MM_CLS2PATCH_ANYRES_PRUNE_LOG_TILES_ENV:
                        sample_count = min(
                            _get_mm_cls2patch_prune_log_topk(),
                            sorted_local.numel(),
                        )
                        logger.info(
                            "[MMAnyresTopKPruneTile] img_idx=%s "
                            "tile_idx=%s valid_count=%s keep=%s "
                            "local_by_score_sample=%s "
                            "local_sorted_sample=%s "
                            "merged_idx_sample=%s",
                            i,
                            tile_idx,
                            tile_valid_count[tile_idx],
                            k_tile,
                            local_idx[:sample_count]
                            .detach().cpu().tolist(),
                            sorted_local[:sample_count]
                            .detach().cpu().tolist(),
                            merged_idx[:sample_count]
                            .detach().cpu().tolist(),
                        )
                newline_keep = current_height
                if newline_keep > 0:
                    r_u = torch.arange(
                        current_height,
                        device=score_img.device,
                    )
                    newline_idx = base_offset + \
                        r_u * (current_width + 1) + current_width
                    keep_indices.append(newline_idx)
                if keep_indices:
                    keep_idx = torch.cat(keep_indices)
                    if keep_idx.numel() > 0:
                        keep_idx = torch.unique(keep_idx)
                    out_of_range = (keep_idx < 0) | \
                        (keep_idx >= merged.shape[0])
                    if out_of_range.any():
                        bad = keep_idx[out_of_range]
                        sample = bad[:8].detach().cpu().tolist()
                        msg = (
                            f"[MMAnyresTopKPrune] img_idx={i} "
                            f"out_of_range={bad.numel()} sample={sample}")
                        logger.error(msg)
                        raise ValueError(msg)
                    keep_idx, _ = torch.sort(keep_idx)
                    merged = merged.index_select(0, keep_idx)
                    keep_count = int(keep_idx.numel())
                    if budget_mode == "relevance":
                        predicted_keep = keep_image_tokens + current_height
                    else:
                        predicted_keep = k_global + tiles_keep_sum + \
                            current_height
                    if DEBUG:
                        keep_sample = keep_idx[:8] \
                            .detach().cpu().tolist()
                        logger.info(
                            "[MMAnyresTopKPrune] img_idx=%s mode=%s "
                            "merged_len=%s keep_count=%s predicted_keep=%s "
                            "global_keep=%s tiles_keep_sum=%s newline_keep=%s "
                            "keep_sample=%s",
                            i,
                            budget_mode,
                            merged.shape[0],
                            keep_count,
                            predicted_keep,
                            k_global,
                            tiles_keep_sum,
                            current_height,
                            keep_sample,
                        )
                    if keep_count != predicted_keep:
                        msg = (
                            f"[MMAnyresTopKPrune] img_idx={i} "
                            f"keep_mismatch keep_count={keep_count} "
                            f"predicted_keep={predicted_keep}")
                        logger.error(msg)
                        raise ValueError(msg)
            if base_only and cls2patch_topk > 0:
                score_vec = None
                if score_mode == "cls2patch":
                    if cls2patch is not None and cls2patch.dim() == 2 \
                            and i < cls2patch.shape[0]:
                        score_vec = cls2patch[i]
                elif score_mode == "relevance":
                    if relevance_img is not None:
                        score_vec = relevance_img[0]
                    elif cls2patch is not None and cls2patch.dim() == 2 \
                            and i < cls2patch.shape[0]:
                        score_vec = cls2patch[i]
                elif score_mode == "fusion":
                    if relevance_img is not None and cls2patch is not None \
                            and cls2patch.dim() == 2 \
                            and i < cls2patch.shape[0]:
                        cls_scores = _minmax_norm(
                            cls2patch[i].unsqueeze(0)).squeeze(0)
                        rel_scores = _minmax_norm(
                            relevance_img[0].unsqueeze(0)).squeeze(0)
                        score_vec = fusion_alpha * cls_scores \
                            + (1.0 - fusion_alpha) * rel_scores
                    elif relevance_img is not None:
                        score_vec = relevance_img[0]
                    elif cls2patch is not None and cls2patch.dim() == 2 \
                            and i < cls2patch.shape[0]:
                        score_vec = cls2patch[i]
                if score_vec is not None:
                    k = min(cls2patch_topk, score_vec.shape[0])
                    if k > 0:
                        topk_idx = torch.topk(score_vec, k=k).indices
                        topk_idx, _ = torch.sort(topk_idx)
                        merged = merged.index_select(0, topk_idx)
                        if DEBUG:
                            sample_count = min(8, topk_idx.numel())
                            sample_idx = topk_idx[:sample_count].detach().cpu()
                            logger.info(
                                "[MMDebug] llava_next_score_select "
                                "img_idx=%s before=%s after=%s topk=%s "
                                "idx_sample=%s",
                                i,
                                patch_features_batch.shape[1],
                                merged.shape[0],
                                k,
                                sample_idx.tolist(),
                            )
                    elif DEBUG:
                        logger.info(
                            "[MMDebug] llava_next_score_select img_idx=%s "
                            "skipped topk=%s",
                            i,
                            cls2patch_topk,
                        )
                elif DEBUG:
                    logger.info(
                        "[MMDebug] llava_next_score_select img_idx=%s "
                        "skipped score_vec",
                        i,
                    )
            merged_embeddings.append(merged)
        return merged_embeddings

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self,
                                  **kwargs: object) -> MultiModalEmbeddings:
        relevance_text_embed = kwargs.pop("relevance_text_embed", None)
        relevance_apply = kwargs.pop("relevance_apply", None)
        if "budget_prune_ratio" in kwargs:
            raise RuntimeError(
                "Goal 3 expects request-level budget_prune_ratios, not the "
                "legacy group-level budget_prune_ratio."
            )
        raw_ratios = kwargs.pop("budget_prune_ratios", None)
        budget_prune_ratios_override: Optional[list[Optional[float]]] = None
        if raw_ratios is not None:
            if not isinstance(raw_ratios, list):
                raise RuntimeError(
                    "Goal 3 requires budget_prune_ratios to be a list aligned "
                    "with the multimodal batch."
                )
            budget_prune_ratios_override = []
            for raw_ratio in raw_ratios:
                if raw_ratio is None:
                    budget_prune_ratios_override.append(None)
                elif isinstance(raw_ratio, (int, float)):
                    budget_prune_ratios_override.append(float(raw_ratio))
                else:
                    raise RuntimeError(
                        f"Invalid request-level prune ratio payload: {raw_ratio!r}"
                    )
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        vision_embeddings = self._process_image_input(
            image_input,
            relevance_text_embed=relevance_text_embed,
            relevance_apply=relevance_apply,
            budget_prune_ratios_override=budget_prune_ratios_override,
        )
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:

        if multimodal_embeddings is None \
            or len(multimodal_embeddings) == 0:
            return self.language_model.get_input_embeddings(input_ids)

        inputs_embeds = embed_multimodal(
            input_ids,
            self.config.image_token_index,
            self.language_model.model.get_input_embeddings,
            multimodal_embeddings,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-NeXT.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"A chat between a curious human and an artificial intelligence
        assistant. The assistant gives helpful, detailed, and polite answers to
        the human's questions.
        USER: <image>\\nWhat is shown in this image? ASSISTANT:"`.

        Tokenizer outputs:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973, 319, 1799,
        9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255,
        29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568,
        6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901,
        29871, 32000, ..., 32000, 13, 5618, 338, 4318, 297, 445, 1967, 29973,
        319, 1799, 9047, 13566, 29901]`.

        Unlike in LLaVA-1.5, the number of image tokens inputted to the language
        model depends on the original size of the input image. Including the
        original image token in the input, the required number of image tokens
        is given by [`LlavaNextProcessingInfo.get_num_image_tokens`][vllm.\
model_executor.models.llava_next.LlavaNextProcessingInfo.get_num_image_tokens].

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.

        Info:
            [`LlavaNextImageInputs`][vllm.model_executor.models.llava_next.LlavaNextImageInputs]
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
