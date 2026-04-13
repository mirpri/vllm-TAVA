# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextvars
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

_MM_TRACE_ENV = os.getenv("VLLM_MM_TRACE", "0") == "1"

_TRACE_KEY = "vllm_mm_trace"
_FORCED_GRID_KEY = "forced_grid"
_FORCED_TOTAL_KEY = "forced_total_patches"
_TRACE_KEYS = {_TRACE_KEY, _FORCED_GRID_KEY, _FORCED_TOTAL_KEY}


@dataclass(frozen=True)
class MMTraceState:
    request_id: str
    forced_grid: Optional[tuple[int, int]]
    forced_total_patches: Optional[int]


_MM_TRACE_CTX: contextvars.ContextVar[Optional[MMTraceState]] = \
    contextvars.ContextVar("vllm_mm_trace", default=None)


def is_env_enabled() -> bool:
    return _MM_TRACE_ENV


def _to_bool(val: object) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def _to_int(val: object) -> Optional[int]:
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        text = val.strip()
        if text.isdigit():
            return int(text)
    return None


def _to_grid(val: object) -> Optional[tuple[int, int]]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        try:
            rows = int(val[0])
            cols = int(val[1])
        except (TypeError, ValueError):
            return None
        return rows, cols
    return None


def parse_mm_trace_kwargs(
    mm_kwargs: Optional[Mapping[str, object]],
) -> tuple[bool, Optional[tuple[int, int]], Optional[int]]:
    if not mm_kwargs:
        return False, None, None
    trace_enabled = _to_bool(mm_kwargs.get(_TRACE_KEY))
    forced_grid = _to_grid(mm_kwargs.get(_FORCED_GRID_KEY))
    forced_total = _to_int(mm_kwargs.get(_FORCED_TOTAL_KEY))
    return trace_enabled, forced_grid, forced_total


def extract_mm_processor_kwargs(
    prompt: object,
) -> Optional[Mapping[str, object]]:
    if isinstance(prompt, dict):
        if "encoder_prompt" in prompt:
            return prompt.get("mm_processor_kwargs")
        return prompt.get("mm_processor_kwargs")
    return None


def strip_mm_trace_kwargs(
    mm_kwargs: Optional[Mapping[str, object]],
) -> dict[str, object]:
    if not mm_kwargs:
        return {}
    if not any(key in mm_kwargs for key in _TRACE_KEYS):
        return dict(mm_kwargs)
    return {k: v for k, v in mm_kwargs.items() if k not in _TRACE_KEYS}


def set_mm_trace(
    request_id: str,
    mm_kwargs: Optional[Mapping[str, object]],
) -> Optional[contextvars.Token[Optional[MMTraceState]]]:
    if not _MM_TRACE_ENV:
        return None
    trace_enabled, forced_grid, forced_total = parse_mm_trace_kwargs(mm_kwargs)
    if not trace_enabled:
        return None
    state = MMTraceState(
        request_id=request_id,
        forced_grid=forced_grid,
        forced_total_patches=forced_total,
    )
    return _MM_TRACE_CTX.set(state)


def reset_mm_trace(
    token: Optional[contextvars.Token[Optional[MMTraceState]]],
) -> None:
    if token is not None:
        _MM_TRACE_CTX.reset(token)


def get_mm_trace_state() -> Optional[MMTraceState]:
    if not _MM_TRACE_ENV:
        return None
    return _MM_TRACE_CTX.get()
