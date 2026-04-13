# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextvars
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MMDebugState:
    req_ids: tuple[str, ...]


_MM_VISION_DEBUG_CTX: contextvars.ContextVar[Optional[MMDebugState]] = \
    contextvars.ContextVar("vllm_mm_vision_debug", default=None)


def set_mm_vision_debug(req_ids: Iterable[str]):
    """Enable per-call multimodal vision debug context for given req_ids."""
    unique_ids: list[str] = []
    for r in req_ids:
        if r is None:
            continue
        if r not in unique_ids:
            unique_ids.append(str(r))
    if not unique_ids:
        return None
    state = MMDebugState(req_ids=tuple(unique_ids))
    return _MM_VISION_DEBUG_CTX.set(state)


def reset_mm_vision_debug(token: Optional[contextvars.Token[Optional[MMDebugState]]]):
    if token is not None:
        _MM_VISION_DEBUG_CTX.reset(token)


def get_mm_vision_debug_state() -> Optional[MMDebugState]:
    return _MM_VISION_DEBUG_CTX.get()
