# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared configuration for multimodal pruning.

All pruning options (except per-request prune_ratio, which comes from
HTTP vllm_xargs) are read from a single YAML config file whose path is
specified by the env-var ``VLLM_MM_PRUNE_CONFIG`` (default:
``./mm_prune_config.yaml``).

Example config file
-------------------
.. code-block:: yaml

    enabled: true
    budget_mode: "relevance"
    score_mode: "fusion"
    fusion_alpha: 0.5
    cls2patch_topk: 256
    tiles_topp: 0.5
    zone_budget_tau: 1.0
    base_only: false
    log:
      enabled: true
      file: "./mm_prune_log.txt"
      log_tiles: false
      log_topk: 8
      budget_log: true
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from vllm.logger import init_logger

logger = init_logger(__name__)

_CONFIG_ENV_VAR = "VLLM_MM_PRUNE_CONFIG"
_DEFAULT_CONFIG_PATH = "./mm_prune_config.yaml"


@dataclass
class MmPruneLogConfig:
    """Logging sub-config."""

    enabled: bool = False
    file: str = "./mm_prune_log.txt"
    log_tiles: bool = False
    log_topk: int = 8
    budget_log: bool = False


@dataclass
class MmPruneConfig:
    """Multimodal pruning configuration.

    Attributes
    ----------
    enabled : bool
        Master switch – corresponds to the former
        ``VLLM_MM_CLS2PATCH_ANYRES_PRUNE`` env-var.
    budget_mode : str
        ``"default"`` or ``"relevance"``.
    score_mode : str
        ``"cls2patch"``, ``"relevance"`` or ``"fusion"``.
    fusion_alpha : float
        Blending weight when ``score_mode="fusion"`` (0.0–1.0).
    cls2patch_topk : int
        Global top-k for cls2patch scoring (≥0).
    tiles_topp : float
        Tile-level top-p fraction (0.0–1.0).
    zone_budget_tau : float
        Softmax temperature for zone budget allocation (>0).
    base_only : bool
        LLaVA-Next specific: use base patch only.
    log : MmPruneLogConfig
        Logging options.
    """

    enabled: bool = False
    budget_mode: str = "default"
    score_mode: str = "cls2patch"
    fusion_alpha: float = 0.5
    cls2patch_topk: int = 0
    tiles_topp: float = 1.0
    zone_budget_tau: float = 1.0
    base_only: bool = False
    log: MmPruneLogConfig = field(default_factory=MmPruneLogConfig)

    def __post_init__(self) -> None:
        # ---- validate & clamp ----
        if self.budget_mode not in ("default", "relevance"):
            logger.warning(
                "[MmPruneConfig] unknown budget_mode=%s, using 'default'",
                self.budget_mode,
            )
            self.budget_mode = "default"
        if self.score_mode not in ("cls2patch", "relevance", "fusion"):
            logger.warning(
                "[MmPruneConfig] unknown score_mode=%s, using 'cls2patch'",
                self.score_mode,
            )
            self.score_mode = "cls2patch"
        self.fusion_alpha = max(0.0, min(1.0, self.fusion_alpha))
        self.cls2patch_topk = max(0, self.cls2patch_topk)
        self.tiles_topp = max(0.0, min(1.0, self.tiles_topp))
        if self.zone_budget_tau <= 0:
            self.zone_budget_tau = 1.0


def _load_config(path: str) -> MmPruneConfig:
    """Load config from a YAML file.  Returns defaults on any failure."""
    try:
        import yaml
    except ImportError:
        logger.warning(
            "[MmPruneConfig] PyYAML not installed; using all defaults"
        )
        return MmPruneConfig()

    if not os.path.isfile(path):
        logger.info(
            "[MmPruneConfig] config file not found at %s; using defaults", path
        )
        return MmPruneConfig()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except Exception as exc:
        logger.warning(
            "[MmPruneConfig] failed to read %s: %s; using defaults", path, exc
        )
        return MmPruneConfig()

    if not isinstance(raw, dict):
        logger.warning(
            "[MmPruneConfig] config root is not a dict; using defaults"
        )
        return MmPruneConfig()

    log_raw = raw.pop("log", None)
    log_cfg = MmPruneLogConfig()
    if isinstance(log_raw, dict):
        log_cfg = MmPruneLogConfig(
            enabled=bool(log_raw.get("enabled", False)),
            file=str(log_raw.get("file", "./mm_prune_log.txt")),
            log_tiles=bool(log_raw.get("log_tiles", False)),
            log_topk=int(log_raw.get("log_topk", 8)),
            budget_log=bool(log_raw.get("budget_log", False)),
        )

    cfg = MmPruneConfig(
        enabled=bool(raw.get("enabled", False)),
        budget_mode=str(raw.get("budget_mode", "default")),
        score_mode=str(raw.get("score_mode", "cls2patch")),
        fusion_alpha=float(raw.get("fusion_alpha", 0.5)),
        cls2patch_topk=int(raw.get("cls2patch_topk", 0)),
        tiles_topp=float(raw.get("tiles_topp", 1.0)),
        zone_budget_tau=float(raw.get("zone_budget_tau", 1.0)),
        base_only=bool(raw.get("base_only", False)),
        log=log_cfg,
    )
    logger.info("[MmPruneConfig] loaded from %s: %s", path, cfg)
    return cfg


# ---- singleton ----
_cached_config: MmPruneConfig | None = None


def get_mm_prune_config() -> MmPruneConfig:
    """Return the singleton pruning config (loaded once on first call)."""
    global _cached_config
    if _cached_config is None:
        path = os.getenv(_CONFIG_ENV_VAR, _DEFAULT_CONFIG_PATH)
        _cached_config = _load_config(path)
    return _cached_config


def append_prune_log(line: str) -> None:
    """Append a line to the prune log file (if configured)."""
    cfg = get_mm_prune_config()
    if not cfg.log.file:
        return
    try:
        with open(cfg.log.file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def append_info_log(line: str) -> None:
    """Log to both the prune log file and the standard logger."""
    append_prune_log(f"[INFO!] {line}")
    logger.info(line)
