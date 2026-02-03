"""Configuration schema and loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class LevelConfig:
    """Per-level settings used during the coarse-to-fine search."""

    scale: float
    grid_stride: int
    rotation_steps: int
    uniform_scales: List[float]
    aspect_ratios: List[float]
    max_sprites: int
    min_gain: float
    min_gain_rate: float


@dataclass(frozen=True)
class PresetConfig:
    """Quality/speed preset configuration."""

    name: str
    top_k: int
    refinement_iters: int
    levels: List[LevelConfig]


@dataclass(frozen=True)
class Config:
    """Top-level configuration for sprite reconstruction."""

    preset: PresetConfig
    sprite_paths: List[Path] = field(default_factory=list)
    sprite_dir: Optional[Path] = None
    sprite_extensions: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])
    pyramid_min_size: int = 64
    pyramid_max_levels: int = 6
    sprite_penalty: float = 0.0
    global_min_gain: float = 0.001
    max_sprites: int = 5000
    enable_profiling: bool = False
    profile_output: Optional[Path] = None
    enable_diagnostics: bool = False
    residual_snapshot_every: int = 0
    debug_output_dir: Optional[Path] = None
    validation_full_residual_every: int = 0
    validation_tolerance: float = 1e-4
    enable_comparative_validation: bool = False
    enable_hsv_refine: bool = False
    hsv_refine_step: float = 0.02
    hsv_presets: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.0, 0.0, 0.0),
            (0.03, 0.05, 0.0),
            (-0.03, -0.05, 0.0),
            (0.0, -0.2, 0.0),
        ]
    )


_PRESET_LEVELS: Dict[str, List[LevelConfig]] = {
    "fast": [
        LevelConfig(
            scale=0.125,
            grid_stride=8,
            rotation_steps=8,
            uniform_scales=[0.5, 1.0, 1.5],
            aspect_ratios=[1.0],
            max_sprites=200,
            min_gain=0.02,
            min_gain_rate=0.001,
        ),
        LevelConfig(
            scale=0.25,
            grid_stride=6,
            rotation_steps=8,
            uniform_scales=[0.75, 1.0, 1.25],
            aspect_ratios=[1.0],
            max_sprites=400,
            min_gain=0.01,
            min_gain_rate=0.0008,
        ),
        LevelConfig(
            scale=0.5,
            grid_stride=4,
            rotation_steps=12,
            uniform_scales=[0.75, 1.0, 1.25],
            aspect_ratios=[0.9, 1.0, 1.1],
            max_sprites=800,
            min_gain=0.005,
            min_gain_rate=0.0005,
        ),
        LevelConfig(
            scale=1.0,
            grid_stride=2,
            rotation_steps=16,
            uniform_scales=[0.9, 1.0, 1.1],
            aspect_ratios=[0.9, 1.0, 1.1],
            max_sprites=1200,
            min_gain=0.003,
            min_gain_rate=0.0003,
        ),
    ],
    "balanced": [
        LevelConfig(
            scale=0.125,
            grid_stride=6,
            rotation_steps=12,
            uniform_scales=[0.5, 0.75, 1.0, 1.25],
            aspect_ratios=[0.9, 1.0, 1.1],
            max_sprites=300,
            min_gain=0.015,
            min_gain_rate=0.0009,
        ),
        LevelConfig(
            scale=0.25,
            grid_stride=4,
            rotation_steps=16,
            uniform_scales=[0.5, 0.75, 1.0, 1.25, 1.5],
            aspect_ratios=[0.85, 1.0, 1.15],
            max_sprites=600,
            min_gain=0.008,
            min_gain_rate=0.0006,
        ),
        LevelConfig(
            scale=0.5,
            grid_stride=3,
            rotation_steps=24,
            uniform_scales=[0.75, 1.0, 1.25, 1.5],
            aspect_ratios=[0.85, 1.0, 1.15],
            max_sprites=1200,
            min_gain=0.004,
            min_gain_rate=0.0004,
        ),
        LevelConfig(
            scale=1.0,
            grid_stride=2,
            rotation_steps=32,
            uniform_scales=[0.9, 1.0, 1.1, 1.2],
            aspect_ratios=[0.9, 1.0, 1.1],
            max_sprites=2000,
            min_gain=0.002,
            min_gain_rate=0.00025,
        ),
    ],
    "high_quality": [
        LevelConfig(
            scale=0.125,
            grid_stride=4,
            rotation_steps=24,
            uniform_scales=[0.5, 0.75, 1.0, 1.25, 1.5],
            aspect_ratios=[0.8, 0.9, 1.0, 1.1, 1.2],
            max_sprites=500,
            min_gain=0.01,
            min_gain_rate=0.0007,
        ),
        LevelConfig(
            scale=0.25,
            grid_stride=3,
            rotation_steps=32,
            uniform_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            aspect_ratios=[0.8, 0.9, 1.0, 1.1, 1.2],
            max_sprites=900,
            min_gain=0.006,
            min_gain_rate=0.0005,
        ),
        LevelConfig(
            scale=0.5,
            grid_stride=2,
            rotation_steps=48,
            uniform_scales=[0.75, 0.9, 1.0, 1.1, 1.25, 1.5],
            aspect_ratios=[0.85, 0.9, 1.0, 1.1, 1.15],
            max_sprites=1800,
            min_gain=0.003,
            min_gain_rate=0.0003,
        ),
        LevelConfig(
            scale=1.0,
            grid_stride=1,
            rotation_steps=64,
            uniform_scales=[0.85, 0.9, 1.0, 1.1, 1.2],
            aspect_ratios=[0.85, 0.9, 1.0, 1.1, 1.15],
            max_sprites=3000,
            min_gain=0.0015,
            min_gain_rate=0.0002,
        ),
    ],
}


def preset_config(name: str) -> PresetConfig:
    """Return a preset configuration by name."""

    key = name.lower()
    if key not in _PRESET_LEVELS:
        raise ValueError(f"Unknown preset '{name}'. Available: {', '.join(_PRESET_LEVELS)}")
    levels = _PRESET_LEVELS[key]
    top_k = 32 if key == "fast" else 64 if key == "balanced" else 128
    refinement_iters = 8 if key == "fast" else 12 if key == "balanced" else 16
    return PresetConfig(name=key, top_k=top_k, refinement_iters=refinement_iters, levels=levels)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge overrides into base without mutating either input."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Optional[Path], preset_name: str) -> Config:
    """Load configuration from JSON and apply preset defaults."""

    base = {
        "preset": preset_name,
        "sprite_paths": [],
        "sprite_dir": None,
        "sprite_extensions": [".png", ".jpg", ".jpeg", ".webp"],
        "pyramid_min_size": 64,
        "pyramid_max_levels": 6,
        "sprite_penalty": 0.0,
        "global_min_gain": 0.001,
        "max_sprites": 5000,
        "enable_profiling": False,
        "profile_output": None,
        "enable_diagnostics": False,
        "residual_snapshot_every": 0,
        "debug_output_dir": None,
        "validation_full_residual_every": 0,
        "validation_tolerance": 1e-4,
        "enable_comparative_validation": False,
        "enable_hsv_refine": False,
        "hsv_refine_step": 0.02,
        "hsv_presets": [
            (0.0, 0.0, 0.0),
            (0.03, 0.05, 0.0),
            (-0.03, -0.05, 0.0),
            (0.0, -0.2, 0.0),
        ],
    }

    if path:
        raw = json.loads(Path(path).read_text())
        merged = _merge_dict(base, raw)
    else:
        merged = base

    preset = preset_config(merged["preset"])
    sprite_paths = [Path(p) for p in merged.get("sprite_paths", [])]
    sprite_dir = Path(merged["sprite_dir"]) if merged.get("sprite_dir") else None

    return Config(
        preset=preset,
        sprite_paths=sprite_paths,
        sprite_dir=sprite_dir,
        sprite_extensions=merged.get("sprite_extensions", base["sprite_extensions"]),
        pyramid_min_size=int(merged.get("pyramid_min_size", base["pyramid_min_size"])),
        pyramid_max_levels=int(merged.get("pyramid_max_levels", base["pyramid_max_levels"])),
        sprite_penalty=float(merged.get("sprite_penalty", base["sprite_penalty"])),
        global_min_gain=float(merged.get("global_min_gain", base["global_min_gain"])),
        max_sprites=int(merged.get("max_sprites", base["max_sprites"])),
        enable_profiling=bool(merged.get("enable_profiling", base["enable_profiling"])),
        profile_output=Path(merged["profile_output"]) if merged.get("profile_output") else None,
        enable_diagnostics=bool(merged.get("enable_diagnostics", base["enable_diagnostics"])),
        residual_snapshot_every=int(
            merged.get("residual_snapshot_every", base["residual_snapshot_every"])
        ),
        debug_output_dir=Path(merged["debug_output_dir"]) if merged.get("debug_output_dir") else None,
        validation_full_residual_every=int(
            merged.get("validation_full_residual_every", base["validation_full_residual_every"])
        ),
        validation_tolerance=float(
            merged.get("validation_tolerance", base["validation_tolerance"])
        ),
        enable_comparative_validation=bool(
            merged.get("enable_comparative_validation", base["enable_comparative_validation"])
        ),
        enable_hsv_refine=bool(merged.get("enable_hsv_refine", base["enable_hsv_refine"])),
        hsv_refine_step=float(merged.get("hsv_refine_step", base["hsv_refine_step"])),
        hsv_presets=[tuple(preset) for preset in merged.get("hsv_presets", base["hsv_presets"])],
    )
