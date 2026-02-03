"""Profiling and diagnostics tracking for the optimizer."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from sprite_recon.io import save_image


@dataclass
class TimingRecord:
    """Timing metrics for a single iteration."""

    level: int
    iteration: int
    scoring_ms: float
    refinement_ms: float
    commit_ms: float
    total_ms: float
    residual_energy: float


@dataclass
class DiagnosticsTracker:
    """Collects timing, residual energy, and debug outputs."""

    enable_profiling: bool = False
    enable_diagnostics: bool = False
    profile_output: Optional[Path] = None
    debug_output_dir: Optional[Path] = None
    residual_snapshot_every: int = 0
    records: List[TimingRecord] = field(default_factory=list)

    def track_iteration(
        self,
        level: int,
        iteration: int,
        scoring_s: float,
        refinement_s: float,
        commit_s: float,
        residual_energy: float,
        total_s: float,
    ) -> None:
        if not (self.enable_profiling or self.enable_diagnostics):
            return
        self.records.append(
            TimingRecord(
                level=level,
                iteration=iteration,
                scoring_ms=scoring_s * 1000.0,
                refinement_ms=refinement_s * 1000.0,
                commit_ms=commit_s * 1000.0,
                total_ms=total_s * 1000.0,
                residual_energy=residual_energy,
            )
        )

    def export(self) -> None:
        """Export timing records to JSON/CSV if configured."""

        if not self.records:
            return

        if self.profile_output:
            self.profile_output.parent.mkdir(parents=True, exist_ok=True)
            payload = [record.__dict__ for record in self.records]
            self.profile_output.write_text(json.dumps(payload, indent=2))

        if self.profile_output:
            csv_path = self.profile_output.with_suffix(".csv")
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.records[0].__dict__.keys()))
                writer.writeheader()
                for record in self.records:
                    writer.writerow(record.__dict__)

    def maybe_save_residual(self, residual: np.ndarray, level: int, iteration: int) -> None:
        """Optionally save a residual heatmap for debugging."""

        if not self.debug_output_dir or self.residual_snapshot_every <= 0:
            return
        if iteration % self.residual_snapshot_every != 0:
            return
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        magnitude = np.linalg.norm(residual[..., :3], axis=-1)
        normed = magnitude / (np.max(magnitude) + 1e-8)
        heat = np.stack([normed, np.zeros_like(normed), 1.0 - normed, np.ones_like(normed)], axis=-1)
        save_image(self.debug_output_dir / f"residual_level{level}_iter{iteration}.png", heat)

    def maybe_save_canvas(self, canvas: np.ndarray, level: int, iteration: int) -> None:
        """Optionally save a canvas snapshot for preview/debugging."""

        if not self.debug_output_dir or self.residual_snapshot_every <= 0:
            return
        if iteration % self.residual_snapshot_every != 0:
            return
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        save_image(self.debug_output_dir / f"canvas_level{level}_iter{iteration}.png", canvas)

    def save_canvas(self, canvas: np.ndarray, level: int) -> None:
        """Save the canvas at the end of a level if debug output is enabled."""

        if not self.debug_output_dir:
            return
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        save_image(self.debug_output_dir / f"canvas_level{level}.png", canvas)

    def save_bounding_boxes(self, canvas: np.ndarray, boxes: List[tuple[int, int, int, int]], level: int) -> None:
        """Save a canvas overlay with sprite bounding boxes."""

        if not self.debug_output_dir or not boxes:
            return
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)
        rgb = (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGBA")
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(box, outline=(255, 0, 0, 255), width=1)
        image.save(self.debug_output_dir / f"boxes_level{level}.png")


class Timer:
    """Simple context timer for profiling blocks."""

    def __init__(self) -> None:
        self.start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
