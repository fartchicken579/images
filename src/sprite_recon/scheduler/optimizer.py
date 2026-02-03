"""Main greedy optimization loop across pyramid levels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from PIL import Image

from sprite_recon.compositor import composite_sprite, render_sprite
from sprite_recon.config.schema import Config
from sprite_recon.gpu_scoring import CandidateScorer
from sprite_recon.diagnostics import DiagnosticsTracker
from sprite_recon.io import load_sprite_images, save_image
from sprite_recon.pyramid import build_pyramid
from sprite_recon.refinement import RefinedCandidate, refine_candidates
from sprite_recon.residual import recompute_residual, update_residual_patch
from sprite_recon.sprites import preprocess_sprites
from sprite_recon.diagnostics.tracker import Timer
from sprite_recon.scheduler.control import RunControl


def _upsample_canvas(canvas: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Upsample canvas to match target shape using bilinear scaling."""

    height, width = target_shape[:2]
    pil_image = Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGBA")
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _select_best(refined: List[RefinedCandidate]) -> RefinedCandidate | None:
    if not refined:
        return None
    return max(refined, key=lambda item: item.improvement)


def run_optimization(
    target: np.ndarray,
    config: Config,
    output_dir: Path,
    control: RunControl | None = None,
    status_callback: Callable[[Dict[str, float | str | int]], None] | None = None,
) -> Dict[str, object]:
    """Run the greedy optimization loop and write outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.enable_profiling and config.profile_output is None:
        config = config.__class__(
            preset=config.preset,
            sprite_paths=config.sprite_paths,
            sprite_dir=config.sprite_dir,
            sprite_extensions=config.sprite_extensions,
            pyramid_min_size=config.pyramid_min_size,
            pyramid_max_levels=config.pyramid_max_levels,
            sprite_penalty=config.sprite_penalty,
            global_min_gain=config.global_min_gain,
            max_sprites=config.max_sprites,
            enable_profiling=config.enable_profiling,
            profile_output=output_dir / "profile.json",
            enable_diagnostics=config.enable_diagnostics,
            residual_snapshot_every=config.residual_snapshot_every,
            debug_output_dir=config.debug_output_dir,
            validation_full_residual_every=config.validation_full_residual_every,
            validation_tolerance=config.validation_tolerance,
            enable_comparative_validation=config.enable_comparative_validation,
            enable_hsv_refine=config.enable_hsv_refine,
            hsv_refine_step=config.hsv_refine_step,
            hsv_presets=config.hsv_presets,
        )
    tracker = DiagnosticsTracker(
        enable_profiling=config.enable_profiling,
        enable_diagnostics=config.enable_diagnostics,
        profile_output=config.profile_output,
        debug_output_dir=config.debug_output_dir,
        residual_snapshot_every=config.residual_snapshot_every,
    )

    sprite_images = load_sprite_images(config)
    sprite_pyramids = preprocess_sprites(sprite_images, config.preset.levels)

    pyramid = build_pyramid(target, config.pyramid_min_size, config.pyramid_max_levels)
    total_sprites: List[Dict[str, object]] = []
    scorer = CandidateScorer(backend="cpu")

    canvas = np.zeros_like(pyramid[0])
    residual = recompute_residual(pyramid[0], canvas)
    boxes: List[tuple[int, int, int, int]] = []

    for level_index, level_target in enumerate(pyramid):
        last_gain = None
        with Timer() as level_timer:
            if level_index > 0:
                canvas = _upsample_canvas(canvas, level_target.shape)
                residual = recompute_residual(level_target, canvas)

            level_config = config.preset.levels[min(level_index, len(config.preset.levels) - 1)]
            iteration = 0
            while True:
                if control:
                    if control.should_stop():
                        break
                    control.wait_if_paused()
                if len(total_sprites) >= config.max_sprites:
                    break
                if float(np.mean(residual**2)) < config.global_min_gain:
                    break
                if not np.isfinite(residual).all():
                    raise ValueError("Residual contains non-finite values.")

                top_k = config.preset.top_k
                with Timer() as score_timer:
                    candidates = scorer.score(
                        residual=residual,
                        sprite_pyramids=sprite_pyramids,
                        level_index=level_index,
                        level_config=level_config,
                        hsv_presets=config.hsv_presets,
                        top_k=top_k,
                    )
                if any(
                    (not np.isfinite(candidate.score)) or candidate.score < 0.0
                    for candidate in candidates
                ):
                    raise ValueError("Invalid score detected during candidate scoring.")
                with Timer() as refine_timer:
                    refined = refine_candidates(
                        candidates=candidates,
                        residual=residual,
                        sprite_pyramids=sprite_pyramids,
                        level_config=level_config,
                        config=config,
                        max_iters=config.preset.refinement_iters,
                    )
                best = _select_best(refined)
                if best is None:
                    break

                accept_threshold = max(level_config.min_gain, config.sprite_penalty)
                if last_gain is not None:
                    gain_rate = best.improvement / max(last_gain, 1e-8)
                    if gain_rate < level_config.min_gain_rate:
                        break
                if best.improvement <= accept_threshold:
                    break
                if best.improvement < 0.0 or not np.isfinite(best.improvement):
                    raise ValueError("Refinement produced invalid improvement value.")

                with Timer() as commit_timer:
                    canvas, bounds = composite_sprite(
                        canvas=canvas,
                        sprite=sprite_pyramids[best.sprite_id],
                        level_index=level_index,
                        position=best.position,
                        rotation_deg=best.rotation_deg,
                        scale=best.scale,
                        aspect_ratio=best.aspect_ratio,
                        hsv_adjust=best.hsv_adjust,
                        opacity=best.opacity,
                    )
                    residual = update_residual_patch(
                        residual=residual,
                        target=level_target,
                        canvas=canvas,
                        bounds=bounds,
                        sanity_check=(
                            config.validation_full_residual_every > 0
                            and iteration % config.validation_full_residual_every == 0
                        ),
                        tolerance=config.validation_tolerance,
                    )

                scale_x = target.shape[1] / level_target.shape[1]
                scale_y = target.shape[0] / level_target.shape[0]
                total_sprites.append(
                    {
                        "sprite_id": best.sprite_id,
                        "x": best.position[0] * scale_x,
                        "y": best.position[1] * scale_y,
                        "rotation": best.rotation_deg,
                        "scale": best.scale,
                        "aspect_ratio": best.aspect_ratio,
                        "hsv": best.hsv_adjust,
                        "opacity": best.opacity,
                        "improvement": best.improvement,
                        "order": len(total_sprites),
                    }
                )

                iteration += 1
                last_gain = best.improvement
                boxes.append(bounds)
                tracker.maybe_save_residual(residual, level_index, iteration)
                tracker.maybe_save_canvas(canvas, level_index, iteration)
                if config.enable_comparative_validation:
                    template, _ = render_sprite(
                        sprite=sprite_pyramids[best.sprite_id],
                        level_index=level_index,
                        position=best.position,
                        rotation_deg=best.rotation_deg,
                        scale=best.scale,
                        aspect_ratio=best.aspect_ratio,
                        hsv_adjust=best.hsv_adjust,
                    )
                    patch = residual[bounds[1]:bounds[3], bounds[0]:bounds[2]]
                    template_rgb = template[..., :3]
                    residual_rgb = patch[..., :3]
                    denom = float(np.sum(template_rgb * template_rgb)) + 1e-8
                    ls_improvement = (float(np.sum(residual_rgb * template_rgb)) ** 2) / denom
                    avg_scale = float(np.mean(residual_rgb)) / (float(np.mean(template_rgb)) + 1e-8)
                    avg_improvement = (avg_scale * avg_scale) * denom
                    print(
                        f"Validation Δimprovement: {ls_improvement - avg_improvement:.6f} "
                        f"(LS {ls_improvement:.6f} vs avg {avg_improvement:.6f})"
                    )

                residual_energy = float(np.mean(residual**2))
                tracker.track_iteration(
                    level=level_index,
                    iteration=iteration,
                    scoring_s=score_timer.elapsed,
                    refinement_s=refine_timer.elapsed,
                    commit_s=commit_timer.elapsed,
                    residual_energy=residual_energy,
                    total_s=score_timer.elapsed + refine_timer.elapsed + commit_timer.elapsed,
                )
                if status_callback:
                    status_callback(
                        {
                            "level": level_index,
                            "sprites": len(total_sprites),
                            "residual": residual_energy,
                            "iteration_ms": (score_timer.elapsed + refine_timer.elapsed + commit_timer.elapsed)
                            * 1000.0,
                            "preview": str(
                                (config.debug_output_dir / f"canvas_level{level_index}_iter{iteration}.png")
                                if config.debug_output_dir
                                else ""
                            ),
                            "message": (
                                f"Level {level_index} | Sprite {len(total_sprites)} "
                                f"| Improvement {best.improvement:.6f}"
                            ),
                        }
                    )
                if config.enable_profiling or config.enable_diagnostics:
                    print(
                        f"Level {level_index} | Sprite {len(total_sprites)} | "
                        f"Improvement {best.improvement:.6f} | "
                        f"Residual {residual_energy:.6f} | "
                        f"t(score/refine/commit/total) "
                        f"{score_timer.elapsed*1000:.2f}/"
                        f"{refine_timer.elapsed*1000:.2f}/"
                        f"{commit_timer.elapsed*1000:.2f}/"
                        f"{(score_timer.elapsed + refine_timer.elapsed + commit_timer.elapsed)*1000:.2f} ms"
                    )
                else:
                    print(
                        f"Level {level_index} | Sprite {len(total_sprites)} | "
                        f"Improvement {best.improvement:.6f}"
                    )

                if iteration >= level_config.max_sprites:
                    break

        if config.enable_profiling:
            print(
                f"Level {level_index} total time: {level_timer.elapsed*1000:.2f} ms"
            )
        tracker.save_canvas(canvas, level_index)
        tracker.save_bounding_boxes(canvas, boxes, level_index)
        boxes = []
        if len(total_sprites) >= config.max_sprites:
            break

    output_image = output_dir / "composite.png"
    save_image(output_image, canvas)

    output_json = output_dir / "placements.json"
    output_json.write_text(json.dumps(total_sprites, indent=2))
    tracker.export()

    return {
        "canvas": canvas,
        "placements": total_sprites,
        "output_image": str(output_image),
        "output_json": str(output_json),
    }
