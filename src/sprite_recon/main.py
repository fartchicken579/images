"""Entry point for sprite reconstruction initialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from sprite_recon.config import load_config
from sprite_recon.io import load_target_image
from sprite_recon.pyramid import build_pyramid
from sprite_recon.residual import initialize_state
from sprite_recon.scheduler import run_optimization


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprite reconstruction initialization")
    parser.add_argument("--target", type=Path, required=True, help="Path to target image")
    parser.add_argument("--sprites-dir", type=Path, help="Directory of sprite images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, help="Optional JSON config path")
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "high_quality"],
        help="Quality preset",
    )
    parser.add_argument("--max-sprites", type=int, help="Override max sprite count")
    parser.add_argument("--sprite-penalty", type=float, help="Override sprite penalty lambda")
    parser.add_argument("--enable-profiling", action="store_true", help="Enable timing/profiling output")
    parser.add_argument("--profile-output", type=Path, help="Write profiling data to JSON/CSV")
    parser.add_argument("--enable-diagnostics", action="store_true", help="Enable residual diagnostics")
    parser.add_argument("--debug-output-dir", type=Path, help="Write debug images to this directory")
    parser.add_argument("--residual-snapshot-every", type=int, help="Save residual every N iterations")
    parser.add_argument(
        "--validation-full-residual-every",
        type=int,
        help="Recompute residual from scratch every N iterations",
    )
    parser.add_argument("--validation-tolerance", type=float, help="Residual validation tolerance")
    parser.add_argument(
        "--enable-comparative-validation",
        action="store_true",
        help="Compare average-color vs least-squares fitting",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config, args.preset)
    if args.sprites_dir:
        config = config.__class__(
            preset=config.preset,
            sprite_paths=config.sprite_paths,
            sprite_dir=args.sprites_dir,
            sprite_extensions=config.sprite_extensions,
            pyramid_min_size=config.pyramid_min_size,
            pyramid_max_levels=config.pyramid_max_levels,
            sprite_penalty=config.sprite_penalty,
            global_min_gain=config.global_min_gain,
            max_sprites=config.max_sprites,
            enable_profiling=config.enable_profiling,
            profile_output=config.profile_output,
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
    if args.max_sprites is not None or args.sprite_penalty is not None:
        config = config.__class__(
            preset=config.preset,
            sprite_paths=config.sprite_paths,
            sprite_dir=config.sprite_dir,
            sprite_extensions=config.sprite_extensions,
            pyramid_min_size=config.pyramid_min_size,
            pyramid_max_levels=config.pyramid_max_levels,
            sprite_penalty=float(args.sprite_penalty if args.sprite_penalty is not None else config.sprite_penalty),
            global_min_gain=config.global_min_gain,
            max_sprites=int(args.max_sprites if args.max_sprites is not None else config.max_sprites),
            enable_profiling=config.enable_profiling,
            profile_output=config.profile_output,
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
    if (
        args.enable_profiling
        or args.profile_output
        or args.enable_diagnostics
        or args.debug_output_dir
        or args.residual_snapshot_every is not None
        or args.validation_full_residual_every is not None
        or args.validation_tolerance is not None
        or args.enable_comparative_validation
    ):
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
            enable_profiling=bool(args.enable_profiling or config.enable_profiling),
            profile_output=args.profile_output or config.profile_output,
            enable_diagnostics=bool(args.enable_diagnostics or config.enable_diagnostics),
            residual_snapshot_every=int(
                args.residual_snapshot_every
                if args.residual_snapshot_every is not None
                else config.residual_snapshot_every
            ),
            debug_output_dir=args.debug_output_dir or config.debug_output_dir,
            validation_full_residual_every=int(
                args.validation_full_residual_every
                if args.validation_full_residual_every is not None
                else config.validation_full_residual_every
            ),
            validation_tolerance=float(
                args.validation_tolerance
                if args.validation_tolerance is not None
                else config.validation_tolerance
            ),
            enable_comparative_validation=bool(
                args.enable_comparative_validation or config.enable_comparative_validation
            ),
            enable_hsv_refine=config.enable_hsv_refine,
            hsv_refine_step=config.hsv_refine_step,
            hsv_presets=config.hsv_presets,
        )

    if not config.sprite_dir and not config.sprite_paths:
        raise ValueError("Sprites directory or sprite paths must be provided.")
    if config.sprite_dir and not config.sprite_dir.exists():
        raise ValueError("Sprites directory does not exist.")

    target = load_target_image(args.target)
    print("Initialization complete")
    print(f"Target size: {target.shape[1]}x{target.shape[0]}")
    pyramid = build_pyramid(
        target,
        min_size=config.pyramid_min_size,
        max_levels=config.pyramid_max_levels,
    )
    coarsest = pyramid[0]
    state = initialize_state(coarsest, level_scale=config.preset.levels[0].scale)
    print(f"Pyramid levels: {len(pyramid)}")
    print(f"Canvas level shape: {state.canvas.image.shape}")

    run_optimization(
        target=target,
        config=config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
