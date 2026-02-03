"""Flask-based UI for running the optimizer with live preview."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from sprite_recon.config import load_config
from sprite_recon.io import load_target_image
from sprite_recon.scheduler import run_optimization
from sprite_recon.scheduler.control import RunControl


@dataclass
class UIState:
    """Shared UI state for progress reporting."""

    running: bool = False
    paused: bool = False
    stopped: bool = False
    message: str = "Idle"
    current_level: int = 0
    sprites_placed: int = 0
    residual_energy: float = 0.0
    iteration_time_ms: float = 0.0
    preview_path: Optional[Path] = None
    log: list[str] = field(default_factory=list)


class OptimizerWorker:
    """Background worker running the optimizer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.state = UIState()
        self.control = RunControl()
        self.thread: Optional[threading.Thread] = None

    def start(self, target_path: Path, sprite_dir: Path, output_dir: Path, preset: str, max_sprites: int) -> None:
        with self.lock:
            if self.state.running:
                return
            self.state = UIState(running=True, message="Starting...")
            self.control = RunControl()

        def _run() -> None:
            try:
                config = load_config(None, preset)
                config = config.__class__(
                    preset=config.preset,
                    sprite_paths=config.sprite_paths,
                    sprite_dir=sprite_dir,
                    sprite_extensions=config.sprite_extensions,
                    pyramid_min_size=config.pyramid_min_size,
                    pyramid_max_levels=config.pyramid_max_levels,
                    sprite_penalty=config.sprite_penalty,
                    global_min_gain=config.global_min_gain,
                    max_sprites=max_sprites,
                    enable_profiling=True,
                    profile_output=output_dir / "profile.json",
                    enable_diagnostics=True,
                    residual_snapshot_every=5,
                    debug_output_dir=output_dir / "debug",
                    validation_full_residual_every=config.validation_full_residual_every,
                    validation_tolerance=config.validation_tolerance,
                    enable_comparative_validation=False,
                    enable_hsv_refine=config.enable_hsv_refine,
                    hsv_refine_step=config.hsv_refine_step,
                    hsv_presets=config.hsv_presets,
                )
                target = load_target_image(target_path)

                def status_callback(payload: Dict[str, float]) -> None:
                    with self.lock:
                        self.state.current_level = int(payload.get("level", 0))
                        self.state.sprites_placed = int(payload.get("sprites", 0))
                        self.state.residual_energy = float(payload.get("residual", 0.0))
                        self.state.iteration_time_ms = float(payload.get("iteration_ms", 0.0))
                        preview = payload.get("preview")
                        if preview:
                            self.state.preview_path = Path(preview)
                        self.state.log.append(payload.get("message", ""))
                        self.state.log = self.state.log[-200:]

                run_optimization(
                    target=target,
                    config=config,
                    output_dir=output_dir,
                    control=self.control,
                    status_callback=status_callback,
                )
                with self.lock:
                    self.state.running = False
                    self.state.message = "Completed"
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    self.state.running = False
                    self.state.message = f"Error: {exc}"

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def pause(self) -> None:
        with self.lock:
            self.state.paused = True
            self.state.message = "Paused"
        self.control.pause()

    def resume(self) -> None:
        with self.lock:
            self.state.paused = False
            self.state.message = "Running"
        self.control.resume()

    def stop(self) -> None:
        with self.lock:
            self.state.stopped = True
            self.state.message = "Stopping..."
        self.control.stop()


worker = OptimizerWorker()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/start", methods=["POST"])
    def start() -> str:
        target_path = Path(request.form["target"]).expanduser()
        sprite_dir = Path(request.form["sprites"]).expanduser()
        output_dir = Path(request.form["output"]).expanduser()
        preset = request.form.get("preset", "balanced")
        max_sprites = int(request.form.get("max_sprites", 500))
        worker.start(target_path, sprite_dir, output_dir, preset, max_sprites)
        return redirect(url_for("index"))

    @app.route("/pause", methods=["POST"])
    def pause() -> str:
        worker.pause()
        return redirect(url_for("index"))

    @app.route("/resume", methods=["POST"])
    def resume() -> str:
        worker.resume()
        return redirect(url_for("index"))

    @app.route("/stop", methods=["POST"])
    def stop() -> str:
        worker.stop()
        return redirect(url_for("index"))

    @app.route("/status")
    def status() -> str:
        with worker.lock:
            payload = {
                "running": worker.state.running,
                "paused": worker.state.paused,
                "message": worker.state.message,
                "level": worker.state.current_level,
                "sprites": worker.state.sprites_placed,
                "residual": worker.state.residual_energy,
                "iteration_ms": worker.state.iteration_time_ms,
                "log": worker.state.log,
            }
        return jsonify(payload)

    @app.route("/preview")
    def preview() -> str:
        with worker.lock:
            preview_path = worker.state.preview_path
        if not preview_path or not preview_path.exists():
            return "", 204
        return send_file(preview_path)

    return app
