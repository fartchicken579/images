# Sprite Recon

Sprite Recon reconstructs a target image by greedily placing and blending sprites. The project includes a minimal web UI and a CLI runner.

## Quick start (recommended)

These scripts create a local virtual environment, install dependencies only when needed, and start the app.

### macOS / Linux

```bash
./run.sh
```

### Windows (PowerShell)

```powershell
.\run.ps1
```

By default, the scripts launch the web UI at `http://127.0.0.1:5000`.

## CLI usage

You can run the CLI through the unified entrypoint:

```bash
python -m app cli -- \
  --target /path/to/target.png \
  --sprites-dir /path/to/sprites \
  --output-dir /path/to/output \
  --preset balanced \
  --max-sprites 500
```

## UI options

Run the web UI directly, optionally choosing host/port:

```bash
python -m app ui --host 127.0.0.1 --port 5000
```

## Requirements

- Python 3.10+
- Internet access for first-time dependency installation

## Notes

- The CLI remains available and unchanged; the scripts simply wrap the same `python -m app` entrypoint.
- Outputs are written to the output directory you specify via the UI or CLI.
