# Technical Plan: GPU-Accelerated Sprite Reconstruction

## Overview
Build a GPU-accelerated system that reconstructs a target image by iteratively placing transformed sprites. The system maintains a residual (target − canvas), searches for the single best sprite placement that reduces error most, and stops when improvement is no longer justified. A multiresolution pyramid is used to capture global structure first and refine details later.

---

## 1) Full Algorithm Flow (Step-by-Step)

1. **Load inputs and configuration**
   - Load target image, sprite atlas/collection, and configuration presets (quality/speed).
   - Configure search grid densities, top-K, refinement iterations, and stopping thresholds.

2. **Precompute assets**
   - Build target image pyramid (coarse → full resolution).
   - Build sprite pyramids (same levels): store alpha masks, luminance/feature templates, and any color basis used for fitting.

3. **Initialize state**
   - Set current level to the coarsest pyramid level.
   - Initialize canvas to zeros at that resolution.
   - Compute residual = target − canvas.

4. **Iterative placement loop (per level)**
   1. **Generate discrete candidate grid**
      - Enumerate positions (x, y), rotation, uniform scale, optional aspect ratio, and HSV presets based on level-specific grid settings.
   2. **GPU candidate scoring (coarse pass)**
      - For every candidate, estimate *upper-bound* error reduction using residual + sprite templates.
      - Produce global top-K candidates (across all sprites and parameters).
   3. **Refinement (continuous, local)**
      - For each top-K candidate, run local refinement with continuous parameters on a small patch.
      - Solve optimal opacity analytically; optionally solve color adjustment on the footprint.
      - Evaluate true error reduction on the local patch.
   4. **Select best refined candidate**
      - Choose the candidate with the highest actual error reduction.
   5. **Update canvas and residual**
      - Composite the chosen sprite onto the canvas (local patch update).
      - Update residual on the affected region (local patch update).
   6. **Check stopping criteria**
      - If improvement is below threshold or improvement rate stalls, stop at this level.
      - Otherwise, repeat another placement iteration.

5. **Resolution escalation**
   - When improvements at current level are exhausted, upscale canvas to next resolution.
   - Recompute residual at the new level (target − upscaled canvas).
   - Continue the loop at higher resolution.

6. **Terminate**
   - Stop when highest resolution is exhausted or global limits (e.g., max sprites) are reached.
   - Output final sprite list and composite image.

---

## 2) Multiresolution Strategy

- **Pyramid levels**: e.g., 1/8 → 1/4 → 1/2 → full resolution.
- **Start**: Coarsest level to capture large-scale structure quickly.
- **Advance to next level** when:
  - Best refined improvement < `min_gain(level)` for N consecutive iterations, or
  - Gain rate falls below `min_gain_rate`, or
  - Max sprites for the level reached.
- **At new level**:
  - Upscale canvas from previous level.
  - Recompute residual against target at new resolution.
  - Adjust grid density (finer for higher resolution).

---

## 3) Discrete vs. Continuous Parameters

**Discrete (GPU candidate grid)**
- Position (x, y): grid with level-dependent stride.
- Rotation: fixed steps (e.g., 16–64 angles).
- Uniform scale: fixed steps (e.g., 0.5–2.0).
- Optional aspect ratio: discrete set (e.g., 0.75, 1.0, 1.25).
- HSV presets: small, configurable palette (e.g., neutral, warm, cool, desaturated).

**Continuous (refinement)**
- Position (x, y): sub-pixel refinement within local neighborhood.
- Rotation: continuous adjustment around candidate angle.
- Scale: continuous adjustment.
- Aspect ratio: continuous adjustment if enabled.
- HSV adjustment: fine-tuning near preset.
- Opacity: solved analytically during local fitting.

---

## 4) Candidate Scoring (High-Level)

**Goal:** estimate maximum possible error reduction *without full rendering*.

- Use the residual and precomputed sprite templates (e.g., alpha-weighted luminance or per-channel basis).
- For each candidate placement:
  - Compute correlation between residual patch and sprite template.
  - Use an analytical upper bound on error reduction assuming optimal opacity (and optional color scaling).
- This yields a score proportional to the *best achievable* reduction for the candidate.
- Output is a ranked list; no canvas updates occur during scoring.

**Key property:** scoring is independent of the number of already placed sprites because it depends only on the current residual and precomputed sprite data.

---

## 5) Refinement vs. Scoring

**Scoring**
- Coarse, discrete, fast.
- Uses approximate correlation; does not render the sprite.
- Produces top-K candidates.

**Refinement**
- Continuous, local, more accurate.
- Renders only small patches to compute true error reduction.
- Solves optimal opacity (and optional color) analytically.
- Produces final candidate parameters for selection.

---

## 6) Stopping Conditions

**Per level**
- Best candidate’s improvement < `min_gain(level)`.
- Improvement rate over M iterations < `min_gain_rate`.
- Max sprites per level reached.

**Global**
- Highest resolution is exhausted.
- Total sprite count >= `max_sprites`.
- Residual error below global threshold.

---

## 7) Module Breakdown and Data Flow

**Modules**
1. **config/**
   - Presets for quality/speed, grid density, top-K, thresholds.
2. **pyramid/**
   - Build target pyramid and sprite pyramids.
3. **residual/**
   - Maintain canvas and residual; provide patch extraction and updates.
4. **gpu_scoring/**
   - GPU kernels for candidate generation, scoring, and top-K selection.
5. **refinement/**
   - Local continuous optimization and analytical fitting.
6. **compositor/**
   - Apply sprite to canvas and update residual locally.
7. **scheduler/**
   - Controls per-level iterations and resolution escalation.
8. **io/**
   - Load target image/sprites, save outputs.

**Data Flow**
Target pyramid → residual (current level) → GPU scoring → top-K candidates → refinement → best candidate → compositor updates canvas/residual → scheduler decides loop/advance.

---

## 8) Performance Risks & Mitigations

1. **Candidate scoring cost**
   - *Risk*: grid too dense → GPU overload.
   - *Mitigation*: level-dependent stride, adaptive grid density, configurable presets.

2. **Top-K selection overhead**
   - *Risk*: global sorting too expensive.
   - *Mitigation*: GPU partial selection, block-wise top-K then merge.

3. **Refinement overhead**
   - *Risk*: too many candidates refined.
   - *Mitigation*: small K, early rejection if score below threshold.

4. **Residual update cost**
   - *Risk*: full image recompute each iteration.
   - *Mitigation*: local patch update only; keep residual on GPU.

5. **High-resolution rendering cost**
   - *Risk*: expensive at full resolution.
   - *Mitigation*: coarse-to-fine strategy; refinement is localized.

---

## 9) Implementation-Oriented Presets

- **Fast**
  - Coarse grid, small K, fewer rotations/scales.
  - Early level escalation and higher min_gain thresholds.

- **Balanced**
  - Moderate grid density and K.
  - Default for general use.

- **High Quality**
  - Dense grid, large K, more refinement iterations.
  - Stricter improvement thresholds, slower but better output.

---

## Notes
- The scoring/refinement split ensures predictable iteration cost dominated by GPU kernels.
- Localized updates keep per-iteration costs stable even as the number of sprites grows.
- Quality/speed is fully controlled by configuration presets.
