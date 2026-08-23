# Milestone 05 — D2Q9 Lid-Driven Cavity

Single-rank lid-driven cavity flow: stationary bounce-back walls on three
sides, moving-wall (lid) boundary condition on the top, iterated to steady
state. This is the last single-process milestone before MPI domain
decomposition and multi-GPU scaling are introduced in later milestones.

- Grid: 256 × 256, `omega = 1.7`, lid velocity `u_lid = 0.1`
- BC order per step: stream → bounce-back → moving wall → collide
- Moving-wall closure: `rho_w` reconstructed from known post-stream
  populations, then `f4, f7, f8` set from the Zou–He-style relation
- Convergence: `max|du| < 1e-6` over consecutive steps, capped at 300,000
  steps; throughput reported in MLUPS
- Validated against Ghia et al. (1982) centerline velocity benchmark

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target milestone05
```

## Run

```bash
./build/executables/05/milestone05
```

Writes `velocity_field.csv` and `centerline.csv` to the working directory on
convergence (see `results/` for a saved reference run).

## Visualize

```bash
python3 executables/05/scripts/m05_py.py
```

## Results

`results/streamline_steady.png` — converged streamlines.
`results/centerline_ghia.png` — centerline velocity vs. Ghia et al. benchmark data.
`results/convergence.png` (+ `convergence_log.csv`) — `max|du|` convergence history.
`results/velocity_animation.gif` — transient evolution to steady state.
