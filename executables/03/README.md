# Milestone 03 — BGK Collision with Configurable Relaxation Time

Adds the BGK collision step to the Milestone 02 streaming kernel, with the
relaxation time `tau` exposed as a CLI argument.

- Grid: 15 × 10, 50 timesteps
- Stability constraint enforced: `omega = 1/tau`, `0 < omega < 2` ⇒ `tau > 0.5`
- Tracks total mass conservation across collision + streaming to validate
  the equilibrium distribution and collision operator.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target milestone03
```

## Run

```bash
./build/executables/03/milestone03 [tau]   # default tau = 1.0
```

## Visualize

```bash
python3 executables/03/scripts/py_milestone03.py
python3 executables/03/scripts/generate_lbm_snapshots.py
```

## Results

`results/lbm_milestone03_bgk_collision.gif` — evolution under collision + streaming.
`results/lbm_milestone03_mass.png` — mass-conservation check across timesteps.
`results/lbm_milestone03_snapshots.png`, `results/lbm_milestone03_speed.png` — field snapshots.
