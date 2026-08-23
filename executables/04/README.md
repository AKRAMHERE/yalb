# Milestone 04 — Shear Wave Decay & Viscosity Validation

Extends Milestone 03 to a 64 × 64 periodic domain and runs a shear-wave
decay test: a sinusoidal velocity perturbation (selectable wavelength mode
via CLI) decays exponentially under BGK collision. Fitting the decay rate
against the analytical LBM shear viscosity, `nu = (tau - 0.5) / 3`, validates
the collision operator's numerical viscosity across a range of `tau`.

- Grid: 64 × 64, 500 timesteps
- CLI: `tau` (must be `> 0.5` for stability), `mode` (perturbation wavelength)

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target milestone04
```

## Run

```bash
./build/executables/04/milestone04 [tau] [mode]   # default tau = 1.0, mode = 1
```

## Visualize / validate

```bash
python3 executables/04/scripts/py_milestone04.py
python3 executables/04/scripts/figure_shearwave_report.py
python3 executables/04/scripts/compare_m04.py
```

## Results

`results/lbm_milestone04_shear_wave.gif`, `results/periodic_lbm_grid.png` — field evolution.
`results/shear_wave_mode_comparison.png`, `results/shear_wave_report.png` — mode comparison and decay-rate report.
`results/viscosity_validation.png`, `results/viscosity_results.txt` — measured vs. analytical `nu(tau)` from a `tau` sweep (`run_visc_sweep`).
