# Milestone 02 — D2Q9 Streaming on a Periodic Lattice

Kokkos implementation of the pure streaming operator for a D2Q9 lattice
(no collision). Verifies that a localized density perturbation advects
correctly across periodic boundaries before BGK collision is introduced
in later milestones.

- Grid: 15 × 10 (`NX × NY`)
- Views: `f`, `cx`, `cy` declared and destroyed inside the same scope block
  as `Kokkos::initialize`/`finalize`, avoiding the "View deallocated after
  Kokkos::finalize" fault present in an earlier draft of this milestone.

## Build

From the repository root:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target milestone02
```

## Run

```bash
./build/executables/02/milestone02
```

Per-timestep density/velocity/population snapshots are written as CSVs to
`Outputs/` in the working directory (ignored by git — regenerate by running
the binary; see `.gitignore`).

## Visualize

```bash
python3 executables/02/scripts/visualize_periodic_lbm.py
```

## Results

`results/explosion_grid.png` and `results/explosion_periodic.gif` show the
density spike streaming through and wrapping around the periodic domain.
