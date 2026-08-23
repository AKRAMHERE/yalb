#!/usr/bin/env python3
"""
Visualize LBM D2Q9 periodic-BC streaming output from m02_1905_fixed.cpp.

Reads the rho_t{t}.csv (x,y,value) and vel_t{t}.csv (x,y,vx,vy) files
written by the simulation and renders density heatmaps with velocity
arrows, styled to match the outflow-BC reference plot. Because this
build uses periodic wrap-around instead of outflow BC, packets that
exit one edge re-enter from the opposite edge instead of being
absorbed -- that's the behavior this script is meant to make visible.

Requires: numpy, matplotlib, pillow (only for --animate)
    pip install numpy matplotlib pillow --break-system-packages

Usage
-----
Run the simulation first so rho_t*.csv / vel_t*.csv exist in a folder:
    ./m02_1905_fixed

Then, from that same folder (or pass --dir):

  Single timestep snapshot:
    python3 visualize_periodic_lbm.py --dir . --snapshot 8 --out t8.png

  Grid of evenly spaced snapshots (default behavior, no flags needed):
    python3 visualize_periodic_lbm.py --dir .

  Full animated GIF over every timestep:
    python3 visualize_periodic_lbm.py --dir . --animate --out periodic_lbm.gif

With NX=15 and N_STEPS=30 (the defaults in m02_1905_fixed.cpp), the
axis-aligned packets (i=1,2,3,4) first cross an edge around t=7-8 and
the diagonal packets (i=5,6,7,8) around t=5-8 -- watch for those steps
specifically to see the wrap-around, since that's the one behavior
that distinguishes this from the outflow-BC version.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

NX_DEFAULT = 15
NY_DEFAULT = 10


def load_rho(path, nx, ny):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    grid = np.zeros((ny, nx))
    xs, ys, vals = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2]
    grid[ys, xs] = vals
    return grid


def load_vel(path, nx, ny):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    gvx, gvy = np.zeros((ny, nx)), np.zeros((ny, nx))
    xs, ys = data[:, 0].astype(int), data[:, 1].astype(int)
    gvx[ys, xs] = data[:, 2]
    gvy[ys, xs] = data[:, 3]
    return gvx, gvy


def discover_steps(directory):
    steps = []
    for f in glob.glob(os.path.join(directory, "rho_t*.csv")):
        m = re.search(r"rho_t(\d+)\.csv$", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def auto_range(directory, steps, nx, ny):
    """Scan all rho_t*.csv to find a sensible color range, instead of
    assuming the old explosion-init scale (baseline 0.9, peak 1.5).
    Using the raw global max is fragile: initializations where multiple
    directions briefly overlap in one cell (e.g. an 8-direction
    'explosion' seed at t=0) produce a huge one-off spike that would
    wash out every other frame's contrast. Taking the MEDIAN of each
    frame's own peak value is robust to that -- it reflects the
    'typical' peak intensity across the run rather than a rare outlier."""
    frame_maxes = []
    for step in steps:
        rho = load_rho(os.path.join(directory, f"rho_t{step}.csv"), nx, ny)
        m = rho.max()
        if m > 0:
            frame_maxes.append(m)
    vmax = float(np.median(frame_maxes)) if frame_maxes else 1.0
    return 0.0, vmax


def draw_frame(ax, directory, step, nx, ny, vmin, vmax, show_title=True):
    rho = load_rho(os.path.join(directory, f"rho_t{step}.csv"), nx, ny)
    vx, vy = load_vel(os.path.join(directory, f"vel_t{step}.csv"), nx, ny)

    ax.clear()
    ax.set_facecolor("#0b0b6b")

    im = ax.pcolormesh(
        np.arange(nx + 1) - 0.5, np.arange(ny + 1) - 0.5, rho,
        cmap="plasma", vmin=vmin, vmax=vmax, shading="flat",
    )

    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    ax.plot(gx, gy, ".", color="white", markersize=1.5, alpha=0.5)

    speed = np.sqrt(vx**2 + vy**2)
    mask = speed > 1e-6
    if mask.any():
        ax.quiver(
            gx[mask], gy[mask], vx[mask], vy[mask],
            color="white", scale=8, scale_units="xy",
            width=0.006, headwidth=4, headlength=5,
        )

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(range(nx))
    ax.set_yticks(range(ny + 1))
    ax.set_xlabel("X Lattice Position")
    ax.set_ylabel("Y Lattice Position")
    if show_title:
        ax.set_title("LBM Periodic Fluid Streaming (Wrap-Around Boundaries)",
                     fontsize=13, fontweight="bold")
    ax.grid(True, color="white", linestyle=":", linewidth=0.4, alpha=0.4)

    ax.text(
        0.02, 0.96, f"Time Step: {step}  (Wrap-Around Active)",
        transform=ax.transAxes, fontsize=10, color="white",
        bbox=dict(boxstyle="round", facecolor="#1a1a1a", alpha=0.85, edgecolor="none"),
        va="top",
    )
    return im


def main():
    ap = argparse.ArgumentParser(description="Visualize periodic-BC LBM streaming output")
    ap.add_argument("--dir", default=".", help="folder containing rho_t*.csv / vel_t*.csv")
    ap.add_argument("--nx", type=int, default=NX_DEFAULT)
    ap.add_argument("--ny", type=int, default=NY_DEFAULT)
    ap.add_argument("--animate", action="store_true", help="render an animated GIF over all steps")
    ap.add_argument("--snapshot", type=int, default=None, help="render one timestep as PNG")
    ap.add_argument("--out", default=None, help="output filename")
    ap.add_argument("--vmin", type=float, default=None,
                     help="colorbar min; omit to auto-scale from the data")
    ap.add_argument("--vmax", type=float, default=None,
                     help="colorbar max; omit to auto-scale from the data")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    steps = discover_steps(args.dir)
    if not steps:
        print(f"No rho_t*.csv files found in {args.dir}. Run the simulation binary first.",
              file=sys.stderr)
        sys.exit(1)

    if args.vmin is None or args.vmax is None:
        auto_vmin, auto_vmax = auto_range(args.dir, steps, args.nx, args.ny)
        if args.vmin is None:
            args.vmin = auto_vmin
        if args.vmax is None:
            args.vmax = auto_vmax
        print(f"Auto-scaled color range: vmin={args.vmin:.4f}, vmax={args.vmax:.4f}")

    if args.snapshot is not None:
        if args.snapshot not in steps:
            print(f"Step {args.snapshot} not found. Available: {steps}", file=sys.stderr)
            sys.exit(1)
        fig, ax = plt.subplots(figsize=(9, 6))
        im = draw_frame(ax, args.dir, args.snapshot, args.nx, args.ny, args.vmin, args.vmax)
        fig.colorbar(im, ax=ax, label="Fluid Density (\u03c1)")
        fig.tight_layout()
        out = args.out or f"periodic_lbm_t{args.snapshot}.png"
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
        return

    if args.animate:
        fig, ax = plt.subplots(figsize=(9, 6))
        im = draw_frame(ax, args.dir, steps[0], args.nx, args.ny, args.vmin, args.vmax)
        fig.colorbar(im, ax=ax, label="Fluid Density (\u03c1)")

        def update(step):
            draw_frame(ax, args.dir, step, args.nx, args.ny, args.vmin, args.vmax)
            return []

        anim = FuncAnimation(fig, update, frames=steps, blit=False)
        out = args.out or "periodic_lbm.gif"
        anim.save(out, writer=PillowWriter(fps=args.fps))
        print(f"Saved {out}")
        return

    # Default: grid of evenly spaced snapshots, chosen to include wrap-around steps
    n_show = min(6, len(steps))
    idxs = np.linspace(0, len(steps) - 1, n_show).astype(int)
    chosen = [steps[i] for i in idxs]

    ncols = 3
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    last_im = None
    for ax_i, step in zip(axes, chosen):
        last_im = draw_frame(ax_i, args.dir, step, args.nx, args.ny, args.vmin, args.vmax,
                              show_title=False)
    for ax_i in axes[len(chosen):]:
        ax_i.axis("off")

    fig.suptitle("LBM Periodic Fluid Streaming (Wrap-Around Boundaries)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.colorbar(last_im, ax=axes.tolist(), label="Fluid Density (\u03c1)", shrink=0.8)
    out = args.out or "periodic_lbm_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out} (steps: {chosen})")


if __name__ == "__main__":
    main()