"""
Static snapshot generator for the Milestone 03 D2Q9 LBM solver.

Produces LaTeX-ready static images instead of an animated GIF:
  1. A multi-panel "filmstrip" of density + velocity quiver at selected
     time steps (e.g. t=0, early peak, mid-relaxation, late/steady).
  2. A standalone total-mass-vs-time plot (conservation check).
  3. A standalone peak-speed-vs-time plot (relaxation/decay check).

Reads the same lbm_step_<N>.txt dumps as py_milestone03.py, so run it
from the same directory (e.g. run/) after the C++ binary has executed.

Usage:
    python3 generate_lbm_snapshots.py [tau] [n_snapshots] [format]

    tau          - optional, only used to label the filmstrip title
    n_snapshots  - optional, number of density panels in the filmstrip (default 6)
    format       - optional, "png" (default, raster) or "pdf" (vector wrapper)
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

STEP_FILE_RE = re.compile(r"lbm_step_(\d+)\.txt$")


def discover_steps(directory="."):
    """Find every lbm_step_<N>.txt file present and return sorted step indices."""
    found = []
    for path in glob.glob(os.path.join(directory, "lbm_step_*.txt")):
        match = STEP_FILE_RE.search(os.path.basename(path))
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def infer_grid_size(step, directory="."):
    data = np.loadtxt(os.path.join(directory, f"lbm_step_{step}.txt"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ny = int(data[:, 0].max()) + 1
    nx = int(data[:, 1].max()) + 1
    return ny, nx


def load_step_data(step, ny, nx, directory="."):
    filename = os.path.join(directory, f"lbm_step_{step}.txt")
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    rho, ux, uy = np.zeros((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx))
    for row in data:
        y, x = int(row[0]), int(row[1])
        rho[y, x], ux[y, x], uy[y, x] = row[2], row[3], row[4]
    return rho, ux, uy


def generate_static_figures(directory=".", tau=None, out_prefix="lbm_milestone03",
                             n_snapshots=6, fmt="png", dpi=300):
    steps = discover_steps(directory)
    if not steps:
        print("Data text logs missing. Run the compiled C++ code first.")
        return

    ny, nx = infer_grid_size(steps[0], directory)

    all_rho, all_ux, all_uy = [], [], []
    for s in steps:
        rho, ux, uy = load_step_data(s, ny, nx, directory)
        all_rho.append(rho)
        all_ux.append(ux)
        all_uy.append(uy)

    n_frames = len(all_rho)
    total_mass = np.array([r.sum() for r in all_rho])
    max_speed = np.array([np.sqrt(all_ux[i] ** 2 + all_uy[i] ** 2).max() for i in range(n_frames)])

    rho0 = 1.0
    vmin, vmax = rho0 - 0.01, rho0 + 0.01

    # ---- 1. Filmstrip of density + quiver at selected steps ----
    n_snapshots = min(n_snapshots, n_frames)
    snap_idx = sorted(set(np.linspace(0, n_frames - 1, n_snapshots).astype(int)))

    fig, axes = plt.subplots(1, len(snap_idx), figsize=(3.2 * len(snap_idx), 3.6), sharey=True)
    if len(snap_idx) == 1:
        axes = [axes]

    X, Y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    im = None
    for ax, idx in zip(axes, snap_idx):
        im = ax.imshow(all_rho[idx], cmap='jet', origin='lower',
                        extent=[0, nx, 0, ny], vmin=vmin, vmax=vmax)
        u, v = all_ux[idx], all_uy[idx]
        mag = np.sqrt(u ** 2 + v ** 2)
        u_masked = np.where(mag > 0.01, u, 0.0)
        v_masked = np.where(mag > 0.01, v, 0.0)
        ax.quiver(X, Y, u_masked, v_masked, color='white', scale=50, width=0.0025, pivot='middle')
        ax.set_title(f"t = {steps[idx]}", fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    tau_suffix = f"  (tau = {tau:.3f})" if tau is not None else ""
    fig.suptitle(f"LBM BGK Collision + Streaming (Milestone 03){tau_suffix}",
                 fontsize=12, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Fluid Density ($\\rho$)", fontsize=10, fontweight='bold')

    filmstrip_path = f"{out_prefix}_snapshots.{fmt}"
    fig.savefig(filmstrip_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # ---- 2. Total mass vs time (standalone) ----
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.plot(steps, total_mass, color='steelblue', lw=1.5)
    ax2.set_title("Total Mass $\\sum \\rho$ (flat = conserved)")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total mass")
    ax2.grid(alpha=0.3)
    mass_path = f"{out_prefix}_mass.{fmt}"
    fig2.savefig(mass_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig2)

    # ---- 3. Peak speed vs time (standalone) ----
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    ax3.plot(steps, max_speed, color='darkorange', lw=1.5)
    ax3.set_title("Peak Speed $|\\mathbf{u}|_{max}$ (decays as flow relaxes)")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Peak speed")
    ax3.grid(alpha=0.3)
    speed_path = f"{out_prefix}_speed.{fmt}"
    fig3.savefig(speed_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig3)

    mass_drift = abs(total_mass[-1] - total_mass[0]) / total_mass[0]
    print(f"Saved: {filmstrip_path}")
    print(f"Saved: {mass_path}")
    print(f"Saved: {speed_path}")
    print(f"Grid: {ny} x {nx}, frames used: {n_frames}, snapshots: {len(snap_idx)}")
    print(f"Relative mass drift over the run: {mass_drift:.3e} (should be ~machine precision)")


if __name__ == '__main__':
    tau_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    n_snap_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    fmt_arg = sys.argv[3] if len(sys.argv) > 3 else "png"
    generate_static_figures(tau=tau_arg, n_snapshots=n_snap_arg, fmt=fmt_arg)
