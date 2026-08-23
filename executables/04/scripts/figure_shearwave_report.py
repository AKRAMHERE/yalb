"""
Combined shear-wave decay report figure (single run).

Produces a 3-panel PNG:
  - top-left    : u_x(y) overlaid at many timesteps, color-coded early -> late
  - bottom-left : final wave profile alone
  - right       : normalized Fourier-mode amplitude A(t)/A(0) vs timestep

Reuses the same directory-parsing logic as py_milestone04.py /
compare_m04.py, so it drops straight into your existing output folder
(one containing lbm_04_step_*.txt from m04_210626.cpp).

Usage:
    python figure_shearwave_report.py <directory> [--n-curves 25] [--mode 1] [--out shear_wave_report.png]

Example:
    python figure_shearwave_report.py run_tau1.0 --n-curves 30
"""
import argparse
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

STEP_RE = re.compile(r"lbm_04_step_(\d+)\.txt$")


def discover_steps(directory):
    steps = []
    for f in glob.glob(os.path.join(directory, "lbm_04_step_*.txt")):
        m = STEP_RE.search(os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_ux_profile(directory, step):
    """Return the y-averaged u_x(y) profile at a given step."""
    data = np.loadtxt(os.path.join(directory, f"lbm_04_step_{step}.txt"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ny = int(data[:, 0].max()) + 1
    nx = int(data[:, 1].max()) + 1
    ux = np.zeros((ny, nx))
    for row in data:
        y, x = int(row[0]), int(row[1])
        ux[y, x] = row[3]
    return ux.mean(axis=1)  # average over x -> u_x(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="folder containing lbm_04_step_*.txt")
    ap.add_argument("--n-curves", type=int, default=25,
                     help="number of overlaid profiles in the top-left panel")
    ap.add_argument("--mode", type=int, default=1,
                     help="Fourier mode used at init (matches --mode passed to the C++ binary)")
    ap.add_argument("--out", default="shear_wave_report.png")
    args = ap.parse_args()

    steps = discover_steps(args.directory)
    if not steps:
        raise SystemExit(f"No lbm_04_step_*.txt files found in {args.directory}")

    profiles = {s: load_ux_profile(args.directory, s) for s in steps}
    ny = len(profiles[steps[0]])
    y = np.arange(ny)
    basis = np.sin(2 * np.pi * args.mode * y / ny)

    amps = np.array([(2.0 / ny) * np.sum(profiles[s] * basis) for s in steps])
    amps_norm = amps / amps[0]

    # evenly-spaced subset of steps for the overlay panel
    idx = np.unique(np.linspace(0, len(steps) - 1, args.n_curves).astype(int))
    curve_steps = [steps[i] for i in idx]
    cmap = plt.colormaps["viridis"].resampled(len(curve_steps))
    norm = Normalize(vmin=curve_steps[0], vmax=curve_steps[-1])

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.3])
    ax_overlay = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[1, 0])
    ax_amp = fig.add_subplot(gs[:, 1])

    for c, s in enumerate(curve_steps):
        ax_overlay.plot(y, profiles[s], color=cmap(c), linewidth=1.2)
    ax_overlay.set_title("Wave decay over time")
    ax_overlay.set_xlabel("y")
    ax_overlay.set_ylabel(r"$u_x(y)$")
    ax_overlay.grid(True)

    sm = ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_overlay, pad=0.02)
    cbar.set_label("Time step")

    ax_final.plot(y, profiles[steps[-1]], color="tab:blue", linewidth=1.5)
    ax_final.set_title(f"Wave profile at step {steps[-1]}")
    ax_final.set_xlabel("y")
    ax_final.set_ylabel(r"$u_x(y)$")
    ax_final.set_ylim(ax_overlay.get_ylim())
    ax_final.grid(True)

    ax_amp.plot(steps, amps_norm, linewidth=2, color="tab:blue")
    ax_amp.set_title("Amplitude decay")
    ax_amp.set_xlabel("Time step")
    ax_amp.set_ylabel(r"$A(t)/A(0)$")
    ax_amp.grid(True)

    fig.suptitle("Shear Wave Decay", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()