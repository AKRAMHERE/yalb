#!/usr/bin/env python3
"""
m05_py.py — LBM Lid-Driven Cavity Visualization
Matches course Figure 2 exactly: streamlines colored by |u|, viridis colormap,
lid arrow, secondary vortex markers.

Usage:
  python3 m05_py.py                              # steady-state plots only
  python3 m05_py.py --re 400                     # Re=400 Ghia comparison
  python3 m05_py.py --snapshots snapshot_        # + animation GIF
  python3 m05_py.py --snapshots snapshot_ --fps 8 --skip 2

Run from inside build/ — that's where C++ writes the CSV files.
"""

import argparse
import glob
import os
import re as re_module
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

# ── Ghia 1982 benchmark data ─────────────────────────────────────────────────
GHIA = {
    100: {
        "y":  [0.0000,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,
               0.4531,0.5000,0.6172,0.7344,0.8516,0.9531,0.9609,
               0.9688,0.9766,1.0000],
        "ux": [-1.0000,-0.0372,-0.0419,-0.0477,-0.0643,-0.1015,-0.1566,
               -0.2109,-0.2058,-0.1364, 0.0033, 0.2315, 0.6872, 0.7372,
                0.7887, 0.8412, 1.0000],
    },
    400: {
        "y":  [0.0000,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,
               0.4531,0.5000,0.6172,0.7344,0.8516,0.9531,0.9609,
               0.9688,0.9766,1.0000],
        "ux": [-1.0000,-0.1811,-0.2021,-0.2222,-0.2973,-0.3829,-0.2973,
               -0.1259,-0.1259,-0.0602, 0.0570, 0.1867, 0.5580, 0.6172,
                0.6827, 0.7372, 1.0000],
    },
    1000: {
        "y":  [0.0000,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,
               0.4531,0.5000,0.6172,0.7344,0.8516,0.9531,0.9609,
               0.9688,0.9766,1.0000],
        "ux": [-1.0000,-0.3894,-0.4276,-0.4612,-0.5765,-0.5765,-0.4612,
               -0.3275,-0.3113,-0.1960,-0.0754, 0.0319, 0.3756, 0.4276,
                0.5000, 0.5765, 1.0000],
    },
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_velocity_field(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"x":np.int32,"y":np.int32,
                                       "ux":np.float64,"uy":np.float64})
    NX = int(df["x"].max()) + 1
    NY = int(df["y"].max()) + 1
    if len(df) != NX * NY:
        raise ValueError(f"Expected {NX*NY} rows, got {len(df)}")
    df = df.sort_values(["y","x"])
    ux = df["ux"].to_numpy().reshape(NY, NX)
    uy = df["uy"].to_numpy().reshape(NY, NX)
    return ux, uy, NX, NY

def extract_step(path: str) -> int:
    m = re_module.search(r"(\d+)\.csv$", os.path.basename(path))
    return int(m.group(1)) if m else 0

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _add_lid_arrow(ax):
    ax.annotate("", xy=(0.85, 1.04), xytext=(0.55, 1.04),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax.text(0.68, 1.055, r"$u_\mathrm{lid}$", transform=ax.transAxes,
            color="red", fontsize=11, ha="center", va="bottom")

def _streamplot_frame(ax, x1d, y1d, ux, uy, norm, title_str):
    """Draw one streamplot frame — the course Fig.2 style."""
    speed = np.sqrt(ux**2 + uy**2)
    strm = ax.streamplot(
        x1d, y1d, ux, uy,
        color=speed,
        cmap="viridis",
        norm=norm,
        linewidth=1.0,
        density=2.0,
        arrowsize=0.7,
        arrowstyle="->",
        minlength=0.05,
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x/L"); ax.set_ylabel("y/L")
    ax.set_title(title_str, fontsize=10)
    _add_lid_arrow(ax)
    return strm

# ── Plot 1: Steady-state streamlines ─────────────────────────────────────────

def plot_streamlines(ux, uy, NX, NY, re=100,
                     out_path="streamline_steady.png"):
    x1d = np.linspace(0, 1, NX)
    y1d = np.linspace(0, 1, NY)
    speed = np.sqrt(ux**2 + uy**2)
    norm  = mcolors.Normalize(vmin=0, vmax=speed.max())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    strm = _streamplot_frame(ax, x1d, y1d, ux, uy, norm,
                              f"Lid-driven cavity — steady state  (Re≈{re})")
    cbar = fig.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Velocity magnitude |u| (lattice units)", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Plot 2: Centerline vs Ghia ────────────────────────────────────────────────

def plot_centerline(csv_path="centerline.csv", re=100,
                    out_path="centerline_ghia.png"):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found — skipping.")
        return
    if re not in GHIA:
        re = min(GHIA, key=lambda r: abs(r-re))

    df     = pd.read_csv(csv_path)
    y_sim  = df["y_norm"].to_numpy()
    ux_sim = df["ux_norm"].to_numpy()

    g      = GHIA[re]
    y_ref  = np.array(g["y"])
    ux_ref = np.array(g["ux"])

    ux_interp = np.interp(y_ref, y_sim, ux_sim)
    ref_range = ux_ref.max() - ux_ref.min()
    pct_err   = 100.0 * np.abs(ux_interp - ux_ref) / ref_range
    rms_err   = np.sqrt(np.mean((ux_interp-ux_ref)**2)) / ref_range * 100.0

    print(f"Ghia Re={re} comparison:")
    print(f"  RMS error: {rms_err:.2f}%  (normalized by velocity range)")
    print(f"  Max error: {pct_err.max():.2f}%  at y={y_ref[np.argmax(pct_err)]:.4f}")
    print(f"  Course target: < ~5%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6),
                                    gridspec_kw={"width_ratios":[2,1]})
    ax1.plot(ux_sim, y_sim, color="#185FA5", lw=2, label="LBM simulation")
    ax1.scatter(ux_ref, y_ref, color="#D85A30", s=45, zorder=5,
                label=f"Ghia et al. Re={re}")
    ax1.axvline(0, color="gray", lw=0.5, ls="--")
    ax1.set_xlabel(r"$u_x\,/\,u_\mathrm{lid}$")
    ax1.set_ylabel("y / L")
    ax1.set_title("Centerline velocity")
    ax1.legend(frameon=False, fontsize=10)
    ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(0, 1)

    ax2.barh(y_ref, pct_err, height=0.03, color="#D85A30", alpha=0.8)
    ax2.axvline(5, color="gray", lw=0.8, ls="--", label="5% target")
    ax2.set_xlabel("% error")
    ax2.set_title(f"RMS: {rms_err:.1f}%")
    ax2.set_xlim(0, max(10, pct_err.max()*1.2))
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, fontsize=9)
    ax2.yaxis.set_ticklabels([])

    fig.suptitle(f"Centerline u_x vs Ghia et al.  Re={re}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Plot 3: Convergence history ───────────────────────────────────────────────

def plot_convergence(csv_path="convergence_log.csv",
                     out_path="convergence.png"):
    if not os.path.exists(csv_path):
        return
    try:
        df = pd.read_csv(csv_path, names=["step","max_du"])
        df = df[pd.to_numeric(df["step"], errors="coerce").notna()]
        df = df.astype({"step":int,"max_du":float})
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(df["step"], df["max_du"], color="#185FA5", lw=1.5)
    ax.axhline(1e-6, color="#D85A30", lw=1, ls="--",
               label=r"threshold $10^{-6}$")
    converged = df[df["max_du"] < 1e-6]
    if not converged.empty:
        s = converged.iloc[0]["step"]
        ax.axvline(s, color="gray", lw=0.8, ls=":")
        ax.text(s, df["max_du"].max()*0.3, f" step {s:,}",
                fontsize=9, color="gray")
    ax.set_xlabel("Step"); ax.set_ylabel(r"max $|\Delta u|$")
    ax.set_title("Convergence history")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── Plot 4: Animation — streamplot per frame, matches course Fig.2 ────────────

def make_animation(snapshot_files: list,
                   out_path: str = "velocity_animation.gif",
                   fps: int = 8,
                   skip: int = 1):
    """
    GIF animation using streamplot per frame — same style as course Fig.2.
    Each frame: streamlines colored by |u|, viridis, consistent colorscale,
    lid arrow, step counter in title.

    streamplot is slow (~2s/frame at 256x256). At skip=2 with ~25 snapshots
    you get ~12 frames, rendering in ~25s total.
    """
    files = snapshot_files[::skip]
    if not files:
        print("No snapshot files found.")
        return

    print(f"Pre-loading {len(files)} snapshots...")
    frames_ux, frames_uy, steps = [], [], []
    for path in files:
        try:
            ux, uy, NX, NY = load_velocity_field(path)
            frames_ux.append(ux); frames_uy.append(uy)
            steps.append(extract_step(path))
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    if not frames_ux:
        print("No valid snapshots.")
        return

    NX, NY = frames_ux[0].shape[1], frames_ux[0].shape[0]
    x1d = np.linspace(0, 1, NX)
    y1d = np.linspace(0, 1, NY)

    # Fixed colorscale — use the FINAL (converged) frame's max speed as vmax
    # so early frames don't look artificially bright
    final_speed = np.sqrt(frames_ux[-1]**2 + frames_uy[-1]**2)
    vmax = final_speed.max()
    vmax = max(vmax, 1e-9)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    print(f"Rendering {len(frames_ux)} frames with streamplot...")
    print(f"  Grid: {NX}x{NY}  vmax={vmax:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

    # Colorbar — drawn once, stays for all frames
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|u| (lattice units)", fontsize=9)

    fig.tight_layout()

    def update(i):
        ax.cla()   # clear axes — must redo streamplot from scratch each frame
        ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("x/L", fontsize=9)
        ax.set_ylabel("y/L", fontsize=9)

        ux, uy = frames_ux[i], frames_uy[i]
        speed  = np.sqrt(ux**2 + uy**2)

        # streamplot — exact same params as the steady-state plot
        ax.streamplot(
            x1d, y1d, ux, uy,
            color=speed,
            cmap="viridis",
            norm=norm,
            linewidth=0.9,
            density=1.8,
            arrowsize=0.6,
            arrowstyle="->",
            minlength=0.04,
        )

        status = "converged" if i == len(frames_ux)-1 else "developing"
        ax.set_title(f"Step {steps[i]:,}  ({status})", fontsize=9, pad=22)

        # Lid arrow — must redraw after ax.cla(). Sits just above the axes,
        # well clear of the title (which now has extra pad).
        ax.annotate("", xy=(0.84, 1.015), xytext=(0.56, 1.015),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.6))
        ax.text(0.68, 1.03, r"$u_\mathrm{lid}$", transform=ax.transAxes,
                color="red", fontsize=9, ha="center", va="bottom")

        if i % 3 == 0:
            print(f"  frame {i+1}/{len(frames_ux)}  step={steps[i]:,}")

    anim = FuncAnimation(fig, update, frames=len(frames_ux),
                         interval=1000//fps, blit=False, repeat=True)

    print(f"Writing {out_path} ...")
    anim.save(out_path, writer=PillowWriter(fps=fps),
              savefig_kwargs={"bbox_inches":"tight"})
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LBM lid-driven cavity visualization")
    p.add_argument("--field",       default="velocity_field.csv")
    p.add_argument("--centerline",  default="centerline.csv")
    p.add_argument("--convergence", default="convergence_log.csv")
    p.add_argument("--re",          type=int, default=100, choices=[100,400,1000])
    p.add_argument("--snapshots",   default=None,
                   help="Snapshot prefix, e.g. snapshot_")
    p.add_argument("--fps",         type=int, default=8)
    p.add_argument("--skip",        type=int, default=1,
                   help="Use every N-th snapshot (default 1 = all)")
    args = p.parse_args()

    if os.path.exists(args.field):
        try:
            ux, uy, NX, NY = load_velocity_field(args.field)
            plot_streamlines(ux, uy, NX, NY, re=args.re)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Field file not found: {args.field}")

    plot_centerline(args.centerline, re=args.re)
    plot_convergence(args.convergence)

    if args.snapshots:
        files = sorted(glob.glob(args.snapshots + "*.csv"))
        if files:
            print(f"Found {len(files)} snapshot files.")
            make_animation(files, fps=args.fps, skip=args.skip)
        else:
            print(f"No files matching: {args.snapshots}*.csv")
            print("Make sure you ran ./lbm_m05 first — it writes snapshot_000000.csv etc.")

if __name__ == "__main__":
    main()