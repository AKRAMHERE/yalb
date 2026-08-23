"""
Milestone 03 visualization for the D2Q9 Lattice Boltzmann solver.

Reads the per-timestep text dumps written by the Kokkos C++ code
("lbm_step_<N>.txt", columns: y x rho ux uy") and renders:

  1. An animated density heat map with a velocity quiver overlay
     (same core visualization as Milestone 02).
  2. A live "total mass" trace, to make the BGK operator's mass
     conservation visible frame by frame.
  3. A live "peak speed" trace, to make the relaxation/dissipation
     introduced by the collision step visible: in Milestone 02 the
     initial pulse just streams forever, in Milestone 03 it should
     decay as the distribution relaxes toward local equilibrium.

The script auto-discovers however many "lbm_step_*.txt" files are
present (instead of assuming a fixed step count) and infers the grid
size (ny, nx) from the data itself, so it stays correct even if NX/NY
or the number of steps in the C++ source change.
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

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
    """Infer (ny, nx) from the max y/x indices present in one step file."""
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


def generate_lbm_animation(directory=".", tau=None,
                            output_gif="lbm_milestone03_bgk_collision.gif"):
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

    # ---- Relaxation diagnostics ------------------------------------
    # Total mass should stay (numerically) constant: the BGK operator
    # conserves rho locally because sum_i f_i^eq = rho.
    #total_mass = np.array([r.sum() for r in all_rho])
    #this code below was added on 21.06.2026
    total_mass = np.array([r.sum() for r in all_rho])

    mass_error = (
        total_mass - total_mass[0]
    ) / total_mass[0]

    # Peak flow speed is expected to decay over time as the collision
    # operator relaxes the distribution toward local equilibrium and
    # dissipates the initial velocity perturbation.
    max_speed = np.array([
        np.sqrt(all_ux[i] ** 2 + all_uy[i] ** 2).max() for i in range(n_frames)
    ])

    # ---- Color scale, fit to the actual data range present ----------
    #rho_min = min(r.min() for r in all_rho)
    #rho_max = max(r.max() for r in all_rho)
    #pad = 0.05 * max(rho_max - rho_min, 1e-6)
    #vmin, vmax = rho_min - pad, rho_max + pad

    rho0 = 1.0

    vmin = rho0 - 0.01
    vmax = rho0 + 0.01

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.0, 1.2], height_ratios=[1, 1])
    ax_main = fig.add_subplot(gs[:, 0])
    ax_mass = fig.add_subplot(gs[0, 1])
    ax_speed = fig.add_subplot(gs[1, 1])
    #ax_nu = fig.add_subplot(gs[2,1])
    '''
    tau_values = np.linspace(0.51,5.0,500)
    nu_values = (tau_values-0.5)/3.0

    ax_nu.plot(tau_values,nu_values)

    if tau is not None:
        current_nu = (tau-0.5)/3.0

        ax_nu.plot(
            [tau],
            [current_nu],
            'ro'
        )

    ax_nu.set_title(r'$\nu$ vs $\tau$')
    ax_nu.set_xlabel(r'$\tau$')
    ax_nu.set_ylabel(r'$\nu$')
    ax_nu.grid(True)'''

    im = ax_main.imshow(all_rho[0], cmap='jet', origin='lower',
                         extent=[0, nx, 0, ny], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax_main, pad=0.03)
    cbar.set_label("Fluid Density ($\\rho$)", fontsize=11, fontweight='bold')

    X, Y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
    Q = ax_main.quiver(X, Y, all_ux[0], all_uy[0], color='white', scale=50,
                        width=0.001, pivot='middle')

    tau_suffix = f"  (tau = {tau:.3f})" if tau is not None else ""
    ax_main.set_title(f"LBM BGK Collision + Streaming (Milestone 03){tau_suffix}",
                       fontsize=12, fontweight='bold', pad=10)
    ax_main.set_xticks(np.arange(0, nx + 1, 1))
    ax_main.set_yticks(np.arange(0, ny + 1, 1))
    ax_main.grid(color='white', linestyle=':', alpha=0.25)

    txt_step = ax_main.text(
        0.5, ny - 0.7, "Time Step: 0", color='white', fontsize=11,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    # ---- Mass-conservation panel ----
    steps_axis = np.arange(n_frames)
    ax_mass.plot(steps_axis, total_mass, color='steelblue', lw=1.5)
    ax_mass.set_title("Total mass $\\sum \\rho$ (flat = conserved)", fontsize=9)
    ax_mass.set_title("Total Mass")
    ax_mass.set_xlabel("Time step", fontsize=8)
    #added on 21.06.2026:
    #ax_mass.set_ylabel(r'$(M-M_0)/M_0$')
    ax_mass.set_ylabel(r'Total Mass')
    ax_mass.tick_params(labelsize=8)
    mass_marker, = ax_mass.plot([0], [total_mass[0]], 'o', color='crimson', ms=6)
    mass_span = max(total_mass.max() - total_mass.min(), 1e-9)
    ax_mass.set_ylim(total_mass.min() - 0.1 * mass_span - 1e-9,
                      total_mass.max() + 0.1 * mass_span + 1e-9)
    #initial_mass = total_mass[0]
    #ax_mass.set_ylim(initial_mass - 1.0, initial_mass + 1.0)

    # ---- Relaxation panel: peak speed decaying as collisions dissipate flow ----
    ax_speed.plot(steps_axis, max_speed, color='darkorange', lw=1.5)
    ax_speed.set_title("Peak speed $|\\mathbf{u}|_{max}$ (decays as flow relaxes)",
                        fontsize=9)
    ax_speed.set_xlabel("Time step", fontsize=8)
    ax_speed.tick_params(labelsize=8)
    speed_marker, = ax_speed.plot([0], [max_speed[0]], 'o', color='crimson', ms=6)

    fig.tight_layout()
    
    def update_frame(frame_idx):
        im.set_array(all_rho[frame_idx])
        u, v = all_ux[frame_idx], all_uy[frame_idx]
        mag = np.sqrt(u ** 2 + v ** 2)
        u_masked = np.where(mag > 0.01, u, 0.0)
        v_masked = np.where(mag > 0.01, v, 0.0)
        #u_masked = u
        #v_masked = v
        Q.set_UVC(u_masked, v_masked)
        txt_step.set_text(f"Time Step: {steps[frame_idx]}")
        mass_marker.set_data([frame_idx], [total_mass[frame_idx]])
        speed_marker.set_data([frame_idx], [max_speed[frame_idx]])
        return im, Q, txt_step, mass_marker, speed_marker 
    '''
    def update_frame(frame_idx):
        im.set_array(all_rho[frame_idx])

        u, v = all_ux[frame_idx], all_uy[frame_idx]

        mag = np.sqrt(u**2 + v**2)

        u_vis = np.divide(
            u,
            mag,
            out=np.zeros_like(u),
            where=mag > 1e-12
        )

        v_vis = np.divide(
            v,
            mag,
            out=np.zeros_like(v),
            where=mag > 1e-12
        )

        Q.set_UVC(u_vis, v_vis)'''

    ani = animation.FuncAnimation(fig, update_frame, frames=n_frames,
                                   interval=400, blit=False)
    ani.save(output_gif, writer='pillow', fps=2.5)
    plt.close()

    mass_drift = abs(total_mass[-1] - total_mass[0]) / total_mass[0]
    print(f"Animation successfully compiled and stored as {output_gif}")
    print(f"Grid: {ny} x {nx}, frames: {n_frames}")
    print(f"Relative mass drift over the run: {mass_drift:.3e} "
          f"(should be ~machine precision)")


if __name__ == '__main__':
    # Optional: python visualize_lbm_milestone03.py [tau]
    # tau is only used to label the plot title; it does not need to match
    # exactly if you forget to pass it.
    tau_arg = float(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_lbm_animation(tau=tau_arg)
