import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

STEP_FILE_RE = re.compile(r"lbm_04_step_(\d+)\.txt$")


def discover_steps(directory="."):
    found = []
    for path in glob.glob(os.path.join(directory, "lbm_04_step_*.txt")):
        match = STEP_FILE_RE.search(os.path.basename(path))
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def infer_grid_size(step, directory="."):
    data = np.loadtxt(
        os.path.join(directory, f"lbm_04_step_{step}.txt")
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ny = int(data[:, 0].max()) + 1
    nx = int(data[:, 1].max()) + 1

    return ny, nx


def load_step_data(step, ny, nx, directory="."):

    filename = os.path.join(
        directory,
        f"lbm_04_step_{step}.txt"
    )

    data = np.loadtxt(filename)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    rho = np.zeros((ny, nx))
    ux  = np.zeros((ny, nx))
    uy  = np.zeros((ny, nx))

    for row in data:

        y = int(row[0])
        x = int(row[1])

        rho[y, x] = row[2]
        ux[y, x]  = row[3]
        uy[y, x]  = row[4]

    return rho, ux, uy


def generate_milestone04_visualization(
        directory=".",
        tau=None,
        output_gif="lbm_milestone04_shear_wave.gif"):

    steps = discover_steps(directory)

    if not steps:
        print("No simulation files found.")
        return

    ny, nx = infer_grid_size(steps[0], directory)

    all_rho = []
    all_ux  = []
    all_uy  = []

    for s in steps:

        rho, ux, uy = load_step_data(
            s,
            ny,
            nx,
            directory
        )

        all_rho.append(rho)
        all_ux.append(ux)
        all_uy.append(uy)

    n_frames = len(all_rho)

    # --------------------------------------------------
    # Mass conservation
    # --------------------------------------------------

    total_mass = np.array([
        r.sum()
        for r in all_rho
    ])

    # --------------------------------------------------
    # Shear-wave amplitude
    # --------------------------------------------------
    '''
    amplitude = np.array([
        np.max(np.abs(all_ux[i]))
        for i in range(n_frames)
    ]) '''

    # --------------------------------------------------
    # Shear-wave amplitude (Fourier mode extraction)
    # --------------------------------------------------

    y = np.arange(ny)

    mode = np.sin(
        2.0 * np.pi * y / ny
    )

    amplitude = []

    for ux_field in all_ux:

        profile = np.mean(ux_field, axis=1)

        amp = (
            2.0 / ny
        ) * np.sum(
            profile * mode
        )

        amplitude.append(amp)

    amplitude = np.array(amplitude)

    # --------------------------------------------------
    # Viscosity extraction
    # --------------------------------------------------

    k = 2.0 * np.pi / ny

    safe_amp = np.maximum(np.abs(amplitude), 1e-15)

    log_amp = np.log(safe_amp)

    steps_axis = np.array(steps)

    slope, intercept = np.polyfit(
        steps_axis,
        log_amp,
        1
    )

    nu_measured = -slope / (k * k)

    if tau is not None:

        omega = 1.0 / tau

        nu_theory = (
            tau - 0.5
        ) / 3.0

        rel_error = (
            abs(
                nu_measured
                - nu_theory
            )
            / abs(nu_theory)
        )

        with open(
            "viscosity_results.txt",
            "a"
        ) as f:

            f.write(
                f"{tau} "
                f"{omega} "
                f"{nu_measured} "
                f"{nu_theory} "
                f"{rel_error}\n"
            )

    

    print("\n===== Viscosity Measurement =====")
    print(f"Measured viscosity : {nu_measured:.8f}")

    if tau is not None:

        nu_theory = (tau - 0.5) / 3.0

        rel_error = (
            abs(nu_measured - nu_theory)
            / abs(nu_theory)
        )

        print(f"Theoretical viscosity : {nu_theory:.8f}")
        print(f"Relative error        : {100.0*rel_error:.4f}%")

    # --------------------------------------------------
    # Figure layout
    # --------------------------------------------------

    fig = plt.figure(figsize=(14, 8))

    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[1.7, 1.0],
        height_ratios=[1, 1]
    )

    ax_profile = fig.add_subplot(gs[:, 0])
    ax_mass    = fig.add_subplot(gs[0, 1])
    ax_log     = fig.add_subplot(gs[1, 1])

    # --------------------------------------------------
    # Velocity profile panel
    # --------------------------------------------------

    y_axis = np.arange(ny)

    line_profile, = ax_profile.plot(
        y_axis,
        all_ux[0][:, 0],
        linewidth=2
    )

    ax_profile.set_title(
        "Shear-Wave Velocity Profile"
    )

    ax_profile.set_xlabel("y")
    ax_profile.set_ylabel(r"$u_x(y)$")

    ax_profile.grid(True)

    max_u = max(
        np.max(np.abs(u))
        for u in all_ux
    )

    ax_profile.set_ylim(
        -1.1 * max_u,
        1.1 * max_u
    )

    txt_step = ax_profile.text(
        0.02,
        0.95,
        "Time Step: 0",
        transform=ax_profile.transAxes,
        fontsize=11,
        bbox=dict(
            facecolor='white',
            alpha=0.8
        )
    )

    # --------------------------------------------------
    # Total mass panel
    # --------------------------------------------------

    #steps_axis = np.arange(steps)
    steps_axis = np.array(steps)

    ax_mass.plot(
        steps_axis,
        total_mass,
        linewidth=2
    )

    mass_marker, = ax_mass.plot(
        [0],
        [total_mass[0]],
        'ro'
    )

    ax_mass.set_title(
        "Total Mass"
    )

    ax_mass.set_xlabel(
        "Time Step"
    )

    ax_mass.set_ylabel(
        r"$\sum \rho$"
    )

    ax_mass.grid(True)

    # --------------------------------------------------
    # Log amplitude panel
    # --------------------------------------------------

    ax_log.plot(
        steps_axis,
        log_amp,
        linewidth=2,
        label=r'$\ln(A)$'
    )

    fit_line = (
        slope * steps_axis
        + intercept
    )

    theory_decay = (
    intercept
    -
    nu_measured *
    (k*k) *
    steps_axis)

    ax_log.plot(
    steps_axis,
    theory_decay,
    ':',
    linewidth=2,
    label='Decay Model')

    ax_log.plot(
        steps_axis,
        fit_line,
        '--',
        linewidth=2,
        label='Linear Fit'
    )

    log_marker, = ax_log.plot(
        [0],
        [log_amp[0]],
        'ro'
    )

    ax_log.set_title(
        r'$\ln(A)$ vs Time'
    )

    ax_log.set_xlabel(
        "Time Step"
    )

    ax_log.set_ylabel(
        r'$\ln(A)$'
    )

    ax_log.legend()

    ax_log.grid(True)

    fig.tight_layout()

    # --------------------------------------------------
    # Animation update
    # --------------------------------------------------

    def update(frame_idx):

        line_profile.set_ydata(
            all_ux[frame_idx][:, 0]
        )

        txt_step.set_text(
            f"Time Step: {steps[frame_idx]}"
        )

        mass_marker.set_data(
            [frame_idx],
            [total_mass[frame_idx]]
        )

        log_marker.set_data(
            [frame_idx],
            [log_amp[frame_idx]]
        )

        return (
            line_profile,
            txt_step,
            mass_marker,
            log_marker
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=250,
        blit=False
    )

    ani.save(
        output_gif,
        writer="pillow",
        fps=4
    )

    plt.close()

    # --------------------------------------------------
    # ν vs ω validation plot
    # --------------------------------------------------

    if os.path.exists("viscosity_results.txt"):

        data = np.loadtxt(
            "viscosity_results.txt"
        )

        if data.ndim == 1:
            data = data.reshape(1, -1)

        tau_vals = data[:,0]
        omega_vals = data[:,1]

        nu_measured_vals = data[:,2]
        nu_theory_vals = data[:,3]

        plt.figure(figsize=(8,6))

        plt.plot(
            omega_vals,
            nu_theory_vals,
            linewidth=2,
            label="Theory"
        )

        plt.scatter(
            omega_vals,
            nu_measured_vals,
            s=80,
            label="Measured"
        )

        plt.xlabel(r'$\omega$')
        plt.ylabel(r'$\nu$')

        plt.title(
            "LBM Viscosity Validation"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

    plt.savefig(
        "viscosity_validation.png",
        dpi=300
    )

    plt.close()

    print(
        f"\nAnimation saved as: "
        f"{output_gif}"
    )


if __name__ == "__main__":

    tau_arg = (
        float(sys.argv[1])
        if len(sys.argv) > 1
        else None
    )

    generate_milestone04_visualization(
        tau=tau_arg
    )
