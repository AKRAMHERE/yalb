import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

rows = 300
cols = 300

vel_files = sorted(glob.glob("data/v/output_velocity_*.txt"))
rho_files = sorted(glob.glob("data/rho/output_rho_*.txt"))

def load_rho(filename):
    data = np.loadtxt(filename)

    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    rho_values = data[:, 2]

    rho = np.zeros((cols, rows))

    for k in range(len(rho_values)):
        rho[y[k], x[k]] = rho_values[k]

    return rho

def load_velocity(filename):
    data = np.loadtxt(filename)

    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)

    vx_values = data[:, 2]
    vy_values = data[:, 3]

    vx = np.zeros((cols, rows))
    vy = np.zeros((cols, rows))

    for k in range(len(vx_values)):
        vx[y[k], x[k]] = vx_values[k]
        vy[y[k], x[k]] = vy_values[k]

    return vx, vy

X, Y = np.meshgrid(np.arange(rows), np.arange(cols))

fig, ax = plt.subplots(figsize=(8, 5))

rho0 = load_rho(rho_files[0])
vx0, vy0 = load_velocity(vel_files[0])

# drho0 = rho0 - 0.1

rho_img = ax.imshow(
    vx0,
    origin="lower",
    aspect="equal",
    interpolation="nearest",
    cmap="seismic",
    vmin=-0.01,
    vmax=0.01
)

plt.colorbar(rho_img, ax=ax, label="ux velocity")

quiv = ax.quiver(
    X, Y, vx0, vy0,
    angles='xy', scale_units='xy', scale=1, color='red'
    )

plt.colorbar(rho_img, ax=ax, label="Density perturbation")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("LBM density and velocity evolution")

ax.set_xticks(np.arange(-0.5, rows, 1), minor=True)
ax.set_yticks(np.arange(-0.5, cols, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
ax.tick_params(which='minor', bottom=False, left=False)

def update(frame):
    vx, vy = load_velocity(vel_files[frame])

    rho_img.set_data(vx)
    quiv.set_UVC(vx, vy)

    ax.set_title(f"LBM ux velocity evolution: step {frame}")

    return rho_img, quiv

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(vel_files),
    interval=150,
    blit=False
)

plt.show()

# save animation
ani.save('lbm_velocity_evolution.gif', writer='imagemagick', fps=5)