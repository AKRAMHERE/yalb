import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

rows = 15
cols = 10

files = sorted(glob.glob("data/rho/output_rho_*.txt"))

fig, ax = plt.subplots(figsize=(8, 5))

rho_img = ax.imshow(
    np.zeros((cols, rows)),
    origin="lower",
    aspect="equal",
    vmin=0,
    vmax=1
)

plt.colorbar(rho_img, ax=ax, label="Density")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("LBM density evolution")

def load_rho(filename):
    data = np.loadtxt(filename)

    x = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    rho_values = data[:, 2]

    rho = np.zeros((cols, rows))

    for k in range(len(rho_values)):
        rho[y[k], x[k]] = rho_values[k]

    return rho

def update(frame):
    rho = load_rho(files[frame])
    rho_img.set_data(rho)
    ax.set_title(f"LBM density evolution: step {frame}")
    return [rho_img]

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(files),
    interval=150,
    blit=False
)

plt.show()