import glob
import numpy as np
import matplotlib.pyplot as plt

rows = 129
cols = 129
step = 92000
u_lid = 0.1

# ------------------------------------------------------------
# Load and combine MPI output files
# ------------------------------------------------------------

files = glob.glob(
    f"data/v/output_velocity_rank*_step{step}.txt"
)

if not files:
    raise FileNotFoundError(
        f"No velocity files found for step {step}"
    )

parts = [np.loadtxt(filename) for filename in files]
data = np.vstack(parts)

global_row = data[:, 0].astype(int)
column = data[:, 1].astype(int)

horizontal_velocity = data[:, 2]
vertical_velocity = data[:, 3]

u = np.zeros((rows, cols))
v = np.zeros((rows, cols))

for row, col, ux, uy in zip(
    global_row,
    column,
    horizontal_velocity,
    vertical_velocity
):
    u[row, col] = ux
    v[row, col] = uy

# Normalize by the lid velocity
u_normalized = u / u_lid
v_normalized = v / u_lid

x = np.linspace(0.0, 1.0, cols)
y = np.linspace(0.0, 1.0, rows)

center_col = cols // 2
center_row = rows // 2

# u along x = 0.5
u_centerline = u_normalized[:, center_col]

# v along y = 0.5
v_centerline = v_normalized[center_row, :]

ghia_y = np.array([
    1.0000,
    0.9766,
    0.9688,
    0.9609,
    0.9531,
    0.8516,
    0.7344,
    0.6172,
    0.5000,
    0.4531,
    0.2813,
    0.1719,
    0.1016,
    0.0703,
    0.0625,
    0.0547,
    0.0000
])

ghia_u_re400 = np.array([
     1.00000,
     0.75837,
     0.68439,
     0.61756,
     0.55892,
     0.29093,
     0.16256,
     0.02135,
    -0.11477,
    -0.17119,
    -0.32726,
    -0.24299,
    -0.14612,
    -0.10338,
    -0.09266,
    -0.08186,
     0.00000
])


u_interp = np.interp(
    ghia_y[::-1],
    y,
    u_centerline
)

u_error = u_interp - ghia_u_re400[::-1]

u_rmse = np.sqrt(np.mean(u_error**2))
u_max_error = np.max(np.abs(u_error))

print(f"u RMSE = {u_rmse:.6e}")
print(f"u max error = {u_max_error:.6e}")



plt.figure(figsize=(6, 6))

plt.plot(
    u_centerline,
    y,
    label="LBM, 129×129"
)

plt.scatter(
    ghia_u_re400,
    ghia_y,
    marker="o",
    facecolors="none",
    edgecolors="black",
    label="Ghia et al., Re = 400"
)

plt.xlabel(r"$u/U_{\mathrm{lid}}$")
plt.ylabel(r"$y/L$")
plt.title("Horizontal velocity on vertical centerline")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

