import numpy as np
import matplotlib.pyplot as plt

rows = 128
cols = 128

data = np.loadtxt("data/v/output_velocity_19000.txt")

x = data[:,0].astype(int)
y = data[:,1].astype(int)
vx_values = data[:,2]
vy_values = data[:,3]

vx = np.zeros((cols, rows))
vy = np.zeros((cols, rows))

for k in range(len(x)):
    vx[y[k], x[k]] = vx_values[k]
    vy[y[k], x[k]] = vy_values[k]

speed = np.sqrt(vx**2 + vy**2)

X, Y = np.meshgrid(
    np.linspace(0, 1, rows),
    np.linspace(0, 1, cols)
)

plt.figure(figsize=(7, 6))

plt.streamplot(
    X, Y,
    vx, vy,
    color=speed,
    cmap="viridis",
    density=2.0,
    linewidth=1.0,
    arrowsize=1.0
)

plt.colorbar(label="Velocity magnitude |u|")
plt.xlabel("x/L")
plt.ylabel("y/L")
plt.title("Lid-driven cavity flow")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect("equal")

plt.show()