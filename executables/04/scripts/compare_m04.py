"""
Milestone 04 comparison helper.

Usage
-----
python compare_modes.py mode1_dir mode2_dir

Each directory should contain:
    lbm_04_step_*.txt

This script compares two shear-wave simulations having different
initial Fourier modes (e.g. mode=1 and mode=2) but the SAME tau.

Outputs:
    shear_wave_mode_comparison.png
"""

import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

STEP_RE = re.compile(r"lbm_04_step_(\d+)\.txt$")


def discover(directory):
    steps = []
    for f in glob.glob(os.path.join(directory, "lbm_04_step_*.txt")):
        m = STEP_RE.search(os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_step(directory, step):
    data = np.loadtxt(os.path.join(directory,
                        f"lbm_04_step_{step}.txt"))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ny = int(data[:,0].max()) + 1
    nx = int(data[:,1].max()) + 1

    ux = np.zeros((ny,nx))
    rho = np.zeros((ny,nx))

    for row in data:
        y = int(row[0])
        x = int(row[1])
        rho[y,x] = row[2]
        ux[y,x] = row[3]

    return rho, ux


def extract(directory, mode_number):
    steps = discover(directory)

    rho0, ux0 = load_step(directory, steps[0])
    ny = ux0.shape[0]

    y = np.arange(ny)
    basis = np.sin(2*np.pi*mode_number*y/ny)

    amps = []
    profiles = []

    for s in steps:
        _, ux = load_step(directory, s)
        profile = ux.mean(axis=1)
        profiles.append(profile)

        amp = (2.0/ny) * np.sum(profile * basis)
        amps.append(amp)

    amps = np.asarray(amps)
    log_amp = np.log(np.maximum(np.abs(amps),1e-15))

    slope, intercept = np.polyfit(steps, log_amp, 1)

    return {
        "steps":np.asarray(steps),
        "profile":profiles,
        "log":log_amp,
        "fit":slope*np.asarray(steps)+intercept,
        "slope":slope,
        "ny":ny
    }


def main():

    if len(sys.argv) != 3:
        print("python compare_modes.py mode1_dir mode2_dir")
        return

    m1 = extract(sys.argv[1],1)
    m2 = extract(sys.argv[2],2)

    fig,ax = plt.subplots(3,1,figsize=(8,10))

    ax[0].plot(m1["profile"][0],label="t=0")
    ax[0].plot(m1["profile"][-1],label="final")
    ax[0].set_title("Mode 1 (λ = Ny)")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(m2["profile"][0],label="t=0")
    ax[1].plot(m2["profile"][-1],label="final")
    ax[1].set_title("Mode 2 (λ = Ny/2)")
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(m1["steps"],m1["log"],label="Mode 1")
    #ax[2].plot(m1["steps"],m1["fit"],'--')

    ax[2].plot(m2["steps"],m2["log"],label="Mode 2")
    #ax[2].plot(m2["steps"],m2["fit"],'--')

    ax[2].set_xlabel("Time step")
    ax[2].set_ylabel("ln(A)")
    ax[2].set_title("Decay comparison")
    ax[2].legend()
    ax[2].grid(True)

    ratio = abs(m2["slope"]/m1["slope"])

    fig.suptitle(f"Slope ratio = {ratio:.2f} (Theory ≈ 4.0)")
    plt.tight_layout()
    plt.savefig("shear_wave_mode_comparison.png",dpi=300)
    print("Saved shear_wave_mode_comparison.png")
    print("Mode 1 slope:",m1["slope"])
    print("Mode 2 slope:",m2["slope"])
    print("Slope ratio :",ratio)

if __name__=="__main__":
    main()
