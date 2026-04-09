'''
The goal of this animation is to show how the 20 matter density profiles evolve as a function of time 
Although some angles can show a difference of a couple of orders of magnitude, they all follow the same trend and adiabaticity is not impacted.
Therefore, the collapsar can be assumed to be spherically symmetric and the whole study of our neutrinos can be done using any angle at any time. 
This is only for simplicity and to avoid studying every single angle.
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import LogFormatter
from matplotlib.animation import PillowWriter


# Constants
file_path = "C:/Users/jujus/Documents/Cours/5ème année/Data/radp.h5"
with h5py.File(file_path, "r") as hdf_file:
    r_data = hdf_file["r"][:]  # Distance values in cm
    phi_data = hdf_file["phi"][:]  # Phi angles
    theta_data = hdf_file["theta"][:]  # Theta angles
    time_data = hdf_file["time"][:]  # Time data
    density_data = hdf_file["rho"][:]  # Density data (t, phi, theta, r)
    Y_e_data = hdf_file["y_e"][:]

# Time data
time_seconds = np.linspace(0.25, 3.55, len(time_data))

# Plot setup
fig, ax = plt.subplots(figsize=(16, 10))
lines = []
for i_phi in range(len(phi_data)):
    for i_theta in range(len(theta_data)):
        label = fr"$\theta = {theta_data[i_theta]:.2f}$ rad, $\varphi = {phi_data[i_phi]:.2f}$ rad"
        (line,) = ax.plot(r_data, density_data[0, i_phi, i_theta, :], label=label, alpha=0.6)
        lines.append(line)

ax.set_xlabel("Distance r (km)", fontsize=14)
ax.set_ylabel("Matter Density (g/cc)", fontsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim([r_data[0], r_data[-1]])
ax.set_ylim([1e-9, 3e15])

if len(lines) <= 15:
    ax.legend(loc="upper right")

class CustomLogFormatter(LogFormatter):
    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        else:
            exponent = int(np.log10(x))
            return f"$10^{{{exponent-5}}}$"

sci_formatter = CustomLogFormatter(base=10.0)
ax.xaxis.set_major_formatter(sci_formatter)

def update(frame_index):
    for i_phi in range(len(phi_data)):
        for i_theta in range(len(theta_data)):
            idx = i_phi * len(theta_data) + i_theta
            rho_values = density_data[frame_index, i_phi, i_theta, :]
            lines[idx].set_ydata(rho_values)
    ax.set_title(f"Time = {time_seconds[frame_index]:.2f} s", fontsize=16)
    return lines


ani = animation.FuncAnimation(
    fig, update, frames=len(time_seconds), blit=False, repeat=False
)

# Save as MP4 (requires ffmpeg)
ani.save("density_evolution.gif", writer=PillowWriter(fps=10), dpi=200)

plt.close(fig)
print("MP4 video saved as 'density_evolution.mp4'")
