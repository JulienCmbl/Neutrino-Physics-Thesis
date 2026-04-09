'''
This code only takes in the data of the matter density profile with the different time steps, angles and distance steps
The collapsar simulation has a total of 20 angles, 5 theta (polar) and 4 phi angles (azimuthal)
Using a slider, the goal is to see how the matter density profile for the 20 different angles evolves 
We can also add a few lines to compute the resonant layers, which will tell us where the MSW effect occurs (see when the layers cross the matter density profile)
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import LogFormatter


# Constants
G_F = 1.166e-23  # Fermi constant in eV^-2
m_N = 1.675e-27  # Nucleon mass in kg
m_N_eV = m_N * 5.609588e35  # Convert nucleon mass to eV
Delta_m2 = 2.455e-3  # Mass-squared difference in eV^2
Delta_m2_high = 4e-3  # eV^2
Delta_m2_low = 7e-6  # eV^2
theta = 0.587252  # Mixing angle in radians
theta_high = 0.624523
theta_low = 0.0387686682
c2t = np.cos(2 * theta)
c2t_high = np.cos(2 * theta_high)
c2t_low = np.cos(2 * theta_low)

# Load HDF5 file
file_path = ""
with h5py.File(file_path, "r") as hdf_file:
    r_data = hdf_file["r"][:]  # Distance values in cm
    phi_data = hdf_file["phi"][:]  # Phi angles
    theta_data = hdf_file["theta"][:]  # Theta angles
    time_data = hdf_file["time"][:]  # Time data
    density_data = hdf_file["rho"][:]  # Density data (t, phi, theta, r)
    Y_e_data = hdf_file["y_e"][:]

time_seconds = np.linspace(0.25, 3.55, len(time_data))

phi_index = 0
time_index = 0



phi_indices, theta_indices = np.meshgrid(np.arange(len(phi_data)), np.arange(len(theta_data)), indexing='ij')

fig, ax = plt.subplots(figsize=(16, 10))
plt.subplots_adjust(bottom=0.15)

lines = []

for i_phi in range(len(phi_data)):
    for i_theta in range(len(theta_data)):
        rho_values = density_data[time_index, i_phi, i_theta, :]
        Y_e_values = Y_e_data[time_index, i_phi, i_theta, :]
        label = fr"$\theta = {theta_data[i_theta]:.2f}$ rad, $\varphi = {phi_data[i_phi]:.2f}$ rad"
        line, = ax.plot(r_data, rho_values, label=label, alpha=0.6)
        lines.append(line)

ax.set_xlabel("Distance r (km)", fontsize=14)
ax.set_ylabel(r"Density Matter (g/cc)", fontsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim([r_data[0], r_data[-1]])
ax.set_ylim([1e-9, 3e15])

if len(lines) <= 15:
    ax.legend(loc="upper right")
else:
    print(f"Plotted {len(lines)} lines. Legend skipped to avoid clutter.")

class CustomLogFormatter(LogFormatter):
    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        else:
            exponent = int(np.log10(x))
            return f"$10^{{{exponent-5}}}$"

sci_formatter = CustomLogFormatter(base=10.0)
ax.xaxis.set_major_formatter(sci_formatter)

ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
time_slider = Slider(ax_slider, "Time (s)", time_seconds[0], time_seconds[-1], valinit=time_seconds[time_index], valstep=0.05)

def update(val):
    time_index = np.argmin(np.abs(time_seconds - time_slider.val))
    line_index = 0
    for i_phi in range(len(phi_data)):
        for i_theta in range(len(theta_data)):
            rho_values = density_data[time_index, i_phi, i_theta, :]
            Y_e_values = Y_e_data[time_index, i_phi, i_theta, :]
            lines[line_index].set_ydata(rho_values)
            line_index += 1
    fig.canvas.draw_idle()

time_slider.on_changed(update)
update(0)

plt.show()
