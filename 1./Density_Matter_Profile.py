'''
This code takes in the data, the matter density profile and the electron fraction with the different time steps, angles and distance steps
The collapsar simulation has a total of 20 angles, 5 theta (polar) and 4 phi angles (azimuthal)
Using a slider, the goal is to see how the matter density profile for the 20 different angles evolves 
We can also add a few lines to compute the resonant layers, which will tell us where the MSW effect occurs (see when the layers cross the matter density profile)
The code and the plot are very dense as they show all the resonant layers for:
    - The low and high modes
    - The range of neutrino energy using 1, 10 and 100 MeV
    - The extra matter potential 
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
m_tau = 1.77686e9
m_W = 8.0379e10
Delta_m2 = 2.455e-3  # Mass-squared difference in eV^2 
Delta_m2_high = 4*10**(-3) # eV^2
Delta_m2_low = 7*10**(-6) # eV^2
Dm32 = 0.0024511
theta = 0.587252  # Mixing angle in radians
theta_high = 0.624523
theta_low = 0.0387686682
THETA_23 = 0.843529
c2t = np.cos(2*theta)
c2t_high = np.cos(2*theta_high)
c2t_low = np.cos(2*theta_low)
c2t_extrm = np.cos(2*THETA_23)
energies = [1e6, 1e7, 1e8]  # Neutrino energy in eV (example for 1 GeV neutrino)


file_path = "C:/Users/jujus/Documents/Cours/5ème année/Data/radp.h5"
with h5py.File(file_path, "r") as hdf_file:
    r_actual = hdf_file["r"][:]         # Actual distance values in cm
    r_data = r_actual                   # Keeping the distance in cm
    phi_data = hdf_file["phi"][:]       # Phi angles
    theta_data = hdf_file["theta"][:]   # Theta angles
    time_data = hdf_file["time"][:]     # Time data
    density_data = hdf_file["rho"][:]   # Density data, shape (t, phi, theta, r)
    Y_e_data = hdf_file["y_e"][:]       # Electron fraction data (t, phi, theta, r)

time_seconds = np.linspace(0.25, 3.55, len(time_data))  

phi_index = 0
theta_index = 0
time_index = 1

rho_values = density_data[time_index, phi_index, theta_index, :]
Y_e_values = Y_e_data[time_index, phi_index, theta_index, :]
Y_n_values = 1 - Y_e_values

fig, ax = plt.subplots(figsize=(16,10))
plt.subplots_adjust(bottom=0.15)

line_rho_res = []
line_rho_res_high = []
line_rho_res_low = []
line_rho_res_extreme = []

for energy, colour, e_name in zip(energies, colours, energy_name):
    line_rr, = ax.plot(r_data, np.full_like(r_data, 1e-6), label=fr"$\rho_{{res}}$ at {e_name}", color = colour, linestyle="-")
    line_rho_res.append(line_rr)

    line_rr_high = ax.plot(r_data, np.full_like(r_data, 1e-6), label=fr"$\rho_{{res H}}$ at {e_name}", color = colour, linestyle="-.")[0]
    line_rho_res_high.append(line_rr_high)

    line_rr_low = ax.plot(r_data, np.full_like(r_data, 1e-6), label=fr"$\rho_{{res L}}$ at {e_name}", color = colour, linestyle="--")[0]
    line_rho_res_low.append(line_rr_low)

    line_rr_extreme = ax.plot(r_data, np.full_like(r_data, 1e-6), label=fr"$\rho_{{res extrm}}$ at {e_name}", color = colour, linestyle=":")[0]
    line_rho_res_extreme.append(line_rr_extreme)


line, = ax.plot(r_data, rho_values, label=r"Density matter $\rho$", color = "Purple")

ax.set_xlabel("Distance r (km)", fontsize=14)
ax.set_ylabel(r"Density Matter $\rho$ (g/cc)", fontsize=14)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim([r_data[0], r_data[-1]])
ax.set_ylim([6e-9, 1.5e15])
ax.grid(True)
ax.legend(loc="upper right")

class CustomLogFormatter(LogFormatter):
    def __call__(self, x, pos=None):
        """Return scientific notation without the 1× for powers of 10."""
        if x == 0:
            return "0"
        else:
            exponent = int(np.log10(x))
            return f"$10^{{{exponent-5}}}$"

sci_formatter = CustomLogFormatter(base=10.0)

ax.xaxis.set_major_formatter(sci_formatter)


def format_pi(value):
    fractions = {0: "0", np.pi/4: "π/4", np.pi/2: "π/2", 3*np.pi/4: "3π/4", np.pi: "π",
                 -np.pi: "-π", -np.pi/2: "-π/2", -3*np.pi/4: "-3π/4"}
    return fractions.get(value,f"{value/np.pi:.2f}π")

ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
time_slider = Slider(ax_slider, "Time (s)", time_seconds[0], time_seconds[-1], valinit=time_seconds[time_index], valstep=0.05)

ax_theta_slider = plt.axes([0.04, 0.2, 0.01, 0.6])
theta_slider = Slider(ax_theta_slider, "Theta", theta_data[0], theta_data[-1], valinit=theta_data[theta_index], valstep=np.pi/4, orientation="vertical")
theta_slider.valtext.set_text(format_pi(theta_slider.val))

ax_phi_slider = plt.axes([0.96, 0.2, 0.01, 0.6])
phi_slider = Slider(ax_phi_slider, "Phi", phi_data[0], phi_data[-1], valinit=phi_data[phi_index], valstep=np.pi/2, orientation="vertical")
phi_slider.valtext.set_text(format_pi(phi_slider.val))


def update(val):
    global time_index, phi_index, theta_index
    time_index = np.argmin(np.abs(time_seconds - time_slider.val))
    theta_index = np.argmin(np.abs(theta_data - theta_slider.val))
    phi_index = np.argmin(np.abs(phi_data - phi_slider.val))
    theta_slider.valtext.set_text(format_pi(theta_slider.val))
    phi_slider.valtext.set_text(format_pi(phi_slider.val))

    rho_values = density_data[time_index, phi_index, theta_index, :]
    Y_e_values = Y_e_data[time_index, phi_index, theta_index, :]

    for i, (energy, rho_res_line, rho_res_high_line, rho_res_low_line, rho_res_extrm_line) in enumerate(zip(energies, line_rho_res, line_rho_res_high, line_rho_res_low, line_rho_res_extreme)):
        rho_res = (Delta_m2 * c2t * m_N_eV) / (2 * np.sqrt(2) * G_F * energy * Y_e_values) * (2.46*10**(-19))
        rho_res_high = (Delta_m2_high * c2t_high * m_N_eV) / (2 * np.sqrt(2) * G_F * energy * Y_e_values) * (2.46*10**(-19))
        rho_res_low = (Delta_m2_low * c2t_low * m_N_eV) / (2 * np.sqrt(2) * G_F * energy * Y_e_values) * (2.46*10**(-19))
        V_mu_tau = (np.sqrt(2) * G_F * energy) * (3 * G_F * (m_tau)**2)/(2 * np.sqrt(2) * (np.pi)**2 * Y_e_values) * (np.log((m_W/m_tau)**2) - 1 + (Y_n_values)/(3))
        rho_res_extrm = (Dm32 * np.abs(c2t_extrm) * m_N_eV) / (2 * V_mu_tau * Y_e_values) * (2.46*10**(-19))

        rho_res_line.set_data(r_data, rho_res)
        rho_res_high_line.set_data(r_data, rho_res_high)
        rho_res_low_line.set_data(r_data, rho_res_low)
        rho_res_extrm_line.set_data(r_data, rho_res_extrm)


    line.set_ydata(rho_values)

    fig.canvas.draw_idle()
    


time_slider.on_changed(update)
theta_slider.on_changed(update)
phi_slider.on_changed(update)

update(0)

plt.show()
