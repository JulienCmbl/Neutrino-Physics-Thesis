import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.ticker import LogFormatter


# Constants
m_N = 1.675e-27  # Nucleon mass in kg
m_N_eV = m_N * 5.609588e35  # Convert nucleon mass to eV
Delta_m2 = 2.455e-3  # eV^2
theta = 0.587252  # radians
sin2_2theta = np.sin(2 * theta)**2
cos2_theta = np.cos(2 * theta)
E = [1e6, 1e7, 1e8]  # Energy in eV
energy_name = ['1 MeV', '10 MeV', '100 Mev']
colours = ['Cyan', 'mediumspringgreen', 'Orange']


file_path = "C:/Users/Data"
with h5py.File(file_path, "r") as hdf_file:
    r_actual = hdf_file["r"][:]  # Distance in cm
    r_values = r_actual          # Keeping values in cm
    rho = hdf_file["rho"][:]    # Matter density values
    Y_e = hdf_file["y_e"][:]    # Electron fraction values
    time_data = hdf_file["time"][:]
    theta_data = hdf_file["theta"][:]
    phi_data = hdf_file["phi"][:]

time_seconds = np.linspace(0.25, 3.55, len(time_data))

phi_index = 0
theta_index = 0
time_index = 1

rho_values = rho[time_index, phi_index, theta_index, :]
Y_e_values = Y_e[time_index, phi_index, theta_index, :]
drho = np.gradient(rho_values, r_values)
dYe = np.gradient(Y_e_values, r_values)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9), sharex=True, squeeze=True, gridspec_kw={'height_ratios':[1, 4]})
plt.subplots_adjust(hspace=0., bottom=0.12, top=.95)

line_gamma = []
line_P_f = []

for energy , colour, e_name in zip(E, colours, energy_name):
    line, = ax2.plot(r_values, np.full_like(r_values, 1e-6), label=rf"Adiabaticity Parameter $\gamma$ ({e_name})", color=colour)
    line_gamma.append(line)
    line_pf, = ax1.plot(r_values, np.zeros_like(r_values), label=f"$P_F$ ({e_name})", color=colour)
    line_P_f.append(line_pf)



ax2.set_xlabel("Distance r (km)", fontsize=12)
ax2.set_ylabel(r"Adiabaticity Parameter $\gamma$", fontsize=12)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_xlim([r_values[0], r_values[-1]])
ax2.set_ylim([1e0, 9.9e9])
ax2.legend(loc ='lower right')
ax2.grid(True)

ax1.set_ylabel(r"Flip Probability $P_f$", fontsize=12)
ax1.set_yscale("linear")
ax1.set_xscale("log")
ax1.set_xlim([r_values[0], r_values[-1]])
ax1.set_ylim([0, 1])
ax1.legend()
ax1.grid(True)

class CustomLogFormatter(LogFormatter):
    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        else:
            exponent = int(np.log10(x))
            return f"$10^{{{exponent-5}}}$"

sci_formatter = CustomLogFormatter(base=10.0)

ax2.xaxis.set_major_formatter(sci_formatter)
ax1.xaxis.set_major_formatter(sci_formatter)



def format_pi(value):
    fractions = {0: "0", np.pi/4: "π/4", np.pi/2: "π/2", 3*np.pi/4: "3π/4", np.pi: "π",
                 -np.pi: "-π", -np.pi/2: "-π/2", -3*np.pi/4: "-3π/4"}
    return fractions.get(value,f"{value/np.pi:.2f}π")

ax_slider = plt.axes([0.2, 0.02, 0.65, 0.03])
time_slider = Slider(ax_slider, "Time (s)", time_seconds[0], time_seconds[-1], valinit=time_seconds[time_index], valstep=0.05)

ax_theta_slider = plt.axes([0.04, 0.2, 0.01, 0.6])
theta_slider = Slider(ax_theta_slider, "Theta", theta_data[0], theta_data[-1], valinit=theta_data[theta_index], valstep=np.pi/4, orientation="vertical")
theta_slider.valtext.set_text(format_pi(theta_slider.val))

ax_phi_slider = plt.axes([0.96, 0.2, 0.01, 0.6])
phi_slider = Slider(ax_phi_slider, "Phi", phi_data[0], phi_data[-1], valinit=phi_data[phi_index], valstep=np.pi/2, orientation="vertical")
phi_slider.valtext.set_text(format_pi(phi_slider.val))

theta_slider.valtext.set_text(format_pi(theta_slider.val))
phi_slider.valtext.set_text(format_pi(phi_slider.val))


def update(val):
    global time_index, phi_index, theta_index
    time_index = np.argmin(np.abs(time_seconds - time_slider.val))
    theta_index = np.argmin(np.abs(theta_data - theta_slider.val))
    phi_index = np.argmin(np.abs(phi_data - phi_slider.val))
    theta_slider.valtext.set_text(format_pi(theta_slider.val))
    phi_slider.valtext.set_text(format_pi(phi_slider.val))
    
    rho_values = rho[time_index, phi_index, theta_index, :]
    Y_e_values = Y_e[time_index, phi_index, theta_index, :]
    drho = np.gradient(rho_values, r_values)
    dYe = np.gradient(Y_e_values, r_values)

    for i, (energy, gamma_line, pf_line) in enumerate(zip(E, line_gamma, line_P_f)):
        gamma = ((Delta_m2 * sin2_2theta) / (2 * energy * cos2_theta) * 1 / np.abs(np.abs((drho / rho_values)) + np.abs((dYe / Y_e_values))) )* (5.07*10**(6))
        P_f = np.exp(-gamma * np.pi / 2)
        gamma_line.set_ydata(gamma)
        pf_line.set_ydata(P_f)


    fig.canvas.draw_idle()

time_slider.on_changed(update)
theta_slider.on_changed(update)
phi_slider.on_changed(update)
update(0)
plt.show()
