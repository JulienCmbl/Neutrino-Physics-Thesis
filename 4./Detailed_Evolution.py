'''
The evolution of the neutrino flavor conversion did not work correctly for several reasons.
To understand where the problem was each time, I needed to see what each quantity was doing and when the evolution became unphysical.
So in this subplot, I put the evolution of all the components of:
  - The density matrix (plot (0,0) and (1,0))
  - The Hamiltonian in matter (plot (0,1))
  - The eigenvalues, which give us the level-crossing diagram (plot (1,1))
  - The electron density with the resonant layers to double-check that the conversion effects take place at the correct distance (plot (2,0))
  - The matter potentials V_ee and V_mu_tau (plot (2,1)). Note that the bump at 10^{20} is due to the decrease of electron fraction when exiting the PNS
'''

'''
Comparing the probability, values & eigenvalues of the Hamiltonian, electron & matter density and the potentials
'''

import numpy as np
import h5py 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
from matplotlib.ticker import LogFormatter
import time
from tqdm import tqdm
from scipy.linalg import expm
from joblib import Parallel, delayed
import numba
from scipy.sparse.linalg import expm_multiply



# Constants
G_F = 1.1663787e-23
CONV_KM_TO_INV_EV = 5.06773e9
CONV_EV_TO_G = 1.783e-33
CONV_G_TO_EV = 1./CONV_EV_TO_G
CONV_CM_TO_INV_EV = CONV_KM_TO_INV_EV * 1.e-5
CONV_GCC_TO_EV = 4.362e18
MASS_NEUTRON = 939.565379e6
MASS_PROTON = 938.272046e6
MASS_ELECTRON = 0.5109989461e6
m_N = 1.675e-27  # Nucleon mass in kg
m_N_eV = m_N * 5.609588e35  # Convert nucleon mass to eV
m_tau = 1.77686e9
m_W = 8.0379e10


DCP = 0             # Try 1.19 \pm 0.22
dm21 = 7.53e-5      # All the values are already squared and are in eV², Normal Hierarchy
dm32 = 0.002455
dm31 = dm32 + dm21

#dm21 = 7.53e-5      # Inverted Hierarchy 0.0000753
#dm32 = -0.002455
#dm31 = dm32 - dm21

THETA_23 = 0.843529
THETA_13 = 0.148532
THETA_12 = 0.587252

squared_s12 = 0.307
squared_s13 = 0.0219
squared_s23 = 0.558
squared_c12 = 1 - squared_s12
squared_c13 = 1 - squared_s13
squared_c23 = 1 - squared_s23

s12 = np.sqrt(squared_s12)
s13 = np.sqrt(squared_s13)
s23 = np.sqrt(squared_s23)
c12 = np.sqrt(squared_c12)
c13 = np.sqrt(squared_c13)
c23 = np.sqrt(squared_c23)

energies = [1e6, 1e7, 1e8]  # Neutrino energy in eV (example for 1 GeV neutrino)
E = 1e7


file_path = ""
with h5py.File(file_path, "r") as hdf_file:
    r_actual = hdf_file["r"][:]  # Distances in cm
    r_data = r_actual
    phi_data = hdf_file["phi"][:]
    theta_data = hdf_file["theta"][:]
    time_data = hdf_file["time"][:]
    density_data = hdf_file["rho"][:]   # Density in g/cm^3
    Y_e_data = hdf_file["y_e"][:]       # Electron fraction data (t, phi, theta, r)

file_path = ""
with h5py.File(file_path, "r") as hdf_file:
    r_data_new = hdf_file["r_fine"][:]
    density_data_new = hdf_file["rho_fine"][:]
    Y_e_data_new = hdf_file["y_e_fine"][:]


time_seconds = np.linspace(0.25, 3.55, len(time_data))

phi_index = 0
theta_index = 0
time_index = 0

rho_values = density_data[time_index, phi_index, theta_index, :] * CONV_GCC_TO_EV
Y_e_values = Y_e_data[time_index, phi_index, theta_index, :]
n_e_values = rho_values * Y_e_values / m_N_eV

rho_values_new = density_data_new[time_index, phi_index, theta_index, :] * CONV_GCC_TO_EV
Y_e_values_new = Y_e_data_new[time_index, phi_index, theta_index, :]
n_e_values_new = rho_values_new * Y_e_values_new / m_N_eV                                           # Unitless
Y_n_values_new = 1 - Y_e_values_new



def solve_and_plot_with_matrix_exponential(n_e, Y_e, Y_n, E):
    m2 = np.array([[0, 0, 0], [0, dm21, 0], [0, 0, dm31]])
    eCP = np.exp(1j *DCP)
    enCP = np.exp(-1j*DCP)

    U11 = c12 * c13
    U12 = s12 * c13
    U13 = s13 * enCP
    U21 = -s12 * c23 - c12 * s23 * s13 * eCP
    U22 = c12 * c23 - s12 * s23 * s13 * eCP
    U23 = s23 * c13
    U31 = s12 * s23 - c12 * c23 * s13 * eCP
    U32 = -c12 * s23 - s12 * c23 * s13 * eCP
    U33 = c23 * c13

    U_PMNS = ([[U11, U12, U13],
             [U21, U22, U23],
             [U31, U32, U33]])
    
    U_PMNS_dagger = np.conjugate(U_PMNS).T

    M2 = U_PMNS @ m2 @ U_PMNS_dagger

    H_vac_fl = (M2)
    H_full_list = []
    for i in range(len(n_e)):
        V_mu_tau_i = (3 * G_F * (m_tau)**2)/(2 * np.sqrt(2) * (np.pi)**2 * Y_e[i]) * (np.log((m_W/m_tau)**2) - 1 + (Y_n[i])/(3))
        H_mat_i = (np.sqrt(2) * G_F * n_e[i]) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, V_mu_tau_i]])
        H_full_i = (1 / (2 * E)) * (H_vac_fl) + H_mat_i
        H_full_list.append(H_full_i)

    H_full = np.array(H_full_list)  

    def evolve_state(H, dt, psi_matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * dt)) @ eigenvectors.conj().T
        psi_next = U @ psi_matrix @ U.conj().T
        psi_next /= np.trace(psi_next)
        return psi_next

    # Initial conditions
    psi = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)

    probability_nu_e = []
    probability_nu_mu = []
    probability_nu_tau = []
    psi12 = []
    psi13 = []
    psi21 = []
    psi23 = []
    psi31 = []
    psi32 = []
    total_proba = []
    distances = []
    n_e_values_plot = []
    n_e_res_extrm = []
    n_e_res_low = []
    n_e_res_high = []

    #max_steps = min(max(r_data_new), len(r_data_new) - 1)
    max_steps = len(r_data_new) - 1

    for i in tqdm(range(max_steps), desc="Matrix Exponential Method", unit="step"):
        dt = (r_data_new[i + 1] - r_data_new[i]) * CONV_CM_TO_INV_EV  # Time step in inverse eV
        V_mu_tau = (3 * G_F * (m_tau)**2)/(2 * np.sqrt(2) * (np.pi)**2 * Y_e[i]) * (np.log((m_W/m_tau)**2) - 1 + (Y_n[i])/(3))
        H_full = (1 / (2 * E)) * H_vac_fl + (np.sqrt(2) * G_F * n_e[i]) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, V_mu_tau]])
        psi = evolve_state(H_full, dt, psi)  # Evolve using matrix exponential

        # Record probabilities
        probability_nu_e.append(abs(psi[0][0]))
        probability_nu_mu.append(abs(psi[1][1]))
        probability_nu_tau.append(abs(psi[2][2]))
        total_proba.append(abs(psi[0][0]) + abs(psi[1][1]) + abs(psi[2][2]))

        n_e_res_extreme = np.abs((dm32 * np.cos(2 * THETA_23))/(2 * np.sqrt(2) * G_F * E * V_mu_tau))
        n_e_res_l = np.abs((dm21 * np.cos(2 * THETA_13)) / (2 * np.sqrt(2) * G_F * E))
        n_e_res_h = np.abs((dm31 * np.cos(2 * THETA_12)) / (2 * np.sqrt(2) * G_F * E))

        psi12.append(abs(psi[0][1]))
        psi13.append(abs(psi[0][2]))
        psi21.append(abs(psi[1][0]))
        psi23.append(abs(psi[1][2]))
        psi31.append(abs(psi[2][0]))
        psi32.append(abs(psi[2][1]))

        distances.append(r_data_new[i + 1])  # Store distance (in cm)
        n_e_values_plot.append(n_e[i])

        n_e_res_extrm.append(np.abs(n_e_res_extreme))
        n_e_res_low.append(np.abs(n_e_res_l))
        n_e_res_high.append(np.abs(n_e_res_h))

    # Convert distances to km
    distance_km = [i * 1e-5 for i in distances]
    n_e_values_plot_km = [i for i in n_e_values_plot]


    
    H11 = []
    H33 = []
    Vmutau = []
    Vee = []

    for i in range(len(n_e)):
        V_mu_tau = (3 * G_F * (m_tau)**2)/(2 * np.sqrt(2) * (np.pi)**2 * Y_e[i]) * (np.log((m_W/m_tau)**2) - 1 + (Y_n[i])/(3))
        H = (1 / (2 * E)) * (H_vac_fl) + ((np.sqrt(2) * G_F * n_e[i]) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, V_mu_tau]]))

        V_ee_full = np.sqrt(2) * G_F * n_e[i]
        V_mu_tau_full = np.sqrt(2) * G_F * n_e[i] * V_mu_tau

        Vee.append(V_ee_full)
        Vmutau.append(V_mu_tau_full)

        H11.append(np.abs(H[0][0]))
        H33.append(np.abs(H[2][2]))

    H11 = np.array(H11)
    H33 = np.array(H33)

    eigenvalues_list = []
    for n_e, Y_e, Y_n in zip(n_e_values_new, Y_e_values_new, Y_n_values_new):
        V_mu_tau = (3 * G_F * (m_tau)**2)/(2 * np.sqrt(2) * (np.pi)**2 * Y_e) * (np.log((m_W/m_tau)**2) - 1 + (Y_n)/(3))
        Hamiltonian = (1 / (2 * E)) * H_vac_fl + (np.sqrt(2) * G_F * n_e) * np.array([[1, 0, 0], [0, 0, 0], [0, 0, V_mu_tau]])
        eigenvalues_list.append(np.sort(np.abs(np.linalg.eigvals(Hamiltonian))))
    eigenvalues_list = np.array(eigenvalues_list).T
    Neutrino_name = ["v1", "v2", "v3"]



    fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex="col", squeeze=True)

    ax2_0 = axs[0, 1]
    ax2_1 = axs[1, 1]
    ax2_2 = axs[2, 1]

    ax2_0.yaxis.set_label_position("right")
    ax2_0.yaxis.tick_right()

    ax2_1.yaxis.set_label_position("right")
    ax2_1.yaxis.tick_right()

    ax2_2.yaxis.set_label_position("right")
    ax2_2.yaxis.tick_right()

    # Top left plot = Probability
    axs[0, 0].plot(distance_km, probability_nu_e, label=r"$P(\nu_e)$", color="orange")
    axs[0, 0].plot(distance_km, probability_nu_mu, label=r"$P(\nu_{\mu})$", color="blueviolet")
    axs[0, 0].plot(distance_km, probability_nu_tau, label=r"$P(\nu_{\tau})$", color="orangered")
    axs[0, 0].legend(loc="center left")
    axs[0, 0].set_xlim([distance_km[0], distance_km[-1]])
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_xscale("log")
    axs[0, 0].grid(True)

    # Top right plot = Hamiltonians
    ax2_0.plot(n_e_values_new, H11, label=r"H$_{11}$", linestyle="dashed", color="blue")
    ax2_0.plot(n_e_values_new, H33, label=r"H$_{33}$", linestyle="dashed", color="red")
    ax2_0.legend()
    ax2_0.set_xlim([n_e_values_new[0], n_e_values_new[-1]])
    ax2_0.set_xscale("log")
    ax2_0.set_yscale("log")
    ax2_0.grid(True)

    # Mid left plot = rho
    axs[1, 0].plot(distance_km, psi21, label=r"$\rho_{21}$", linestyle="dotted", color="green")
    axs[1, 0].plot(distance_km, psi13, label=r"$\rho_{13}$", linestyle="dotted", color="salmon")
    axs[1, 0].plot(distance_km, psi23, label=r"$\rho_{23}$", linestyle="dotted", color="purple")
    axs[1, 0].legend(loc="center left")
    axs[1, 0].set_xlim([distance_km[0], distance_km[-1]])
    axs[1, 0].set_ylim([0, 1])
    axs[1, 0].set_xscale("log")
    axs[1, 0].grid(True)

    # Mid right plot = Eigenvalues
    for i, (eigenvalues, colour, neutrino) in enumerate(zip(eigenvalues_list, colours, Neutrino_name)):
        ax2_1.plot(n_e_values_new, eigenvalues, label=fr"{neutrino} With $V_{{\mu\tau}}$", color=colour)
    ax2_1.legend()
    ax2_1.set_xscale("log")
    ax2_1.set_yscale("log")
    ax2_1.set_xlim([n_e_values_new[0], n_e_values_new[-1]])
    ax2_1.set_ylim([1e-22, 1e1])
    ax2_1.grid(True)

    # Bot left plot = n_e
    axs[2, 0].plot(distance_km, n_e_values_plot_km, label=r"n$_e$ (eV$^3$)", linestyle="solid", color="indigo")
    #axs[2, 0].plot(distance_km, rho_values_new, label=r"$\rho$ (eV$^4$)", linestyle="solid", color="orangered")
    axs[2, 0].plot(distance_km, n_e_res_extrm, color="grey", linestyle="dashed")
    axs[2, 0].plot(distance_km, n_e_res_low, color="grey", linestyle="dashed")
    axs[2, 0].plot(distance_km, n_e_res_high, color="grey", linestyle="dashed")
    axs[2, 0].set_xlim([distance_km[0], distance_km[-1]])
    axs[2, 0].set_xscale("log")
    axs[2, 0].set_yscale("log")
    axs[2, 0].legend(loc="center left")
    axs[2, 0].grid(True)

    # Bot right plot = Potential V
    ax2_2.plot(n_e_values_new, Vee, label=r"V$_{ee}$", linestyle="solid", color="orange")
    ax2_2.plot(n_e_values_new, Vmutau, label=r"V$_{\mu\tau}$", linestyle="solid", color="blueviolet")
    ax2_2.set_xlim([n_e_values_new[0], n_e_values_new[-1]])
    ax2_2.set_xscale("log")
    ax2_2.set_yscale("log")
    ax2_2.legend()
    ax2_2.grid(True)



    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# Call the function
solve_and_plot_with_matrix_exponential(n_e_values_new, Y_e_values_new, Y_n_values_new, E)
