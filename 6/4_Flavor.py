'''
After analysing the 3 flavor evolution of neutrinos in the collapsar, the next goal was to try to implement a fourth flavor
There isn't much change in the code since the 3x3 matrix just becomes 4x4.
However, to do so, we need the values of the mass-squared difference between 4 and 1, and we need the 4 theta angles. 
This required diving into the literature to see which values were possible, since we do not want a strong probability of nu_s.
In the end, the values work for the exception of an initial nu_e, which gives a final probability of nu_s that is above 0.8
'''

import numpy as np
import h5py 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.linalg import expm
from joblib import Parallel, delayed
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


DCP = 0 
dm21 = 7.53e-5      # All the values are already squared and are in eV², Normal Hierarchy
dm31 = 0.002455
dm32 = dm31 - dm21
dm41 = 1

#dm21 = 7.53e-5      # Inverted Hierarchy 0.0000753
#dm32 = -0.002455
#dm31 = dm32 + dm21

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


THETA_14 = 0.0872665 
THETA_24 = 0.122173 
THETA_34 = 0.174533

s14 = np.sin(THETA_14)
s24 = np.sin(THETA_24)
s34 = np.sin(THETA_34)
c14 = np.cos(THETA_14)
c24 = np.cos(THETA_24)
c34 = np.cos(THETA_34)


energies = [1e6, 1e7, 1e8]  # Neutrino energy in eV (example for 1 GeV neutrino)
E = 1e7


# Load data
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


def Hamiltonian(n_e, Y_e, Y_n, E):
    m2 = np.array([[0, 0, 0, 0], [0, dm21, 0, 0], [0, 0, dm31, 0], [0, 0, 0, dm41]])
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

    U_pmns = np.array([[U11, U12, U13, 0],
             [U21, U22, U23, 0],
             [U31, U32, U33, 0],
             [0, 0, 0, 1]])
    
    R34 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c34, s34], [0, 0, -s34, c34]])

    R24 = np.array([[1, 0, 0, 0], [0, c24, 0, s24], [0, 0, 1, 0], [0, -s24, 0, c24]])

    R14 = np.array([[c14, 0, 0, s14], [0, 1, 0, 0], [0, 0, 1, 0], [-s14, 0, 0, c14]])

    U_PMNS = R34 @ R24 @ R14 @ U_pmns

    U_PMNS_dagger = np.conjugate(U_PMNS).T

    M2 = U_PMNS @ m2 @ U_PMNS_dagger


    N_n = n_e * (1 - Y_e) / Y_e
    V_mu_tau = ((3 * G_F * (m_tau)**2)/(2 * np.sqrt(2) * (np.pi)**2 * Y_e) * (np.log((m_W/m_tau)**2) - 1 + (Y_n)/(3)))

    H_vac_fl = (M2)
    H_NC = -(np.sqrt(2) / 2) * G_F * N_n * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    H_CC = (np.sqrt(2) * G_F * n_e) * np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, V_mu_tau, 0], [0, 0, 0, 0]])
    #H_CC = (np.sqrt(2) * G_F * n_e) * np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    H_full = (1 / (2 * E)) * (H_vac_fl) + (H_CC) + (H_NC)
    return H_full

def evolve_state(H, dt, psi_matrix):
    eigenvalues, V = np.linalg.eigh(H)
    V_inv = np.linalg.inv(V)
    exp_diag = np.diag(np.exp(-1j * eigenvalues * dt))
    U = V @ exp_diag @ V_inv
    psi_next = U @ psi_matrix
    psi_next /= np.linalg.norm(psi_next)
    return psi_next

def solve_and_plot_with_matrix_exponential():
    psi = np.array([1, 0, 0, 0], dtype=complex)
    precomputed_H = [Hamiltonian(n_e_values_new[i], Y_e_values_new[i], Y_n_values_new[i], E) for i in range(len(r_data_new))]

    probability_nu_e = []
    probability_nu_mu = []
    probability_nu_tau = []
    probability_nu_sterile = []
    total_proba = []
    distances = []

    max_steps = min(max(r_data_new), len(r_data_new) - 1)

    for i in tqdm(range(max_steps), desc="Matrix Exponential Method", unit="step"):
        dt = (r_data_new[i + 1] - r_data_new[i]) * CONV_CM_TO_INV_EV  # Time step in inverse eV
        H = precomputed_H[i]
        psi = evolve_state(H, dt, psi)

        # Record probabilities
        probability_nu_e.append(abs(psi[0])**2)
        probability_nu_mu.append(abs(psi[1])**2)
        probability_nu_tau.append(abs(psi[2])**2)
        probability_nu_sterile.append(abs(psi[3])**2)
        total_proba.append(abs(psi[0])**2 + abs(psi[1])**2 + abs(psi[2])**2 + abs(psi[3])**2)
        distances.append(r_data_new[i + 1])  # Store distance (in cm)

    distance_km = [i * 1e-5 for i in distances]

    plt.figure(figsize=[4,3])
    plt.plot(distance_km, probability_nu_e, label=r"$P(\nu_{e} \rightarrow \nu_e)$", color="orange")
    plt.plot(distance_km, probability_nu_mu, label=r"$P(\nu_{e} \rightarrow \nu_{\mu})$", color="blueviolet")
    plt.plot(distance_km, probability_nu_tau, label=r"$P(\nu_{e} \rightarrow \nu_{\tau})$", color="orangered")
    plt.plot(distance_km, probability_nu_sterile, label=r"$P(\nu_{e} \rightarrow \nu_{s})$", color="grey")
    #plt.plot(distance_km, total_proba, label="Total Probability", color="Black")
    plt.xlabel("Distance (km)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.xscale("log")
    plt.yscale("linear")
    plt.xlim([distance_km[0], distance_km[-1]])
    plt.ylim([0, 1])
    #plt.title("Neutrino Oscillation Probability", fontsize=14)
    plt.legend(loc="upper right")
    plt.show()

# Call the function
solve_and_plot_with_matrix_exponential()
