import numpy as np
import h5py 
import matplotlib.pyplot as plt
from tqdm import tqdm




# Constants
G_F = 1.1663787e-23
Conv_km_to_inv_eV = 5.06773e9
Conv_eV_to_g = 1.783e-33
Conv_g_to_eV = 1./Conv_eV_to_g
Conv_cm_to_inv_eV = Conv_km_to_inv_eV * 1.e-5
Conv_gcc_to_eV = 4.362e18
M_Neutron = 939.565379e6
M_Proton = 938.272046e6
M_Eelectron = 0.5109989461e6
m_N = 1.675e-27  # Nucleon mass in kg
m_N_eV = m_N * 5.609588e35  # Convert nucleon mass to eV

#Dm = 7.5e-5  # Mass-squared difference in eV^2 
Dm = 0.002455
theta = 0.587252        # Theta 12
#theta = 0.843529        # Theta 23
#theta = 0.148532        # Theta 13
s = np.sin(theta)
c = np.cos(theta)
E = 1e7



file_path = "C:/Users/Data/Data_interpolated_quadratic_new.h5"
with h5py.File(file_path, "r") as hdf_file:
    r_data_new = hdf_file["r_fine"][:]
    density_data_new = hdf_file["rho_fine"][:]
    Y_e_data_new = hdf_file["y_e_fine"][:]


phi_index = 0
theta_index = 0
time_index = 0

rho_values_new = density_data_new[time_index, phi_index, theta_index, :] * Conv_gcc_to_eV
Y_e_values_new = Y_e_data_new[time_index, phi_index, theta_index, :]
n_e_values_new = rho_values_new * Y_e_values_new / m_N_eV                                           # Unitless



def Hamiltonian(n_e, E):
    m2 = np.array([[0, 0], [0, Dm]])

    U11 = c
    U12 = s
    U21 = -s
    U22 = c

    U_PMNS = [[U11, U12],
             [U21, U22]]
    
    U_PMNS_dagger = np.conjugate(U_PMNS).T

    M2 = U_PMNS @ m2 @ U_PMNS_dagger

    H_vac_fl = (M2)
    H_mat = (np.sqrt(2) * G_F * n_e) * np.array([[1, 0], [0, 0]])
    H_full = (1 / (2 * E)) * (H_vac_fl) - (H_mat)
    return H_full

def evolve_state(H, dt, psi_matrix):
    eigenvalues, V = np.linalg.eig(H)
    V_inv = np.linalg.inv(V)
    exp_diag = np.diag(np.exp(-1j * eigenvalues * dt))
    U = V @ exp_diag @ V_inv
    psi_next = U @ psi_matrix
    psi_next /= np.linalg.norm(psi_next)
    return psi_next

def solve_and_plot_with_matrix_exponential():
    # Initial conditions
    psi = np.array([1, 0], dtype=complex)    # Define the initial neutrino flavour, first number for electron, second for muon. 
                                             # It gives the probability of the flavour
    precomputed_H = [Hamiltonian(n_e_values_new[i], E) for i in range(len(r_data_new))]

    probability_nu_e = []
    probability_nu_x = []
    total_proba = []
    distances = []

    max_steps = min(max(r_data_new), len(r_data_new) - 1)

    for i in tqdm(range(max_steps), desc="Matrix Exponential Method", unit="step"):
        dt = (r_data_new[i + 1] - r_data_new[i]) * Conv_cm_to_inv_eV  # Time step in inverse eV
        H = precomputed_H[i]
        psi = evolve_state(H, dt, psi)  # Evolve using matrix exponential

        # Record probabilities
        probability_nu_e.append(abs(psi[0])**2)
        probability_nu_x.append(abs(psi[1])**2)
        total_proba.append(abs(psi[0])**2 + abs(psi[1])**2)
        distances.append(r_data_new[i + 1])  # Store distance (in cm)

    # Convert distances to km
    distance_km = [i * 1e-5 for i in distances]

    # Plot results
    plt.figure(figsize=[4, 3])
    plt.plot(distance_km, probability_nu_e, label=r"$P(\nu_{e} \rightarrow \nu_e)$", color="orange")
    plt.plot(distance_km, probability_nu_x, label=r"$P(\nu_{e} \rightarrow \nu_x)$", color="blueviolet")
    #plt.plot(distance_km, total_proba, label="Total Probability", color="Black")
    plt.xlabel("Distance (km)", fontsize=16)
    plt.ylabel("Probability", fontsize=16)
    plt.xscale("log")
    plt.yscale("linear")
    plt.xlim([distance_km[0], distance_km[-1]])
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


solve_and_plot_with_matrix_exponential()
