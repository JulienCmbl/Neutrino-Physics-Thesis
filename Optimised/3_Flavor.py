'''
The goal is to optimise the longest code simulating neutrino flavor evolution 
In this code, I optimise as much as possible in Python and passed from ~54sec to ~35sec
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
from numba import njit, prange
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────
G_F = 1.1663787e-23
CONV_KM_TO_INV_EV = 5.06773e9
CONV_CM_TO_INV_EV = CONV_KM_TO_INV_EV * 1e-5
CONV_GCC_TO_EV = 4.362e18
m_N_eV = 1.675e-27 * 5.609588e35
m_tau = 1.77686e9
m_W = 8.0379e10

DCP = 0.0
dm21 = 7.53e-5
dm31 = 0.002455
dm32 = dm31 - dm21

s12, c12 = np.sqrt(0.307),  np.sqrt(1 - 0.307)
s13, c13 = np.sqrt(0.0219), np.sqrt(1 - 0.0219)
s23, c23 = np.sqrt(0.558),  np.sqrt(1 - 0.558)

E = 1e7
eCP = np.exp( 1j * DCP)
enCP = np.exp(-1j * DCP)

file_path = ""
with h5py.File(file_path, "r") as f:
    r_data = f["r_fine"][:]
    rho_data = f["rho_fine"][:]
    Y_e_data = f["y_e_fine"][:]


U_PMNS = np.array([
    [ c12*c13, s12*c13, s13*enCP],
    [-s12*c23 - c12*s23*s13*eCP, c12*c23 - s12*s23*s13*eCP, s23*c13],
    [ s12*s23 - c12*c23*s13*eCP, -c12*s23 - s12*c23*s13*eCP, c23*c13],
], dtype=complex)

m2_diag = np.diag([0.0, dm21, dm31]).astype(complex)
M2_flavour = U_PMNS @ m2_diag @ U_PMNS.conj().T   

rho_1d = rho_data[0, 0, 0, :] * CONV_GCC_TO_EV
Y_e_1d = Y_e_data[0, 0, 0, :]
n_e_1d = rho_1d * Y_e_1d / m_N_eV
Y_n_1d = 1.0 - Y_e_1d

N = len(r_data)


log_ratio = np.log((m_W / m_tau)**2) - 1.0
V_mu_tau = (3.0 * G_F * m_tau**2) / (2.0 * np.sqrt(2) * np.pi**2 * Y_e_1d) \
              * (log_ratio + Y_n_1d / 3.0)
V_ee = np.sqrt(2) * G_F * n_e_1d 
V_tt = V_ee * V_mu_tau 
H_vac = M2_flavour / (2.0 * E)
H_stack = np.broadcast_to(H_vac, (N, 3, 3)).copy() 
H_stack[:, 0, 0] += V_ee
H_stack[:, 2, 2] += V_tt

dt = np.diff(r_data) * CONV_CM_TO_INV_EV


@njit(cache=True)
def evolve_loop(H_stack, dt, psi0):
    n_steps = len(dt)
    prob_e = np.empty(n_steps, dtype=np.float64)
    prob_mu = np.empty(n_steps, dtype=np.float64)
    prob_tau = np.empty(n_steps, dtype=np.float64)

    psi = psi0.copy()

    for i in range(n_steps):
        H = H_stack[i]
        eigenvalues, V = np.linalg.eigh(H)

        phase = np.exp(-1j * eigenvalues * dt[i])

        tmp = np.dot(V.conj().T, psi)
        tmp *= phase
        psi = np.dot(V, tmp)

        prob_e[i] = (psi[0].real**2 + psi[0].imag**2)
        prob_mu[i] = (psi[1].real**2 + psi[1].imag**2)
        prob_tau[i] = (psi[2].real**2 + psi[2].imag**2)

    return prob_e, prob_mu, prob_tau


psi0 = np.array([0.0, 0.0, 1.0], dtype=complex)
prob_e, prob_mu, prob_tau = evolve_loop(H_stack, dt, psi0)
distance_km = r_data[1:] * 1e-5

plt.figure(figsize=(16, 10))
plt.plot(distance_km, prob_e, label=r"$P(\nu_\tau \to \nu_e)$",   color="orange")
plt.plot(distance_km, prob_mu, label=r"$P(\nu_\tau \to \nu_\mu)$", color="blueviolet")
plt.plot(distance_km, prob_tau, label=r"$P(\nu_\tau \to \nu_\tau)$",color="orangered")
plt.xlabel("Distance (km)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.xscale("log")
plt.xlim([distance_km[0], distance_km[-1]])
plt.ylim([0, 1])
plt.title("Neutrino Oscillation Probability", fontsize=16)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
