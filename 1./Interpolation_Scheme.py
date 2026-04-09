import h5py
import numpy as np
from scipy.interpolate import interp1d

# Define file paths
original_file_path = "C:/Users/Data.h5"
new_file_path = "C:/Users/Data_interpolated_quadratic_new.h5"

# Open original file
with h5py.File(original_file_path, "r") as original_file:
    # Load the original data
    r_actual = original_file["r"][:]  # Distances in cm
    density_data = original_file["rho"][:]  # Density (t, phi, theta, r)
    Y_e_data = original_file["y_e"][:]  # Electron fraction (t, phi, theta, r)
    phi_data = original_file["phi"][:]
    theta_data = original_file["theta"][:]
    time_data = original_file["time"][:]

# Step indices for interpolation
steps = np.arange(len(r_actual))
steps_fine = np.linspace(steps[0], steps[-1], num=len(steps) * 1000)

# Logarithmic interpolation of the radius
log_r_actual = np.log10(r_actual)
interp_r_func = interp1d(steps, log_r_actual, kind="slinear")
log_r_interpolated = interp_r_func(steps_fine)
r_interpolated = 10**log_r_interpolated  # Convert back from log-space

# Initialize interpolated arrays
rho_fine_shape = (density_data.shape[0], density_data.shape[1], density_data.shape[2], len(r_interpolated))
rho_fine = np.zeros(rho_fine_shape)
Y_e_fine = np.zeros(rho_fine_shape)

# Interpolate for each (time, phi, theta)
for t in range(density_data.shape[0]):  # Loop over time
    for phi in range(density_data.shape[1]):  # Loop over phi
        for theta in range(density_data.shape[2]):  # Loop over theta
            # Extract original values
            rho_original = density_data[t, phi, theta, :]
            Y_e_original = Y_e_data[t, phi, theta, :]

            # Log-transform density for interpolation
            log_rho_original = np.log10(rho_original)

            # Density interpolation (quadratic)
            interp_rho_func = interp1d(np.log10(r_actual), log_rho_original, kind="quadratic", fill_value="extrapolate")
            log_rho_interpolated = interp_rho_func(np.log10(r_interpolated))
            rho_fine[t, phi, theta, :] = 10**log_rho_interpolated  # Convert back from log-space

            # Electron fraction interpolation (slinear)
            interp_Y_e_func = interp1d(np.log10(r_actual), np.log10(Y_e_original), kind="slinear", fill_value="extrapolate")
            log_Y_e_interpolated = interp_Y_e_func(np.log10(r_interpolated))
            Y_e_fine[t, phi, theta, :] = 10**log_Y_e_interpolated  # Convert back from log-space

# Save interpolated data
with h5py.File(new_file_path, "w") as new_hdf_file:
    # Save original datasets
    new_hdf_file.create_dataset("r", data=r_actual, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("rho", data=density_data, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("phi", data=phi_data, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("theta", data=theta_data, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("time", data=time_data, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("y_e", data=Y_e_data, compression="gzip", compression_opts=9)

    # Save the interpolated values
    new_hdf_file.create_dataset("r_fine", data=r_interpolated, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("rho_fine", data=rho_fine, compression="gzip", compression_opts=9)
    new_hdf_file.create_dataset("y_e_fine", data=Y_e_fine, compression="gzip", compression_opts=9)

    # Add metadata
    new_hdf_file.attrs["description"] = "File containing original and interpolated data with step-based interpolation"
    new_hdf_file.attrs["interpolation_method"] = "r: slinear, rho: quadratic, Y_e: slinear"
    new_hdf_file.attrs["units"] = "r: cm, rho: g/cm^3, y_e: dimensionless"

print("Interpolation and saving completed successfully!")
