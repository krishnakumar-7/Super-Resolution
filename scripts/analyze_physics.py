import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "../data/raw/turbulence_data.npy")
TARGET_OUTPUT = os.path.join(BASE_DIR, "../data/processed/target_spectrum.npy")
PLOT_OUTPUT = os.path.join(
    BASE_DIR, "../results/figures/spectrum_analysis.png")

# Ensure output directories exist
os.makedirs(os.path.dirname(TARGET_OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)


def get_energy_spectrum(u, v, w):
    """Computes the radially averaged energy spectrum E(k)."""
    N = u.shape[0]

    # Compute Energy Density in Frequency Domain
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    w_hat = np.fft.fft2(w)

    energy_2d = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)

    # Radial Wavenumbers
    k_freq = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(k_freq, k_freq)
    k_mag = np.sqrt(kx**2 + ky**2).flatten()
    energy_flat = energy_2d.flatten()

    # Binning
    k_bins = np.arange(0.5, N // 2 + 1, 1.0)
    k_vals = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_k = np.zeros(len(k_vals))

    for i, _ in enumerate(k_vals):
        indices = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        E_k[i] = np.sum(energy_flat[indices])

    return k_vals, E_k


def kolmogorov_law(k, C):
    """Theoretical turbulence scaling: E(k) = C * k^(-5/3)"""
    return C * k**(-5/3)


def main():

    if not os.path.exists(INPUT_PATH):
        sys.exit(f"Error: Data file not found at {INPUT_PATH}")

    print(f"Loading data from {INPUT_PATH}...")
    data = np.load(INPUT_PATH)
    num_snapshots, N, _, _ = data.shape
    print(f"Loaded {num_snapshots} snapshots of size {N}x{N}")

    # Compute Ensemble Average Spectrum
    print("Computing ensemble statistics...")
    avg_E_k = np.zeros(N // 2)
    k_vals = None

    for i in range(num_snapshots):
        u, v, w = data[i, :, :, 0], data[i, :, :, 1], data[i, :, :, 2]
        k_curr, E_curr = get_energy_spectrum(u, v, w)

        if k_vals is None:
            k_vals = k_curr
        avg_E_k += E_curr

    avg_E_k /= num_snapshots

    # Fit Physics Model (Inertial Subrange: k > 10)
    fit_mask = k_vals > 10
    popt, _ = curve_fit(kolmogorov_law, k_vals[fit_mask], avg_E_k[fit_mask])
    C_fit = popt[0]
    print(f"Fitted Kolmogorov Constant C: {C_fit:.4f}")

    # Generate Target Curve (Extrapolate to 1024)
    target_N = 1024
    target_k = np.arange(1, target_N // 2)
    target_E = kolmogorov_law(target_k, C_fit)

    np.save(TARGET_OUTPUT, target_E)
    print(f"Saved target spectrum to {TARGET_OUTPUT}")

    # Plot Results
    plt.figure(figsize=(10, 6))
    plt.loglog(k_vals, avg_E_k, 'bo-',
               label='Input Data (Coarse)', linewidth=2)
    plt.loglog(target_k, target_E, 'r--',
               label='Target Physics (-5/3 Law)', linewidth=1)

    plt.axvline(N // 2, color='k', linestyle=':', label='Nyquist Limit')
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy E(k)')
    plt.title(f'Spectrum Analysis (N={num_snapshots})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print(f"Plot saved to {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
