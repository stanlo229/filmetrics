# -*- coding: utf-8 -*-
"""
thin_film_analysis.py — Extract n(λ)/k(λ) from reflectance using Sellmeier + TMM

Stack assumed:  air (n=1)  /  thin film (Sellmeier, k=0)  /  Si substrate

Usage:
    python thin_film_analysis.py <reflectance_csv> <summary_csv> [--output <dir>]

Example:
    python _code/thin_film_analysis.py \\
        results/test_run_2026-03-03_102945_measured.csv \\
        results/test_run_2026-03-03_102945_summary.csv

Inputs:
    reflectance_csv  _measured.csv or _calculated.csv
                     columns: wavelength_nm, reflectance  (values in 0–1)
    summary_csv      _summary.csv  (must contain column thickness_nm)

Outputs (written to --output dir, default: same folder as reflectance_csv):
    <stem>_sellmeier.fitnk   Sellmeier n/k in Filmetrics v3 tabulated format
    <stem>_nk.csv            wavelength_nm, n, k
    <stem>_sellmeier.png     two-panel plot: reflectance fit + n(λ)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Si substrate optical constants
# UV  188–361 nm : Aspnes & Studna, Phys. Rev. B 27, 985 (1983)
#                  (188 nm: Palik 1985 extension)
# VIS–NIR 400–1200 nm: Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008)
# ---------------------------------------------------------------------------
_SI_WL_NM = np.array([
    188,  207,  225,  248,  263,  276,  285,  291,  297,
    303,  310,  319,  329,  338,  361,
    400,  450,  500,  550,  600,  650,
    700,  750,  800,  850,  900,  950, 1000, 1050,
   1100, 1200,
], dtype=float)

_SI_N = np.array([
    # --- UV 188–361 nm: Aspnes & Studna (1983) + Palik at 188 nm ---
    0.69, 0.579, 0.618, 1.036, 1.524, 2.264, 2.334, 2.449, 3.043,
    3.939, 5.082, 5.500, 5.633, 5.596, 5.576,
    # --- VIS–NIR 400–1200 nm: Green (2008) ---
    5.587, 4.676, 4.293, 4.077, 3.939, 3.844,
    3.774, 3.723, 3.681, 3.650, 3.620, 3.592, 3.570, 3.554,
    3.541, 3.520,
])

_SI_K = np.array([
    # --- UV 188–361 nm: Aspnes & Studna (1983) + Palik at 188 nm ---
    2.89, 2.976, 2.944, 3.111, 3.491, 3.584, 3.635, 3.473, 2.944,
    2.349, 1.063, 0.619, 0.425, 0.375, 0.350,
    # --- VIS–NIR 400–1200 nm: Green (2008) ---
    3.040e-1, 9.22e-2, 4.52e-2, 2.82e-2, 1.99e-2, 1.47e-2,
    1.07e-2, 7.82e-3, 5.44e-3, 3.43e-3, 1.94e-3, 9.47e-4, 3.64e-4,
    1.46e-4, 4.98e-5, 2.80e-7,
])

# Module-level PCHIP interpolators — built once, evaluated many times.
# PCHIP is shape-preserving (no overshoot) and safe for non-monotone UV data.
_SI_N_INTERP    = scipy.interpolate.PchipInterpolator(_SI_WL_NM, _SI_N,         extrapolate=True)
_SI_LOGK_INTERP = scipy.interpolate.PchipInterpolator(_SI_WL_NM, np.log(_SI_K), extrapolate=True)


def _si_nk(wl_nm: np.ndarray):
    """Interpolate Si n, k at requested wavelengths (nm).

    n uses PCHIP (shape-preserving, no overshoot through non-monotone UV data).
    k uses PCHIP on log(k) to handle the many-decade IR range.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    n = _SI_N_INTERP(wl_nm)
    k = np.exp(_SI_LOGK_INTERP(wl_nm))
    k = np.maximum(k, 0.0)
    return n, k


# ---------------------------------------------------------------------------
# Sellmeier dispersion model   n²(λ) = 1 + Σ Bᵢλ²/(λ²-Cᵢ²)    k = 0
#
# Oscillator wavelengths fixed to Malitson (1965) fused-silica positions
# (λ in µm).  Only the oscillator strengths B1, B2, B3 are fitted.
#
#   C1 = 0.0684043 µm   UV oscillator  (~68 nm)
#   C2 = 0.1162414 µm   UV oscillator  (~116 nm)
#   C3 = 9.896161  µm   IR oscillator  (~9896 nm)
#
# Fixing the pole positions keeps n(λ) finite and positive throughout the
# 160–11000 nm window, unlike the Cauchy polynomial which can diverge in
# the UV when fit with noisy data.
# ---------------------------------------------------------------------------
_SM_C1 = 0.0684043   # µm
_SM_C2 = 0.1162414   # µm
_SM_C3 = 9.896161    # µm

# Malitson (1965) amplitudes for fused silica — used as initial guess
_SM_B1_0 = 0.6961663
_SM_B2_0 = 0.4079426
_SM_B3_0 = 0.8974794


def sellmeier_n(wl_nm: np.ndarray, B1: float, B2: float, B3: float) -> np.ndarray:
    """Evaluate Sellmeier n(λ) for SiO2 with fixed pole positions."""
    lam = wl_nm / 1000.0   # nm → µm
    n2 = (1.0
          + B1 * lam**2 / (lam**2 - _SM_C1**2)
          + B2 * lam**2 / (lam**2 - _SM_C2**2)
          + B3 * lam**2 / (lam**2 - _SM_C3**2))
    return np.sqrt(np.maximum(n2, 1.0))   # clamp n² ≥ 1 for stability


# ---------------------------------------------------------------------------
# Transfer Matrix Method — normal incidence, single film on substrate
# ---------------------------------------------------------------------------
def tmm_reflectance(
    wl_nm: np.ndarray, d_nm: float, B1: float, B2: float, B3: float
) -> np.ndarray:
    """Compute R(λ) for air / Sellmeier-film / Si at normal incidence."""
    n0 = np.ones(len(wl_nm), dtype=complex)            # air
    n1 = sellmeier_n(wl_nm, B1, B2, B3).astype(complex)  # film (k = 0)
    n_si, k_si = _si_nk(wl_nm)
    n2 = (n_si - 1j * k_si)                            # Si substrate

    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    delta = 2.0 * np.pi * n1 * d_nm / wl_nm            # phase thickness
    exp2id = np.exp(2j * delta)

    r_tot = (r01 + r12 * exp2id) / (1.0 + r01 * r12 * exp2id)
    return np.abs(r_tot) ** 2


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_sellmeier(
    wl_nm: np.ndarray,
    R_meas: np.ndarray,
    d_nm: float,
    B10: float = _SM_B1_0,
    B20: float = _SM_B2_0,
    B30: float = _SM_B3_0,
):
    """Fit Sellmeier B1, B2, B3 to minimise RMS(R_meas - R_model).

    Oscillator pole wavelengths C1, C2, C3 are fixed to Malitson (1965)
    fused-silica values.  Only the oscillator strengths are free.

    Returns
    -------
    (B1, B2, B3) : fitted Sellmeier oscillator strengths
    rms          : RMS residual (same units as R_meas, i.e. reflectance 0–1)
    """
    def objective(params):
        B1, B2, B3 = params
        R_model = tmm_reflectance(wl_nm, d_nm, B1, B2, B3)
        return np.mean((R_meas - R_model) ** 2)

    bounds = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]
    result = scipy.optimize.minimize(
        objective,
        [B10, B20, B30],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if not result.success:
        print(f"  Warning: optimizer did not fully converge — {result.message}")
    rms = float(np.sqrt(result.fun))
    return result.x, rms


# ---------------------------------------------------------------------------
# .fitnk writer  (Filmetrics v3 tabulated format)
# ---------------------------------------------------------------------------
def write_fitnk(
    path,
    wl_nm: np.ndarray,
    n_arr: np.ndarray,
    k_arr: np.ndarray,
):
    """Write a Filmetrics v3 .fitnk file with tabulated n/k data."""
    with open(path, "w") as f:
        f.write("Version, 3\n")
        f.write("Compat. Version, 1\n")
        f.write("Material Type, 3\n")
        f.write("File Information, \n")
        f.write("Wavelength (nm), n, k\n")
        for w, n, k in zip(wl_nm, n_arr, k_arr):
            f.write(f"{w:.2f}, {n:.5f}, {k:.5f}\n")
        f.write("\n")
        f.write("2\n")
        f.write("0, 1\n")
        f.write("0, 1\n")
        f.write("0, 1\n")
        f.write("0, 1\n")
        f.write("0, 1\n")
        f.write("0, 1\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot(wl_nm, R_meas, R_fit, n_arr, stem, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(stem, fontsize=9)

    axes[0].plot(wl_nm, R_meas, color="blue", lw=1, label="Measured")
    axes[0].plot(wl_nm, R_fit, color="red", lw=1, linestyle="--", label="Sellmeier fit")
    axes[0].set_ylabel("Reflectance")
    axes[0].legend(loc="upper right")

    axes[1].plot(wl_nm, n_arr, color="black", lw=1.5)
    axes[1].set_ylabel("n  (refractive index)")
    axes[1].set_xlabel("Wavelength (nm)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programmatic entry point (no argparse — callable from other scripts)
# ---------------------------------------------------------------------------
def run_analysis(reflectance_csv, summary_csv, output_dir=None):
    """Fit Sellmeier n(λ) to a saved reflectance CSV and write outputs.

    Parameters
    ----------
    reflectance_csv : str or Path
        Path to a _measured.csv or _calculated.csv file.
    summary_csv : str or Path
        Path to the matching _summary.csv (must contain thickness_nm).
    output_dir : str or Path, optional
        Where to write outputs.  Defaults to the same folder as reflectance_csv.

    Returns
    -------
    dict with keys: B1, B2, B3, rms, n_632, fitnk_path, nk_csv_path, plot_path
    """
    refl_path = Path(reflectance_csv)
    summ_path = Path(summary_csv)
    out_dir = Path(output_dir) if output_dir else refl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(refl_path, delimiter=",", skiprows=1)
    wl_nm = data[:, 0]
    R_meas = data[:, 1]
    print(f"Loaded {len(wl_nm)} points from {refl_path.name}")
    print(f"  wavelength range: {wl_nm[0]:.1f} – {wl_nm[-1]:.1f} nm")

    with open(summ_path) as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    d_nm = float(row["thickness_nm"])
    print(f"  film thickness (from summary): {d_nm:.2f} nm")

    print("Fitting Sellmeier model (L-BFGS-B) ...")
    (B1, B2, B3), rms = fit_sellmeier(wl_nm, R_meas, d_nm)
    n_632 = float(sellmeier_n(np.array([632.8]), B1, B2, B3)[0])
    print(f"  B1 = {B1:.6f}   (UV oscillator strength ~{_SM_C1*1000:.0f} nm)")
    print(f"  B2 = {B2:.6f}   (UV oscillator strength ~{_SM_C2*1000:.0f} nm)")
    print(f"  B3 = {B3:.6f}   (IR oscillator strength ~{_SM_C3*1000:.0f} nm)")
    print(f"  RMS residual    = {rms:.5f}")
    print(f"  n(632.8 nm)     = {n_632:.5f}")

    n_arr = sellmeier_n(wl_nm, B1, B2, B3)
    k_arr = np.zeros_like(n_arr)
    R_fit = tmm_reflectance(wl_nm, d_nm, B1, B2, B3)

    stem = refl_path.stem

    fitnk_path = out_dir / f"{stem}_sellmeier.fitnk"
    write_fitnk(fitnk_path, wl_nm, n_arr, k_arr)
    print(f"Saved: {fitnk_path}")

    nk_csv_path = out_dir / f"{stem}_nk.csv"
    np.savetxt(
        nk_csv_path,
        np.column_stack([wl_nm, n_arr, k_arr]),
        delimiter=",",
        header="wavelength_nm,n,k",
        comments="",
    )
    print(f"Saved: {nk_csv_path}")

    plot_path = out_dir / f"{stem}_sellmeier.png"
    _plot(wl_nm, R_meas, R_fit, n_arr, stem, plot_path)
    print(f"Saved: {plot_path}")

    return dict(B1=B1, B2=B2, B3=B3, rms=rms, n_632=n_632,
                fitnk_path=fitnk_path, nk_csv_path=nk_csv_path, plot_path=plot_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit Sellmeier n(λ) to a reflectance spectrum (air/film/Si TMM)"
    )
    parser.add_argument("reflectance_csv", help="_measured.csv or _calculated.csv")
    parser.add_argument("summary_csv", help="_summary.csv  (needs column: thickness_nm)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same folder as reflectance_csv)",
    )
    args = parser.parse_args()
    run_analysis(args.reflectance_csv, args.summary_csv, args.output)


if __name__ == "__main__":
    main()
