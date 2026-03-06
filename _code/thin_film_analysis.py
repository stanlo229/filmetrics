# -*- coding: utf-8 -*-
"""
thin_film_analysis.py — Extract n(λ)/k(λ) from reflectance using Sellmeier + TMM

Stack assumed:  air (n=1)  /  thin film (Sellmeier, k=0)  /  Si substrate

Usage:
    python thin_film_analysis.py <reflectance_csv> <summary_csv> [--output <dir>]

Example:
    python _code/thin_film_analysis.py \
        results/test_run_2026-03-03_145827_measured.csv \
        results/test_run_2026-03-03_145827_summary.csv

Inputs:
    reflectance_csv  _measured.csv or _calculated.csv
                     columns: wavelength_nm, reflectance  (values in 0–1)
    summary_csv      _summary.csv  (must contain column thickness_nm)

Options:
    --model {sellmeier,cauchy,both}   dispersion model to fit (default: sellmeier)

Outputs (written to --output dir, default: same folder as reflectance_csv):
    <stem>_sellmeier.fitnk   Sellmeier n/k in Filmetrics v3 tabulated format
    <stem>_cauchy.fitnk      Cauchy    n/k in Filmetrics v3 tabulated format
    <stem>_<model>_nk.csv    wavelength_nm, n, k
    <stem>_<model>.png       two-panel plot: reflectance fit(s) + n(λ)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Si substrate optical constants — Palik (1985)
# Source: Si_optical_constants_Palik.csv  (wavelength in µm, n, k)
# Filtered to 175–2100 nm to cover the full measurement range (188–2000 nm).
# ---------------------------------------------------------------------------
_SI_CSV = Path(__file__).parent / "Si_optical_constants_Palik.csv"


def _build_si_interpolators():
    data = np.loadtxt(_SI_CSV, delimiter=",", skiprows=1)
    wl_nm = data[:, 0] * 1000.0  # µm → nm
    n = data[:, 1]
    k = data[:, 2]
    mask = (wl_nm >= 175) & (wl_nm <= 2100)
    wl_nm, n, k = wl_nm[mask], n[mask], k[mask]
    n_interp = scipy.interpolate.PchipInterpolator(wl_nm, n, extrapolate=True)
    # log(k) only on k > 0 points; PCHIP extrapolates smoothly into the NIR
    kpos = k > 0
    logk_interp = scipy.interpolate.PchipInterpolator(
        wl_nm[kpos], np.log(k[kpos]), extrapolate=True
    )
    return n_interp, logk_interp


_SI_N_INTERP, _SI_LOGK_INTERP = _build_si_interpolators()


def _si_nk(wl_nm: np.ndarray):
    """Interpolate Si n, k at requested wavelengths (nm) using Palik (1985) data.

    n: PCHIP — shape-preserving, safe for non-monotone UV data.
    k: PCHIP on log(k) using k>0 points only; extrapolates smoothly into NIR.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    n = _SI_N_INTERP(wl_nm)
    k = np.exp(_SI_LOGK_INTERP(wl_nm))
    return n, np.maximum(k, 0.0)


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
_SM_C1 = 0.0684043  # µm
_SM_C2 = 0.1162414  # µm
_SM_C3 = 9.896161  # µm

# Malitson (1965) amplitudes for fused silica — used as initial guess
_SM_B1_0 = 0.6961663
_SM_B2_0 = 0.4079426
_SM_B3_0 = 0.8974794


def sellmeier_n(wl_nm: np.ndarray, B1: float, B2: float, B3: float) -> np.ndarray:
    """Evaluate Sellmeier n(λ) for SiO2 with fixed pole positions."""
    lam = wl_nm / 1000.0  # nm → µm
    n2 = (
        1.0
        + B1 * lam**2 / (lam**2 - _SM_C1**2)
        + B2 * lam**2 / (lam**2 - _SM_C2**2)
        + B3 * lam**2 / (lam**2 - _SM_C3**2)
    )
    return np.sqrt(np.maximum(n2, 1.0))  # clamp n² ≥ 1 for stability


# ---------------------------------------------------------------------------
# Cauchy dispersion model   n(λ) = A + B/λ²(µm) + C/λ⁴(µm)    k = 0
#
# Initial guess from fused silica (Malitson 1965 Cauchy approximation):
#   A ≈ 1.4580,  B ≈ 0.00354 µm²,  C ≈ 0 µm⁴
# ---------------------------------------------------------------------------
_CA_A0 = 1.458
_CA_B0 = 0.00354  # µm²
_CA_C0 = 0.0  # µm⁴


def cauchy_n(wl_nm: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Evaluate Cauchy n(λ) = A + B/λ²(µm) + C/λ⁴(µm)."""
    lam = wl_nm / 1000.0  # nm → µm
    return A + B / lam**2 + C / lam**4


# ---------------------------------------------------------------------------
# Transfer Matrix Method — normal incidence, single film on substrate
# ---------------------------------------------------------------------------
def _tmm_r(wl_nm: np.ndarray, n1_arr: np.ndarray, d_nm: float) -> np.ndarray:
    """Core TMM: R(λ) for air / film(n1_arr, k=0) / Si substrate."""
    n0 = np.ones(len(wl_nm), dtype=complex)
    n1 = n1_arr.astype(complex)
    n_si, k_si = _si_nk(wl_nm)
    n2 = n_si - 1j * k_si

    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    delta = 2.0 * np.pi * n1 * d_nm / wl_nm
    exp2id = np.exp(2j * delta)

    r_tot = (r01 + r12 * exp2id) / (1.0 + r01 * r12 * exp2id)
    return np.abs(r_tot) ** 2


def tmm_reflectance(
    wl_nm: np.ndarray, d_nm: float, B1: float, B2: float, B3: float
) -> np.ndarray:
    """Compute R(λ) for air / Sellmeier-film / Si at normal incidence."""
    return _tmm_r(wl_nm, sellmeier_n(wl_nm, B1, B2, B3), d_nm)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def fit_cauchy(
    wl_nm: np.ndarray,
    R_meas: np.ndarray,
    d_nm: float,
    A0: float = _CA_A0,
    B0: float = _CA_B0,
    C0: float = _CA_C0,
):
    """Fit Cauchy A, B, C to minimise RMS(R_meas - R_model).

    Returns
    -------
    (A, B, C) : fitted Cauchy coefficients
    rms       : RMS residual (reflectance units, 0–1)
    """

    def objective(params):
        A, B, C = params
        R_model = _tmm_r(wl_nm, cauchy_n(wl_nm, A, B, C), d_nm)
        return np.mean((R_meas - R_model) ** 2)

    bounds = [(1.0, 3.0), (-0.1, 0.5), (-0.01, 0.01)]
    result = scipy.optimize.minimize(
        objective,
        [A0, B0, C0],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if not result.success:
        print(f"  Warning: optimizer did not fully converge — {result.message}")
    rms = float(np.sqrt(result.fun))
    return result.x, rms


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
_MODEL_STYLES = {
    "sellmeier": dict(color="red", linestyle="--"),
    "cauchy": dict(color="green", linestyle=":"),
}


def _plot(wl_nm, R_meas, model_results, stem, out_path):
    """Plot reflectance fit(s) and n(λ) curve(s).

    Parameters
    ----------
    model_results : dict mapping model name → {"R_fit": ..., "n_arr": ...}
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(stem, fontsize=9)

    axes[0].plot(wl_nm, R_meas, color="blue", lw=1, label="Measured")
    for name, res in model_results.items():
        style = _MODEL_STYLES.get(name, {})
        axes[0].plot(
            wl_nm, res["R_fit"], lw=1, label=f"{name.capitalize()} fit", **style
        )
        axes[1].plot(wl_nm, res["n_arr"], lw=1.5, label=name.capitalize(), **style)

    axes[0].set_ylabel("Reflectance")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("n  (refractive index)")
    axes[1].set_xlabel("Wavelength (nm)")
    if len(model_results) > 1:
        axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programmatic entry point (no argparse — callable from other scripts)
# ---------------------------------------------------------------------------
def run_analysis(reflectance_csv, summary_csv, output_dir=None, model="sellmeier"):
    """Fit dispersion model(s) to a saved reflectance CSV and write outputs.

    Parameters
    ----------
    reflectance_csv : str or Path
        Path to a _measured.csv or _calculated.csv file.
    summary_csv : str or Path
        Path to the matching _summary.csv (must contain thickness_nm).
    output_dir : str or Path, optional
        Where to write outputs.  Defaults to the same folder as reflectance_csv.
    model : {"sellmeier", "cauchy", "both"}
        Which dispersion model(s) to fit.

    Returns
    -------
    dict keyed by model name, each containing fit parameters and output paths.
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

    stem = refl_path.stem
    models_to_run = ["sellmeier", "cauchy"] if model == "both" else [model]
    model_results = {}
    results = {}

    for m in models_to_run:
        k_arr = np.zeros(len(wl_nm))

        if m == "sellmeier":
            print("Fitting Sellmeier model (L-BFGS-B) ...")
            (B1, B2, B3), rms = fit_sellmeier(wl_nm, R_meas, d_nm)
            n_arr = sellmeier_n(wl_nm, B1, B2, B3)
            R_fit = _tmm_r(wl_nm, n_arr, d_nm)
            n_632 = float(sellmeier_n(np.array([632.8]), B1, B2, B3)[0])
            print(f"  B1 = {B1:.6f}   (UV oscillator strength ~{_SM_C1*1000:.0f} nm)")
            print(f"  B2 = {B2:.6f}   (UV oscillator strength ~{_SM_C2*1000:.0f} nm)")
            print(f"  B3 = {B3:.6f}   (IR oscillator strength ~{_SM_C3*1000:.0f} nm)")
            print(f"  RMS residual    = {rms:.5f}")
            print(f"  n(632.8 nm)     = {n_632:.5f}")
            fit_params = dict(B1=B1, B2=B2, B3=B3, rms=rms, n_632=n_632)

        else:  # cauchy
            print("Fitting Cauchy model (L-BFGS-B) ...")
            (A, B, C), rms = fit_cauchy(wl_nm, R_meas, d_nm)
            n_arr = cauchy_n(wl_nm, A, B, C)
            R_fit = _tmm_r(wl_nm, n_arr, d_nm)
            n_632 = float(cauchy_n(np.array([632.8]), A, B, C)[0])
            print(f"  A  = {A:.6f}")
            print(f"  B  = {B:.6e}  µm²")
            print(f"  C  = {C:.6e}  µm⁴")
            print(f"  RMS residual    = {rms:.5f}")
            print(f"  n(632.8 nm)     = {n_632:.5f}")
            fit_params = dict(A=A, B=B, C=C, rms=rms, n_632=n_632)

        model_results[m] = {"R_fit": R_fit, "n_arr": n_arr}

        fitnk_path = out_dir / f"{stem}_{m}.fitnk"
        write_fitnk(fitnk_path, wl_nm, n_arr, k_arr)
        print(f"Saved: {fitnk_path}")

        nk_csv_path = out_dir / f"{stem}_{m}_nk.csv"
        np.savetxt(
            nk_csv_path,
            np.column_stack([wl_nm, n_arr, k_arr]),
            delimiter=",",
            header="wavelength_nm,n,k",
            comments="",
        )
        print(f"Saved: {nk_csv_path}")

        results[m] = dict(**fit_params, fitnk_path=fitnk_path, nk_csv_path=nk_csv_path)

    suffix = "_both" if model == "both" else f"_{model}"
    plot_path = out_dir / f"{stem}{suffix}.png"
    _plot(wl_nm, R_meas, model_results, stem, plot_path)
    print(f"Saved: {plot_path}")

    for m in results:
        results[m]["plot_path"] = plot_path

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit Sellmeier/Cauchy n(λ) to a reflectance spectrum (air/film/Si TMM)"
    )
    parser.add_argument("reflectance_csv", help="_measured.csv or _calculated.csv")
    parser.add_argument(
        "summary_csv", help="_summary.csv  (needs column: thickness_nm)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same folder as reflectance_csv)",
    )
    parser.add_argument(
        "--model",
        choices=["sellmeier", "cauchy", "both"],
        default="sellmeier",
        help="Dispersion model to fit (default: sellmeier)",
    )
    args = parser.parse_args()
    run_analysis(args.reflectance_csv, args.summary_csv, args.output, args.model)


if __name__ == "__main__":
    main()
