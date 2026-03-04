# -*- coding: utf-8 -*-
"""
polyolefin_analysis.py — Extract n(λ) from reflectance using 1-oscillator Sellmeier + TMM

Film model:  air (n=1)  /  polyolefin film (1-osc. Sellmeier, k=0)  /  substrate

    n²(λ) = A  +  B·λ²/(λ²−C²)

    A : background constant (IR oscillator contributions + 1)
    B : UV oscillator strength
    C : effective UV pole wavelength (µm, free parameter, typically 0.08–0.20 µm)

This form is physically stable: n stays bounded and positive for all λ > C.
Unlike the 3-term polynomial Cauchy model, it cannot diverge in the UV.

Supported substrates (--substrate flag):
    si           Silicon wafer (Aspnes & Studna 1983 + Green 2008)
    bk7          BK7 borosilicate glass (Schott formula, k ≈ 0 in 350–2000 nm)
    fused_silica Fused silica (Malitson 1965, k ≈ 0 in 160–2500 nm)

Usage:
    python _code/polyolefin_analysis.py <reflectance_csv> <summary_csv> [options]

Options:
    --substrate {si,bk7,fused_silica}   default: si
    --output <dir>                       default: same folder as reflectance_csv

Inputs:
    reflectance_csv  _measured.csv or _calculated.csv
                     columns: wavelength_nm, reflectance  (values in 0–1)
    summary_csv      _summary.csv  (must contain column thickness_nm)

Outputs (written to --output dir):
    <stem>_poly_sellmeier.fitnk   tabulated n/k in Filmetrics v3 format
    <stem>_nk.csv                 wavelength_nm, n, k
    <stem>_poly_sellmeier.png     two-panel plot: reflectance fit + n(λ)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Substrate 1 — Silicon
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
    0.69, 0.579, 0.618, 1.036, 1.524, 2.264, 2.334, 2.449, 3.043,
    3.939, 5.082, 5.500, 5.633, 5.596, 5.576,
    5.587, 4.676, 4.293, 4.077, 3.939, 3.844,
    3.774, 3.723, 3.681, 3.650, 3.620, 3.592, 3.570, 3.554,
    3.541, 3.520,
])

_SI_K = np.array([
    2.89, 2.976, 2.944, 3.111, 3.491, 3.584, 3.635, 3.473, 2.944,
    2.349, 1.063, 0.619, 0.425, 0.375, 0.350,
    3.040e-1, 9.22e-2, 4.52e-2, 2.82e-2, 1.99e-2, 1.47e-2,
    1.07e-2, 7.82e-3, 5.44e-3, 3.43e-3, 1.94e-3, 9.47e-4, 3.64e-4,
    1.46e-4, 4.98e-5, 2.80e-7,
])

_SI_N_INTERP    = scipy.interpolate.PchipInterpolator(_SI_WL_NM, _SI_N,         extrapolate=True)
_SI_LOGK_INTERP = scipy.interpolate.PchipInterpolator(_SI_WL_NM, np.log(_SI_K), extrapolate=True)


def _si_nk(wl_nm: np.ndarray):
    n = _SI_N_INTERP(wl_nm)
    k = np.exp(_SI_LOGK_INTERP(wl_nm))
    return n, np.maximum(k, 0.0)


# ---------------------------------------------------------------------------
# Substrate 2 — BK7 borosilicate glass
# Schott formula (λ in µm), valid 310–2325 nm.  k ≈ 0 in 350–2000 nm.
# ---------------------------------------------------------------------------
def _bk7_nk(wl_nm: np.ndarray):
    lam = wl_nm / 1000.0
    n2 = (1.0
          + 1.03961212  * lam**2 / (lam**2 - 0.00600069867)
          + 0.231792344 * lam**2 / (lam**2 - 0.0200179144)
          + 1.01046945  * lam**2 / (lam**2 - 103.560653))
    return np.sqrt(np.maximum(n2, 1.0)), np.zeros_like(wl_nm)


# ---------------------------------------------------------------------------
# Substrate 3 — Fused silica (SiO2 amorphous)
# Malitson (1965), λ in µm, valid 210–3710 nm.  k ≈ 0 throughout.
# ---------------------------------------------------------------------------
def _fused_silica_nk(wl_nm: np.ndarray):
    lam = wl_nm / 1000.0
    n2 = (1.0
          + 0.6961663 * lam**2 / (lam**2 - 0.0684043**2)
          + 0.4079426 * lam**2 / (lam**2 - 0.1162414**2)
          + 0.8974794 * lam**2 / (lam**2 - 9.896161**2))
    return np.sqrt(np.maximum(n2, 1.0)), np.zeros_like(wl_nm)


_SUBSTRATE_FUNCS = {
    "si":           _si_nk,
    "bk7":          _bk7_nk,
    "fused_silica": _fused_silica_nk,
}


# ---------------------------------------------------------------------------
# 1-oscillator Sellmeier film model   n²(λ) = A + B·λ²/(λ²−C²)    k = 0
#
# A : background (n² as λ→∞), absorbs IR oscillator contributions
# B : UV oscillator strength  (dimensionless)
# C : UV pole wavelength (µm, free — fitted per material)
#
# Typical polyolefin starting values (Malitson-inspired):
#   A = 1.0,  B = 1.5,  C = 0.10 µm  →  n(590 nm) ≈ 1.50
# ---------------------------------------------------------------------------
def sellmeier_n(wl_nm: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    """Evaluate 1-oscillator Sellmeier n(λ) for a transparent polymer film."""
    lam = wl_nm / 1000.0   # nm → µm
    n2 = A + B * lam**2 / (lam**2 - C**2)
    return np.sqrt(np.maximum(n2, 1.0))


# ---------------------------------------------------------------------------
# Transfer Matrix Method — normal incidence, single film on substrate
# ---------------------------------------------------------------------------
def tmm_reflectance(
    wl_nm: np.ndarray,
    d_nm: float,
    A: float,
    B: float,
    C: float,
    substrate: str = "si",
) -> np.ndarray:
    """Compute R(λ) for air / polymer-film / substrate at normal incidence."""
    nk_fn = _SUBSTRATE_FUNCS[substrate]
    n0 = np.ones(len(wl_nm), dtype=complex)
    n1 = sellmeier_n(wl_nm, A, B, C).astype(complex)
    n_sub, k_sub = nk_fn(np.asarray(wl_nm, dtype=float))
    n2 = (n_sub - 1j * k_sub)

    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    delta = 2.0 * np.pi * n1 * d_nm / wl_nm
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
    substrate: str = "si",
    A0: float = 1.0,
    B0: float = 1.5,
    C0: float = 0.10,
):
    """Fit 1-oscillator Sellmeier A, B, C to minimise RMS(R_meas - R_model).

    Returns
    -------
    (A, B, C) : fitted parameters  (A, B dimensionless; C in µm)
    rms       : RMS residual (reflectance units, 0–1)
    """
    def objective(params):
        a, b, c = params
        R_model = tmm_reflectance(wl_nm, d_nm, a, b, c, substrate)
        return np.mean((R_meas - R_model) ** 2)

    bounds = [
        (0.8, 3.0),    # A
        (0.1, 4.0),    # B
        (0.05, 0.35),  # C (µm) — UV pole between 50 and 350 nm
    ]
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


# ---------------------------------------------------------------------------
# .fitnk writer  (Filmetrics v3 tabulated format)
# ---------------------------------------------------------------------------
def write_fitnk(path, wl_nm: np.ndarray, n_arr: np.ndarray, k_arr: np.ndarray):
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
        for _ in range(6):
            f.write("0, 1\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _plot(wl_nm, R_meas, R_fit, n_arr, stem, out_path, substrate):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(f"{stem}  [substrate: {substrate}]", fontsize=9)

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
# Programmatic entry point (callable from other scripts)
# ---------------------------------------------------------------------------
def run_analysis(reflectance_csv, summary_csv, output_dir=None, substrate="si"):
    """Fit 1-oscillator Sellmeier n(λ) to a saved reflectance CSV.

    Parameters
    ----------
    reflectance_csv : str or Path
    summary_csv     : str or Path  (must contain column thickness_nm)
    output_dir      : str or Path, optional
    substrate       : 'si', 'bk7', or 'fused_silica'

    Returns
    -------
    dict with keys: A, B, C, C_nm, rms, n_632, fitnk_path, nk_csv_path, plot_path
    """
    if substrate not in _SUBSTRATE_FUNCS:
        raise ValueError(f"Unknown substrate '{substrate}'. "
                         f"Choose from: {list(_SUBSTRATE_FUNCS)}")

    refl_path = Path(reflectance_csv)
    summ_path = Path(summary_csv)
    out_dir = Path(output_dir) if output_dir else refl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(refl_path, delimiter=",", skiprows=1)
    wl_nm = data[:, 0]
    R_meas = data[:, 1]
    print(f"Loaded {len(wl_nm)} points from {refl_path.name}")
    print(f"  wavelength range: {wl_nm[0]:.1f} – {wl_nm[-1]:.1f} nm")
    print(f"  substrate: {substrate}")

    with open(summ_path) as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
    d_nm = float(row["thickness_nm"])
    print(f"  film thickness (from summary): {d_nm:.2f} nm")

    print("Fitting 1-oscillator Sellmeier model (L-BFGS-B) ...")
    (A, B, C), rms = fit_sellmeier(wl_nm, R_meas, d_nm, substrate)
    n_632 = float(sellmeier_n(np.array([632.8]), A, B, C)[0])
    print(f"  A = {A:.6f}   (background)")
    print(f"  B = {B:.6f}   (UV oscillator strength)")
    print(f"  C = {C:.6f} µm  ({C*1000:.1f} nm UV pole)")
    print(f"  RMS residual    = {rms:.5f}")
    print(f"  n(632.8 nm)     = {n_632:.5f}")

    n_arr = sellmeier_n(wl_nm, A, B, C)
    k_arr = np.zeros_like(n_arr)
    R_fit = tmm_reflectance(wl_nm, d_nm, A, B, C, substrate)

    stem = refl_path.stem

    fitnk_path = out_dir / f"{stem}_poly_sellmeier.fitnk"
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

    plot_path = out_dir / f"{stem}_poly_sellmeier.png"
    _plot(wl_nm, R_meas, R_fit, n_arr, stem, plot_path, substrate)
    print(f"Saved: {plot_path}")

    return dict(A=A, B=B, C=C, C_nm=C*1000, rms=rms, n_632=n_632,
                fitnk_path=fitnk_path, nk_csv_path=nk_csv_path, plot_path=plot_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit 1-oscillator Sellmeier n(λ) to a polyolefin reflectance spectrum"
    )
    parser.add_argument("reflectance_csv", help="_measured.csv or _calculated.csv")
    parser.add_argument("summary_csv",     help="_summary.csv  (needs column: thickness_nm)")
    parser.add_argument(
        "--substrate",
        choices=list(_SUBSTRATE_FUNCS),
        default="si",
        help="Substrate material (default: si)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same folder as reflectance_csv)",
    )
    args = parser.parse_args()
    run_analysis(args.reflectance_csv, args.summary_csv, args.output, args.substrate)


if __name__ == "__main__":
    main()
