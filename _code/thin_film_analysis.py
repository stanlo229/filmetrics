# -*- coding: utf-8 -*-
"""
thin_film_analysis.py — Extract n(λ)/k(λ) from reflectance spectra via TMM fitting

Stack assumed:  air (n=1)  /  thin film  /  Si substrate (Palik 1985, PCHIP)

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------

1. Fit a single dispersion model (thickness fixed from summary CSV):

    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv>
    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv> --model cauchy
    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv> --model tauc_lorentz

2. Fit multiple models simultaneously:

    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv> --model both
        # 'both' = sellmeier + cauchy

    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv> --model all
        # 'all'  = sellmeier + cauchy + tauc_lorentz

3. Free-d mode — fit n and thickness d simultaneously (use when the film
   material is not in the Filmetrics library and d from the instrument is
   unreliable):

    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv> --model tauc_lorentz --free-d
        # uses summary CSV d as the initial guess for the grid search

    python _code/thin_film_analysis.py <reflectance_csv> <summary_csv> --model sellmeier --free-d --d0 500
        # --d0 NM overrides the initial thickness guess (nm)

4. Plot Si substrate optical constants (Palik 1985 + PCHIP interpolation):

    python _code/thin_film_analysis.py --plot-si

-------------------------------------------------------------------------------
INPUTS
-------------------------------------------------------------------------------
    reflectance_csv   _measured.csv or _calculated.csv
                      columns: wavelength_nm, reflectance  (values in 0–1)
    summary_csv       _summary.csv  (must contain column: thickness_nm)

-------------------------------------------------------------------------------
OPTIONS
-------------------------------------------------------------------------------
    --model {sellmeier,cauchy,tauc_lorentz,both,all}
                      Dispersion model to fit (default: sellmeier).
                      sellmeier     — 3-param Sellmeier, k=0 (transparent films)
                      cauchy        — 3-param Cauchy A+B/λ²+C/λ⁴, k=0
                      tauc_lorentz  — 5-param Tauc-Lorentz, KK-consistent n+k
                                      (use for UV-absorbing / organic films)
                      both          — sellmeier + cauchy
                      all           — sellmeier + cauchy + tauc_lorentz

    --free-d          Fit thickness d simultaneously with n (joint optimisation).
                      A log-spaced grid search over d is used to escape fringe-
                      order local minima. Useful when n is unknown (new polymer).

    --d0 NM           Override the initial d guess (nm) for --free-d.
                      Defaults to the thickness in summary_csv.

    --output DIR      Output directory (default: same folder as reflectance_csv).

    --plot-si         Plot Si Palik data + PCHIP interpolation and exit.

-------------------------------------------------------------------------------
OUTPUTS  (written to --output dir)
-------------------------------------------------------------------------------
    <stem>_<model>.fitnk       n/k table in Filmetrics v3 tabulated format
    <stem>_<model>_nk.csv      wavelength_nm, n, k
    <stem>_<model[s]>.png      reflectance fit(s), n(λ), [k(λ) if non-zero]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt

_HC_EV_NM = 1239.8419  # h·c in eV·nm  (used by Tauc-Lorentz model)

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
    # Store raw tabulated data for plotting
    global _SI_WL_NM, _SI_N, _SI_K
    _SI_WL_NM, _SI_N, _SI_K = wl_nm, n, k
    n_interp = scipy.interpolate.PchipInterpolator(wl_nm, n, extrapolate=True)
    # log(k) only on k > 0 points; PCHIP extrapolates smoothly into the NIR
    kpos = k > 0
    logk_interp = scipy.interpolate.PchipInterpolator(
        wl_nm[kpos], np.log(k[kpos]), extrapolate=True
    )
    return n_interp, logk_interp


_SI_N_INTERP, _SI_LOGK_INTERP = _build_si_interpolators()


def _si_nk(wl_nm: np.ndarray):
    """Evaluate Si n, k at requested wavelengths (nm) via PCHIP on Palik (1985) data.

    n: PCHIP — shape-preserving, safe for non-monotone UV data.
    k: PCHIP on log(k) using k>0 points only; extrapolates smoothly into NIR.
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    n = _SI_N_INTERP(wl_nm)
    k = np.exp(_SI_LOGK_INTERP(wl_nm))
    return n, np.maximum(k, 0.0)


def plot_si_nk(output_path=None):
    """Plot Si n(λ) and k(λ): Palik tabulated points + PCHIP interpolation.

    Parameters
    ----------
    output_path : str or Path, optional
        PNG path.  Defaults to Si_optical_constants_Palik.png next to the CSV.
    """
    wl_dense = np.linspace(_SI_WL_NM[0], _SI_WL_NM[-1], 2000)
    n_dense, k_dense = _si_nk(wl_dense)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(
        "Si optical constants — Palik (1985) + PCHIP interpolation", fontsize=9
    )

    axes[0].plot(wl_dense, n_dense, color="royalblue", lw=1.5, label="PCHIP")
    axes[0].scatter(
        _SI_WL_NM, _SI_N, color="royalblue", s=20, zorder=5, label="Palik data"
    )
    axes[0].set_ylabel("n  (refractive index)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    k_pos_mask = _SI_K > 0
    axes[1].plot(wl_dense, k_dense, color="firebrick", lw=1.5, label="PCHIP")
    axes[1].scatter(
        _SI_WL_NM[k_pos_mask],
        _SI_K[k_pos_mask],
        color="firebrick",
        s=20,
        zorder=5,
        label="Palik data",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("k  (extinction coefficient, log scale)")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.4, which="both")

    fig.tight_layout()
    if output_path is None:
        output_path = _SI_CSV.with_suffix(".png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


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
    wl_um = wl_nm / 1000.0  # nm → µm
    n2 = (
        1.0
        + B1 * wl_um**2 / (wl_um**2 - _SM_C1**2)
        + B2 * wl_um**2 / (wl_um**2 - _SM_C2**2)
        + B3 * wl_um**2 / (wl_um**2 - _SM_C3**2)
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
# Tauc-Lorentz dispersion model  (Jellison & Modine 1996)
#
# Models an amorphous/organic film with a UV absorption band and optical
# bandgap Eg.  Both n(λ) and k(λ) are returned; they are Kramers-Kronig
# consistent by construction.
#
# ε₂(E) = A·E₀·C·(E−Eg)² / [(E²−E₀²)² + C²E²] / E   (E > Eg)
#        = 0                                            (E ≤ Eg)
# ε₁(E) = ε∞ + (2/π) P ∫_Eg^∞  ξ·ε₂(ξ)/(ξ²−E²) dξ    [numerical KK]
#
# Parameters (all in eV):
#   eps_inf : high-frequency dielectric constant  (n²→∞)
#   A       : oscillator amplitude (eV)
#   E0      : oscillator peak energy (eV)  — absorption-band centre
#   C       : broadening / damping (eV)
#   Eg      : optical bandgap (eV)         — absorption onset
#
# Typical polynorbornene starting values:
#   eps_inf=2.25,  A=10,  E0=5.0 (~248 nm),  C=2.0,  Eg=3.1 (~400 nm)
# ---------------------------------------------------------------------------
_TL_EI_0 = 2.25  # eps_inf
_TL_A_0 = 10.0  # eV
_TL_E0_0 = 5.0  # eV  (~248 nm)
_TL_C_0 = 2.0  # eV
_TL_EG_0 = 3.1  # eV  (~400 nm)


def _tl_eps2(E: np.ndarray, A: float, E0: float, C: float, Eg: float) -> np.ndarray:
    """Tauc-Lorentz imaginary dielectric function ε₂(E). Vectorised."""
    return np.where(
        E > Eg,
        A * E0 * C * (E - Eg) ** 2 / ((E**2 - E0**2) ** 2 + C**2 * E**2) / E,
        0.0,
    )


def tauc_lorentz_nk(
    wl_nm: np.ndarray,
    eps_inf: float,
    A: float,
    E0: float,
    C: float,
    Eg: float,
) -> tuple:
    """Evaluate Tauc-Lorentz n(λ) and k(λ).

    ε₁ is obtained from ε₂ via a vectorised numerical Kramers-Kronig integral
    on a dense energy grid (Eg → max(E+5, E0+20C) eV, 3000 points).
    Near-singular values in the integrand are zeroed to handle the principal
    value numerically.

    Returns
    -------
    (n_arr, k_arr) : arrays with the same shape as wl_nm
    """
    E = _HC_EV_NM / np.asarray(wl_nm, dtype=float)

    eps2 = _tl_eps2(E, A, E0, C, Eg)

    # Dense energy grid for KK integration
    E_hi = max(float(E0) + 20.0 * float(C), float(E.max()) + 5.0)
    E_grid = np.linspace(float(Eg) + 1e-4, E_hi, 3000)
    e2g = _tl_eps2(E_grid, A, E0, C, Eg)

    # Vectorised PV integral: shape (3000, N_wl)
    denom = E_grid[:, None] ** 2 - E[None, :] ** 2
    intgd = E_grid[:, None] * e2g[:, None] / denom
    # Zero near-singular entries (principal value handling)
    dE_min = (E_grid[1] - E_grid[0]) * 0.1
    intgd = np.where(np.abs(denom) < dE_min, 0.0, intgd)

    eps1 = eps_inf + (2.0 / np.pi) * np.trapezoid(intgd, E_grid, axis=0)

    mod = np.sqrt(eps1**2 + eps2**2)
    n = np.sqrt(np.maximum((mod + eps1) / 2.0, 0.0))
    k = np.sqrt(np.maximum((mod - eps1) / 2.0, 0.0))
    return n, k


# ---------------------------------------------------------------------------
# Transfer Matrix Method — normal incidence, single film on substrate
# ---------------------------------------------------------------------------
def _tmm_r(wl_nm: np.ndarray, n1_arr: np.ndarray, d_nm: float) -> np.ndarray:
    """Core TMM: R(λ) for air / film / Si substrate at normal incidence.

    n1_arr may be real (k=0, Sellmeier/Cauchy) or complex (n−ik, Tauc-Lorentz).
    """
    n_si, k_si = _si_nk(wl_nm)
    n0 = np.ones(len(wl_nm), dtype=complex)  # air refractive index (complex, n=1, k=0)
    n1 = n1_arr.astype(complex)  # film refractive index (real, k=0)
    n2 = n_si - 1j * k_si  # Si substrate refractive index (complex)

    r01 = (n0 - n1) / (n0 + n1)  # Fresnel reflection coefficient at air/film interface
    r12 = (n1 - n2) / (n1 + n2)  # Fresnel reflection coefficient at film/Si interface
    delta = (
        2.0 * np.pi * n1 * d_nm / wl_nm
    )  # phase thickness of the film -> indicates phase delay upon reflection from the film/substrate interface
    exp2id = np.exp(2j * delta)  # phase factor for round-trip through the film

    r_tot = (r01 + r12 * exp2id) / (
        1.0 + r01 * r12 * exp2id
    )  # Total reflection coefficient for the three-layer system
    return (
        np.abs(r_tot) ** 2
    )  # reflectance is the magnitude squared of the total reflection coefficient


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


def fit_tauc_lorentz(
    wl_nm: np.ndarray,
    R_meas: np.ndarray,
    d_nm: float,
    ei0: float = _TL_EI_0,
    A0: float = _TL_A_0,
    E0_0: float = _TL_E0_0,
    C0: float = _TL_C_0,
    Eg0: float = _TL_EG_0,
):
    """Fit Tauc-Lorentz (eps_inf, A, E0, C, Eg) to minimise RMS(R_meas − R_model).

    The film is modelled as absorbing (k ≥ 0) — the complex refractive index
    n̂ = n − ik is passed directly to the TMM.

    Returns
    -------
    (eps_inf, A, E0, C, Eg) : fitted parameters  (all in eV)
    rms                     : RMS residual (reflectance units, 0–1)
    """

    def objective(params):
        ei, A, E0, C, Eg = params
        if Eg >= E0 or C >= 2.0 * E0:  # physical constraints not captured by bounds
            return 1.0
        n, k = tauc_lorentz_nk(wl_nm, ei, A, E0, C, Eg)
        R_model = _tmm_r(wl_nm, n - 1j * k, d_nm)
        return float(np.mean((R_meas - R_model) ** 2))

    bounds = [
        (1.0, 5.0),  # eps_inf
        (0.1, 200.0),  # A  (eV)
        (3.0, 8.0),  # E0 (eV) — UV range 155–413 nm
        (0.1, 8.0),  # C  (eV)
        (1.5, 6.5),  # Eg (eV) — onset 190–827 nm
    ]
    result = scipy.optimize.minimize(
        objective,
        [ei0, A0, E0_0, C0, Eg0],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if not result.success:
        print(f"  Warning: optimizer did not fully converge — {result.message}")
    rms = float(np.sqrt(result.fun))
    return result.x, rms


# ---------------------------------------------------------------------------
# Simultaneous n + d fitting  (free-d mode)
#
# When the film material is not in the Filmetrics library, the thickness d
# reported by the instrument is unreliable (it assumed a wrong n).  The
# functions below treat d as an additional free parameter and fit it
# alongside the dispersion model coefficients.
#
# Strategy: grid search over d initial values (log-spaced around d0) to
# escape local minima caused by fringe-order ambiguity, then L-BFGS-B
# refinement from each starting point.  The global minimum is returned.
# ---------------------------------------------------------------------------

def _grid_search_nd(wl_nm, R_meas, d0_nm, model_fn, p0, bounds_model, n_grid=9):
    """Fit dispersion model + thickness simultaneously, with grid search over d.

    Parameters
    ----------
    wl_nm, R_meas : measurement arrays
    d0_nm         : initial thickness estimate (nm) — sets the grid range
    model_fn      : callable(wl_nm, model_params_list) → n1_arr (real or complex)
    p0            : list of initial model parameter values
    bounds_model  : list of (lo, hi) tuples for model parameters
    n_grid        : number of d starting points in the grid search

    Returns
    -------
    model_params : ndarray of fitted model parameters
    d_nm         : fitted thickness (nm)
    rms          : RMS residual (reflectance units, 0–1)
    """
    d_lo = max(d0_nm * 0.25, 10.0)
    d_hi = d0_nm * 4.0
    d_grid = np.geomspace(d_lo, d_hi, n_grid)
    bounds = list(bounds_model) + [(10.0, 200_000.0)]

    best_rms, best_x = np.inf, None
    for d_init in d_grid:
        x0 = list(p0) + [d_init]

        def objective(params, _wl=wl_nm, _R=R_meas, _fn=model_fn):
            *mp, d = params
            n1 = _fn(_wl, mp)
            return float(np.mean((_R - _tmm_r(_wl, n1, d)) ** 2))

        result = scipy.optimize.minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10},
        )
        if result.fun < best_rms:
            best_rms, best_x = result.fun, result.x

    return best_x[:-1], float(best_x[-1]), float(np.sqrt(best_rms))


def fit_sellmeier_nd(wl_nm, R_meas, d0_nm):
    """Fit Sellmeier B1, B2, B3 and thickness d simultaneously.

    Returns (B1, B2, B3), d_nm, rms
    """
    p, d, rms = _grid_search_nd(
        wl_nm, R_meas, d0_nm,
        lambda wl, mp: sellmeier_n(wl, *mp),
        [_SM_B1_0, _SM_B2_0, _SM_B3_0],
        [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0)],
    )
    return p, d, rms


def fit_cauchy_nd(wl_nm, R_meas, d0_nm):
    """Fit Cauchy A, B, C and thickness d simultaneously.

    Returns (A, B, C), d_nm, rms
    """
    p, d, rms = _grid_search_nd(
        wl_nm, R_meas, d0_nm,
        lambda wl, mp: cauchy_n(wl, *mp),
        [_CA_A0, _CA_B0, _CA_C0],
        [(1.0, 3.0), (-0.1, 0.5), (-0.01, 0.01)],
    )
    return p, d, rms


def fit_tauc_lorentz_nd(wl_nm, R_meas, d0_nm):
    """Fit Tauc-Lorentz (eps_inf, A, E0, C, Eg) and thickness d simultaneously.

    Returns (eps_inf, A, E0, C, Eg), d_nm, rms
    """
    def tl_fn(wl, mp):
        ei, A, E0, C, Eg = mp
        if Eg >= E0 or C >= 2.0 * E0:
            return np.ones(len(wl), dtype=complex)  # unphysical — forces high residual
        n, k = tauc_lorentz_nk(wl, ei, A, E0, C, Eg)
        return n - 1j * k

    p, d, rms = _grid_search_nd(
        wl_nm, R_meas, d0_nm, tl_fn,
        [_TL_EI_0, _TL_A_0, _TL_E0_0, _TL_C_0, _TL_EG_0],
        [(1.0, 5.0), (0.1, 200.0), (3.0, 8.0), (0.1, 8.0), (1.5, 6.5)],
    )
    return p, d, rms


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
    "tauc_lorentz": dict(color="purple", linestyle="-."),
}

_MODEL_LABELS = {
    "sellmeier": "Sellmeier",
    "cauchy": "Cauchy",
    "tauc_lorentz": "Tauc-Lorentz",
}


def _plot(wl_nm, R_meas, model_results, stem, out_path):
    """Plot reflectance fit(s), n(λ), and k(λ) (when non-zero).

    Parameters
    ----------
    model_results : dict mapping model name →
                    {"R_fit": ..., "n_arr": ..., "k_arr": ...}
    """
    has_k = any(np.any(res["k_arr"] > 0) for res in model_results.values())
    nrows = 3 if has_k else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 3.5 * nrows), sharex=True)
    fig.suptitle(stem, fontsize=9)

    axes[0].plot(wl_nm, R_meas, color="blue", lw=1, label="Measured")
    for name, res in model_results.items():
        style = _MODEL_STYLES.get(name, {})
        label = _MODEL_LABELS.get(name, name.capitalize())
        axes[0].plot(wl_nm, res["R_fit"], lw=1, label=f"{label} fit", **style)
        axes[1].plot(wl_nm, res["n_arr"], lw=1.5, label=label, **style)
        if has_k:
            axes[2].plot(wl_nm, res["k_arr"], lw=1.5, label=label, **style)

    axes[0].set_ylabel("Reflectance")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("n  (refractive index)")
    if len(model_results) > 1:
        axes[1].legend(loc="upper right", fontsize=8)
    if has_k:
        axes[2].set_ylabel("k  (extinction coefficient)")
        axes[2].set_xlabel("Wavelength (nm)")
        if len(model_results) > 1:
            axes[2].legend(loc="upper right", fontsize=8)
    else:
        axes[1].set_xlabel("Wavelength (nm)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programmatic entry point (no argparse — callable from other scripts)
# ---------------------------------------------------------------------------
def run_analysis(reflectance_csv, summary_csv, output_dir=None, model="sellmeier", free_d=False, d0_nm=None):
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
    d_nm_csv = float(row["thickness_nm"])
    # d0 for fitting: user override takes priority, else use Filmetrics value
    d0 = d0_nm if d0_nm is not None else d_nm_csv
    if free_d:
        print(f"  thickness initial guess (d0): {d0:.2f} nm  [free — will be fitted]")
    else:
        print(f"  film thickness (from summary): {d0:.2f} nm  [fixed]")

    stem = refl_path.stem
    if model == "both":
        models_to_run = ["sellmeier", "cauchy"]
    elif model == "all":
        models_to_run = ["sellmeier", "cauchy", "tauc_lorentz"]
    else:
        models_to_run = [model]
    model_results = {}
    results = {}

    for m in models_to_run:
        k_arr = np.zeros(len(wl_nm))
        d_fit = d0  # may be updated when free_d=True

        if m == "sellmeier":
            if free_d:
                print("Fitting Sellmeier + d (grid search + L-BFGS-B) ...")
                (B1, B2, B3), d_fit, rms = fit_sellmeier_nd(wl_nm, R_meas, d0)
            else:
                print("Fitting Sellmeier model (L-BFGS-B) ...")
                (B1, B2, B3), rms = fit_sellmeier(wl_nm, R_meas, d0)
            n_arr = sellmeier_n(wl_nm, B1, B2, B3)
            R_fit = _tmm_r(wl_nm, n_arr, d_fit)
            n_632 = float(sellmeier_n(np.array([632.8]), B1, B2, B3)[0])
            print(f"  B1 = {B1:.6f}   (UV oscillator strength ~{_SM_C1*1000:.0f} nm)")
            print(f"  B2 = {B2:.6f}   (UV oscillator strength ~{_SM_C2*1000:.0f} nm)")
            print(f"  B3 = {B3:.6f}   (IR oscillator strength ~{_SM_C3*1000:.0f} nm)")
            if free_d:
                print(f"  d  = {d_fit:.2f}  nm  (fitted)")
            print(f"  RMS residual    = {rms:.5f}")
            print(f"  n(632.8 nm)     = {n_632:.5f}")
            fit_params = dict(B1=B1, B2=B2, B3=B3, d_nm=d_fit, rms=rms, n_632=n_632)

        elif m == "cauchy":
            if free_d:
                print("Fitting Cauchy + d (grid search + L-BFGS-B) ...")
                (A, B, C), d_fit, rms = fit_cauchy_nd(wl_nm, R_meas, d0)
            else:
                print("Fitting Cauchy model (L-BFGS-B) ...")
                (A, B, C), rms = fit_cauchy(wl_nm, R_meas, d0)
            n_arr = cauchy_n(wl_nm, A, B, C)
            R_fit = _tmm_r(wl_nm, n_arr, d_fit)
            n_632 = float(cauchy_n(np.array([632.8]), A, B, C)[0])
            print(f"  A  = {A:.6f}")
            print(f"  B  = {B:.6e}  µm²")
            print(f"  C  = {C:.6e}  µm⁴")
            if free_d:
                print(f"  d  = {d_fit:.2f}  nm  (fitted)")
            print(f"  RMS residual    = {rms:.5f}")
            print(f"  n(632.8 nm)     = {n_632:.5f}")
            fit_params = dict(A=A, B=B, C=C, d_nm=d_fit, rms=rms, n_632=n_632)

        else:  # tauc_lorentz
            if free_d:
                print("Fitting Tauc-Lorentz + d (grid search + L-BFGS-B) ...")
                (ei, A, E0, C, Eg), d_fit, rms = fit_tauc_lorentz_nd(wl_nm, R_meas, d0)
            else:
                print("Fitting Tauc-Lorentz model (L-BFGS-B) ...")
                (ei, A, E0, C, Eg), rms = fit_tauc_lorentz(wl_nm, R_meas, d0)
            n_arr, k_arr = tauc_lorentz_nk(wl_nm, ei, A, E0, C, Eg)
            R_fit = _tmm_r(wl_nm, n_arr - 1j * k_arr, d_fit)
            n_632 = float(tauc_lorentz_nk(np.array([632.8]), ei, A, E0, C, Eg)[0][0])
            k_300 = float(tauc_lorentz_nk(np.array([300.0]), ei, A, E0, C, Eg)[1][0])
            print(f"  eps_inf = {ei:.5f}")
            print(f"  A       = {A:.5f}  eV")
            print(f"  E0      = {E0:.5f}  eV  ({_HC_EV_NM/E0:.1f} nm)")
            print(f"  C       = {C:.5f}  eV")
            print(f"  Eg      = {Eg:.5f}  eV  ({_HC_EV_NM/Eg:.1f} nm onset)")
            if free_d:
                print(f"  d  = {d_fit:.2f}  nm  (fitted)")
            print(f"  RMS residual    = {rms:.5f}")
            print(f"  n(632.8 nm)     = {n_632:.5f}")
            print(f"  k(300 nm)       = {k_300:.5f}")
            fit_params = dict(
                eps_inf=ei, A=A, E0=E0, C=C, Eg=Eg, d_nm=d_fit, rms=rms, n_632=n_632, k_300=k_300
            )

        model_results[m] = {"R_fit": R_fit, "n_arr": n_arr, "k_arr": k_arr}

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

    suffix = {"both": "_both", "all": "_all"}.get(model, f"_{model}")
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
    parser.add_argument(
        "reflectance_csv", nargs="?", help="_measured.csv or _calculated.csv"
    )
    parser.add_argument(
        "summary_csv", nargs="?", help="_summary.csv  (needs column: thickness_nm)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same folder as reflectance_csv)",
    )
    parser.add_argument(
        "--model",
        choices=["sellmeier", "cauchy", "tauc_lorentz", "both", "all"],
        default="sellmeier",
        help="Dispersion model to fit (default: sellmeier); "
        "'both'=sellmeier+cauchy, 'all'=all three",
    )
    parser.add_argument(
        "--free-d",
        action="store_true",
        help="Fit thickness d simultaneously with n (joint n+d fit). "
             "Uses Filmetrics d as initial guess unless --d0 is given.",
    )
    parser.add_argument(
        "--d0",
        type=float,
        default=None,
        metavar="NM",
        help="Override the initial thickness guess (nm) used by --free-d.",
    )
    parser.add_argument(
        "--plot-si",
        action="store_true",
        help="Plot the Si Palik data + PCHIP interpolation and exit",
    )
    args = parser.parse_args()
    if args.plot_si:
        plot_si_nk()
        return
    run_analysis(args.reflectance_csv, args.summary_csv, args.output, args.model,
                 free_d=args.free_d, d0_nm=args.d0)


if __name__ == "__main__":
    main()
