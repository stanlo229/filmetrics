# -*- coding: utf-8 -*-
"""
Filmetrics FIRemote example — Python 3, pythonnet/clr approach
@author: stanlo

Before running:
    0. source ~/Research/envs/filmetrics32/Scripts/activate
    1. Open FILMeasure and complete a Baseline
    2. Save the recipe so the Baseline can be recovered
"""
import sys
import time
sys.path.append(r"C:\Program Files (x86)\Filmetrics\FILMeasure")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import pythonnet
pythonnet.load("netfx")
import clr
clr.AddReference('FILMeasure')
clr.AddReference('System.Windows.Forms')

import System
from Filmetrics import FIRemote
from System.Windows.Forms import FormWindowState
from System import String

def baseline_acquisition(theFIRemote, reflectance_standard="Si"):
    # Acquire baseline and reference
    input("Place the sample and press Enter to continue...")
    theFIRemote.BaselineAcquireSpectrumFromSample()
    time.sleep(1)
    input("Place the reference sample and press Enter to continue...")
    theFIRemote.BaselineSetRefMat(reflectance_standard);
    time.sleep(1)
    theFIRemote.BaselineAcquireReference();
    time.sleep(1)
    input("Remove the reference sample and press Enter to continue...")
    theFIRemote.BaselineAcquireBackgroundAfterRef()
    time.sleep(1)
    theFIRemote.BaselineCommit()

def measure_sample_plot(theFIRemote, id="test_run"):
    # Measure sample and plot and save to results with labelled data
    input("Place the sample and press Enter to continue...")
    theResult = theFIRemote.Measure(False)
    print(f"{theResult.get_ResultsSummary()=}")
    wavelengths = np.array(theResult.get_PrimaryWavelengths())
    reflectances = np.array(theResult.get_PrimarySpectrum())
    thickness = theResult.get_LayerThicknesses()[1]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    stem = f"{id}_{timestamp}"

    # Save spectrum CSV
    np.savetxt(results_dir / f"{stem}.csv", np.column_stack([wavelengths, reflectances]),
               delimiter=",", header="wavelength_nm,reflectance", comments="")

    gof = theResult.GOF

    # Save summary CSV
    with open(results_dir / f"{stem}_summary.csv", "w") as f:
        f.write("id,timestamp,thickness_nm,GOF\n")
        f.write(f"{id},{timestamp},{thickness},{gof}\n")

    # Plot and save
    fig, ax = plt.subplots()
    ax.plot(wavelengths, reflectances)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"{id} — {timestamp}")
    ax.text(0.98, 0.95, f"Thickness: {thickness:.1f} nm\nGOF: {gof:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.savefig(results_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)