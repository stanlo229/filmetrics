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

import filmetrics_api
import thin_film_analysis

if __name__ == "__main__":
    theFIRemote = FIRemote(
        FIRemote.GraphicalUserInterfaceType.Standard,
        FIRemote.GraphicalUserInterfaceStartupState.Shown,
        FormWindowState.Normal,
        getattr(FIRemote.ConstructorWarning, 'None'),
        String.Empty
    )
    # theFIRemote.SetRecipe("polynorbornene") # whenever we change recipe, new baseline is needed
    theFIRemote.SetRecipe("SiO2 16kA on Si")

    baseline = False
    if baseline:
        filmetrics_api.baseline_acquisition(theFIRemote)
    else:
        theFIRemote.AuthenticateRefBac()

    stem, results_dir = filmetrics_api.measure_sample_plot(theFIRemote, "test_run")

    thin_film_analysis.run_analysis(
        reflectance_csv=results_dir / f"{stem}_measured.csv",
        summary_csv=results_dir / f"{stem}_summary.csv",
        output_dir=results_dir,
    )
    # NOTE: requires encrypted/proprietary n, k file from FILMetrics. But why?