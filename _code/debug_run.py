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

import pythonnet
pythonnet.load("netfx")
import clr
clr.AddReference('FILMeasure')
clr.AddReference('System.Windows.Forms')

import numpy as np
from matplotlib import pyplot as plt

import System
from Filmetrics import FIRemote
from System.Windows.Forms import FormWindowState
from System import String

# Instantiate FIRemote object.
# NOTE: FIRemote.ConstructorWarning.None cannot be written directly —
#       'None' is a reserved keyword in Python 3. Use getattr() instead.
theWarning = getattr(FIRemote.ConstructorWarning, 'None')
warningMessage = String.Empty

theFIRemote = FIRemote(
    FIRemote.GraphicalUserInterfaceType.Standard,
    FIRemote.GraphicalUserInterfaceStartupState.Shown,
    FormWindowState.Normal,
    theWarning,
    warningMessage
)
# Set recipe
# theFIRemote.SetRecipe("polynorbornene")
theFIRemote.SetRecipe("SiO2 16kA on Si")

baseline = False
if baseline == True:
    # Acquire baseline and reference
    # input("Place the sample and press Enter to continue...")
    theFIRemote.BaselineAcquireSpectrumFromSample()
    # time.sleep(1)
    input("Place the reference sample and press Enter to continue...")
    reflectance_standard = "Si"
    theFIRemote.BaselineSetRefMat(reflectance_standard);
    # time.sleep(1)
    theFIRemote.BaselineAcquireReference();
    # time.sleep(1)
    input("Remove the reference sample and press Enter to continue...")
    theFIRemote.BaselineAcquireBackgroundAfterRef()
    # time.sleep(1)
    theFIRemote.BaselineCommit()
# Recover saved baseline and measure
elif baseline == False:
    theFIRemote.AuthenticateRefBac()
theResult = theFIRemote.Measure(False)
import pdb; pdb.set_trace()
# Retrieve GOF, number of spectrum points, first and last points
# numpoints = theResult.PrimaryWavelengths.Length
# print('GOF = ' + str(theResult.GOF))
# print('Number of Points = ' + str(numpoints))
# print('First Point = ('
#       + str(theResult.PrimaryWavelengths[0])
#       + ', ' + str(theResult.PrimarySpectrum[0]) + ')')
# print('Last Point = ('
#       + str(theResult.PrimaryWavelengths[numpoints - 1])
#       + ', ' + str(theResult.PrimarySpectrum[numpoints - 1]) + ')')

