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
import win32gui
import win32api
import win32con
import pywinauto
from pywinauto.application import Application
import pythonnet
pythonnet.load("netfx")
import clr
clr.AddReference('FILMeasure')
clr.AddReference('System.Windows.Forms')

import System
from Filmetrics import FIRemote
from System.Windows.Forms import FormWindowState
from System import String

def _win32_click_button(window_title: str, button_text: str, timeout: float = 5.0):
    """Find a visible top-level window and click a named child button via raw Win32."""
    deadline = time.time() + timeout
    popup_hwnd = None
    while popup_hwnd is None:
        found = []
        def _enum_windows(hwnd, _):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == window_title:
                found.append(hwnd)
        win32gui.EnumWindows(_enum_windows, None)
        if found:
            popup_hwnd = found[0]
            break
        if time.time() > deadline:
            raise RuntimeError(f"Timed out waiting for window '{window_title}'")
        time.sleep(0.1)

    buttons = []
    def _enum_children(hwnd, _):
        if win32gui.GetWindowText(hwnd) == button_text:
            buttons.append(hwnd)
    win32gui.EnumChildWindows(popup_hwnd, _enum_children, None)
    if not buttons:
        raise RuntimeError(f"Button '{button_text}' not found in '{window_title}'")

    win32gui.SetForegroundWindow(popup_hwnd)
    time.sleep(0.1)
    rect = win32gui.GetWindowRect(buttons[0])
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    win32api.SetCursorPos((cx, cy))
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def export_nk_via_gui(save_path: str):
    """Triggers File > Save Measured n and k... in FILMeasure and saves to save_path."""
    from pywinauto import findwindows
    elements = findwindows.find_elements(title="FILMeasure", backend="uia")
    # Pick the main window (auto_id="MainForm")
    main = next(e for e in elements if e.automation_id == "MainForm")
    pid = main.process_id

    app_uia = Application(backend="uia").connect(process=pid)
    win = app_uia.window(auto_id="MainForm")
    win.set_focus()
    win.menu_select("File->Save Measured n and k...")
    time.sleep(1)  # wait for the owned dialog to appear

    # Click OK on the WinForms popup via raw Win32 — bypasses pywinauto backend issues.
    _win32_click_button("Save N and K to File", "OK")
    time.sleep(0.5)

    # File save dialog — switch to win32 backend which handles common dialogs well.
    app_win32 = Application(backend="win32").connect(process=pid)
    save_dlg = app_win32.window(title="Save Material File")
    save_dlg.wait("ready", timeout=5)
    save_dlg.set_focus()
    time.sleep(0.2)
    fn_edit = save_dlg.child_window(class_name="Edit", found_index=0)
    fn_edit.set_edit_text(save_path)
    time.sleep(0.2)
    save_dlg.child_window(title="Save", class_name="Button").click_input()


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
    # input("Place the sample and press Enter to continue...")
    theResult = theFIRemote.Measure(False)
    print(f"{theResult.get_ResultsSummary()=}")
    wavelengths = np.array(theResult.get_PrimaryWavelengths())
    reflectances = np.array(theResult.get_PrimarySpectrum())
    thickness = theResult.get_LayerThicknesses()[1]
    n_value = theResult.get_LayerNValues()[1]
    k_value = theResult.get_LayerKValues()[1]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    stem = f"{id}_{timestamp}"

    # Save spectrum CSVs (measured and calculated may have different wavelength grids)
    np.savetxt(results_dir / f"{stem}_measured.csv", np.column_stack([wavelengths, reflectances]),
               delimiter=",", header="wavelength_nm,reflectance", comments="")
    calc_reflectances = np.array(theResult.get_PrimaryCalcSpectrum())
    calc_wavelengths = np.array(theResult.get_PrimaryCalcWavelengths())
    np.savetxt(results_dir / f"{stem}_calculated.csv", np.column_stack([calc_wavelengths, calc_reflectances]),
               delimiter=",", header="wavelength_nm,calc_reflectance", comments="")

    gof = theResult.GOF

    # Save summary CSV
    with open(results_dir / f"{stem}_summary.csv", "w") as f:
        f.write("id,timestamp,thickness_nm,GOF,n_value,k_value\n")
        f.write(f"{id},{timestamp},{thickness},{gof},{n_value},{k_value}\n")

    # Plot and save
    fig, ax = plt.subplots()
    ax.plot(wavelengths, reflectances, color="blue", label="Measured")
    ax.plot(calc_wavelengths, calc_reflectances, color="red", label="Calculated")
    ax.legend(loc="upper right")
    ax.text(0.98, 0.60, f"Thickness: {thickness:.1f} nm\nGOF: {gof:.4f}\nn(632.8 nm): {n_value:.4f}\nk(632.8 nm): {k_value:.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"{id} — {timestamp}")
    fig.savefig(results_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # export_nk_via_gui(str(results_dir / f"{stem}_nk.txt")) #.fitnk

    return stem, results_dir