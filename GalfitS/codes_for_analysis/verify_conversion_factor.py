"""
Compare two methods for computing the Ia9 conversion factor for MIRI bands.

Method 1 (original): lambda_pivot from filter transmission curves, hardcoded PIXEL_SR
Method 2 (RC-Li):    PIXAR_SR from FITS header, then ABmag_to_covf() from GalfitS
"""
import numpy as np
import os
import sys
from astropy.io import fits

# Add galfits to path
sys.path.insert(0, '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src')
from galfits import gsutils

# --- Configuration ---
FILTER_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src/data/filters'
CUTOUT_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v2/images'
PIXEL_SR = 8.46159499407524e-14  # sr for 60mas pixels
C_ANGSTROM_S = 2.99792458e18      # speed of light in Angstrom/s

BAND_CENTERS = {
    'F560W': 56000, 'F770W': 77000, 'F1000W': 100000,
    'F1130W': 113000, 'F1280W': 128000, 'F1500W': 150000,
    'F1800W': 180000, 'F2100W': 210000, 'F2550W': 255000
}

ALL_BANDS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
             'F1500W', 'F1800W', 'F2100W', 'F2550W']

JWST_BAND_NAMES = ['miri_f770w', 'miri_f1000w', 'miri_f1280w',
                   'miri_f1500w', 'miri_f1800w', 'miri_f2100w']

print("=" * 95)
print("Ia9 Conversion Factor Verification for MIRI Bands")
print("=" * 95)

# =========================================================================
# Method 1: Original approach (lambda_pivot from filter, hardcoded PIXEL_SR)
# =========================================================================
print("\n" + "=" * 95)
print("METHOD 1 — Original: lambda_pivot from filter × hardcoded PIXEL_SR")
print("  factor = (lambda_pivot² / c) / PIXEL_SR × 1e17")
print("=" * 95)
print(f"{'Band':<10} {'Filter File':<6} {'λ_pivot (Å)':<16} {'Ia9':<24}")
print("-" * 65)

results_m1 = {}
for band in ALL_BANDS:
    fname = f'miri_{band.lower()}'
    fpath = os.path.join(FILTER_DIR, fname)

    if os.path.exists(fpath):
        data = np.loadtxt(fpath)
        wl = data[:, 0]
        T = data[:, 1]
        # Ensure non-negative throughput for integration
        T = np.clip(T, 0, None)
        num = np.trapezoid(T * wl, wl)
        den = np.trapezoid(T / wl, wl)
        if den > 0:
            lambda_pivot = np.sqrt(num / den)
        else:
            lambda_pivot = BAND_CENTERS[band]
        has_filter = "YES"
    else:
        lambda_pivot = BAND_CENTERS[band]
        has_filter = "NO"

    factor = (lambda_pivot**2 / C_ANGSTROM_S) / PIXEL_SR * 1e17
    results_m1[band] = factor
    note = " ⚠ NO FILTER FILE" if not os.path.exists(fpath) else ""
    print(f"{band:<10} {has_filter:<6} {lambda_pivot:<16.1f} {factor:<24.8e}{note}")

# =========================================================================
# Method 2: RC-Li approach (PIXAR_SR from header + ABmag_to_covf)
# =========================================================================
print("\n" + "=" * 95)
print("METHOD 2 — RC-Li: PIXAR_SR from header + ABmag_to_covf()")
print("  ZP = 2.5×log10(3631 / (PIXAR_SR × 1e6))")
print("  Ia9 = ABmag_to_covf(ZP, effective_wave[band])")
print("=" * 95)

# Get PIXAR_SR from actual FITS header
pixar_sr_from_header = None
for band in ALL_BANDS:
    fpath = os.path.join(CUTOUT_DIR, f'cos87259_{band}_cut.fits')
    if os.path.exists(fpath):
        with fits.open(fpath) as hdu:
            pixar_sr_from_header = hdu[0].header.get('PIXAR_SR', None)
        if pixar_sr_from_header is not None:
            print(f"  Read PIXAR_SR from {band} cutout header: {pixar_sr_from_header:.14e} sr")
            break

if pixar_sr_from_header is None:
    print("  WARNING: Could not read PIXAR_SR from any FITS header!")
    pixar_sr_from_header = PIXEL_SR

print(f"  Hardcoded  PIXEL_SR:                     {PIXEL_SR:.14e} sr")
print(f"  Difference:                               {abs(pixar_sr_from_header - PIXEL_SR):.2e} sr")
print()

ZP = 2.5 * np.log10(3631.0 / (pixar_sr_from_header * 1e6))
print(f"  ZP_GALFIT = 2.5×log10(3631/({pixar_sr_from_header:.6e}×1e6)) = {ZP:.6f} AB mag")
print()
print(f"{'Band':<10} {'eff_wave (Å)':<16} {'ZP (AB)':<12} {'Ia9':<24}")
print("-" * 70)

results_m2 = {}
for jband in JWST_BAND_NAMES:
    wave = gsutils.effective_wave[jband]
    ia9 = gsutils.ABmag_to_covf(ZP, wave)
    band_short = jband.replace('miri_', '').upper()
    results_m2[band_short] = ia9
    print(f"{band_short:<10} {wave:<16.1f} {ZP:<12.6f} {ia9:<24.8e}")

# =========================================================================
# Comparison
# =========================================================================
print("\n" + "=" * 95)
print("COMPARISON: Method 1 vs Method 2 vs GalfitS phys_to_image")
print("=" * 95)
print(f"{'Band':<10} {'M1 (pivot)':<22} {'M2 (RC-Li)':<22} {'M2/M1':<12} {'phys_to_image':<24} {'M1/phys':<12} {'M2/phys':<12}")
print("-" * 115)

for band in ALL_BANDS:
    v1 = results_m1.get(band)
    v2 = results_m2.get(band)
    jband = f'miri_{band.lower()}'
    gs_val = gsutils.phys_to_image.get(jband, None)

    v1s = f"{v1:.8e}" if v1 else "N/A"
    v2s = f"{v2:.8e}" if v2 else "N/A       "
    r21 = f"{v2/v1:.6f}" if (v1 and v2) else "N/A"
    gss = f"{gs_val:.8e}" if gs_val else "— (not in dict)"
    r1g = f"{v1/gs_val:.6f}" if (v1 and gs_val) else "—"
    r2g = f"{v2/gs_val:.6f}" if (v2 and gs_val) else "—"

    print(f"{band:<10} {v1s:<22} {v2s:<22} {r21:<12} {gss:<24} {r1g:<12} {r2g:<12}")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 95)
print("SUMMARY")
print("=" * 95)

# Check F770W specifically
print(f"\nF770W check against GalfitS built-in:")
print(f"  phys_to_image['miri_f770w'] = {gsutils.phys_to_image['miri_f770w']:.8e}")
print(f"  Method 1 (pivot)             = {results_m1['F770W']:.8e}")
if 'F770W' in results_m2:
    print(f"  Method 2 (RC-Li)             = {results_m2['F770W']:.8e}")
    print(f"  M2 / GalfitS                 = {results_m2['F770W']/gsutils.phys_to_image['miri_f770w']:.8f}")
print(f"  M1 / GalfitS                 = {results_m1['F770W']/gsutils.phys_to_image['miri_f770w']:.8f}")

# For bands with filter files
print(f"\nBands with filter files (M2 available):")
for band in ['F770W', 'F1000W', 'F1280W', 'F1500W', 'F1800W', 'F2100W']:
    v1 = results_m1[band]
    v2 = results_m2.get(band)
    if v2:
        print(f"  {band}: M1={v1:.6e}  M2={v2:.6e}  ratio={v2/v1:.8f}")

print(f"\nBands without filter files (M1 uses central λ, M2 unavailable):")
for band in ['F560W', 'F1130W', 'F2550W']:
    v1 = results_m1[band]
    print(f"  {band}: M1={v1:.6e} (central wavelength)")
