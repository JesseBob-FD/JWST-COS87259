"""Compute Ia9 conversion factors using RC-Li's method for all bands."""
import numpy as np
import os
import sys
sys.path.insert(0, '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src')
from galfits import gsutils

FILTER_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src/data/filters'
C_ANGSTROM_S = 2.99792458e18

def pivot_from_filter(band_name):
    """Compute pivot wavelength from filter file, or None if unavailable."""
    fpath = os.path.join(FILTER_DIR, band_name)
    if not os.path.exists(fpath):
        return None
    data = np.loadtxt(fpath)
    wl = data[:, 0]
    T = np.clip(data[:, 1], 0, None)
    num = np.trapezoid(T * wl, wl)
    den = np.trapezoid(T / wl, wl)
    if den > 0:
        return np.sqrt(num / den)
    return None

def get_effective_wave(band):
    """Get effective wavelength, falling back to filter-pivot or central."""
    # Try gsutils dict first
    if band in gsutils.effective_wave:
        return gsutils.effective_wave[band]
    # Try computing pivot from filter file
    pivot = pivot_from_filter(band)
    if pivot:
        return pivot
    # Fallback: parse from band name
    name = band.split('_')[-1]
    for suffix in ['W', 'M']:
        if suffix in name:
            return float(name.replace('f', '').replace(suffix, '')) * 100
    return None

# MIRI PIXAR_SR = 8.46159499407524e-14 sr (60 mas pixels)
PIXAR_SR_MIRI = 8.46159499407524e-14
ZP_MIRI = 2.5 * np.log10(3631.0 / (PIXAR_SR_MIRI * 1e6))

# NIRCam PIXAR_SR = 2.11539874851881e-14 sr (30 mas pixels)
PIXAR_SR_NIRCAM = 2.11539874851881e-14
ZP_NIRCAM = 2.5 * np.log10(3631.0 / (PIXAR_SR_NIRCAM * 1e6))

print("=" * 80)
print("Ia9 Conversion Factors (RC-Li method)")
print("=" * 80)

print(f"\nMIRI:  ZP={ZP_MIRI:.6f} AB")
print(f"{'Band':<14} {'eff_wave (A)':<16} {'Source':<20} {'Ia9':<24} {'Ia9 (compact)'}")
print("-" * 85)

miri_bands = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
              'F1500W', 'F1800W', 'F2100W', 'F2550W']

for band in miri_bands:
    jband = f'miri_{band.lower()}'
    wave = get_effective_wave(jband)
    source = 'gsutils' if jband in gsutils.effective_wave else 'filter pivot'
    ia9 = gsutils.ABmag_to_covf(ZP_MIRI, wave)
    print(f"{band:<14} {wave:<16.1f} {source:<20} {ia9:<24.8e} {ia9:.6e}")

print(f"\nNIRCam: ZP={ZP_NIRCAM:.6f} AB")
print(f"{'Band':<14} {'eff_wave (A)':<16} {'Ia9':<24} {'Ia9 (compact)'}")
print("-" * 85)

nircam_bands = ['F115W', 'F200W', 'F410M']
for band in nircam_bands:
    jband = f'nircam_{band.lower()}'
    wave = gsutils.effective_wave[jband]
    ia9 = gsutils.ABmag_to_covf(ZP_NIRCAM, wave)
    print(f"{band:<14} {wave:<16.1f} {ia9:<24.8e} {ia9:.6e}")

print("\n=== Copy-paste ready for .lyric files ===")
print("# MIRI bands:")
for band in miri_bands:
    jband = f'miri_{band.lower()}'
    wave = get_effective_wave(jband)
    ia9 = gsutils.ABmag_to_covf(ZP_MIRI, wave)
    print(f"I_9) {ia9:.6e}  # {band}")
print("# NIRCam bands:")
for band in nircam_bands:
    jband = f'nircam_{band.lower()}'
    wave = gsutils.effective_wave[jband]
    ia9 = gsutils.ABmag_to_covf(ZP_NIRCAM, wave)
    print(f"I_9) {ia9:.6e}  # {band}")
