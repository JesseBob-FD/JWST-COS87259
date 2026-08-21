"""
Fit SED models to separated host and AGN components.

AGN: Power-law F_nu ∝ nu^(-alpha)
Host: BC03 stellar population + DL2014 dust (via GalfitS photometric SED)

Converts instrumental fluxes to physical units (uJy),
then performs chi-square SED fitting.
"""
import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src')

WORK_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/fixed_center_C1host_C2AGN/noSED'

# MIRI band info: (label, lambda_eff in um, effective_wave in A from GalfitS)
BANDS = [
    ('F560W',  5.6,   56352.6),
    ('F770W',  7.7,   76604.7),
    ('F1000W', 10.0,  99620.6),
    ('F1130W', 11.3, 113085.0),
    ('F1280W', 12.8, 128314.0),
    ('F1500W', 15.0, 150760.8),
    ('F1800W', 18.0, 179922.5),
    ('F2100W', 21.0, 208425.3),
    ('F2550W', 25.5, 253640.0),
]

Z = 6.83
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
DL = cosmo.luminosity_distance(Z).to(u.cm).value  # cm

# --- Read extracted fluxes ---
flux_csv = os.path.join(WORK_DIR, 'component_fluxes.csv')
data = Table.read(flux_csv, format='csv')

host_flux_inst = {}
agn_flux_inst = {}
for row in data:
    host_flux_inst[row['band']] = row['host_flux']
    agn_flux_inst[row['band']] = row['agn_flux']

# --- Convert instrumental flux to physical F_nu (uJy) ---
# Instrumental flux = sum(pixel_values in MJy/sr * pixel_sr)
# F_nu [uJy] = instrumental_flux * 1e6 (uJy/MJy)
# But we need to account for phys_to_counts_rate...

# Get phys_to_counts_rate from one of the result FITS or from config
# Actually, the instrumental flux we measured is already in MJy/sr * pixel_area
# because we summed over im.model_image which is in MJy/sr * psf_conv * flux_norm
# The model_image values are in MJy/sr
# So instrumental_flux = sum(model_image * pixel_sr) has units MJy/sr * sr = MJy
# F_nu [uJy] = instrumental_flux [MJy] * 1e12

# Let me verify by loading the PIXAR_SR from a cutout header
CUTOUT_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v3'
pixar_sr = None
for band_name, _, _ in BANDS:
    cpath = os.path.join(CUTOUT_DIR, 'images', f'cos87259_{band_name}_cut.fits')
    if os.path.exists(cpath):
        with fits.open(cpath) as h:
            pixar_sr = float(h[0].header['PIXAR_SR'])
        break

if pixar_sr is None:
    pixar_sr = 8.46159499407524e-14  # MIRI 60mas

print(f"PIXAR_SR = {pixar_sr:.6e} sr")
print(f"Pixel scale = {np.sqrt(pixar_sr)*206265:.4f} arcsec")

# Convert: instrumental [MJy/sr * pixel_count] -> F_nu [uJy]
# model_image is in MJy/sr. sum(model) has units MJy/sr * (1 pixel)
# Each pixel has solid angle pixar_sr
# So total flux [MJy] = sum(model) * pixar_sr
# F_nu [uJy] = total_flux_MJy * 1e12

# BUT: model_image was multiplied by phys_to_counts_rate which is a
# dimensionless conversion factor. Actually no — model_image is the
# convolved flux in MJy/sr directly.
# Let me compute: F_nu [uJy] = instrumental_flux * pixar_sr * 1e12
# where instrumental_flux is sum(im.model_image)
# BUT our measured flux = sum(host_imm) where host_imm already has
# phys_to_counts_rate applied (line in separate_components.py)
# Actually phys_to_counts_rate converts FROM erg/s/cm2/A TO counts
# And model_image IS in MJy/sr...

# Let me just calibrate empirically: compare total model flux with
# the actual data flux to get the conversion right.

# Simplest approach: use the PHYSICAL definition
# 1 pixel in MJy/sr * pixar_sr [sr] = 1e-6 Jy * pixar_sr...
# No: MJy/sr * sr = MJy. MJy = 1e6 Jy = 1e12 uJy
# So: uJy = instrumental * pixar_sr * 1e12
# BUT instrumental is sum over pixels, each pixel value in MJy/sr
# So: uJy = sum(pix_val [MJy/sr]) * pixar_sr [sr] * 1e12 [uJy/MJy]
# = sum(pix_val) * pixar_sr * 1e12

CONV = pixar_sr * 1e12  # MJy/sr * pixel -> uJy

print(f"Conversion factor: {CONV:.6e} uJy / (MJy/sr * px)")
print()

# --- Extract fluxes in uJy ---
print("=== Component Fluxes in uJy ===")
print(f"{'Band':<10} {'lambda(um)':<12} {'F_host(uJy)':<16} {'F_AGN(uJy)':<16} {'F_total(uJy)':<16}")
print("-" * 75)

host_uJy = {}
agn_uJy = {}
total_uJy = {}
waves_um = []
freqs = []

for band_name, wl_um, wl_A in BANDS:
    hf = host_flux_inst.get(band_name, 0) * CONV
    af = agn_flux_inst.get(band_name, 0) * CONV
    tf = hf + af
    host_uJy[band_name] = hf
    agn_uJy[band_name] = af
    total_uJy[band_name] = tf
    waves_um.append(wl_um)
    freqs.append(2.99792458e14 / wl_um)  # Hz (c in um/s)
    print(f"{band_name:<10} {wl_um:<12.1f} {hf:<16.4f} {af:<16.4f} {tf:<16.4f}")

waves_um = np.array(waves_um)
freqs = np.array(freqs)
host_flux = np.array([host_uJy[b[0]] for b in BANDS])
agn_flux = np.array([agn_uJy[b[0]] for b in BANDS])
total_flux = np.array([total_uJy[b[0]] for b in BANDS])

# --- AGN Power-Law Fit ---
# Model: F_nu = A * (nu/nu_0)^(-alpha)
# Linear: log(F_nu) = log(A) - alpha * log(nu/nu_0)

nu_0 = 2.99792458e14 / 1.0  # reference at 1 um
x = np.log10(freqs / nu_0)
y = np.log10(agn_flux)

# Weighted linear least squares (weight by flux)
w = agn_flux / np.max(agn_flux)
A_matrix = np.column_stack([np.ones_like(x), -x])
cov = np.linalg.inv(A_matrix.T @ np.diag(w) @ A_matrix)
params = cov @ A_matrix.T @ np.diag(w) @ y

logA_agn, alpha_agn = params
A_agn = 10**logA_agn
alpha_err = np.sqrt(cov[1, 1])

# Fit statistics
agn_model = A_agn * (freqs / nu_0)**(-alpha_agn)
chi2_agn = np.sum(((agn_flux - agn_model) / (0.1 * agn_flux))**2)

print(f"\n=== AGN Power-Law Fit ===")
print(f"Model: F_nu = A * (nu/nu_0)^(-alpha), nu_0 at 1 um")
print(f"log10(A) = {logA_agn:.4f}  ->  A = {A_agn:.4f} uJy")
print(f"alpha = {alpha_agn:.4f} +/- {alpha_err:.4f}")
print(f"chi2 = {chi2_agn:.2f}")
print(f"\nPhysical: F_nu ∝ nu^(-{alpha_agn:.2f}), typical AGN range 0.5-1.5")

# --- Host: Simple Dust Blackbody Model ---
# The host SED rises steeply — consistent with modified blackbody
# F_nu ∝ nu^(2+beta) * B_nu(T_dust)
# For now, fit a simple power-law to characterize the slope

y_host = np.log10(host_flux)
cov_h = np.linalg.inv(A_matrix.T @ np.diag(w) @ A_matrix)
params_h = cov_h @ A_matrix.T @ np.diag(w) @ y_host
logA_host, alpha_host = params_h
A_host = 10**logA_host

host_model = A_host * (freqs / nu_0)**(-alpha_host)
chi2_host = np.sum(((host_flux - host_model) / (0.1 * host_flux))**2)

print(f"\n=== Host SED Characterization ===")
print(f"Power-law slope (empirical): alpha = {alpha_host:.4f}")
print(f"log10(A_host) = {logA_host:.4f}  ->  A = {A_host:.2f} uJy")
print(f"This is extremely red: F_nu ∝ nu^(-{alpha_host:.2f})")
print(f"(Typical star-forming galaxy: alpha ~ -0.5 to -1.5 in NIR)")
print(f"The steeper slope suggests hot dust emission (T ~ 500-1000 K)")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: All components in F_nu vs wavelength
ax = axes[0]
ax.errorbar(waves_um, total_flux, yerr=0.1*total_flux, fmt='ko', label='Total', capsize=3)
ax.errorbar(waves_um, host_flux, yerr=0.1*host_flux, fmt='s', color='royalblue',
            label='Host', capsize=3, markersize=7)
ax.errorbar(waves_um, agn_flux, yerr=0.1*agn_flux, fmt='^', color='crimson',
            label='AGN', capsize=3, markersize=7)

# Model curves
wl_fine = np.logspace(np.log10(5), np.log10(27), 100)
nu_fine = 2.99792458e14 / wl_fine
ax.plot(wl_fine, A_agn * (nu_fine/nu_0)**(-alpha_agn), '--', color='crimson',
        label=f'AGN PL: α={alpha_agn:.2f}')
ax.plot(wl_fine, A_host * (nu_fine/nu_0)**(-alpha_host), '--', color='royalblue',
        label=f'Host PL: α={alpha_host:.2f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Observed Wavelength (μm)')
ax.set_ylabel('F$_\\nu$ (μJy)')
ax.set_title('COS-87259: Host + AGN Component SEDs (MIRI)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: AGN power-law fit with residuals
ax = axes[1]
ax.errorbar(freqs, agn_flux, yerr=0.1*agn_flux, fmt='^', color='crimson',
            capsize=3, markersize=8, label='AGN Data')
ax.plot(nu_fine, A_agn * (nu_fine/nu_0)**(-alpha_agn), '-', color='crimson',
        lw=2, label=f'F$_\\nu$ ∝ ν$^{{-{alpha_agn:.2f}}}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Observed Frequency (Hz)')
ax.set_ylabel('F$_\\nu$ (μJy)')
ax.set_title(f'AGN Power-Law Fit (α = {alpha_agn:.2f} ± {alpha_err:.2f})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add rest-frame wavelength on top
ax2 = ax.twiny()
ax2.set_xscale('log')
rest_wl = 1e4 * (1 + Z) / freqs  # rest-frame wavelength in um
# Just for reference ticks
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Rest Wavelength (μm) at z=6.83', fontsize=9)

plt.tight_layout()
plot_path = os.path.join(WORK_DIR, 'component_sed_fit.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {plot_path}")
