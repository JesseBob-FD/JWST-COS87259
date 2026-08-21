"""Fit SED to separated components. Usage: python fit_component_sed.py [work_dir]"""
import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

BANDS = [
    ('F560W',  5.6,   56352.6), ('F770W',  7.7,   76604.7),
    ('F1000W', 10.0,  99620.6), ('F1130W', 11.3, 113085.0),
    ('F1280W', 12.8, 128314.0), ('F1500W', 15.0, 150760.8),
    ('F1800W', 18.0, 179922.5), ('F2100W', 21.0, 208425.3),
    ('F2550W', 25.5, 253640.0),
]
Z = 6.83
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
DL = cosmo.luminosity_distance(Z).to(u.cm).value

# Read fluxes
data = Table.read(os.path.join(WORK_DIR, 'component_fluxes.csv'), format='csv')
host_inst = {r['band']: r['host_flux'] for r in data}
agn_inst  = {r['band']: r['agn_flux']  for r in data}

# Get PIXAR_SR
CUT = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v3'
pixar = None
for b,_,_ in BANDS:
    cp = os.path.join(CUT, 'images', f'cos87259_{b}_cut.fits')
    if os.path.exists(cp):
        with fits.open(cp) as h: pixar = float(h[0].header['PIXAR_SR']); break
pixar = pixar or 8.46159499407524e-14
CONV = pixar * 1e12  # MJy/sr*px -> uJy

# Extract arrays
waves_um = np.array([b[1] for b in BANDS])
freqs = 2.99792458e14 / waves_um
host_uJy = np.array([host_inst.get(b[0],0)*CONV for b in BANDS])
agn_uJy  = np.array([agn_inst.get(b[0],0)*CONV  for b in BANDS])
total    = host_uJy + agn_uJy

# --- AGN Power-Law ---
nu_0 = 2.99792458e14 / 1.0
x = np.log10(freqs / nu_0)
y_agn = np.log10(agn_uJy)
w = agn_uJy / np.max(agn_uJy)
Amat = np.column_stack([np.ones_like(x), -x])
cov_a = np.linalg.inv(Amat.T @ np.diag(w) @ Amat)
p_a = cov_a @ Amat.T @ np.diag(w) @ y_agn
A_agn, alpha_agn = 10**p_a[0], p_a[1]
agn_model = A_agn * (freqs/nu_0)**(-alpha_agn)

# --- Host Power-Law ---
y_host = np.log10(host_uJy)
cov_h = np.linalg.inv(Amat.T @ np.diag(w) @ Amat)
p_h = cov_h @ Amat.T @ np.diag(w) @ y_host
A_host, alpha_host = 10**p_h[0], p_h[1]
host_model = A_host * (freqs/nu_0)**(-alpha_host)

# Print
print(f"=== {os.path.basename(WORK_DIR)} ===")
print(f"AGN: A={A_agn:.4f} uJy, alpha={alpha_agn:.4f}+/-{np.sqrt(cov_a[1,1]):.4f}")
print(f"Host: A={A_host:.4f} uJy, alpha={alpha_host:.4f}")

print(f"\n{'Band':<10} {'F_host(uJy)':<14} {'F_AGN(uJy)':<14} {'F_tot(uJy)':<14} {'Host%':<8}")
for i,b in enumerate(BANDS):
    print(f"{b[0]:<10} {host_uJy[i]:<14.4f} {agn_uJy[i]:<14.4f} {total[i]:<14.4f} {100*host_uJy[i]/total[i]:<8.1f}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.errorbar(waves_um, total, yerr=0.1*total, fmt='ko', label='Total', capsize=3)
ax.errorbar(waves_um, host_uJy, yerr=0.1*host_uJy, fmt='s', color='royalblue', label='Host (C2)', capsize=3, markersize=7)
ax.errorbar(waves_um, agn_uJy, yerr=0.1*agn_uJy, fmt='^', color='crimson', label='AGN (C1)', capsize=3, markersize=7)
wl_f = np.logspace(np.log10(5), np.log10(27), 100)
nu_f = 2.99792458e14 / wl_f
ax.plot(wl_f, A_agn*(nu_f/nu_0)**(-alpha_agn), '--', color='crimson', label=f'AGN PL: α={alpha_agn:.2f}')
ax.plot(wl_f, A_host*(nu_f/nu_0)**(-alpha_host), '--', color='royalblue', label=f'Host PL: α={alpha_host:.2f}')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Observed Wavelength (μm)'); ax.set_ylabel('F$_\\nu$ (μJy)')
ax.set_title('COS-87259: C2=Host, C1=AGN — Component SEDs')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.errorbar(freqs, agn_uJy, yerr=0.1*agn_uJy, fmt='^', color='crimson', capsize=3, markersize=8)
ax.plot(nu_f, A_agn*(nu_f/nu_0)**(-alpha_agn), '-', color='crimson', lw=2,
        label=f'F$_\\nu$ ∝ ν$^{{-{alpha_agn:.2f}}}$')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Observed Frequency (Hz)'); ax.set_ylabel('F$_\\nu$ (μJy)')
ax.set_title(f'C2host_C1AGN — AGN Power-Law Fit (α={alpha_agn:.2f})')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(WORK_DIR, 'component_sed_fit.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot: {out}")
