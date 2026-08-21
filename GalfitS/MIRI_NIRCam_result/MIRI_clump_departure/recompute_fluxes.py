"""Recompute fluxes from correct per-component model images and replot SEDs."""
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/two_sersic_C1C2/noSED'
OUT  = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/MIRI_clump_departure'
PIXAR_SR = 8.46159499407524e-14
CONV_uJy = PIXAR_SR * 1e12  # MJy/sr * px -> uJy

BANDS = ['F560W','F770W','F1000W','F1130W','F1280W',
         'F1500W','F1800W','F2100W','F2550W']
waves_um = np.array([5.6, 7.7, 10.0, 11.3, 12.8, 15.0, 18.0, 21.0, 25.5])

# Read per-component and total fluxes
c1_uJy, c2_uJy, total_uJy = [], [], []
for band in BANDS:
    c1 = fits.getdata(f'{WORK}/C1_model_{band}_correct.fits')
    c2 = fits.getdata(f'{WORK}/C2_model_{band}_correct.fits')
    with fits.open(f'{WORK}/C1C2_noSED_miri_{band.lower()}_result.fits') as hdu:
        total = hdu[3].data  # ext3 = model
        mask = hdu[1].data   # ext1 = mask

    good = (mask == 0) & np.isfinite(c1) & np.isfinite(c2)
    f1 = np.sum(c1[good]) * CONV_uJy
    f2 = np.sum(c2[good]) * CONV_uJy
    ft = np.sum(total[good]) * CONV_uJy
    c1_uJy.append(f1); c2_uJy.append(f2); total_uJy.append(ft)

c1_uJy = np.array(c1_uJy); c2_uJy = np.array(c2_uJy); total_uJy = np.array(total_uJy)

# Print
print(f"{'Band':<10} {'C1 (uJy)':<14} {'C2 (uJy)':<14} {'Total':<14} {'C1/C2':<10} {'C1%':<8}")
for i, b in enumerate(BANDS):
    print(f"{b:<10} {c1_uJy[i]:<14.4f} {c2_uJy[i]:<14.4f} {total_uJy[i]:<14.4f} "
          f"{c1_uJy[i]/c2_uJy[i]:<10.3f} {100*c1_uJy[i]/total_uJy[i]:<8.1f}")

# Save CSV
with open(f'{OUT}/clump_fluxes_corrected.csv', 'w') as f:
    f.write('band,C1_uJy,C2_uJy,total_uJy,C1_frac\n')
    for i, b in enumerate(BANDS):
        f.write(f'{b},{c1_uJy[i]:.4f},{c2_uJy[i]:.4f},{total_uJy[i]:.4f},{c1_uJy[i]/total_uJy[i]:.4f}\n')

# --- Plot: F_nu ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.errorbar(waves_um, total_uJy, yerr=0.1*total_uJy, fmt='ko', label='Total', capsize=3)
ax.errorbar(waves_um, c1_uJy, yerr=0.1*c1_uJy, fmt='o', color='royalblue',
            markersize=9, capsize=3, label='Center (C1=bulge)')
ax.errorbar(waves_um, c2_uJy, yerr=0.1*c2_uJy, fmt='s', color='crimson',
            markersize=9, capsize=3, label='West (C2=disk)')

# Power-law fits
nu_0 = 2.99792458e14 / 1.0
freqs = 2.99792458e14 / waves_um
w = c1_uJy / np.max(c1_uJy)
x = np.log10(freqs / nu_0)

# C1 power-law
Amat = np.column_stack([np.ones_like(x), -x])
p1 = np.linalg.inv(Amat.T @ np.diag(w) @ Amat) @ Amat.T @ np.diag(w) @ np.log10(c1_uJy)
A1, a1 = 10**p1[0], p1[1]

# C2 power-law
p2 = np.linalg.inv(Amat.T @ np.diag(w) @ Amat) @ Amat.T @ np.diag(w) @ np.log10(c2_uJy)
A2, a2 = 10**p2[0], p2[1]

wl_f = np.logspace(np.log10(5), np.log10(27), 100)
nu_f = 2.99792458e14 / wl_f
ax.plot(wl_f, A1*(nu_f/nu_0)**(-a1), '--', color='royalblue', lw=1.5,
        label=f'C1: F$_\\nu$∝ν$^{{-{a1:.2f}}}$')
ax.plot(wl_f, A2*(nu_f/nu_0)**(-a2), '--', color='crimson', lw=1.5,
        label=f'C2: F$_\\nu$∝ν$^{{-{a2:.2f}}}$')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
ax.set_ylabel('F$_\\nu$ (μJy)', fontsize=12)
ax.set_title('COS-87259 (z=6.83): MIRI Component SEDs (corrected)', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(4.5, 28)

# Rest-frame axis
ax2 = ax.twiny()
ax2.set_xscale('log')
tick_um = np.array([0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
ax2.set_xticks(tick_um * (1+6.83))
ax2.set_xticklabels([f'{t:.1f}' for t in tick_um])
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Rest-Frame Wavelength (μm)', fontsize=10)

plt.tight_layout()
out = f'{OUT}/clump_seds_corrected.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'\nSaved: {out}')
