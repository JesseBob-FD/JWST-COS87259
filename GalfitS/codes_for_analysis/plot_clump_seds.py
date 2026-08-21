"""Plot MIRI SEDs for all 5 clumps: Center, West (detected) + North/South/SE (upper limits)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/MIRI_clump_departure'

# MIRI bands
bands = ['F560W','F770W','F1000W','F1130W','F1280W','F1500W','F1800W','F2100W','F2550W']
waves_um = np.array([5.6, 7.7, 10.0, 11.3, 12.8, 15.0, 18.0, 21.0, 25.5])

# Detected fluxes (uJy) from two-Sersic C1C2 separation
center_uJy = np.array([1.389, 2.013, 3.566, 3.469, 4.844, 10.173, 19.319, 34.612, 90.366])
west_uJy   = np.array([1.389, 1.828, 3.122, 3.468, 4.295,  6.959, 14.412, 32.035, 90.355])

# 5-sigma upper limits (uJy)
north_ul = np.array([0.028, 0.039, 0.072, 0.086, 0.137, 0.243, 0.500, 0.545, 0.895])
south_ul = np.array([0.017, 0.015, 0.033, 0.043, 0.074, 0.060, 0.164, 0.211, 0.608])
se_ul    = np.array([0.015, 0.017, 0.036, 0.055, 0.059, 0.087, 0.158, 0.197, 0.615])

fig, ax = plt.subplots(figsize=(10, 7))

# Detected: points with error bars
ax.errorbar(waves_um, center_uJy, yerr=0.1*center_uJy, fmt='o', color='royalblue',
            markersize=9, capsize=3, label='Center (C1)', markeredgewidth=1)
ax.errorbar(waves_um, west_uJy, yerr=0.1*west_uJy, fmt='s', color='crimson',
            markersize=9, capsize=3, label='West (C2)', markeredgewidth=1)

# Upper limits: downward arrows
for wl, ul, c, lbl in [(waves_um, north_ul, 'darkorange', 'North (<5σ)'),
                         (waves_um, south_ul, 'green', 'South (<5σ)'),
                         (waves_um, se_ul, 'purple', 'Southeast (<5σ)')]:
    ax.scatter(wl, ul, marker='v', color=c, s=50, alpha=0.7)
    ax.plot(wl, ul, ':', color=c, alpha=0.4, lw=0.8)

# Labels for limit clumps (only once in legend)
for wl, ul, c, lbl in [(waves_um, north_ul, 'darkorange', 'North (<5σ)'),
                         (waves_um, south_ul, 'green', 'South (<5σ)'),
                         (waves_um, se_ul, 'purple', 'Southeast (<5σ)')]:
    ax.plot([], [], 'v', color=c, markersize=7, label=lbl)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
ax.set_ylabel('F$_\\nu$ (μJy)', fontsize=12)
ax.set_title('COS-87259 (z=6.83): MIRI Clump SEDs', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(4.5, 28)

# Rest-frame axis on top
ax2 = ax.twiny()
ax2.set_xscale('log')
rest_wl = waves_um / (1 + 6.83)
tick_um = np.array([0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
ax2.set_xticks(tick_um * (1 + 6.83))
ax2.set_xticklabels([f'{t:.1f}' for t in tick_um])
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Rest-Frame Wavelength (μm)', fontsize=10)

plt.tight_layout()
out = f'{OUT_DIR}/clump_seds.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')

# Also save a version in νLν
fig2, ax = plt.subplots(figsize=(10, 7))
c_kms = 2.99792458e5  # km/s
nu = c_kms / (waves_um * 1e-4)  # Hz (waves in um -> cm)
nuLnu_center = nu * center_uJy * 1e-6 * 1e-23 * nu  # uJy -> erg/s
# Actually for νLν in L⊙: νLν = 4π D_L² ν Fν
# Simpler: just plot νFν in relative units
nuFnu_c = waves_um * center_uJy  # λ·Fλ ∝ ν·Fν
nuFnu_w = waves_um * west_uJy

ax.errorbar(waves_um, nuFnu_c, yerr=0.1*nuFnu_c, fmt='o', color='royalblue',
            markersize=9, capsize=3, label='Center (C1)')
ax.errorbar(waves_um, nuFnu_w, yerr=0.1*nuFnu_w, fmt='s', color='crimson',
            markersize=9, capsize=3, label='West (C2)')
for wl, ul, c, lbl in [(waves_um, waves_um*north_ul, 'darkorange', 'North (<5σ)'),
                         (waves_um, waves_um*south_ul, 'green', 'South (<5σ)'),
                         (waves_um, waves_um*se_ul, 'purple', 'Southeast (<5σ)')]:
    ax.scatter(wl, ul, marker='v', color=c, s=50, alpha=0.7)
    ax.plot([], [], 'v', color=c, markersize=7, label=lbl)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
ax.set_ylabel('λ F$_\\lambda$ (arbitrary units)', fontsize=12)
ax.set_title('COS-87259 (z=6.83): MIRI Clump νF$_\\nu$ SEDs', fontsize=14)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(4.5, 28)

ax2 = ax.twiny()
ax2.set_xscale('log')
ax2.set_xticks(tick_um * (1 + 6.83))
ax2.set_xticklabels([f'{t:.1f}' for t in tick_um])
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Rest-Frame Wavelength (μm)', fontsize=10)

plt.tight_layout()
out2 = f'{OUT_DIR}/clump_seds_nuFnu.png'
plt.savefig(out2, dpi=200, bbox_inches='tight')
print(f'Saved: {out2}')
