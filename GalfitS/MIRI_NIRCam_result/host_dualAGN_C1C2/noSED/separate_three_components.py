"""
Separate 3 components from host_dualAGN_C1C2_noSED fit:
  Host Sérsic, AGN_C1 (Center), AGN_C2 (West)

CORRECTED METHOD v2:
  Temporarily swap fitter.psfmodel_list to isolate components,
  then call cal_model_image() which handles all PSF/size matching internally.

  - All AGNs active → total model
  - psfmodel_list = [AGN_C2] → Host + AGN_C2 model; subtract from total → AGN_C1
  - psfmodel_list = [AGN_C1] → Host + AGN_C1 model; subtract from total → AGN_C2
  - psfmodel_list = [] → Host-only model
  - Verify: host + agn_c1 + agn_c2 ≈ total

Outputs: FITS, CSV, PNG (component images + SED)
"""
import os, sys, numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

GALFITS_SRC = '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src'
sys.path.insert(0, GALFITS_SRC)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.numpy as jnp
import galfits.gsutils as gsutils
_old = jnp.array
def _safe(obj, *a, **kw):
    if isinstance(obj, np.ndarray) and obj.dtype.byteorder in ('>', 'B'):
        obj = obj.byteswap().view(obj.dtype.newbyteorder('='))
    return _old(obj, *a, **kw)
jnp.array = _safe

WORK = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(WORK, 'host_dualAGN_C1C2_noSED.lyric')
GSFILE  = os.path.join(WORK, 'host_dualAGN_C1C2_noSED.gssummary')

ALL_MIRI = ['miri_f560w','miri_f770w','miri_f1000w','miri_f1130w',
            'miri_f1280w','miri_f1500w','miri_f1800w','miri_f2100w','miri_f2550w']
ALL_NIRCam = ['nircam_f115w','nircam_f200w','nircam_f410m']
ALL_BANDS = ALL_MIRI + ALL_NIRCam
BAND_LABELS_MIRI = ['F560W','F770W','F1000W','F1130W','F1280W','F1500W',
                    'F1800W','F2100W','F2550W']
BAND_LABELS_NIRCam = ['F115W','F200W','F410M']
BAND_LABELS = BAND_LABELS_MIRI + BAND_LABELS_NIRCam
WAVES_UM = np.array([5.6,7.7,10.0,11.3,12.8,15.0,18.0,21.0,25.5,
                     1.15,2.00,4.10])
PIXAR_SR = 8.46159499407524e-14
CONV_uJy = PIXAR_SR * 1e12

print("Loading fit...")
fitter, targ, fs = gsutils.read_config_file(CONFIG, WORK)
smfile = Table.read(GSFILE, format='ascii')
for row in smfile:
    pname = row['pname']
    if pname in fitter.lmParameters:
        fitter.lmParameters[pname].set(value=row['best_value'],
                                        min=row['best_value']-0.1,
                                        max=row['best_value']+0.1)
fitter.loose_fix_pars()
for key in fitter.pardict:
    if key in fitter.lmParameters:
        fitter.pardict[key] = fitter.lmParameters[key].value

# Identify models
gal = fitter.gmodel_list[0]
print(f"Galaxy: {gal.name}, subcomponents: {gal.subnames}")

agn_models = []
for i, m in enumerate(fitter.model_list):
    if fitter.mtype_list[i] == 'agn':
        agn_models.append(m)
        print(f"AGN: {m.name}")

# Save original psfmodel_list
original_psf_models = list(fitter.psfmodel_list)
# Find which indices correspond to AGN models
agn_indices = []
for idx, m in enumerate(original_psf_models):
    if hasattr(m, 'sedModel') and m.sedModel == 'NP':
        agn_indices.append(idx)
        print(f"  psfmodel_list[{idx}]: {m.name} (sedModel={m.sedModel})")

agn_C1 = agn_models[0]
agn_C2 = agn_models[1]
nimage = fitter.GSdata.Nimages

# ============================================================
# Step 1: Total model
# ============================================================
print("\n=== Step 1: Total model ===")
fitter.psfmodel_list = list(original_psf_models)  # restore
fitter.cal_model_image()
totals, data_imgs, sigma_imgs, mask_imgs = [], [], [], []
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
    data_imgs.append(np.array(im.cut_image) - sky_median)
    totals.append(np.array(im.model_image))
    sigma_imgs.append(np.array(im.cut_sigma_image))
    mask_imgs.append(np.array(im.cut_mask_image, dtype=float))

# ============================================================
# Step 2: Host-only (no AGNs)
# ============================================================
print("=== Step 2: Host-only (psfmodel_list = []) ===")
fitter.psfmodel_list = []
fitter.cal_model_image()
hosts = [np.array(fitter.GSdata.get_image(i).model_image) for i in range(nimage)]

# ============================================================
# Step 3: Host + AGN_C1 only (then subtract host → AGN_C1)
# ============================================================
print("=== Step 3: Host + AGN_C1 ===")
fitter.psfmodel_list = [agn_C1]
fitter.cal_model_image()
host_agn1 = [np.array(fitter.GSdata.get_image(i).model_image) for i in range(nimage)]
agn1_imgs = [h1 - hosts[i] for i, h1 in enumerate(host_agn1)]

# ============================================================
# Step 4: Host + AGN_C2 only (then subtract host → AGN_C2)
# ============================================================
print("=== Step 4: Host + AGN_C2 ===")
fitter.psfmodel_list = [agn_C2]
fitter.cal_model_image()
host_agn2 = [np.array(fitter.GSdata.get_image(i).model_image) for i in range(nimage)]
agn2_imgs = [h2 - hosts[i] for i, h2 in enumerate(host_agn2)]

# Restore
fitter.psfmodel_list = list(original_psf_models)

# ============================================================
# Verify
# ============================================================
print("\nVerification (host + AGN_C1 + AGN_C2 vs total):")
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    bn = im.band
    recon = hosts[i] + agn1_imgs[i] + agn2_imgs[i]
    diff = recon - totals[i]
    max_diff = np.max(np.abs(diff))
    max_total = np.max(np.abs(totals[i]))
    print(f"  {bn}: max|recon - total| = {max_diff:.4e}  (peak={max_total:.4f}, "
          f"rel_err={100*max_diff/max_total:.2f}%)")

# ============================================================
# Flux measurement
# ============================================================
print(f"\n{'Band':<12} {'Host(uJy)':<14} {'AGN_C1(uJy)':<14} {'AGN_C2(uJy)':<14} "
      f"{'Sum(uJy)':<14} {'GalfitS(uJy)':<14} {'Host%':<8} {'C1%':<8} {'C2%':<8}")
print("-" * 130)

all_fluxes = {}
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    band_label = im.band.replace('miri_','').replace('nircam_','').upper()
    mask = im.cut_mask_image
    h, a1, a2, t = hosts[i], agn1_imgs[i], agn2_imgs[i], totals[i]
    good = (mask==0) & np.isfinite(h) & np.isfinite(a1) & np.isfinite(a2)

    f_host = np.sum(h[good]) * CONV_uJy
    f_agn1 = np.sum(a1[good]) * CONV_uJy
    f_agn2 = np.sum(a2[good]) * CONV_uJy
    f_sum = f_host + f_agn1 + f_agn2
    f_total = np.sum(t[good]) * CONV_uJy
    p_host = 100*f_host/f_sum if f_sum>0 else 0
    p_agn1 = 100*f_agn1/f_sum if f_sum>0 else 0
    p_agn2 = 100*f_agn2/f_sum if f_sum>0 else 0
    all_fluxes[band_label] = {'host':f_host, 'agn1':f_agn1, 'agn2':f_agn2,
                               'sum':f_sum, 'total':f_total}
    print(f"{band_label:<12} {f_host:<14.4f} {f_agn1:<14.4f} {f_agn2:<14.4f} "
          f"{f_sum:<14.4f} {f_total:<14.4f} {p_host:<8.1f} {p_agn1:<8.1f} {p_agn2:<8.1f}")

# Save CSV
csv_path = os.path.join(WORK, 'three_component_fluxes.csv')
with open(csv_path, 'w') as f:
    f.write('band,Host_uJy,AGN_C1_uJy,AGN_C2_uJy,Sum_uJy,GalfitS_Total_uJy,Host_pct,AGN_C1_pct,AGN_C2_pct\n')
    for bl in BAND_LABELS:
        if bl in all_fluxes:
            r = all_fluxes[bl]
            f.write(f'{bl},{r["host"]:.4f},{r["agn1"]:.4f},{r["agn2"]:.4f},'
                    f'{r["sum"]:.4f},{r["total"]:.4f},'
                    f'{100*r["host"]/r["sum"]:.2f},{100*r["agn1"]/r["sum"]:.2f},{100*r["agn2"]/r["sum"]:.2f}\n')
print(f"\nFluxes saved: {csv_path}")

# ============================================================
# Save FITS
# ============================================================
print("\nSaving component FITS...")
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    bl = im.band.replace('miri_','').replace('nircam_','').upper()
    fits.writeto(os.path.join(WORK, f'Host_model_{bl}.fits'),
                 hosts[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'AGN_C1_model_{bl}.fits'),
                 agn1_imgs[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'AGN_C2_model_{bl}.fits'),
                 agn2_imgs[i].astype(np.float32), overwrite=True)
print("  Done.")

# ============================================================
# Plot 1: All 12 bands
# ============================================================
print("Plotting all-band component images...")
fig, axes = plt.subplots(nrows=nimage, ncols=4, figsize=(20, 5*nimage), squeeze=True)
for row, i in enumerate(range(nimage)):
    im = fitter.GSdata.get_image(i)
    bn = im.band.replace('miri_','').replace('nircam_','').upper()
    data = data_imgs[i]; sigma = sigma_imgs[i]; mask = mask_imgs[i]
    host = hosts[i]; agn1 = agn1_imgs[i]; agn2 = agn2_imgs[i]; total = totals[i]

    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    immin = 5*sky_std; immax = float(jnp.nanmax(data)); frac = 0.4
    m_vis = np.ma.masked_where(mask==0, mask)
    ny, nx = data.shape

    recon = host + agn1 + agn2
    max_diff = np.max(np.abs(recon - total))
    rel_err = 100 * max_diff / immax if immax > 0 else 0

    for col, (img, title) in enumerate([
        (host, 'Host (Sérsic)'),
        (agn1, 'AGN_C1 (Center)'),
        (agn2, 'AGN_C2 (West)'),
        ((data - total)/sigma, 'Residual/σ'),
    ]):
        ax = axes[row, col]
        if 'Residual' in title:
            cb = ax.imshow(np.where(sigma>0, img, 0), cmap='seismic', origin='lower',
                           vmin=-10, vmax=10, interpolation='nearest')
        else:
            ax.imshow(gsutils.normimg(img, immin, immax, frac=frac), cmap='seismic',
                      origin='lower', vmin=-1, vmax=1, interpolation='nearest')
        ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5,
                  interpolation='nearest', vmin=0, vmax=1)
        if row == 0:
            ax.text(0.03*nx, 0.85*ny, title, size=16, color='k', weight='bold')
        ax.text(0.03*nx, 0.06*ny, f'{bn} (Δ={rel_err:.1f}%)', size=14, color='k', weight='light')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 3 and row == 0:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='5%', pad=0.05)
            plt.colorbar(cb, cax=cax, orientation='horizontal')

plt.tight_layout()
img_out = os.path.join(WORK, 'three_component_model_images.png')
plt.savefig(img_out, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {img_out}')

# ============================================================
# Plot 2: MIRI-only (9 bands)
# ============================================================
print("Plotting MIRI-only component images...")
miri_bands = [i for i in range(nimage) if fitter.GSdata.get_image(i).band.startswith('miri_')]
nrows_m = len(miri_bands)
fig, axes = plt.subplots(nrows=nrows_m, ncols=4, figsize=(20, 5*nrows_m), squeeze=True)

for row, i in enumerate(miri_bands):
    im = fitter.GSdata.get_image(i)
    bn = im.band.replace('miri_','').upper()
    data = data_imgs[i]; sigma = sigma_imgs[i]; mask = mask_imgs[i]
    host = hosts[i]; agn1 = agn1_imgs[i]; agn2 = agn2_imgs[i]; total = totals[i]

    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    immin = 5*sky_std; immax = float(jnp.nanmax(data)); frac = 0.4
    m_vis = np.ma.masked_where(mask==0, mask)
    ny, nx = data.shape

    recon = host + agn1 + agn2
    max_diff = np.max(np.abs(recon - total))
    rel_err = 100 * max_diff / immax if immax > 0 else 0

    for col, (img, title) in enumerate([
        (host, 'Host (Sérsic)'),
        (agn1, 'AGN_C1 (Center)'),
        (agn2, 'AGN_C2 (West)'),
        ((data - total)/sigma, 'Residual/σ'),
    ]):
        ax = axes[row, col]
        if 'Residual' in title:
            cb = ax.imshow(np.where(sigma>0, img, 0), cmap='seismic', origin='lower',
                           vmin=-10, vmax=10, interpolation='nearest')
        else:
            ax.imshow(gsutils.normimg(img, immin, immax, frac=frac), cmap='seismic',
                      origin='lower', vmin=-1, vmax=1, interpolation='nearest')
        ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5,
                  interpolation='nearest', vmin=0, vmax=1)
        if row == 0:
            ax.text(0.03*nx, 0.85*ny, title, size=16, color='k', weight='bold')
        ax.text(0.03*nx, 0.06*ny, f'{bn} (Δ={rel_err:.1f}%)', size=14, color='k', weight='light')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 3 and row == 0:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='5%', pad=0.05)
            plt.colorbar(cb, cax=cax, orientation='horizontal')

plt.tight_layout()
img_miri = os.path.join(WORK, 'three_component_model_images_MIRI.png')
plt.savefig(img_miri, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {img_miri}')

# ============================================================
# Plot 3: SED with power-law fits
# ============================================================
print("Plotting SED...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax_idx, (ax, ylabel, title_suffix) in enumerate([
    (axes[0], 'F$_\\nu$ (μJy)', 'F_ν'),
    (axes[1], 'ν F$_\\nu$ (μJy · Hz)', 'νF_ν'),
]):
    bands_ok = [bl for bl in BAND_LABELS if bl in all_fluxes]
    waves_arr = np.array([WAVES_UM[BAND_LABELS.index(bl)] for bl in bands_ok])

    host_f = np.array([all_fluxes[bl]['host'] for bl in bands_ok])
    agn1_f = np.array([all_fluxes[bl]['agn1'] for bl in bands_ok])
    agn2_f = np.array([all_fluxes[bl]['agn2'] for bl in bands_ok])
    sum_f = host_f + agn1_f + agn2_f
    total_f = np.array([all_fluxes[bl]['total'] for bl in bands_ok])

    if ax_idx == 1:
        nu = 2.99792458e14 / waves_arr
        host_plot = nu * host_f * 1e-6
        agn1_plot = nu * agn1_f * 1e-6
        agn2_plot = nu * agn2_f * 1e-6
        sum_plot = nu * sum_f * 1e-6
        total_plot = nu * total_f * 1e-6
    else:
        host_plot = host_f; agn1_plot = agn1_f; agn2_plot = agn2_f
        sum_plot = sum_f; total_plot = total_f

    ax.errorbar(waves_arr, total_plot, yerr=0.1*total_plot, fmt='ko',
                label='Total (GalfitS)', capsize=3)
    ax.errorbar(waves_arr, host_plot, yerr=0.1*host_plot, fmt='o',
                color='teal', markersize=9, capsize=3, label='Host (Sérsic)')
    ax.errorbar(waves_arr, agn1_plot, yerr=0.1*agn1_plot, fmt='s',
                color='royalblue', markersize=9, capsize=3, label='AGN_C1 (Center)')
    ax.errorbar(waves_arr, agn2_plot, yerr=0.1*agn2_plot, fmt='^',
                color='crimson', markersize=9, capsize=3, label='AGN_C2 (West)')

    # Power-law fits for AGN components
    nu_0 = 2.99792458e14 / 1.0
    freqs = 2.99792458e14 / waves_arr
    x = np.log10(freqs/nu_0)
    Amat = np.column_stack([np.ones_like(x), -x])

    for flux, color, name in [(agn1_f, 'royalblue', 'AGN_C1'),
                                (agn2_f, 'crimson', 'AGN_C2')]:
        fpos = np.maximum(flux, 1e-10)
        w = fpos/np.max(fpos)
        try:
            p = np.linalg.inv(Amat.T @ np.diag(w) @ Amat) @ Amat.T @ np.diag(w) @ np.log10(fpos)
            A, alpha = 10**p[0], p[1]
            wl_f = np.logspace(np.log10(0.9), np.log10(28), 200)
            nu_f = 2.99792458e14 / wl_f
            if ax_idx == 1:
                pl_plot = nu_f * A * (nu_f/nu_0)**(-alpha) * 1e-6
            else:
                pl_plot = A * (nu_f/nu_0)**(-alpha)
            ax.plot(wl_f, pl_plot, '--', color=color, lw=1.5,
                    label=f'{name}: α={alpha:.2f}')
        except Exception as e:
            print(f"  Power-law fail for {name}: {e}")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'COS-87259 (z=6.83): 3-Component SED ({title_suffix})', fontsize=14)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0.8, 28)

    ax2 = ax.twiny(); ax2.set_xscale('log')
    tick_um = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    ax2.set_xticks(tick_um * (1+6.83))
    ax2.set_xticklabels([f'{t:.1f}' for t in tick_um])
    ax2.set_xlim(ax.get_xlim()); ax2.set_xlabel('Rest-Frame Wavelength (μm)', fontsize=10)

plt.tight_layout()
sed_out = os.path.join(WORK, 'three_component_SED.png')
plt.savefig(sed_out, dpi=200, bbox_inches='tight')
plt.close()
print(f'  Saved: {sed_out}')

print("\n=== All done! ===")
