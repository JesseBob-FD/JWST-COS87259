"""
Separate 4 components from two_sersic_two_AGN_C1C2_noSED fit:
  sersic_C1, sersic_C2, AGN_C1, AGN_C2

Method: psfmodel_list isolation + logNorm zeroing, combining both techniques.

Group 1 (full): 4 individual components
  - Sérsic_C1: psfmodel_list=[], zero logNorm_sersic_C2
  - Sérsic_C2: psfmodel_list=[], zero logNorm_sersic_C1
  - AGN_C1:   psfmodel_list=[AGN_C1], zero both logNorms
  - AGN_C2:   psfmodel_list=[AGN_C2], zero both logNorms

Group 2 (by clump): 2 paired groups
  - C1 group (sersic_C1 + AGN_C1): psfmodel_list=[AGN_C1], zero logNorm_sersic_C2
  - C2 group (sersic_C2 + AGN_C2): psfmodel_list=[AGN_C2], zero logNorm_sersic_C1

Outputs:
  1) 4-component model images (PNG: 4 columns × 12 rows)
  2) 2-group model images (PNG: 3 columns [C1/C2/residual] × 12 rows)
  3) MIRI-only versions of both
  4) 4-component SED (F_nu + nuF_nu)
  5) 2-group SED
  6) Flux CSV for both groupings
  7) Per-component FITS files
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
CONFIG = os.path.join(WORK, 'two_sersic_two_AGN_C1C2_noSED.lyric')
GSFILE  = [f for f in os.listdir(WORK) if f.endswith('.gssummary')]
if not GSFILE:
    print("ERROR: No .gssummary found! Run fitting first.")
    sys.exit(1)
GSFILE = os.path.join(WORK, GSFILE[0])

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

# ============================================================
# Load fit
# ============================================================
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
sc1_name, sc2_name = gal.subnames[0], gal.subnames[1]
print(f"Galaxy: {gal.name}, Sérsic components: {gal.subnames}")

agn_models = []
for i, m in enumerate(fitter.model_list):
    if fitter.mtype_list[i] == 'agn':
        agn_models.append(m)
        print(f"AGN: {m.name}")
agn_C1, agn_C2 = agn_models[0], agn_models[1]

original_psf_models = list(fitter.psfmodel_list)
nimage = fitter.GSdata.Nimages

# Build parameter key lists
sc1_logNorm_keys = [k for k in fitter.pardict if k.startswith(f'logNorm_{sc1_name}_')]
sc2_logNorm_keys = [k for k in fitter.pardict if k.startswith(f'logNorm_{sc2_name}_')]

def zero_logNorm(comp_name):
    """Zero a Sérsic component's logNorm params."""
    keys = [k for k in fitter.pardict if k.startswith(f'logNorm_{comp_name}_')]
    saved = {k: fitter.pardict[k] for k in keys}
    for k in keys:
        fitter.pardict[k] = -30.0
    return saved

def restore_logNorm(saved):
    for k, v in saved.items():
        fitter.pardict[k] = v

def get_model_images(psf_list):
    """Run cal_model_image with given psfmodel_list, return all model images."""
    fitter.psfmodel_list = list(psf_list)
    fitter.cal_model_image()
    return [np.array(fitter.GSdata.get_image(i).model_image) for i in range(nimage)]

# ============================================================
# Data images
# ============================================================
fitter.psfmodel_list = list(original_psf_models)
fitter.cal_model_image()
data_imgs, sigma_imgs, mask_imgs = [], [], []
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
    data_imgs.append(np.array(im.cut_image) - sky_median)
    sigma_imgs.append(np.array(im.cut_sigma_image))
    mask_imgs.append(np.array(im.cut_mask_image, dtype=float))

# ============================================================
# Total model
# ============================================================
print("Computing total model...")
fitter.psfmodel_list = list(original_psf_models)
# restore all logNorms
for k in sc1_logNorm_keys:
    fitter.pardict[k] = fitter.lmParameters[k].value
for k in sc2_logNorm_keys:
    fitter.pardict[k] = fitter.lmParameters[k].value
totals = get_model_images(original_psf_models)

# ============================================================
# 4-component separation
# ============================================================
print("Separating 4 components...")

# Sérsic_C1-only: no AGNs, zero Sérsic_C2 logNorm
sc2_saved = zero_logNorm(sc2_name)
sc1_imgs = get_model_images([])
restore_logNorm(sc2_saved)

# Sérsic_C2-only: no AGNs, zero Sérsic_C1 logNorm
sc1_saved = zero_logNorm(sc1_name)
sc2_imgs = get_model_images([])
restore_logNorm(sc1_saved)

# AGN_C1-only: AGN_C1 active, both logNorms zeroed
sc1_saved = zero_logNorm(sc1_name)
sc2_saved = zero_logNorm(sc2_name)
agn1_imgs = get_model_images([agn_C1])
restore_logNorm(sc1_saved); restore_logNorm(sc2_saved)

# AGN_C2-only: AGN_C2 active, both logNorms zeroed
sc1_saved = zero_logNorm(sc1_name)
sc2_saved = zero_logNorm(sc2_name)
agn2_imgs = get_model_images([agn_C2])
restore_logNorm(sc1_saved); restore_logNorm(sc2_saved)

# ============================================================
# 2-group separation
# ============================================================
print("Separating 2 groups...")

# C1 group (sersic_C1 + AGN_C1): zero Sérsic_C2, AGN_C1 active
sc2_saved = zero_logNorm(sc2_name)
c1_group_imgs = get_model_images([agn_C1])
restore_logNorm(sc2_saved)

# C2 group (sersic_C2 + AGN_C2): zero Sérsic_C1, AGN_C2 active
sc1_saved = zero_logNorm(sc1_name)
c2_group_imgs = get_model_images([agn_C2])
restore_logNorm(sc1_saved)

# Restore everything
fitter.psfmodel_list = list(original_psf_models)

# ============================================================
# Verify
# ============================================================
print("\nVerification (4-component: sc1 + sc2 + agn1 + agn2 vs total):")
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    recon = sc1_imgs[i] + sc2_imgs[i] + agn1_imgs[i] + agn2_imgs[i]
    diff = recon - totals[i]
    max_diff = np.max(np.abs(diff))
    max_total = np.max(np.abs(totals[i]))
    print(f"  {im.band}: max|recon-total|={max_diff:.2e} (rel={100*max_diff/max_total:.2f}%)")

print("\nVerification (2-group: C1_grp + C2_grp vs total):")
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    recon = c1_group_imgs[i] + c2_group_imgs[i]
    diff = recon - totals[i]
    max_diff = np.max(np.abs(diff))
    max_total = np.max(np.abs(totals[i]))
    print(f"  {im.band}: max|recon-total|={max_diff:.2e} (rel={100*max_diff/max_total:.2f}%)")

# ============================================================
# Flux measurement (4-component)
# ============================================================
print(f"\n{'Band':<10} {'Sersic_C1':<14} {'Sersic_C2':<14} {'AGN_C1':<14} {'AGN_C2':<14} {'Sum':<14} {'Total':<14}")
print("-" * 100)

flux_4comp = {}
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    bl = im.band.replace('miri_','').replace('nircam_','').upper()
    mask = im.cut_mask_image
    s1, s2, a1, a2, t = sc1_imgs[i], sc2_imgs[i], agn1_imgs[i], agn2_imgs[i], totals[i]
    good = (mask==0) & np.isfinite(s1) & np.isfinite(s2) & np.isfinite(a1) & np.isfinite(a2)
    f_s1 = np.sum(s1[good]) * CONV_uJy
    f_s2 = np.sum(s2[good]) * CONV_uJy
    f_a1 = np.sum(a1[good]) * CONV_uJy
    f_a2 = np.sum(a2[good]) * CONV_uJy
    f_sum = f_s1 + f_s2 + f_a1 + f_a2
    f_tot = np.sum(t[good]) * CONV_uJy
    flux_4comp[bl] = {'s1':f_s1, 's2':f_s2, 'a1':f_a1, 'a2':f_a2, 'sum':f_sum, 'total':f_tot}
    pct = lambda x: 100*x/f_sum if f_sum>0 else 0
    print(f"{bl:<10} {f_s1:<14.4f} {f_s2:<14.4f} {f_a1:<14.4f} {f_a2:<14.4f} {f_sum:<14.4f} {f_tot:<14.4f}")

# Flux measurement (2-group)
flux_2grp = {}
print(f"\n{'Band':<10} {'C1_group':<14} {'C2_group':<14} {'Sum':<14} {'Total':<14} {'C1%':<8} {'C2%':<8}")
print("-" * 80)
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    bl = im.band.replace('miri_','').replace('nircam_','').upper()
    mask = im.cut_mask_image
    c1, c2, t = c1_group_imgs[i], c2_group_imgs[i], totals[i]
    good = (mask==0) & np.isfinite(c1) & np.isfinite(c2)
    f_c1 = np.sum(c1[good]) * CONV_uJy
    f_c2 = np.sum(c2[good]) * CONV_uJy
    f_sum = f_c1 + f_c2
    f_tot = np.sum(t[good]) * CONV_uJy
    flux_2grp[bl] = {'c1':f_c1, 'c2':f_c2, 'sum':f_sum, 'total':f_tot}
    pct1 = 100*f_c1/f_sum if f_sum>0 else 0
    pct2 = 100*f_c2/f_sum if f_sum>0 else 0
    print(f"{bl:<10} {f_c1:<14.4f} {f_c2:<14.4f} {f_sum:<14.4f} {f_tot:<14.4f} {pct1:<8.1f} {pct2:<8.1f}")

# Save flux CSVs
csv_4 = os.path.join(WORK, 'four_component_fluxes.csv')
with open(csv_4, 'w') as f:
    f.write('band,Sersic_C1_uJy,Sersic_C2_uJy,AGN_C1_uJy,AGN_C2_uJy,Sum_uJy,Total_uJy\n')
    for bl in BAND_LABELS:
        if bl in flux_4comp:
            r = flux_4comp[bl]
            f.write(f'{bl},{r["s1"]:.4f},{r["s2"]:.4f},{r["a1"]:.4f},{r["a2"]:.4f},{r["sum"]:.4f},{r["total"]:.4f}\n')

csv_2 = os.path.join(WORK, 'two_group_fluxes.csv')
with open(csv_2, 'w') as ff:
    ff.write('band,C1_group_uJy,C2_group_uJy,Sum_uJy,Total_uJy,C1_pct,C2_pct\n')
    for bl in BAND_LABELS:
        if bl in flux_2grp:
            r = flux_2grp[bl]
            ff.write(f'{bl},{r["c1"]:.4f},{r["c2"]:.4f},{r["sum"]:.4f},{r["total"]:.4f},'
                     f'{100*r["c1"]/r["sum"]:.2f},{100*r["c2"]/r["sum"]:.2f}\n')
print(f"\nCSV saved: {csv_4}, {csv_2}")

# ============================================================
# Save FITS
# ============================================================
print("\nSaving FITS...")
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    bl = im.band.replace('miri_','').replace('nircam_','').upper()
    fits.writeto(os.path.join(WORK, f'sersic_C1_model_{bl}.fits'), sc1_imgs[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'sersic_C2_model_{bl}.fits'), sc2_imgs[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'AGN_C1_model_{bl}.fits'), agn1_imgs[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'AGN_C2_model_{bl}.fits'), agn2_imgs[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'C1_group_model_{bl}.fits'), c1_group_imgs[i].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK, f'C2_group_model_{bl}.fits'), c2_group_imgs[i].astype(np.float32), overwrite=True)
print("  Done.")

# ============================================================
# Plot helper
# ============================================================
def do_image_plot(miri_only, ncols, col_items, out_name):
    """col_items = list of (img_array_fn, title) tuples"""
    indices = [i for i in range(nimage) if not miri_only or fitter.GSdata.get_image(i).band.startswith('miri_')]
    nrows = len(indices)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), squeeze=True)

    for row, i in enumerate(indices):
        im = fitter.GSdata.get_image(i)
        bn = im.band.replace('miri_','').replace('nircam_','').upper()
        data = data_imgs[i]; sigma = sigma_imgs[i]; mask = mask_imgs[i]
        total = totals[i]
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        immin = 5*sky_std; immax = float(jnp.nanmax(data)); frac = 0.4
        m_vis = np.ma.masked_where(mask==0, mask)
        ny, nx = data.shape

        for col, (img_list, title) in enumerate(col_items):
            img = img_list[i]
            ax = axes[row, col] if nrows > 1 else axes[col]
            if 'Residual' in title:
                cb = ax.imshow(np.where(sigma>0, img, 0), cmap='seismic', origin='lower',
                               vmin=-10, vmax=10, interpolation='nearest')
            else:
                ax.imshow(gsutils.normimg(img, immin, immax, frac=frac), cmap='seismic',
                          origin='lower', vmin=-1, vmax=1, interpolation='nearest')
            ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5, vmin=0, vmax=1)
            if row == 0:
                ax.text(0.03*nx, 0.85*ny, title, size=14, color='k', weight='bold')
            ax.text(0.03*nx, 0.06*ny, bn, size=13, color='k', weight='light')
            ax.set_xticks([]); ax.set_yticks([])
            if 'Residual' in title and row == 0:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('top', size='5%', pad=0.05)
                plt.colorbar(cb, cax=cax, orientation='horizontal')

    plt.tight_layout()
    path = os.path.join(WORK, out_name)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# Residual images (data - total) / sigma
residual_imgs = []
for i in range(nimage):
    sigma = sigma_imgs[i]
    res = (data_imgs[i] - totals[i]) / sigma
    residual_imgs.append(np.where(sigma>0, res, 0))

# ============================================================
# Plot 1: 4-component (all bands)
# ============================================================
print("\nPlotting 4-component images (all bands)...")
col_fn_4 = [
    (sc1_imgs, 'Sérsic_C1 (Center)'),
    (sc2_imgs, 'Sérsic_C2 (West)'),
    (agn1_imgs, 'AGN_C1 (Center)'),
    (agn2_imgs, 'AGN_C2 (West)'),
    (residual_imgs, 'Residual/σ'),
]
do_image_plot(False, 5, col_fn_4, 'four_component_model_images.png')

# Plot 1b: 4-component (MIRI only)
print("Plotting 4-component images (MIRI only)...")
do_image_plot(True, 5, col_fn_4, 'four_component_model_images_MIRI.png')

# ============================================================
# Plot 2: 2-group (all bands)
# ============================================================
print("Plotting 2-group images (all bands)...")
col_fn_2 = [
    (c1_group_imgs, 'C1 Group (Sérsic+AGN)'),
    (c2_group_imgs, 'C2 Group (Sérsic+AGN)'),
    (residual_imgs, 'Residual/σ'),
]
do_image_plot(False, 3, col_fn_2, 'two_group_model_images.png')

# Plot 2b: 2-group (MIRI only)
print("Plotting 2-group images (MIRI only)...")
do_image_plot(True, 3, col_fn_2, 'two_group_model_images_MIRI.png')

# ============================================================
# SED plots
# ============================================================
def do_sed_plot(flux_dict, groups, out_name, title_prefix, band_labels, waves_arr_in):
    """groups = list of (key, label, color, marker) tuples"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax_idx, (ax, ylabel, sfx) in enumerate([
        (axes[0], 'F$_\\nu$ (μJy)', 'F_ν'),
        (axes[1], 'ν F$_\\nu$ (μJy · Hz)', 'νF_ν'),
    ]):
        bands_ok = [bl for bl in band_labels if bl in flux_dict]
        waves_arr = np.array([waves_arr_in[band_labels.index(bl)] for bl in bands_ok])

        for key, label, color, marker in groups:
            vals = np.array([flux_dict[bl][key] for bl in bands_ok])
            if ax_idx == 1:
                nu = 2.99792458e14 / waves_arr
                plot_vals = nu * vals * 1e-6
            else:
                plot_vals = vals
            ax.errorbar(waves_arr, plot_vals, yerr=0.1*plot_vals, fmt=marker,
                        color=color, markersize=9, capsize=3, label=label)

            # Power-law for AGN-like components
            if 'AGN' in label:
                nu_0 = 2.99792458e14 / 1.0; freqs = 2.99792458e14 / waves_arr
                x = np.log10(freqs/nu_0)
                Amat = np.column_stack([np.ones_like(x), -x])
                fpos = np.maximum(vals, 1e-10); w = fpos/np.max(fpos)
                try:
                    p = np.linalg.inv(Amat.T @ np.diag(w) @ Amat) @ Amat.T @ np.diag(w) @ np.log10(fpos)
                    A, alpha = 10**p[0], p[1]
                    wl_f = np.logspace(np.log10(4.5), np.log10(28), 200)
                    nu_f = 2.99792458e14 / wl_f
                    if ax_idx == 1:
                        lp = nu_f * A * (nu_f/nu_0)**(-alpha) * 1e-6
                    else:
                        lp = A * (nu_f/nu_0)**(-alpha)
                    ax.plot(wl_f, lp, '--', color=color, lw=1.2, label=f'{label}: α={alpha:.2f}')
                except: pass

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{title_prefix} ({sfx})', fontsize=13)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3, which='both'); ax.set_xlim(4.5, 28)

        ax2 = ax.twiny(); ax2.set_xscale('log')
        tick_um = np.array([0.6,0.8,1.0,1.5,2.0,2.5,3.0])
        ax2.set_xticks(tick_um*(1+6.83))
        ax2.set_xticklabels([f'{t:.1f}' for t in tick_um])
        ax2.set_xlim(ax.get_xlim()); ax2.set_xlabel('Rest-Frame (μm)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(WORK, out_name)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')

# 4-component SED (MIRI only)
print("\nPlotting 4-component SED (MIRI)...")
do_sed_plot(flux_4comp, [
    ('s1', 'Sérsic_C1 (Center)', 'teal', 'o'),
    ('s2', 'Sérsic_C2 (West)', 'purple', 'o'),
    ('a1', 'AGN_C1 (Center)', 'royalblue', 's'),
    ('a2', 'AGN_C2 (West)', 'crimson', '^'),
], 'four_component_SED.png', 'COS-87259: 4-Component SED', BAND_LABELS_MIRI, WAVES_UM)

# 2-group SED (MIRI only)
print("Plotting 2-group SED (MIRI)...")
do_sed_plot(flux_2grp, [
    ('c1', 'C1 Group (Center)', 'teal', 'o'),
    ('c2', 'C2 Group (West)', 'crimson', '^'),
], 'two_group_SED.png', 'COS-87259: C1 vs C2 Group SED', BAND_LABELS_MIRI, WAVES_UM)

print("\n=== All done! ===")
