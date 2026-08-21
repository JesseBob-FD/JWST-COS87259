"""
Plot per-component model images using GalfitS's own cal_model_image().
Technique from plot_resu.ipynb + zijian.ipynb:
  - call cal_model_image() → im.model_image = total
  - zero one component's logNorm in pardict → call cal_model_image() → other component
  - subtract → get the zeroed component
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

# --- Load fit and set parameters to best-fit values ---
WORK = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/two_sersic_C1C2/noSED'
OUT  = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/MIRI_clump_departure'
CONFIG = os.path.join(WORK, 'C1C2_noSED.lyric')

fitter, targ, fs = gsutils.read_config_file(CONFIG, WORK)
smfile = Table.read(os.path.join(WORK, 'C1C2_noSED.gssummary'), format='ascii')
for row in smfile:
    pname = row['pname']
    if pname in fitter.lmParameters:
        fitter.lmParameters[pname].set(value=row['best_value'],
                                        min=row['best_value']-0.1,
                                        max=row['best_value']+0.1)
fitter.loose_fix_pars()
# Sync pardict with updated lmParameters
for key in fitter.pardict:
    if key in fitter.lmParameters:
        fitter.pardict[key] = fitter.lmParameters[key].value

gal = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='galaxy'][0]
c1_name, c2_name = gal.subnames[0], gal.subnames[1]
print(f"Components: C1={c1_name}, C2={c2_name}")

MIRI = ['miri_f560w','miri_f770w','miri_f1000w','miri_f1130w',
        'miri_f1280w','miri_f1500w','miri_f1800w','miri_f2100w','miri_f2550w']
BANDS = ['F560W','F770W','F1000W','F1130W','F1280W','F1500W','F1800W','F2100W','F2550W']
WAVES = ['5.6um','7.7um','10.0um','11.3um','12.8um','15.0um','18.0um','21.0um','25.5um']

# === Step 1: Total model (all components active) ===
pltwave, Sedcomp, Sedlabel, z = fitter.cal_model_image()
nimage = fitter.GSdata.Nimages
total_models = []
data_imgs = []
sigma_imgs = []
mask_imgs = []
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
    data_imgs.append(np.array(im.cut_image) - sky_median)
    total_models.append(np.array(im.model_image))
    sigma_imgs.append(np.array(im.cut_sigma_image))
    mask_imgs.append(np.array(im.cut_mask_image, dtype=float))

# === Step 2: C2-only (zero C1's logNorm in pardict) ===
saved_c1 = {}
for jband in MIRI:
    key = f'logNorm_{c1_name}_{jband}'
    saved_c1[jband] = fitter.pardict.get(key, 0)
    fitter.pardict[key] = -30.0
fitter.cal_model_image()
c2_models = []
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    c2_models.append(np.array(im.model_image))

# === Step 3: C1-only (restore C1, zero C2's logNorm) ===
for jband in MIRI:
    fitter.pardict[f'logNorm_{c1_name}_{jband}'] = saved_c1[jband]
    fitter.pardict[f'logNorm_{c2_name}_{jband}'] = -30.0
fitter.cal_model_image()
c1_models = []
for i in range(nimage):
    im = fitter.GSdata.get_image(i)
    c1_models.append(np.array(im.model_image))

# Restore C2
for jband in MIRI:
    fitter.pardict[f'logNorm_{c2_name}_{jband}'] = fitter.lmParameters[f'logNorm_{c2_name}_{jband}'].value

# Filter to MIRI bands only (skip NIRCam)
miri_idx = [i for i in range(nimage)
            if fitter.GSdata.get_image(i).band.startswith('miri_')]

print(f"Total images: {nimage}, MIRI bands: {len(miri_idx)}")

# === Plot ===
nrows = len(miri_idx)
fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5*nrows), squeeze=True)

for row, i in enumerate(miri_idx):
    im = fitter.GSdata.get_image(i)
    band_name = im.band.replace('miri_', '').upper()
    data = data_imgs[i]
    total = total_models[i]
    c1 = c1_models[i]
    c2 = c2_models[i]
    sigma = sigma_imgs[i]
    mask = mask_imgs[i]
    ny, nx = data.shape

    sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
    immin = 5 * sky_std
    immax = float(jnp.nanmax(data))
    frac = 0.4

    m_vis = np.ma.masked_where(mask == 0, mask)

    # Verify: C1 + C2 should equal total
    diff = c1 + c2 - total
    max_diff = np.max(np.abs(diff))

    # --- C1 model ---
    ax = axes[row, 0]
    ax.imshow(gsutils.normimg(c1, immin, immax, frac=frac), cmap='seismic',
              origin='lower', vmin=-1, vmax=1, interpolation='nearest')
    ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5,
              interpolation='nearest', vmin=0, vmax=1)
    ax.text(0.03*nx, 0.06*ny, f'{band_name}', size=20, color='k', weight='light')
    ax.text(0.03*nx, 0.80*ny, f'diff max={max_diff:.2e}', size=12, color='k')
    if row == 0:
        ax.text(0.03*nx, 0.90*ny, f'C1={c1_name}', size=20, color='k', weight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # --- C2 model ---
    ax = axes[row, 1]
    ax.imshow(gsutils.normimg(c2, immin, immax, frac=frac), cmap='seismic',
              origin='lower', vmin=-1, vmax=1, interpolation='nearest')
    ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5,
              interpolation='nearest', vmin=0, vmax=1)
    if row == 0:
        ax.text(0.03*nx, 0.90*ny, f'C2={c2_name}', size=20, color='k', weight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # --- Residual (data - total) / sigma ---
    ax = axes[row, 2]
    residual = data - total
    res_sigma = np.where(sigma > 0, residual / sigma, 0)
    cb = ax.imshow(res_sigma, cmap='seismic', origin='lower',
                   vmin=-10, vmax=10, interpolation='nearest')
    ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5,
              interpolation='nearest', vmin=0, vmax=1)
    if row == 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.05)
        cbar = plt.colorbar(cb, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=20)
        ax.text(0.03*nx, 0.80*ny, 'Residual/sigma', size=20, color='k', weight='bold')
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
out = f'{OUT}/component_model_images.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()

# === Save correct per-component FITS ===
print("Saving correct component FITS...")
for row, i in enumerate(miri_idx):
    im = fitter.GSdata.get_image(i)
    band = im.band.replace('miri_', '').upper()
    fits.writeto(f'{WORK}/C1_model_{band}_correct.fits',
                 c1_models[i].astype(np.float32), overwrite=True)
    fits.writeto(f'{WORK}/C2_model_{band}_correct.fits',
                 c2_models[i].astype(np.float32), overwrite=True)
    # Verify: max difference between C1+C2 and total
    diff = np.max(np.abs(c1_models[i] + c2_models[i] - total_models[i]))
    print(f'  {band}: max(|C1+C2-total|) = {diff:.6e}')
print('Done.')
