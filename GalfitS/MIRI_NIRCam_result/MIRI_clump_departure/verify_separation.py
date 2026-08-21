"""
Verify the two-Sersic component separation by visualizing spatial extent.
Regenerates per-component model images on the fly via GalfitS API.
Saves to MIRI_clump_departure/.
"""
import os, sys, numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import binary_erosion

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

WORK = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/two_sersic_C1C2/noSED'
OUT  = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/MIRI_clump_departure'
CONFIG = os.path.join(WORK, 'C1C2_noSED.lyric')
PARAMS = os.path.join(WORK, 'C1C2_noSED.params')

# Load fitting setup
fitter, targ, fs = gsutils.read_config_file(CONFIG, WORK)
ptab = Table.read(PARAMS, format='ascii')
for row in ptab: fitter.pardict[row['name']] = row['value']
gal = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='galaxy'][0]
c1_name, c2_name = gal.subnames[0], gal.subnames[1]

BANDS = ['F560W','F770W','F1000W','F1130W','F1280W',
         'F1500W','F1800W','F2100W','F2550W']
MIRI = ['miri_f560w','miri_f770w','miri_f1000w','miri_f1130w',
        'miri_f1280w','miri_f1500w','miri_f1800w','miri_f2100w','miri_f2550w']

# Generate per-band component images
c1_imgs, c2_imgs, data_imgs, mask_imgs = {}, {}, {}, {}

for jband, band in zip(MIRI, BANDS):
    im = None
    for atl in fitter.GSdata.image_atlas_list:
        for i,b in enumerate(atl.band_list):
            if b == jband: im = atl[i]; break
        if im: break

    if im.wcsshift:
        im.coordinates_transfer_para['x0shift'] = fitter.pardict.get(f'csx_{im.imglabel}',0)
        im.coordinates_transfer_para['y0shift'] = fitter.pardict.get(f'csy_{im.imglabel}',0)

    ny,nx = im.cut_image.shape
    noSED = (fitter.GSdata.fitMode == 'images - photometry')
    for c in gal.subnames: gal.update_subC_para(c, fitter.pardict)
    gal.generate_mass_map((ny,nx), transpar=im.coordinates_transfer_para)

    mm1 = gal.mass_map[c1_name]
    mm2 = gal.mass_map[c2_name]
    ln1 = fitter.pardict.get(f'logNorm_{c1_name}_{jband}', -30)
    ln2 = fitter.pardict.get(f'logNorm_{c2_name}_{jband}', -30)

    c1_imm = (10**ln1) * mm1 * im.phys_to_counts_rate
    c2_imm = (10**ln2) * mm2 * im.phys_to_counts_rate

    c1_imgs[band] = np.array(c1_imm)
    c2_imgs[band] = np.array(c2_imm)
    data_imgs[band] = np.array(im.cut_image)
    mask_imgs[band] = np.array(im.cut_mask_image)

# ---- Plot 1: dominance boundaries ----
fig, axes = plt.subplots(3, 3, figsize=(16, 16))
axes = axes.flatten()

for idx, band in enumerate(BANDS):
    ax = axes[idx]
    data = data_imgs[band]
    mask = mask_imgs[band].astype(bool)
    c1 = c1_imgs[band]
    c2 = c2_imgs[band]

    data_masked = np.where(mask, np.nan, data)
    total_model = c1 + c2
    total_model = np.where(mask, np.nan, total_model)

    with np.errstate(divide='ignore', invalid='ignore'):
        c1_frac = np.where(total_model > 0, c1 / total_model, np.nan)

    vmin = np.nanpercentile(data_masked, 10)
    vmax = np.nanpercentile(data_masked, 99)
    ax.imshow(data_masked, origin='lower', cmap='gray_r',
              vmin=vmin, vmax=vmax, aspect='auto')

    # C1 dominance region edge
    c1_dom = (c1_frac > 0.5)
    c1_edge = c1_dom.astype(float) - binary_erosion(c1_dom).astype(float)
    c1_rgba = np.zeros((*c1_edge.shape, 4))
    c1_rgba[c1_edge > 0] = [0.2, 0.5, 1.0, 0.8]
    ax.imshow(c1_rgba, origin='lower', aspect='auto')

    # C2 dominance region edge
    c2_dom = (c1_frac < 0.5)
    c2_edge = c2_dom.astype(float) - binary_erosion(c2_dom).astype(float)
    c2_rgba = np.zeros((*c2_edge.shape, 4))
    c2_rgba[c2_edge > 0] = [1.0, 0.2, 0.3, 0.8]
    ax.imshow(c2_rgba, origin='lower', aspect='auto')

    cy, cx = data.shape[0]//2, data.shape[1]//2
    c1_px = cx + int(0.124 / 0.06); c1_py = cy + int(0.139 / 0.06)
    c2_px = cx + int(-0.116 / 0.06); c2_py = cy + int(0.169 / 0.06)
    ax.plot(c1_px, c1_py, '+', color='cyan', markersize=12, markeredgewidth=2)
    ax.plot(c2_px, c2_py, 'x', color='yellow', markersize=10, markeredgewidth=2)

    c1_mean = np.nanmean(c1_frac)
    ax.set_title(f'{band}  <C1/(C1+C2)>={c1_mean:.3f}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0],[0],marker='+',color='cyan',markersize=10,label='C1=Center'),
    Line2D([0],[0],marker='x',color='yellow',markersize=10,label='C2=West'),
    plt.Rectangle((0,0),1,1,facecolor='blue',alpha=0.5,label='C1>50%'),
    plt.Rectangle((0,0),1,1,facecolor='red',alpha=0.5,label='C2>50%'),
], loc='lower center', ncol=4, fontsize=10)
fig.suptitle('Separation Verification: Component Dominance Regions\nC1=Center, C2=West. Numbers = mean C1 fraction',
             fontsize=13, y=0.99)
plt.tight_layout(rect=[0,0.04,1,0.96])
out1 = f'{OUT}/separation_boundaries.png'
plt.savefig(out1, dpi=200, bbox_inches='tight')
print(f'Saved: {out1}')
plt.close()

# ---- Plot 2: contour overlay on total model ----
fig2, axes2 = plt.subplots(3, 3, figsize=(14, 14))
axes2 = axes2.flatten()

for idx, band in enumerate(BANDS):
    ax = axes2[idx]
    c1 = c1_imgs[band]; c2 = c2_imgs[band]
    total = c1 + c2
    total[total <= 0] = 1e-30
    with np.errstate(divide='ignore', invalid='ignore'):
        c1_frac = c1 / total

    im = ax.imshow(total, origin='lower', cmap='hot', aspect='auto',
                   norm=LogNorm(vmin=np.percentile(total[total>0],5),
                                vmax=np.percentile(total[total>0],99)))
    X, Y = np.meshgrid(np.arange(c1.shape[1]), np.arange(c1.shape[0]))
    ax.contour(X, Y, c1_frac, levels=[0.3, 0.5, 0.7],
               colors=['cyan','cyan','cyan'], linewidths=[0.8,1.5,0.8],
               linestyles=['--','-','--'])
    ax.contour(X, Y, 1-c1_frac, levels=[0.3, 0.5, 0.7],
               colors=['lime','lime','lime'], linewidths=[0.8,1.5,0.8],
               linestyles=['--','-','--'])

    cy,cx = c1.shape[0]//2, c1.shape[1]//2
    c1_px = cx + int(0.124/0.06); c1_py = cy + int(0.139/0.06)
    c2_px = cx + int(-0.116/0.06); c2_py = cy + int(0.169/0.06)
    ax.plot(c1_px,c1_py,'+',color='cyan',markersize=10,markeredgewidth=2)
    ax.plot(c2_px,c2_py,'x',color='lime',markersize=8,markeredgewidth=2)
    ax.set_title(f'{band}  <C1/(C1+C2)>={np.nanmean(c1_frac):.3f}',fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig2.suptitle('Total Model + Dominance Contours\nCyan=C1(Center) --30% -50% --70%  |  Green=C2(West) --30% -50% --70%',
              fontsize=12, y=0.99)
plt.tight_layout(rect=[0,0.02,1,0.94])
out2 = f'{OUT}/separation_contours.png'
plt.savefig(out2, dpi=200, bbox_inches='tight')
print(f'Saved: {out2}')
plt.close()

# Print summary
print("\n=== Mean C1 fraction per band ===")
for band in BANDS:
    total = c1_imgs[band] + c2_imgs[band]
    total[total<=0] = 1e-30
    c1f = np.nanmean(c1_imgs[band] / total)
    print(f"  {band}: <C1/(C1+C2)> = {c1f:.3f}")
