"""
Separate C1 (Center=bulge) and C2 (West=disk) from two-Sersic fit.
Uses mass_map * 10^logNorm directly — avoids Galaxy.generate_image() issues.
"""
import os, sys, numpy as np
from astropy.io import fits
from astropy.table import Table

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
CONFIG = os.path.join(WORK, 'C1C2_noSED.lyric')
PARAMS = os.path.join(WORK, 'C1C2_noSED.params')

print("Loading...")
fitter, targ, fs = gsutils.read_config_file(CONFIG, WORK)
ptab = Table.read(PARAMS, format='ascii')
for row in ptab:
    fitter.pardict[row['name']] = row['value']

gal = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='galaxy'][0]
c1_name, c2_name = gal.subnames[0], gal.subnames[1]
print(f"Components: C1=Center={c1_name}, C2=West={c2_name}")

# Check actual pardict logNorm values
for key in sorted(fitter.pardict.keys()):
    if 'logNorm' in key and ('miri_f770w' in key or 'miri_f2100w' in key):
        print(f"  {key} = {fitter.pardict[key]}")

MIRI = ['miri_f560w','miri_f770w','miri_f1000w','miri_f1130w',
        'miri_f1280w','miri_f1500w','miri_f1800w','miri_f2100w','miri_f2550w']

PIXAR_SR = 8.46159499407524e-14
CONV_uJy = PIXAR_SR * 1e12

print(f"\n{'Band':<12} {'C1=Center(uJy)':<18} {'C2=West(uJy)':<18} {'Ratio C1/C2':<14} {'C1%':<8} {'C2%':<8}")
print("-" * 85)

results = {}
for jband in MIRI:
    short = jband.replace('miri_', '').upper()
    im = None
    for atl in fitter.GSdata.image_atlas_list:
        for i,b in enumerate(atl.band_list):
            if b == jband: im = atl[i]; break
        if im: break
    if im is None: continue

    if im.wcsshift:
        im.coordinates_transfer_para['x0shift'] = fitter.pardict.get(f'csx_{im.imglabel}',0)
        im.coordinates_transfer_para['y0shift'] = fitter.pardict.get(f'csy_{im.imglabel}',0)

    ny,nx = im.cut_image.shape
    noSED = (fitter.GSdata.fitMode == 'images - photometry')

    # Update and generate mass maps
    for c in gal.subnames: gal.update_subC_para(c, fitter.pardict)
    totalmass, apertures = gal.generate_mass_map((ny,nx), transpar=im.coordinates_transfer_para)

    # Get individual mass maps
    mm1 = gal.mass_map[c1_name]  # C1=Center=bulge
    mm2 = gal.mass_map[c2_name]  # C2=West=disk

    # Get logNorm values
    ln1 = fitter.pardict.get(f'logNorm_{c1_name}_{jband}', -30)
    ln2 = fitter.pardict.get(f'logNorm_{c2_name}_{jband}', -30)

    # Compute per-component model image (before phys_to_counts_rate)
    c1_imm = (10**ln1) * mm1
    c2_imm = (10**ln2) * mm2

    # PSF convolution — generate_image normally does this, but mass_map
    # already IS PSF-convolved? Let me check...
    # Actually generate_image() does: imm += flux * mass_map[key]
    # where mass_map is already convolved with PSF. So the above is correct.

    # Apply phys_to_counts_rate
    c1_imm *= im.phys_to_counts_rate
    c2_imm *= im.phys_to_counts_rate

    mask = im.cut_mask_image
    good = (mask==0) & np.isfinite(np.array(c1_imm)) & np.isfinite(np.array(c2_imm))
    f1 = float(jnp.sum(jnp.where(good, c1_imm, 0.)))
    f2 = float(jnp.sum(jnp.where(good, c2_imm, 0.)))

    # Convert to uJy
    f1_uJy = f1 * CONV_uJy
    f2_uJy = f2 * CONV_uJy
    tt = f1_uJy + f2_uJy
    p1 = 100*f1_uJy/tt if tt>0 else 0
    p2 = 100*f2_uJy/tt if tt>0 else 0

    results[short] = {'C1':f1_uJy, 'C2':f2_uJy, 'total':tt}
    ratio = f1_uJy/f2_uJy if f2_uJy>0 else float('inf')
    print(f"{short:<12} {f1_uJy:<18.4f} {f2_uJy:<18.4f} {ratio:<14.3f} {p1:<8.1f} {p2:<8.1f}")

# Save
with open(os.path.join(WORK,'component_fluxes.csv'),'w') as f:
    f.write('band,C1_Center_uJy,C2_West_uJy,total_uJy,C1_pct,C2_pct\n')
    for s in [m.replace('miri_','').upper() for m in MIRI]:
        if s in results:
            r = results[s]
            f.write(f'{s},{r["C1"]:.4f},{r["C2"]:.4f},{r["total"]:.4f},'
                    f'{100*r["C1"]/r["total"]:.2f},{100*r["C2"]/r["total"]:.2f}\n')
print("\nDone.")
