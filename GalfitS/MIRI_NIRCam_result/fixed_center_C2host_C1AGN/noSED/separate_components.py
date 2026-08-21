"""
Separate host (C2) and AGN (C1) components from C2host_C1AGN noSED fit.
Same approach as C1host_C2AGN: generates per-component model images via
GalfitS API, then measures total flux. No GalfitS source modifications.
"""
import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table

GALFITS_SRC = '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src'
sys.path.insert(0, GALFITS_SRC)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import galfits.gsutils as gsutils

# JAX big-endian fix
_old = jnp.array
def _safe(obj, *a, **kw):
    if isinstance(obj, np.ndarray) and obj.dtype.byteorder in ('>', 'B'):
        obj = obj.byteswap().view(obj.dtype.newbyteorder('='))
    return _old(obj, *a, **kw)
jnp.array = _safe

WORK_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/fixed_center_C2host_C1AGN/noSED'
CONFIG = os.path.join(WORK_DIR, 'C2host_C1AGN_noSED.lyric')
PARAMS = os.path.join(WORK_DIR, 'C2host_C1AGN_noSED.params')

print("Loading config + params...")
fitter, targ, fs = gsutils.read_config_file(CONFIG, WORK_DIR)
ptab = Table.read(PARAMS, format='ascii')
for row in ptab:
    fitter.pardict[row['name']] = row['value']

gal_models = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='galaxy']
agn_models = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='agn']

MIRI = ['miri_f560w','miri_f770w','miri_f1000w','miri_f1130w',
        'miri_f1280w','miri_f1500w','miri_f1800w','miri_f2100w','miri_f2550w']

print(f"\n{'Band':<12} {'Host Flux':<16} {'AGN Flux':<16} {'Ratio H/A':<12} {'Host%':<8} {'AGN%':<8}")
print("-" * 80)

results = {}
for jband in MIRI:
    short = jband.replace('miri_', '').upper()
    im = None; aname = None
    for atl in fitter.GSdata.image_atlas_list:
        for i,b in enumerate(atl.band_list):
            if b == jband:
                im = atl[i]; aname = atl.name; break
        if im: break
    if im is None: continue

    if im.wcsshift:
        im.coordinates_transfer_para['x0shift'] = fitter.pardict.get(f'csx_{im.imglabel}',0)
        im.coordinates_transfer_para['y0shift'] = fitter.pardict.get(f'csy_{im.imglabel}',0)

    ny,nx = im.cut_image.shape
    noSED = (fitter.GSdata.fitMode == 'images - photometry')

    for m in gal_models:
        for c in m.subnames: m.update_subC_para(c, fitter.pardict)
        m.generate_mass_map((ny,nx), transpar=im.coordinates_transfer_para)

    # Host (galaxy) only
    host_im = jnp.zeros((ny,nx), dtype=jnp.float32)
    for m in gal_models:
        host_im += m.generate_image(jband, im.PSF, resp=im.resp,
                                     noSED=noSED, nebularpar=fitter.pardict)

    # AGN only
    agn_im = jnp.zeros((ny,nx), dtype=jnp.float32)
    for m in agn_models:
        ni = fitter.pardict.get(f'Ni_{m.prefix}_{aname}', 1.0)
        agn_im += ni * m.generate_image(
            [fitter.pardict[f'xcen_{m.prefix}'], fitter.pardict[f'ycen_{m.prefix}']],
            [ny,nx], jband, im.PSF, resp=im.resp,
            transpar=im.coordinates_transfer_para, pardict=fitter.pardict)

    host_im *= im.phys_to_counts_rate
    agn_im *= im.phys_to_counts_rate

    mask = im.cut_mask_image
    good = (mask==0) & np.isfinite(np.array(host_im)) & np.isfinite(np.array(agn_im))
    hf = float(jnp.sum(jnp.where(good, host_im, 0.)))
    af = float(jnp.sum(jnp.where(good, agn_im, 0.)))
    tt = hf + af
    hp = 100*hf/tt if tt>0 else 0; ap = 100*af/tt if tt>0 else 0
    ratio = hf/af if af>0 else float('inf')

    results[short] = {'host':hf,'agn':af,'total':tt,'h_img':np.array(host_im),'a_img':np.array(agn_im)}
    print(f"{short:<12} {hf:<16.6e} {af:<16.6e} {ratio:<12.3f} {hp:<8.1f} {ap:<8.1f}")

# Save
print("\nSaving...")
for s,d in results.items():
    fits.writeto(os.path.join(WORK_DIR, f'host_model_{s}.fits'), d['h_img'].astype(np.float32), overwrite=True)
    fits.writeto(os.path.join(WORK_DIR, f'agn_model_{s}.fits'),  d['a_img'].astype(np.float32), overwrite=True)

with open(os.path.join(WORK_DIR,'component_fluxes.csv'),'w') as f:
    f.write('band,host_flux,agn_flux,total,host_pct,agn_pct\n')
    for s in [m.replace('miri_','').upper() for m in MIRI]:
        if s in results:
            r = results[s]
            f.write(f'{s},{r["host"]:.6e},{r["agn"]:.6e},{r["total"]:.6e},{100*r["host"]/r["total"]:.2f},{100*r["agn"]/r["total"]:.2f}\n')
print("Done.")
