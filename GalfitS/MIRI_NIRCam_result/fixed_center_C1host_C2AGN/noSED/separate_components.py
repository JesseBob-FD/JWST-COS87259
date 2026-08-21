"""
Separate host and AGN components from C1host_C2AGN noSED fit.
Generates per-component model images via GalfitS Galaxy.generate_image()
and AGN.generate_image(), then measures total flux per component per band.

Does NOT modify any GalfitS source files.
"""
import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table

# Add GalfitS to path
GALFITS_SRC = '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src'
sys.path.insert(0, GALFITS_SRC)

# Must set JAX preallocation before import
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import galfits.gsutils as gsutils

# --- JAX big-endian fix (needed for FITS data) ---
_old_array = jnp.array
def _safe_jax_array(obj, *args, **kwargs):
    if isinstance(obj, np.ndarray) and obj.dtype.byteorder in ('>', 'B'):
        obj = obj.byteswap().view(obj.dtype.newbyteorder('='))
    return _old_array(obj, *args, **kwargs)
jnp.array = _safe_jax_array

# --- Config ---
WORK_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/fixed_center_C1host_C2AGN/noSED'
CONFIG_FILE = os.path.join(WORK_DIR, 'C1host_C2AGN_noSED.lyric')
PARAMS_FILE = os.path.join(WORK_DIR, 'C1host_C2AGN_noSED.params')

# --- Load fitting setup ---
print("Loading GalfitS config and params...")
fitter, targ, fs = gsutils.read_config_file(CONFIG_FILE, WORK_DIR)

ptab = Table.read(PARAMS_FILE, format='ascii')
for row in ptab:
    fitter.pardict[row['name']] = row['value']

# --- Identify galaxy and AGN models ---
galaxy_models = []
agn_models = []
for loop, model in enumerate(fitter.model_list):
    if fitter.mtype_list[loop] == 'galaxy':
        galaxy_models.append(model)
    elif fitter.mtype_list[loop] == 'agn':
        agn_models.append(model)

print(f"Galaxy models: {len(galaxy_models)}")
print(f"AGN models: {len(agn_models)}")

# --- Per-band separation ---
MIRI_LABELS = ['miri_f560w', 'miri_f770w', 'miri_f1000w', 'miri_f1130w',
               'miri_f1280w', 'miri_f1500w', 'miri_f1800w', 'miri_f2100w',
               'miri_f2550w']

print(f"\n{'Band':<12} {'Host Flux':<16} {'AGN Flux':<16} {'Ratio H/A':<12} {'Host%':<8} {'AGN%':<8}")
print("-" * 80)

results = {}
for jband in MIRI_LABELS:
    band_short = jband.replace('miri_', '').upper()

    # Find image object for this band
    im = None
    atlas_name = None
    for image_atlas in fitter.GSdata.image_atlas_list:
        for loop, b in enumerate(image_atlas.band_list):
            if b == jband:
                im = image_atlas[loop]
                atlas_name = image_atlas.name
                break
        if im is not None:
            break

    if im is None:
        print(f"{band_short:<12} SKIP")
        continue

    # Apply coordinate shifts if any
    if im.wcsshift:
        im.coordinates_transfer_para['x0shift'] = fitter.pardict.get(f'csx_{im.imglabel}', 0)
        im.coordinates_transfer_para['y0shift'] = fitter.pardict.get(f'csy_{im.imglabel}', 0)

    ny, nx = im.cut_image.shape
    noSED = (fitter.GSdata.fitMode == 'images - photometry')

    # --- Generate galaxy mass maps (needed for both) ---
    for model in galaxy_models:
        for com in model.subnames:
            model.update_subC_para(com, fitter.pardict)
        model.generate_mass_map((ny, nx), transpar=im.coordinates_transfer_para)

    # --- Host-only: galaxy models ---
    host_imm = jnp.zeros((ny, nx), dtype=jnp.float32)
    for model in galaxy_models:
        host_imm += model.generate_image(jband, im.PSF, resp=im.resp,
                                          noSED=noSED, nebularpar=fitter.pardict)

    # --- AGN-only: AGN models ---
    agn_imm = jnp.zeros((ny, nx), dtype=jnp.float32)
    for model in agn_models:
        ni_key = f'Ni_{model.prefix}_{atlas_name}'
        ni = fitter.pardict.get(ni_key, 1.0)
        agn_imm += ni * model.generate_image(
            [fitter.pardict[f'xcen_{model.prefix}'],
             fitter.pardict[f'ycen_{model.prefix}']],
            [ny, nx], jband, im.PSF, resp=im.resp,
            transpar=im.coordinates_transfer_para, pardict=fitter.pardict)

    # --- Convert to physical flux units (counts -> MJy/sr) ---
    host_imm = host_imm * im.phys_to_counts_rate
    agn_imm = agn_imm * im.phys_to_counts_rate

    # --- Measure flux (sum over unmasked pixels) ---
    mask = im.cut_mask_image
    good = (mask == 0) & np.isfinite(np.array(host_imm)) & np.isfinite(np.array(agn_imm))

    host_flux = float(jnp.sum(jnp.where(good, host_imm, 0.0)))
    agn_flux = float(jnp.sum(jnp.where(good, agn_imm, 0.0)))
    total = host_flux + agn_flux

    host_pct = 100 * host_flux / total if total > 0 else 0
    agn_pct = 100 * agn_flux / total if total > 0 else 0

    results[band_short] = {
        'host_flux': host_flux, 'agn_flux': agn_flux, 'total': total,
        'host_img': np.array(host_imm), 'agn_img': np.array(agn_imm)
    }

    ratio = host_flux / agn_flux if agn_flux > 0 else float('inf')
    print(f"{band_short:<12} {host_flux:<16.6e} {agn_flux:<16.6e} {ratio:<12.3f} {host_pct:<8.1f} {agn_pct:<8.1f}")

# --- Save results ---
print("\nSaving per-component model images...")
for band_short, data in results.items():
    band_upper = band_short.replace('miri_', '').upper()
    # Save host image
    fits.writeto(os.path.join(WORK_DIR, f'host_model_{band_upper}.fits'),
                 data['host_img'].astype(np.float32), overwrite=True)
    # Save AGN image
    fits.writeto(os.path.join(WORK_DIR, f'agn_model_{band_upper}.fits'),
                 data['agn_img'].astype(np.float32), overwrite=True)

# --- Save flux table ---
flux_table_path = os.path.join(WORK_DIR, 'component_fluxes.csv')
with open(flux_table_path, 'w') as f:
    f.write('band,host_flux,agn_flux,total,host_pct,agn_pct\n')
    for band_short in MIRI_LABELS:
        band_upper = band_short.replace('miri_', '').upper()
        if band_upper in results:
            r = results[band_upper]
            f.write(f'{band_upper},{r["host_flux"]:.6e},{r["agn_flux"]:.6e},'
                    f'{r["total"]:.6e},{100*r["host_flux"]/r["total"]:.2f},'
                    f'{100*r["agn_flux"]/r["total"]:.2f}\n')

print(f"Flux table saved to: {flux_table_path}")
print("Done.")
