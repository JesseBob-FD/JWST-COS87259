"""
For C1host_C2AGN/noSED and C2host_C1AGN/noSED:
  Use Ni_AGN_*=0 trick to isolate host (photometric fitter has no ndisagn).
  Then AGN = total - host.
  1) Component model images (host / AGN / residual)
  2) Individual SEDs with power-law fits
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

MIRI = ['miri_f560w','miri_f770w','miri_f1000w','miri_f1130w',
        'miri_f1280w','miri_f1500w','miri_f1800w','miri_f2100w','miri_f2550w']
BANDS = ['F560W','F770W','F1000W','F1130W','F1280W','F1500W','F1800W','F2100W','F2550W']
WAVES = ['5.6um','7.7um','10.0um','11.3um','12.8um','15.0um','18.0um','21.0um','25.5um']
waves_um = np.array([5.6,7.7,10.0,11.3,12.8,15.0,18.0,21.0,25.5])
PIXAR_SR = 8.46159499407524e-14; CONV_uJy = PIXAR_SR * 1e12
BASE = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result'
OUT  = os.path.join(BASE, 'MIRI_clump_departure')

def load_fit(subdir):
    wd = os.path.join(BASE, subdir)
    config = [f for f in os.listdir(wd) if f.endswith('.lyric')][0]
    gsfile  = [f for f in os.listdir(wd) if f.endswith('.gssummary')][0]
    fitter, _, _ = gsutils.read_config_file(os.path.join(wd, config), wd)
    sm = Table.read(os.path.join(wd, gsfile), format='ascii')
    for row in sm:
        if row['pname'] in fitter.lmParameters:
            fitter.lmParameters[row['pname']].set(
                value=row['best_value'], min=row['best_value']-0.1, max=row['best_value']+0.1)
    fitter.loose_fix_pars()
    for k in fitter.pardict:
        if k in fitter.lmParameters: fitter.pardict[k] = fitter.lmParameters[k].value
    return fitter

for fit_name, host_label, agn_label, host_color, agn_color in [
    ('C1host_C2AGN', 'Host at C1 (Center)', 'AGN at C2 (West)', 'teal', 'orange'),
    ('C2host_C1AGN', 'Host at C2 (West)', 'AGN at C1 (Center)', 'purple', 'brown'),
]:
    subdir = f'fixed_center_{fit_name}/noSED'
    print(f"=== {fit_name} ===")
    fitter = load_fit(subdir)

    # In imagefitter_phot, AGN is in psfmodel_list (NOT gmodel_list).
    # The AGN's generate_image() uses pardict['AGN_{band}_logL'] directly.
    # To isolate host: zero all AGN per-band logL in pardict.

    # 1) Total model
    fitter.cal_model_image()
    totals, data_imgs, sigma_imgs, mask_imgs = [], [], [], []
    for i in range(fitter.GSdata.Nimages):
        im = fitter.GSdata.get_image(i)
        sky_mean, sky_median, sky_std = sigma_clipped_stats(im.cut_image, sigma=3.0, maxiters=5)
        data_imgs.append(np.array(im.cut_image) - sky_median)
        totals.append(np.array(im.model_image))
        sigma_imgs.append(np.array(im.cut_sigma_image))
        mask_imgs.append(np.array(im.cut_mask_image, dtype=float))

    # 2) Host-only: zero all AGN per-band logL values
    agn_logl_keys = [k for k in fitter.pardict if 'AGN_' in k and '_logL' in k and 'torus' not in k]
    saved_logl = {}
    for k in agn_logl_keys:
        saved_logl[k] = fitter.pardict[k]
        fitter.pardict[k] = -30.0
    fitter.cal_model_image()
    hosts = [np.array(fitter.GSdata.get_image(i).model_image) for i in range(fitter.GSdata.Nimages)]

    # 3) AGN = total - host
    agns = [t - h for t, h in zip(totals, hosts)]

    # Restore
    for k, v in saved_logl.items():
        fitter.pardict[k] = v

    # Filter MIRI
    miri_idx = [i for i in range(fitter.GSdata.Nimages)
                if fitter.GSdata.get_image(i).band.startswith('miri_')]

    # Measure fluxes
    host_f, agn_f = [], []
    for i in miri_idx:
        im = fitter.GSdata.get_image(i)
        mask = im.cut_mask_image
        good = (mask==0) & np.isfinite(hosts[i]) & np.isfinite(agns[i])
        host_f.append(np.sum(hosts[i][good]) * CONV_uJy)
        agn_f.append(np.sum(agns[i][good]) * CONV_uJy)
    host_f = np.array(host_f); agn_f = np.array(agn_f)
    total_f = host_f + agn_f
    # Handle negative AGN (shouldn't happen if Ni trick worked)
    agn_f = np.maximum(agn_f, 0)

    # --- 1) Model images ---
    nrows = len(miri_idx)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5*nrows), squeeze=True)
    for row, i in enumerate(miri_idx):
        im = fitter.GSdata.get_image(i)
        bn = im.band.replace('miri_','').upper()
        data = data_imgs[i]; sigma = sigma_imgs[i]; mask = mask_imgs[i]
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, maxiters=5)
        immin = 5*sky_std; immax = float(jnp.nanmax(data)); frac = 0.4
        m_vis = np.ma.masked_where(mask==0, mask)
        ny, nx = data.shape

        for col, (img, title) in enumerate([
            (hosts[i], 'Host'), (agns[i], 'AGN'),
            ((data - totals[i])/sigma, 'Residual/σ')]):
            ax = axes[row, col]
            if 'Residual' in title:
                cb = ax.imshow(np.where(sigma>0, img, 0), cmap='seismic', origin='lower',
                               vmin=-10, vmax=10, interpolation='nearest')
            else:
                ax.imshow(gsutils.normimg(img, immin, immax, frac=frac), cmap='seismic',
                          origin='lower', vmin=-1, vmax=1, interpolation='nearest')
            ax.imshow(m_vis, cmap='Blues', origin='lower', alpha=0.5,
                      interpolation='nearest', vmin=0, vmax=1)
            if row==0: ax.text(0.03*nx, 0.85*ny, title, size=18, color='k', weight='bold')
            ax.text(0.03*nx, 0.06*ny, bn, size=18, color='k', weight='light')
            ax.set_xticks([]); ax.set_yticks([])
            if col==2 and row==0:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('top', size='5%', pad=0.05)
                plt.colorbar(cb, cax=cax, orientation='horizontal')

    plt.tight_layout()
    img_out = os.path.join(OUT, f'component_model_images_{fit_name}.png')
    plt.savefig(img_out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Image: {img_out}')

    # --- 2) SED plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(waves_um, total_f, yerr=0.1*total_f, fmt='ko', label='Total', capsize=3)
    ax.errorbar(waves_um, host_f, yerr=0.1*host_f, fmt='o', color=host_color,
                markersize=9, capsize=3, label=host_label)
    ax.errorbar(waves_um, agn_f, yerr=0.1*agn_f, fmt='s', color=agn_color,
                markersize=9, capsize=3, label=agn_label)

    nu_0 = 2.99792458e14 / 1.0; freqs = 2.99792458e14 / waves_um
    x = np.log10(freqs/nu_0); Amat = np.column_stack([np.ones_like(x), -x])
    for flux, color, name in [(host_f, host_color, host_label), (agn_f, agn_color, agn_label)]:
        fpos = np.maximum(flux, 1e-10)
        w = fpos/np.max(fpos)
        try:
            cov = np.linalg.inv(Amat.T @ np.diag(w) @ Amat)
            p = cov @ Amat.T @ np.diag(w) @ np.log10(fpos)
            A, alpha = 10**p[0], p[1]
            wl_f = np.logspace(np.log10(5), np.log10(27), 100)
            nu_f = 2.99792458e14 / wl_f
            ax.plot(wl_f, A*(nu_f/nu_0)**(-alpha), '--', color=color, lw=1.5,
                    label=f'{name}: α={alpha:.2f}')
        except: pass

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
    ax.set_ylabel('F$_\\nu$ (μJy)', fontsize=12)
    ax.set_title(f'COS-87259: {fit_name} Component SEDs', fontsize=14)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both'); ax.set_xlim(4.5, 28)
    ax2 = ax.twiny(); ax2.set_xscale('log')
    tick_um = np.array([0.6,0.8,1.0,1.5,2.0,2.5,3.0])
    ax2.set_xticks(tick_um*(1+6.83)); ax2.set_xticklabels([f'{t:.1f}' for t in tick_um])
    ax2.set_xlim(ax.get_xlim()); ax2.set_xlabel('Rest-Frame Wavelength (μm)', fontsize=10)
    plt.tight_layout()
    sed_out = os.path.join(OUT, f'clump_seds_{fit_name}.png')
    plt.savefig(sed_out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  SED: {sed_out}')

    # Flux table
    print(f"  {'Band':<10} {'Host':<14} {'AGN':<14} {'Total':<14} {'Host%':<8} {'AGN%':<8}")
    for i, b in enumerate(BANDS):
        hp = 100*host_f[i]/total_f[i] if total_f[i]>0 else 0
        ap = 100*agn_f[i]/total_f[i] if total_f[i]>0 else 0
        print(f"  {b:<10} {host_f[i]:<14.4f} {agn_f[i]:<14.4f} {total_f[i]:<14.4f} {hp:<8.1f} {ap:<8.1f}")

print('Done.')
