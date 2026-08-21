"""
Extract per-component fluxes using cal_model_image() for:
  1. two_sersic_C1C2/noSED  (C1=bulge/Center, C2=disk/West)
  2. C1host_C2AGN/noSED     (host at C1, AGN at C2)
  3. C2host_C1AGN/noSED     (host at C2, AGN at C1)

Then plot all SEDs on one figure with upper limits.
"""
import os, sys, numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
waves_um = np.array([5.6,7.7,10.0,11.3,12.8,15.0,18.0,21.0,25.5])
PIXAR_SR = 8.46159499407524e-14
CONV_uJy = PIXAR_SR * 1e12
BASE = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result'
OUT  = os.path.join(BASE, 'MIRI_clump_departure')

# Upper limits (from previous run)
ul = {
    'North': np.array([0.028,0.039,0.072,0.086,0.137,0.243,0.500,0.545,0.895]),
    'South': np.array([0.017,0.015,0.033,0.043,0.074,0.060,0.164,0.211,0.608]),
    'Southeast': np.array([0.015,0.017,0.036,0.055,0.059,0.087,0.158,0.197,0.615]),
}

def load_fit(work_subdir):
    """Load fitter with best-fit params."""
    wd = os.path.join(BASE, work_subdir)
    config = None
    for f in os.listdir(wd):
        if f.endswith('.lyric'): config = os.path.join(wd, f); break
    gsfile = None
    for f in os.listdir(wd):
        if f.endswith('.gssummary'): gsfile = os.path.join(wd, f); break

    fitter, targ, fs = gsutils.read_config_file(config, wd)
    smfile = Table.read(gsfile, format='ascii')
    for row in smfile:
        if row['pname'] in fitter.lmParameters:
            fitter.lmParameters[row['pname']].set(
                value=row['best_value'],
                min=row['best_value']-0.1, max=row['best_value']+0.1)
    fitter.loose_fix_pars()
    for key in fitter.pardict:
        if key in fitter.lmParameters:
            fitter.pardict[key] = fitter.lmParameters[key].value
    return fitter

def extract_two_sersic(fitter):
    """Extract C1(bulge) and C2(disk) from two-Sersic noSED fit."""
    gal = [m for i,m in enumerate(fitter.model_list) if fitter.mtype_list[i]=='galaxy'][0]
    c1n, c2n = gal.subnames[0], gal.subnames[1]
    print(f"  Components: {c1n}, {c2n}")

    # Total
    fitter.cal_model_image()
    totals = []
    for i in range(fitter.GSdata.Nimages):
        totals.append(np.array(fitter.GSdata.get_image(i).model_image))

    # C2-only (zero C1)
    saved = {}
    for jb in MIRI:
        k = f'logNorm_{c1n}_{jb}'; saved[jb] = fitter.pardict.get(k,0)
        fitter.pardict[k] = -30.0
    fitter.cal_model_image()
    c2s = [np.array(fitter.GSdata.get_image(i).model_image) for i in range(fitter.GSdata.Nimages)]

    # C1-only (restore C1, zero C2)
    for jb in MIRI:
        fitter.pardict[f'logNorm_{c1n}_{jb}'] = saved[jb]
        k2 = f'logNorm_{c2n}_{jb}'; saved[jb] = fitter.pardict.get(k2,0)
        fitter.pardict[k2] = -30.0
    fitter.cal_model_image()
    c1s = [np.array(fitter.GSdata.get_image(i).model_image) for i in range(fitter.GSdata.Nimages)]

    # Restore
    for jb in MIRI:
        fitter.pardict[f'logNorm_{c2n}_{jb}'] = saved[jb]

    return measure_fluxes(fitter, c1s, c2s, totals)

def extract_host_agn(fitter):
    """Extract host and AGN using ndisagn trick."""
    # Total
    fitter.cal_model_image()
    totals = [np.array(fitter.GSdata.get_image(i).model_image)
              for i in range(fitter.GSdata.Nimages)]

    # Host-only (ndisagn=True -> skip AGN)
    fitter.cal_model_image(ndisagn=True)
    hosts = [np.array(fitter.GSdata.get_image(i).model_image)
             for i in range(fitter.GSdata.Nimages)]

    # AGN = total - host
    agns = [t - h for t, h in zip(totals, hosts)]

    return measure_fluxes(fitter, hosts, agns, totals)

def measure_fluxes(fitter, comp1_imgs, comp2_imgs, total_imgs):
    """Sum fluxes over MIRI bands in uJy."""
    c1_f, c2_f, tot_f = [], [], []
    for i in range(fitter.GSdata.Nimages):
        im = fitter.GSdata.get_image(i)
        if not im.band.startswith('miri_'): continue
        mask = im.cut_mask_image
        good = (mask == 0) & np.isfinite(comp1_imgs[i]) & np.isfinite(comp2_imgs[i])
        c1_f.append(np.sum(comp1_imgs[i][good]) * CONV_uJy)
        c2_f.append(np.sum(comp2_imgs[i][good]) * CONV_uJy)
        tot_f.append(np.sum(total_imgs[i][good]) * CONV_uJy)
    return np.array(c1_f), np.array(c2_f), np.array(tot_f)

# ===== Run extraction for all 3 fits =====
results = {}

# 1) two_sersic
print("=== two_sersic_C1C2/noSED ===")
ft = load_fit('two_sersic_C1C2/noSED')
results['two_sersic'] = {
    'c1': extract_two_sersic(ft), 'label1': 'Center (C1=bulge)', 'label2': 'West (C2=disk)',
    'color1': 'royalblue', 'color2': 'crimson'
}

# 2) C1host_C2AGN
print("=== C1host_C2AGN/noSED ===")
ft = load_fit('fixed_center_C1host_C2AGN/noSED')
c1h, c2a, tot = extract_host_agn(ft)
results['C1host_C2AGN'] = {
    'c1': (c1h, c2a, tot), 'label1': 'Host at C1 (Center)', 'label2': 'AGN at C2 (West)',
    'color1': 'teal', 'color2': 'orange'
}

# 3) C2host_C1AGN
print("=== C2host_C1AGN/noSED ===")
ft = load_fit('fixed_center_C2host_C1AGN/noSED')
c2h, c1a, tot = extract_host_agn(ft)
results['C2host_C1AGN'] = {
    'c1': (c2h, c1a, tot), 'label1': 'Host at C2 (West)', 'label2': 'AGN at C1 (Center)',
    'color1': 'purple', 'color2': 'brown'
}

# ===== Save CSVs =====
for name, r in results.items():
    c1f, c2f, totf = r['c1']
    csv_path = os.path.join(OUT, f'clump_fluxes_{name}.csv')
    with open(csv_path, 'w') as f:
        f.write('band,comp1_uJy,comp2_uJy,total_uJy\n')
        for i, b in enumerate(BANDS):
            f.write(f'{b},{c1f[i]:.4f},{c2f[i]:.4f},{totf[i]:.4f}\n')
    print(f'Saved: {csv_path}')

# ===== Plot: all SEDs + upper limits =====
fig, ax = plt.subplots(figsize=(12, 8))

for name, r in results.items():
    c1f, c2f, totf = r['c1']
    lbl1 = f"{r['label1']} [{name}]"
    lbl2 = f"{r['label2']} [{name}]"
    ax.errorbar(waves_um, c1f, yerr=0.1*c1f, fmt='o', color=r['color1'],
                markersize=7, capsize=3, label=lbl1, alpha=0.8)
    ax.errorbar(waves_um, c2f, yerr=0.1*c2f, fmt='s', color=r['color2'],
                markersize=7, capsize=3, label=lbl2, alpha=0.8)

# Upper limits
ul_colors = {'North': 'gray', 'South': 'silver', 'Southeast': 'darkgray'}
for name, vals in ul.items():
    ax.scatter(waves_um, vals, marker='v', color=ul_colors[name], s=40, alpha=0.7)
    ax.plot(waves_um, vals, ':', color=ul_colors[name], alpha=0.5, lw=0.8)
    ax.plot([], [], 'v', color=ul_colors[name], markersize=7, label=f'{name} (<5σ)')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Observed Wavelength (μm)', fontsize=12)
ax.set_ylabel('F$_\\nu$ (μJy)', fontsize=12)
ax.set_title('COS-87259 (z=6.83): All Component SEDs + Upper Limits', fontsize=14)
ax.legend(fontsize=7, loc='upper left', ncol=2)
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
out = os.path.join(OUT, 'clump_seds_all.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f'Saved: {out}')
plt.close()
print('Done.')
