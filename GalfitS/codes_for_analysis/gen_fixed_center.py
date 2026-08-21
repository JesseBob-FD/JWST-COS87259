"""Generate configs for fixed-center fitting (C1, C2 from F410M)."""
import os

BASE_NOSED = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/host_freeAGN/noSED/COS87259_R010_NPnoSED.lyric'
BASE_SED   = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/host_freeAGN/SED/COS87259_R009_NPSED.lyric'

# Clump offsets from fitting center (arcsec)
C1 = ('0.124', '0.139')
C2 = ('-0.116', '0.169')

configs = [
    ('C1host_C2AGN_noSED', 'fixed_center_C1host_C2AGN/noSED', BASE_NOSED, C1, C2),
    ('C1host_C2AGN_SED',   'fixed_center_C1host_C2AGN/SED',   BASE_SED,   C1, C2),
    ('C2host_C1AGN_noSED', 'fixed_center_C2host_C1AGN/noSED', BASE_NOSED, C2, C1),
    ('C2host_C1AGN_SED',   'fixed_center_C2host_C1AGN/SED',   BASE_SED,   C2, C1),
]

RESULT = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result'

for tag, subdir, base, host_xy, agn_xy in configs:
    with open(base) as f:
        content = f.read()

    # Replace the R1 tag
    old_tag = content.split('\n')[0].replace('# ', '').strip()
    content = content.replace(old_tag, tag)

    # Fix host center (Pa3=ΔRA, Pa4=ΔDec): force to clump position, vary=0
    # Original: Pa3) [0, -1.5, 1.5, 0.01, 1]
    # New:      Pa3) [0.124, -1.5, 1.5, 0.01, 0]
    import re
    # Host x-center (Pa3)
    content = re.sub(r'(Pa3\)\s*\[)[^\]]+(\])',
                     f'Pa3) [{host_xy[0]}, -1.5, 1.5, 0.01, 0]',
                     content)
    # Host y-center (Pa4)
    content = re.sub(r'(Pa4\)\s*\[)[^\]]+(\])',
                     f'Pa4) [{host_xy[1]}, -1.5, 1.5, 0.01, 0]',
                     content)
    # AGN x-center (Na4)
    content = re.sub(r'(Na4\)\s*\[)[^\]]+(\])',
                     f'Na4) [{agn_xy[0]}, -1.5, 1.5, 0.01, 0]',
                     content)
    # AGN y-center (Na5)
    content = re.sub(r'(Na5\)\s*\[)[^\]]+(\])',
                     f'Na5) [{agn_xy[1]}, -1.5, 1.5, 0.01, 0]',
                     content)

    outdir = os.path.join(RESULT, subdir)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'{tag}.lyric')
    with open(outpath, 'w') as f:
        f.write(content)
    print(f'Created: {outpath}')

print('Done.')
