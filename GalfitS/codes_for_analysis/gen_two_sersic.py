"""Generate two-Sersic C1+C2 configs (no AGN)."""
import os, re

BASE = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result'

for mode in ['noSED', 'SED']:
    src_dir = os.path.join(BASE, 'bulge_disk', mode)
    src_files = [f for f in os.listdir(src_dir) if f.endswith('.lyric')]
    if not src_files: continue
    src = os.path.join(src_dir, src_files[0])

    with open(src) as f:
        content = f.read()

    # Fix centers
    content = re.sub(r'Pa3\) \[0, -1\.5, 1\.5, 0\.01, 1\]',
                     'Pa3) [0.124, -1.5, 1.5, 0.01, 0]', content)
    content = re.sub(r'Pa4\) \[0, -1\.5, 1\.5, 0\.01, 1\]',
                     'Pa4) [0.139, -1.5, 1.5, 0.01, 0]', content)
    content = re.sub(r'Pb3\) \[0, -1\.5, 1\.5, 0\.01, 1\]',
                     'Pb3) [-0.116, -1.5, 1.5, 0.01, 0]', content)
    content = re.sub(r'Pb4\) \[0, -1\.5, 1\.5, 0\.01, 1\]',
                     'Pb4) [0.169, -1.5, 1.5, 0.01, 0]', content)

    # Update tag
    content = content.replace(f'COS87259_R003_{mode}', f'C1C2_{mode}')

    out_dir = os.path.join(BASE, 'two_sersic_C1C2', mode)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'C1C2_{mode}.lyric')
    with open(out, 'w') as f:
        f.write(content)
    print(f'Created: {out}')
    # Verify
    for line in content.split('\n'):
        if 'Pa3)' in line or 'Pa4)' in line or 'Pb3)' in line or 'Pb4)' in line:
            print(f'  {line.strip()}')
    print()
