"""Convert C1, C2 pixel coords to sky coords and arcsec offsets for GalfitS."""
import numpy as np
from astropy.io import fits
from astropy import wcs

band = 'F410M'
file_path = f'NIRCam/{band}_cos87259_sci.fits'
RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373

with fits.open(file_path) as hdul:
    header = hdul['SCI'].header
    nircam_wcs = wcs.WCS(header)

# Clump pixel coordinates (global)
clumps = {'C1': (4740, 3457), 'C2': (4748, 3458)}

for name, (px, py) in clumps.items():
    ra, dec = nircam_wcs.pixel_to_world_values(px, py)
    ra, dec = float(ra), float(dec)
    dra = (ra - RA_TARGET) * 3600 * np.cos(np.radians(DEC_TARGET))
    ddec = (dec - DEC_TARGET) * 3600
    print(f'{name}: pixel ({px},{py}) -> RA={ra:.8f} Dec={dec:.8f}')
    print(f'      offset from fitting center: ΔRA={dra:.4f}", ΔDec={ddec:.4f}"')
    print(f'      GalfitS P_3, P_4: [{dra:.4f}, ...], [{ddec:.4f}, ...]')
    print()
