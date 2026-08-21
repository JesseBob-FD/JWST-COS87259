import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# MIRI
h = fits.open('MIRI_cutout_v3/images/cos87259_F770W_cut.fits')
w = WCS(h[0].header)
pc = w.wcs.get_pc()
print('MIRI F770W PC:', pc[0,0], pc[0,1])
cx,cy = w.wcs.crpix
ra1,dec1 = w.all_pix2world(cx,cy,0)
ra2,dec2 = w.all_pix2world(cx+10,cy,0)
dra = 3600*(ra2-ra1)*np.cos(np.radians(dec1))
print(f'dRA for +10px in MIRI: {dra:.3f} arcsec  (+ = east/right)')
h.close()

# NIRCam
h2 = fits.open('NIRCam/F410M_cos87259_sci.fits')
w2 = WCS(h2[1].header)
pc2 = w2.wcs.get_pc()
print('NIRCam F410M PC:', pc2[0,0], pc2[0,1])
cx2,cy2 = w2.wcs.crpix
ra1_2,dec1_2 = w2.all_pix2world(cx2,cy2,0)
ra2_2,dec2_2 = w2.all_pix2world(cx2+10,cy2,0)
dra2 = 3600*(ra2_2-ra1_2)*np.cos(np.radians(dec1_2))
print(f'dRA for +10px in NIRCam: {dra2:.3f} arcsec')
h2.close()

# Check MIRI cutout mask patterns
print("\n=== MIRI mask stats ===")
for band in ['F560W','F770W','F1130W','F2550W']:
    h = fits.open(f'MIRI_cutout_v3/images/cos87259_{band}_cut.fits')
    mask = h[1].data
    sci = h[0].data
    print(f'{band}: mask_pct={100*np.sum(mask>0)/mask.size:.1f}%  '
          f'sci_median={np.median(sci):.4f} sci_std={np.std(sci):.4f}  '
          f'center_5px_masked={np.sum(mask[28:38,28:38]>0)}/100 pixels')
    h.close()
