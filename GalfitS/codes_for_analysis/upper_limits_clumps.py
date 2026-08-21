"""
Estimate 5-sigma upper limits for North, South, Southeast clumps in 9 MIRI bands.
Uses sigma-clipped RMS in small apertures at each clump position.
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

CUTOUT_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v3/images'
RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373

# Clump sky coordinates from NIRCam notebook (IFU WCS conversion)
clumps = ['North', 'South', 'Southeast']

# IFU WCS — use known sky coords from the notebook output
# These were already computed in the NIRCam notebook:
clump_sky = {
    'North':     (149.742756, 1.655655),
    'South':     (149.742773, 1.655292),
    'Southeast': (149.742972, 1.655440),
}
for name in clumps:
    print(f"{name}: RA={clump_sky[name][0]:.6f}, Dec={clump_sky[name][1]:.6f}")

MIRI_BANDS = ['F560W','F770W','F1000W','F1130W','F1280W',
              'F1500W','F1800W','F2100W','F2550W']
PIXAR_SR = 8.46159499407524e-14
CONV_uJy = PIXAR_SR * 1e12

print(f"\n{'Band':<10}", end='')
for name in clumps:
    print(f" {'RMS':>8} {'5sig':>10} {'5sig(uJy)':>12}", end='')
print()
print("-" * 100)

results = {}
for band in MIRI_BANDS:
    fpath = f'{CUTOUT_DIR}/cos87259_{band}_cut.fits'
    with fits.open(fpath) as hdu:
        sci = hdu[0].data.astype(np.float64)
        header = hdu[0].header
        w = WCS(header)

    results[band] = {}
    print(f"{band:<10}", end='')

    for name in clumps:
        ra, dec = clump_sky[name]
        px, py = w.world_to_pixel_values(ra, dec)
        px, py = int(float(px)), int(float(py))

        # Small aperture: 3x3 pixel box (MIRI PSF is larger but this gives RMS)
        half = 5  # ~0.3" radius
        x1 = max(0, px - half); x2 = min(sci.shape[1], px + half)
        y1 = max(0, py - half); y2 = min(sci.shape[0], py + half)
        ap = sci[y1:y2, x1:x2]
        ap = ap[np.isfinite(ap)]

        # Sigma-clipped RMS (exclude bright sources in aperture)
        mean, median, std = sigma_clipped_stats(ap, sigma=3.0, maxiters=5)
        rms = std

        # 5-sigma upper limit in MJy/sr
        ul_mjysr = 5.0 * rms
        # Convert to uJy (per pixel)
        ul_uJy = ul_mjysr * CONV_uJy

        results[band][name] = {'rms': rms, 'ul_5sig': ul_mjysr, 'ul_uJy': ul_uJy}
        print(f" {rms:8.4f} {ul_mjysr:10.4f} {ul_uJy:12.4f}", end='')
    print()

# Summary
print(f"\n=== 5-sigma Upper Limits (uJy) ===")
print(f"{'Band':<10} {'North':<12} {'South':<12} {'Southeast':<12}")
print("-" * 50)
for band in MIRI_BANDS:
    print(f"{band:<10}", end='')
    for name in clumps:
        ul = results[band][name]['ul_uJy']
        print(f" {ul:<12.4f}", end='')
    print()
