#!/usr/bin/env python3
"""Reprocess MIRI cutouts for MIRI+NIRCam joint fitting.

Changes from v2:
  - Cutout half-size: 2 arcsec (was 5 arcsec)
  - All 9 MIRI bands (filter files now available for F560W, F1130W, F2550W)
"""
import os, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
warnings.filterwarnings('ignore')

MIRI_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI'
IMAGE_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v3/images'
PSF_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v3/psf'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PSF_DIR, exist_ok=True)

RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373
CUTOUT_HALF_SIZE = 2.0  # arcsec (was 5.0)

BANDS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
         'F1500W', 'F1800W', 'F2100W', 'F2550W']

# MIRI PSF FWHM (Gaussian approximation, FWHM = lambda / D)
MIRI_FWHM = {
    'F560W': 0.20, 'F770W': 0.27, 'F1000W': 0.33,
    'F1130W': 0.36, 'F1280W': 0.40, 'F1500W': 0.45,
    'F1800W': 0.54, 'F2100W': 0.63, 'F2550W': 0.77,
}

print(f"Reprocessing MIRI with cutout half-size = {CUTOUT_HALF_SIZE} arcsec")
print(f"Target: RA={RA_TARGET:.6f} Dec={DEC_TARGET:.6f}\n")

for band in BANDS:
    fpath = os.path.join(MIRI_DIR, f'cos87259_{band}_60mas.fits.gz')
    if not os.path.exists(fpath):
        print(f'{band:8s} - SKIP (file not found)')
        continue

    with fits.open(fpath) as hdu:
        sci_data = hdu['SCI'].data.astype(np.float64)
        err_data = hdu['ERR'].data.astype(np.float64)
        sci_header = hdu['SCI'].header
        wcs = WCS(sci_header)

    # WCS pixel lookup
    px, py = wcs.world_to_pixel_values(RA_TARGET, DEC_TARGET)
    px, py = int(round(float(px))) - 1, int(round(float(py))) - 1

    # Pixel scale from WCS (astropy returns Quantity; extract value)
    pix_area = wcs.proj_plane_pixel_area()
    pix_area_deg2 = pix_area.value  # deg²/pixel
    pix_scale_arcsec = np.sqrt(pix_area_deg2) * 3600  # arcsec/pixel
    half_pix = int(CUTOUT_HALF_SIZE / pix_scale_arcsec)

    # Sky subtraction: sigma-clipped stats + polynomial fit
    sci_clean = np.nan_to_num(sci_data, nan=0.0)
    mean, median, std = sigma_clipped_stats(sci_clean, sigma=3.0, maxiters=5)
    source_mask = sci_clean > (median + 5.0 * std)
    source_mask |= np.isnan(sci_data)

    y, x = np.indices(sci_clean.shape)
    step = 8
    sky_px = ~source_mask[::step, ::step]
    sky_y = y[::step, ::step][sky_px]
    sky_x = x[::step, ::step][sky_px]
    sky_vals = sci_clean[::step, ::step][sky_px]

    if len(sky_vals) > 100:
        A = np.column_stack([np.ones_like(sky_x), sky_x, sky_y,
                             sky_x**2, sky_y**2, sky_x*sky_y])
        coeffs = np.linalg.lstsq(A, sky_vals, rcond=None)[0]
        sky_model = (coeffs[0] + coeffs[1]*x + coeffs[2]*y +
                     coeffs[3]*x**2 + coeffs[4]*y**2 + coeffs[5]*x*y)
    else:
        sky_model = np.full_like(sci_clean, median)

    sci_skysub = sci_data - sky_model

    # Cutout
    x1, x2 = max(0, px - half_pix), min(sci_data.shape[1], px + half_pix)
    y1, y2 = max(0, py - half_pix), min(sci_data.shape[0], py + half_pix)
    sci_cut = sci_skysub[y1:y2, x1:x2]
    err_cut = err_data[y1:y2, x1:x2]

    # Mask: flag bright non-target sources
    cut_median = np.median(sci_cut)
    cut_std = np.std(sci_cut)
    mask = np.zeros(sci_cut.shape, dtype=np.uint8)
    bright = sci_cut > (cut_median + 5 * cut_std)
    cy, cx = sci_cut.shape[0] // 2, sci_cut.shape[1] // 2
    Y, X = np.indices(sci_cut.shape)
    center_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask[bright & (center_dist > 5)] = 1  # smaller exclusion zone for smaller cutout
    mask[np.isnan(sci_cut)] = 1

    # NaN handling
    sci_cut = np.nan_to_num(sci_cut, nan=0.0)
    err_cut = np.nan_to_num(err_cut, nan=1e10)

    # Update WCS header for cutout
    cut_header = sci_header.copy()
    cut_header['CRPIX1'] -= x1
    cut_header['CRPIX2'] -= y1

    # Write cutout FITS
    out_path = os.path.join(IMAGE_DIR, f'cos87259_{band}_cut.fits')
    hdu_out = fits.HDUList([
        fits.PrimaryHDU(data=sci_cut.astype(np.float32), header=cut_header),
        fits.ImageHDU(data=mask, name='MASK'),
        fits.ImageHDU(data=err_cut.astype(np.float32), name='SIGMA'),
    ])
    hdu_out.writeto(out_path, overwrite=True)

    # Generate Gaussian PSF
    fwhm_arcsec = MIRI_FWHM[band]
    fwhm_pix = fwhm_arcsec / pix_scale_arcsec
    sigma_pix = fwhm_pix / 2.355
    psf_size = 51
    py_grid, px_grid = np.indices((psf_size, psf_size))
    center = psf_size // 2
    psf = np.exp(-((px_grid - center)**2 + (py_grid - center)**2) / (2 * sigma_pix**2))
    psf /= psf.sum()

    psf_path = os.path.join(PSF_DIR, f'cos87259_{band}_psf.fits')
    fits.writeto(psf_path, psf.astype(np.float32), overwrite=True)

    print(f'{band:8s} cut={sci_cut.shape} px={px},{py} sky={median:.4f}')

print('\nDone. Output: MIRI_cutout_v3/')
