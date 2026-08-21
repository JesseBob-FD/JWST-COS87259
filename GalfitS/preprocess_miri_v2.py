#!/usr/bin/env python3
"""Reprocess MIRI cutouts with corrected target coordinates."""
import os, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
warnings.filterwarnings('ignore')

MIRI_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI'
IMAGE_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v2/images'
os.makedirs(IMAGE_DIR, exist_ok=True)

RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373
CUTOUT_HALF_SIZE = 5.0

BANDS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
         'F1500W', 'F1800W', 'F2100W', 'F2550W']

print("Reprocessing with corrected target: RA=%.6f Dec=%.6f" % (RA_TARGET, DEC_TARGET))

for band in BANDS:
    fpath = os.path.join(MIRI_DIR, f'cos87259_{band}_60mas.fits.gz')
    if not os.path.exists(fpath):
        continue
    
    with fits.open(fpath) as hdu:
        sci_data = hdu['SCI'].data.astype(np.float64)
        err_data = hdu['ERR'].data.astype(np.float64)
        sci_header = hdu['SCI'].header
    
    wcs = WCS(sci_header)
    px, py = wcs.world_to_pixel_values(RA_TARGET, DEC_TARGET)
    px = int(round(float(px))) - 1
    py = int(round(float(py))) - 1
    
    pix_scale = np.sqrt(wcs.pixel_scale_matrix[0,0]**2 + wcs.pixel_scale_matrix[0,1]**2) * 3600
    half_pix = int(CUTOUT_HALF_SIZE / pix_scale)
    
    x1 = max(0, px - half_pix)
    x2 = min(sci_data.shape[1], px + half_pix)
    y1 = max(0, py - half_pix)
    y2 = min(sci_data.shape[0], py + half_pix)
    
    # Sky subtraction
    sci_clean = np.nan_to_num(sci_data, nan=0.0)
    _, median, std = sigma_clipped_stats(sci_clean, sigma=3.0, maxiters=5)
    source_mask = sci_clean > (median + 5.0 * std)
    
    step = 8
    y_idx, x_idx = np.indices(sci_data.shape)
    sky_pixels = ~source_mask & ~np.isnan(sci_data)
    sky_y = y_idx[::step, ::step][sky_pixels[::step, ::step]]
    sky_x = x_idx[::step, ::step][sky_pixels[::step, ::step]]
    sky_vals = sci_data[::step, ::step][sky_pixels[::step, ::step]]
    
    if len(sky_vals) > 100:
        A = np.column_stack([np.ones_like(sky_x), sky_x, sky_y, sky_x**2, sky_y**2, sky_x*sky_y])
        coeffs, _, _, _ = np.linalg.lstsq(A, sky_vals, rcond=None)
        sky_model = (coeffs[0] + coeffs[1]*x_idx + coeffs[2]*y_idx 
                     + coeffs[3]*x_idx**2 + coeffs[4]*y_idx**2 + coeffs[5]*x_idx*y_idx)
    else:
        sky_model = np.full_like(sci_data, median)
    
    sci_skysub = sci_data - sky_model
    sci_cut = np.nan_to_num(sci_skysub[y1:y2, x1:x2], nan=0.0)
    err_cut = np.nan_to_num(err_data[y1:y2, x1:x2], nan=1e10)
    
    # Mask
    cut_median = np.median(sci_cut)
    cut_std = np.std(sci_cut)
    mask_cut = np.zeros(sci_cut.shape, dtype=np.uint8)
    bright_mask = sci_cut > (cut_median + 5.0 * cut_std)
    cy_c, cx_c = sci_cut.shape[0]//2, sci_cut.shape[1]//2
    Y, X = np.indices(sci_cut.shape)
    center_dist = np.sqrt((X - cx_c)**2 + (Y - cy_c)**2)
    mask_cut[bright_mask & (center_dist > 10)] = 1
    mask_cut[np.isnan(sci_cut)] = 1
    
    # WCS header for cutout
    cut_header = sci_header.copy()
    cut_header['CRPIX1'] = sci_header['CRPIX1'] - x1
    cut_header['CRPIX2'] = sci_header['CRPIX2'] - y1
    
    outname = os.path.join(IMAGE_DIR, f'cos87259_{band}_cut.fits')
    hdu_out = fits.HDUList([
        fits.PrimaryHDU(data=sci_cut.astype(np.float32), header=cut_header),
        fits.ImageHDU(data=mask_cut, name='MASK'),
        fits.ImageHDU(data=err_cut.astype(np.float32), name='SIGMA'),
    ])
    hdu_out.writeto(outname, overwrite=True)
    hdu_out.close()
    print(f'  {band}: cutout {sci_cut.shape} saved, peak={sci_cut.max():.4f}')

# Generate PSFs
PSF_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout_v2/psf'
os.makedirs(PSF_DIR, exist_ok=True)
PSF_FWHM = {'F560W': 0.20, 'F770W': 0.27, 'F1000W': 0.33, 'F1130W': 0.36,
            'F1280W': 0.40, 'F1500W': 0.45, 'F1800W': 0.54, 'F2100W': 0.63, 'F2550W': 0.77}
PIX_SCALE = 0.06

for band in BANDS:
    fwhm_arcsec = PSF_FWHM[band]
    sigma_pix = (fwhm_arcsec / PIX_SCALE) / 2.355
    size = 51
    y, x = np.indices((size, size))
    c = (size - 1) / 2
    psf = np.exp(-((x - c)**2 + (y - c)**2) / (2 * sigma_pix**2))
    psf /= psf.sum()
    outname = os.path.join(PSF_DIR, f'cos87259_{band}_psf.fits')
    fits.writeto(outname, psf.astype(np.float32), overwrite=True)
    print(f'  {band} PSF: FWHM={fwhm_arcsec:.2f}" sigma={sigma_pix:.1f}px')

print("Done!")
