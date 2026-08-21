#!/usr/bin/env python3
"""
MIRI Data Preprocessing Script for GalfitS - Manual Robust Version
==================================================================
Performs: sky subtraction, cutout, mask generation, sigma calculation
for JWST MIRI imaging data of COS-87259.
"""

import os, glob, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
MIRI_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI'
OUTPUT_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_cutout'
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
PSF_DIR = os.path.join(OUTPUT_DIR, 'psf')

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PSF_DIR, exist_ok=True)

TARGET_NAME = 'COS87259'
RA_TARGET = 149.74061852485357
DEC_TARGET = 1.6496036806533318
TARGET_Z = 6.83
CUTOUT_HALF_SIZE = 5.0  # arcsec

BANDS = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
         'F1500W', 'F1800W', 'F2100W', 'F2550W']

print("=" * 60)
print("MIRI Data Preprocessing for GalfitS")
print(f"Target: {TARGET_NAME}  RA={RA_TARGET:.6f}  Dec={DEC_TARGET:.6f}")
print(f"Cutout half-size: {CUTOUT_HALF_SIZE} arcsec")
print("=" * 60)

for band in BANDS:
    fpath = os.path.join(MIRI_DIR, f'cos87259_{band}_60mas.fits.gz')
    if not os.path.exists(fpath):
        print(f"  SKIP {band}: file not found")
        continue
    
    print(f"\n{'='*40}")
    print(f"Processing {band}")
    print(f"{'='*40}")
    
    with fits.open(fpath) as hdu:
        sci_data = hdu['SCI'].data.astype(np.float64)
        err_data = hdu['ERR'].data.astype(np.float64)
        sci_header = hdu['SCI'].header
    
    # --------------------------------------------------------
    # 1. Get target pixel coordinates from WCS
    # --------------------------------------------------------
    wcs = WCS(sci_header)
    px, py = wcs.world_to_pixel_values(RA_TARGET, DEC_TARGET)
    px = int(round(float(px))) - 1  # 0-indexed
    py = int(round(float(py))) - 1
    
    # Pixel scale
    pix_scale = np.sqrt(wcs.pixel_scale_matrix[0,0]**2 
                       + wcs.pixel_scale_matrix[0,1]**2) * 3600
    half_pix = int(CUTOUT_HALF_SIZE / pix_scale)
    
    x1 = max(0, px - half_pix)
    x2 = min(sci_data.shape[1], px + half_pix)
    y1 = max(0, py - half_pix)
    y2 = min(sci_data.shape[0], py + half_pix)
    
    print(f"  Target pixel: ({px}, {py})")
    print(f"  Pixel scale: {pix_scale:.4f} arcsec/pix")
    print(f"  Cutout region: x=[{x1},{x2}] y=[{y1},{y2}]  size={x2-x1}x{y2-y1}")
    
    # --------------------------------------------------------
    # 2. Sky subtraction (polynomial fit to background)
    # --------------------------------------------------------
    sci_clean = np.nan_to_num(sci_data, nan=0.0)
    _, median, std = sigma_clipped_stats(sci_clean, sigma=3.0, maxiters=5)
    
    # Create rough source mask for sky fitting
    source_mask = sci_clean > (median + 5.0 * std)
    
    # Downsample for efficiency
    step = 8
    y_idx, x_idx = np.indices(sci_data.shape)
    sky_pixels = ~source_mask & ~np.isnan(sci_data)
    
    sky_y = y_idx[::step, ::step][sky_pixels[::step, ::step]]
    sky_x = x_idx[::step, ::step][sky_pixels[::step, ::step]]
    sky_vals = sci_data[::step, ::step][sky_pixels[::step, ::step]]
    
    if len(sky_vals) > 100:
        # Fit 2nd order polynomial
        A = np.column_stack([
            np.ones_like(sky_x),
            sky_x, sky_y,
            sky_x**2, sky_y**2, sky_x*sky_y
        ])
        coeffs, _, _, _ = np.linalg.lstsq(A, sky_vals, rcond=None)
        sky_model = (coeffs[0] + coeffs[1]*x_idx + coeffs[2]*y_idx 
                    + coeffs[3]*x_idx**2 + coeffs[4]*y_idx**2 
                    + coeffs[5]*x_idx*y_idx)
        print(f"  Sky poly fit: median={median:.6f}, sky_model range=[{sky_model.min():.6f},{sky_model.max():.6f}]")
    else:
        sky_model = np.full_like(sci_data, median)
        print(f"  Sky: using median={median:.6f}")
    
    sci_skysub = sci_data - sky_model
    
    # --------------------------------------------------------
    # 3. Cutout
    # --------------------------------------------------------
    sci_cut = sci_skysub[y1:y2, x1:x2]
    err_cut = err_data[y1:y2, x1:x2]
    
    # Replace NaN with 0 in cutout
    sci_cut = np.nan_to_num(sci_cut, nan=0.0)
    err_cut = np.nan_to_num(err_cut, nan=1e10)
    
    # --------------------------------------------------------
    # 4. Mask generation
    # --------------------------------------------------------
    # Simple sigma-clipping mask: flag pixels > 5 sigma above median as sources
    cut_median = np.median(sci_cut)
    cut_std = np.std(sci_cut)
    # Mask bright sources (excluding the central target)
    mask_cut = np.zeros(sci_cut.shape, dtype=np.uint8)
    
    # Flag pixels that are significantly bright but not at center
    bright_mask = sci_cut > (cut_median + 5.0 * cut_std)
    cy, cx = sci_cut.shape[0]//2, sci_cut.shape[1]//2
    # Exclude central 10-pixel radius from masking
    Y, X = np.indices(sci_cut.shape)
    center_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    bright_mask = bright_mask & (center_dist > 10)
    mask_cut[bright_mask] = 1
    
    # Also flag NaN regions
    mask_cut[np.isnan(sci_cut)] = 1
    
    print(f"  Cutout stats: median={cut_median:.6f}, std={cut_std:.6f}")
    print(f"  Masked pixels: {mask_cut.sum()} / {mask_cut.size}")
    
    # --------------------------------------------------------
    # 5. Write 3-extension FITS
    # --------------------------------------------------------
    outname = os.path.join(IMAGE_DIR, f'cos87259_{band}_cut.fits')
    
    # Update WCS header for cutout
    cut_header = sci_header.copy()
    cut_header['CRPIX1'] = sci_header['CRPIX1'] - x1
    cut_header['CRPIX2'] = sci_header['CRPIX2'] - y1
    
    hdu_primary = fits.PrimaryHDU(data=sci_cut.astype(np.float32), header=cut_header)
    hdu_mask = fits.ImageHDU(data=mask_cut, name='MASK')
    hdu_sigma = fits.ImageHDU(data=err_cut.astype(np.float32), name='SIGMA')
    
    hdu_list = fits.HDUList([hdu_primary, hdu_mask, hdu_sigma])
    hdu_list.writeto(outname, overwrite=True)
    hdu_list.close()
    
    print(f"  Saved: {outname}")

print("\n" + "=" * 60)
print(f"Preprocessing complete! Output: {IMAGE_DIR}")
print("=" * 60)
