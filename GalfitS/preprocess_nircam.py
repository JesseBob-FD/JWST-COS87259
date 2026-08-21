#!/usr/bin/env python3
"""Preprocess NIRCam data for MIRI+NIRCam joint fitting.

Covers F115W, F200W, F410M for COS-87259.
- Unit conversion: 10*nanoJansky → MJy/sr
- Sky subtraction: sigma-clipped median (no ERR extension)
- Cutout half-size: 2 arcsec (matching MIRI)
- Sigma: computed from data statistics
- PSF: Gaussian approximation (FWHM = lambda/D)
"""
import os, warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
warnings.filterwarnings('ignore')

NIRCAM_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/NIRCam'
IMAGE_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/NIRCam_cutout/images'
PSF_DIR = '/mnt/d/Fudan_University/Research/JWST/GalfitS/NIRCam_cutout/psf'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PSF_DIR, exist_ok=True)

RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373
CUTOUT_HALF_SIZE = 2.0  # arcsec

BANDS = {
    'F115W': {'fname': 'F115W_cos87259_sci.fits', 'fwhm': 0.037, 'wave': 1.154},
    'F200W': {'fname': 'F200W_cos87259_sci.fits', 'fwhm': 0.064, 'wave': 1.989},
    'F410M': {'fname': 'F410M_cos87259_sci.fits', 'fwhm': 0.130, 'wave': 4.082},
}

print(f"NIRCam preprocessing: cutout half-size = {CUTOUT_HALF_SIZE} arcsec")
print(f"Target: RA={RA_TARGET:.6f} Dec={DEC_TARGET:.6f}\n")

for band, info in BANDS.items():
    fpath = os.path.join(NIRCAM_DIR, info['fname'])
    if not os.path.exists(fpath):
        print(f'{band:8s} - SKIP (file not found)')
        continue

    with fits.open(fpath) as hdu:
        sci_data = hdu['SCI'].data.astype(np.float64)
        sci_header = hdu['SCI'].header
        pixar_sr = sci_header.get('PIXAR_SR', 2.11539874851881e-14)
        photmjsr = sci_header.get('PHOTMJSR', 1.0)
        wcs = WCS(sci_header)

    # Unit conversion: 10*nanoJansky → MJy/sr
    # MJy/sr = (pixel_value * 10e-9 Jy) / PIXAR_SR / 1e6 = pixel_value * 1e-14 / PIXAR_SR
    # Or equivalently: MJy/sr = pixel_value / PHOTMJSR
    sci_mjysr = sci_data / photmjsr

    # WCS pixel lookup
    px, py = wcs.world_to_pixel_values(RA_TARGET, DEC_TARGET)
    px, py = int(round(float(px))) - 1, int(round(float(py))) - 1

    # Pixel scale
    pix_scale = np.sqrt(pixar_sr)  # rad/pixel
    pix_scale_arcsec = pix_scale * 180 / np.pi * 3600  # arcsec/pixel
    half_pix = int(CUTOUT_HALF_SIZE / pix_scale_arcsec)

    # Sky subtraction: sigma-clipped median (no polynomial for NIRCam)
    sci_clean = np.nan_to_num(sci_mjysr, nan=0.0)
    mean, median, std = sigma_clipped_stats(sci_clean, sigma=3.0, maxiters=5)
    sci_skysub = sci_mjysr - median

    # Cutout
    x1, x2 = max(0, px - half_pix), min(sci_data.shape[1], px + half_pix)
    y1, y2 = max(0, py - half_pix), min(sci_data.shape[0], py + half_pix)
    sci_cut = sci_skysub[y1:y2, x1:x2]

    # Sigma: from data statistics (NIRCam has no ERR extension in mosaic)
    cut_mean, cut_median, cut_std = sigma_clipped_stats(sci_cut, sigma=3.0, maxiters=5)
    sigma_cut = np.full_like(sci_cut, max(cut_std, 1e-10))

    # Mask: flag bright non-target sources
    mask = np.zeros(sci_cut.shape, dtype=np.uint8)
    bright = sci_cut > (cut_median + 5 * cut_std)
    cy, cx = sci_cut.shape[0] // 2, sci_cut.shape[1] // 2
    Y, X = np.indices(sci_cut.shape)
    center_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask[bright & (center_dist > 5)] = 1
    mask[np.isnan(sci_cut)] = 1

    sci_cut = np.nan_to_num(sci_cut, nan=0.0)

    # Update WCS header for cutout
    cut_header = sci_header.copy()
    cut_header['CRPIX1'] -= x1
    cut_header['CRPIX2'] -= y1

    # Write cutout FITS
    out_path = os.path.join(IMAGE_DIR, f'cos87259_{band}_cut.fits')
    hdu_out = fits.HDUList([
        fits.PrimaryHDU(data=sci_cut.astype(np.float32), header=cut_header),
        fits.ImageHDU(data=mask, name='MASK'),
        fits.ImageHDU(data=sigma_cut.astype(np.float32), name='SIGMA'),
    ])
    hdu_out.writeto(out_path, overwrite=True)

    # Generate Gaussian PSF
    fwhm_pix = info['fwhm'] / pix_scale_arcsec
    sigma_pix = fwhm_pix / 2.355
    psf_size = max(31, int(10 * sigma_pix) // 2 * 2 + 1)  # odd size ~10 sigma
    py_grid, px_grid = np.indices((psf_size, psf_size))
    center = psf_size // 2
    psf = np.exp(-((px_grid - center)**2 + (py_grid - center)**2) / (2 * sigma_pix**2))
    psf /= psf.sum()

    psf_path = os.path.join(PSF_DIR, f'cos87259_{band}_psf.fits')
    fits.writeto(psf_path, psf.astype(np.float32), overwrite=True)

    print(f'{band:8s} cut={sci_cut.shape} px={px},{py} '
          f'scale={pix_scale_arcsec:.4f}"/px sky={median:.4f} sigma={cut_std:.4f}')

print('\nDone. Output: NIRCam_cutout/')
