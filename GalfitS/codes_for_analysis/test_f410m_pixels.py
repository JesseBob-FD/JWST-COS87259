"""Test the F410M brightest pixel code."""
import numpy as np
from astropy.io import fits
from astropy import wcs

band = 'F410M'
file_path = f'NIRCam/{band}_cos87259_sci.fits'
RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373
HALF_WINDOW = 60

with fits.open(file_path) as hdul:
    sci_data = hdul['SCI'].data.astype(np.float64)
    header = hdul['SCI'].header
    nircam_wcs = wcs.WCS(header)

cx, cy = nircam_wcs.world_to_pixel_values(RA_TARGET, DEC_TARGET)
cx, cy = int(float(cx)), int(float(cy))

x1 = max(0, cx - HALF_WINDOW)
x2 = min(sci_data.shape[1], cx + HALF_WINDOW)
y1 = max(0, cy - HALF_WINDOW)
y2 = min(sci_data.shape[0], cy + HALF_WINDOW)
cutout = sci_data[y1:y2, x1:x2]

print(f'Target center: RA={RA_TARGET}, Dec={DEC_TARGET}')
print(f'Pixel center:  cx={cx}, cy={cy}')
print(f'Search box:    x=[{x1}:{x2}], y=[{y1}:{y2}]  ({x2-x1}x{y2-y1} px)')

# Find top 2 pixels within the search box
data_flat = cutout.ravel()
idx_sorted = np.argsort(data_flat)[::-1]
top2_idx_flat = idx_sorted[:2]
top2_flux = data_flat[top2_idx_flat]

# Local -> global coordinates
top2_y_local, top2_x_local = np.unravel_index(top2_idx_flat, cutout.shape)
top2_x_global = top2_x_local + x1
top2_y_global = top2_y_local + y1

# Offset from center
pix_scale = np.sqrt(header['PIXAR_SR']) * 180 / np.pi * 3600  # arcsec/pix
top2_dx = (top2_x_global - cx) * pix_scale
top2_dy = (top2_y_global - cy) * pix_scale

print(f'\nBrightest pixels within +/-{HALF_WINDOW} px of fitting center:')
for i in range(2):
    print(f'  #{i+1}: global ({top2_x_global[i]}, {top2_y_global[i]})  '
          f'offset ({top2_dx[i]:+.3f}", {top2_dy[i]:+.3f}")  '
          f'flux = {top2_flux[i]:.4f} MJy/sr')
print('\nSUCCESS')
