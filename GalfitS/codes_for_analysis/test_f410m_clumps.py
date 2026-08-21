"""Test clump finding via local maxima in F410M."""
import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage import maximum_filter

band = 'F410M'
file_path = f'NIRCam/{band}_cos87259_sci.fits'
RA_TARGET = 149.74276239
DEC_TARGET = 1.6555373
HALF_WINDOW = 60
MIN_SEPARATION = 8
PEAK_THRESHOLD = 0.3

with fits.open(file_path) as hdul:
    sci_data = hdul['SCI'].data.astype(np.float64)
    header = hdul['SCI'].header
    nircam_wcs = wcs.WCS(header)

cx, cy = nircam_wcs.world_to_pixel_values(RA_TARGET, DEC_TARGET)
cx, cy = int(float(cx)), int(float(cy))

x1 = max(0, cx - HALF_WINDOW); x2 = min(sci_data.shape[1], cx + HALF_WINDOW)
y1 = max(0, cy - HALF_WINDOW); y2 = min(sci_data.shape[0], cy + HALF_WINDOW)
cutout = sci_data[y1:y2, x1:x2]
pix_scale = np.sqrt(header['PIXAR_SR']) * 180 / np.pi * 3600

# Local maxima
footprint = np.ones((3, 3), dtype=bool)
local_max = (cutout == maximum_filter(cutout, footprint=footprint))
local_max &= (cutout > PEAK_THRESHOLD)
local_max[0, :] = local_max[-1, :] = local_max[:, 0] = local_max[:, -1] = False

peak_y, peak_x = np.where(local_max)
peak_flux = cutout[peak_y, peak_x]
sort_idx = np.argsort(peak_flux)[::-1]
peak_x_s = peak_x[sort_idx]; peak_y_s = peak_y[sort_idx]; peak_f_s = peak_flux[sort_idx]

# Cluster
clusters = []
assigned = np.zeros(len(peak_x_s), dtype=bool)
for i in range(len(peak_x_s)):
    if assigned[i]: continue
    members = [i]; assigned[i] = True
    for j in range(i+1, len(peak_x_s)):
        if assigned[j]: continue
        dist = np.sqrt((peak_x_s[j]-peak_x_s[i])**2 + (peak_y_s[j]-peak_y_s[i])**2)
        if dist < MIN_SEPARATION:
            members.append(j); assigned[j] = True
    clusters.append((peak_x_s[i], peak_y_s[i], members))

print(f'Local maxima found: {len(peak_x_s)}')
print(f'Distinct clumps (min sep={MIN_SEPARATION}px, peak>{PEAK_THRESHOLD} MJy/sr):')
for ci, (cl_x, cl_y, members) in enumerate(clusters):
    gx, gy = cl_x + x1, cl_y + y1
    dx = (gx - cx) * pix_scale; dy = (gy - cy) * pix_scale
    print(f'  C{ci+1}: global ({gx},{gy})  offset=({dx:+.3f}",{dy:+.3f}")  '
          f'peak={peak_f_s[members[0]]:.4f} MJy/sr  ({len(members)} peaks merged)')

if len(clusters) >= 2:
    print(f'\nInter-clump distances:')
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist = np.sqrt((clusters[j][0]-clusters[i][0])**2 + (clusters[j][1]-clusters[i][1])**2)
            print(f'  C{i+1}-C{j+1}: {dist:.1f} px ({dist*pix_scale:.3f}")')
