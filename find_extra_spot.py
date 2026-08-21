import numpy as np
import json
from astropy.io import fits
import sep

s3d = fits.open('NIRSpec_IFU/COS-87259_BGSUB_g395h-f290lp_s3d.fits')
f3d, e3d = s3d['sci'].data, s3d['err'].data
hdr = s3d['sci'].header

wl0, dwl, wl_len = hdr['CRVAL3'], hdr['CDELT3'], hdr['NAXIS3']
wl = wl0 + np.arange(wl_len) * dwl
wl_det = ((wl>3.7515) & (wl<3.7601)) | ((wl>3.7707) & (wl<4.0274))

ivw = np.nanmean(f3d[wl_det]/e3d[wl_det]**2,axis=0) / np.nanmean(1./e3d[wl_det]**2,axis=0)
nan_mask = np.isnan(ivw)
ivw[nan_mask] = np.nan

bkg = sep.Background(ivw, mask=nan_mask, bw=5, bh=5)
stack = ivw - bkg
stack[nan_mask] = np.nan

objects = sep.extract(stack, 1.5, err=bkg.globalrms, mask=nan_mask)

output = []
for i, obj in enumerate(objects):
    x, y = float(obj['x']), float(obj['y'])
    if 15 < x < 25 and 20 < y < 35:
        output.append({'id': i, 'x': x, 'y': y, 'flux': float(obj['flux']), 'a': float(obj['a']), 'b': float(obj['b']), 'theta': float(obj['theta'])})

with open('temp_spots.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
