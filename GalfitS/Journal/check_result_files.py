from astropy.io import fits
import numpy as np

band = 'F770W'
base = 'MIRI_NIRCam_result/two_sersic_C1C2/noSED'

# GalfitS result file (--saveimgs output)
rf = f'{base}/C1C2_noSED_miri_{band.lower()}_result.fits'
print('=== GalfitS result file ===')
h = fits.open(rf)
print(f'Extensions: {len(h)}')
for i, ext in enumerate(h):
    if ext.data is not None:
        d = ext.data
        print(f'  Ext {i} ({ext.name}): {d.shape} {d.dtype}  min={np.min(d):.4f} max={np.max(d):.4f} mean={np.mean(d):.4f}')
h.close()

# C1 model file (from my separate_components.py)
m1 = f'{base}/C1_model_{band}.fits'
print(f'\n=== C1_model file ===')
h1 = fits.open(m1)
d = h1[0].data
print(f'  {d.shape} {d.dtype}  min={np.min(d):.6f} max={np.max(d):.6f} mean={np.mean(d):.6f}')
h1.close()

# Compare: total model from GalfitS vs C1+C2 from my code
print(f'\n=== Comparing total models ===')
h = fits.open(rf)
galfit_model = h[3].data  # ext 3 = model
h.close()

c1 = fits.getdata(m1)
c2 = fits.getdata(f'{base}/C2_model_{band}.fits')
my_total = c1 + c2
diff = galfit_model - my_total
print(f'  GalfitS total model: min={np.min(galfit_model):.6f} max={np.max(galfit_model):.6f} mean={np.mean(galfit_model):.6f}')
print(f'  My C1+C2 total:      min={np.min(my_total):.6f} max={np.max(my_total):.6f} mean={np.mean(my_total):.6f}')
print(f'  Difference (GalfitS - mine): max_abs_diff={np.max(np.abs(diff)):.6e} mean_abs_diff={np.mean(np.abs(diff)):.6e}')

# Pixel-by-pixel ratio
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(galfit_model > 1e-10, my_total / galfit_model, np.nan)
    print(f'  C1+C2 / GalfitS ratio: median={np.nanmedian(ratio):.6f}  std={np.nanstd(ratio):.6f}')
