import numpy as np
from astropy.io import fits
import sep
import matplotlib.pyplot as plt
import os

print("Starting extraction for North of Southeast Clump...")

output_dir = 'd:/Fudan_University/Research/JWST/extracted_spectra'
os.makedirs(output_dir, exist_ok=True)

s3d_file = 'NIRSpec_IFU/COS-87259_BGSUB_g395h-f290lp_s3d.fits'
s3d = fits.open(s3d_file)
hdr = s3d['sci'].header
f3d = s3d['sci'].data
e3d = s3d['err'].data

# Wavelength array
wl0, dwl, wl_len = hdr['CRVAL3'], hdr['CDELT3'], hdr['NAXIS3']
wl = wl0 + np.arange(wl_len) * dwl

# Clump definition (North of Southeast)
clump_id = 'North_of_Southeast'
x = 18.7524
y = 24.9155
a = 0.9931 # Very small aperture based on user request
b = 0.8873
r = 1.5 # Keeping r small
theta = -0.4654 

# Flux conversion unit
fnu_0 = hdr['pixar_sr']

f1d, e1d = np.zeros_like(wl), np.zeros_like(wl)

print("Running aperture extraction over all wavelength layers...")
for i in range(wl_len):
    f2d, e2d = f3d[i], e3d[i]
    f2d = f2d.astype(f2d.dtype.newbyteorder('='))
    e2d = e2d.astype(e2d.dtype.newbyteorder('='))
    
    # We don't have a rigid NaN mask here in this simple script 
    # but we will mask out pure NaN areas simply
    mask = np.isnan(f2d)
    
    aper_flux, aper_flux_err, aper_flux_flag = sep.sum_ellipse(f2d, [x], [y], [a], [b], [theta], [r], mask=mask)
    f1d[i] = aper_flux[0] * fnu_0
    
    aper_err, aper_err_err, aper_err_flag = sep.sum_ellipse(e2d**2, [x], [y], [a], [b], [theta], [r], mask=mask)
    e1d[i] = np.sqrt(aper_err[0]) * fnu_0

# Fix scale units to erg/s/cm^2/Angstrom just like the original code
f1d = f1d * 1e-17 * 3e18 / (wl * 1e4)**2
e1d = e1d * 1e-17 * 3e18 / (wl * 1e4)**2

# Save the spectrum to TXT
txt_path = os.path.join(output_dir, f'{clump_id}_G395H_spectrum.txt')
np.savetxt(txt_path, np.column_stack((wl, f1d, e1d)), header='Wavelength_um Flux_erg_s_cm2_A Err', comments='')
print(f"Saved numerical spectrum to: {txt_path}")

# Plot the 1D spectrum
plt.figure(figsize=(10, 5))
plt.plot(wl, f1d * 1e19, 'k-', ds='steps-mid', linewidth=1)
plt.xlabel(r'$\lambda_{\rm obs}$ ($\mu$m)')
plt.ylabel(r'$f_{\lambda}$ ($10^{-19}$ erg s$^{-1}$ cm$^{-2}$ $\rm \AA^{-1}$)')
plt.xlim([2.8, 5.3])
plt.title(f'1D Spectrum: {clump_id} Clump (G395H)')

# Provide some smoothing to show the lines better visually
from scipy.signal import savgol_filter
try:
    valid_mask = ~np.isnan(f1d)
    f1d_smooth = savgol_filter(f1d[valid_mask] * 1e19, window_length=15, polyorder=2)
    plt.plot(wl[valid_mask], f1d_smooth, 'r-', label='Smoothed', alpha=0.8, linewidth=1.5)
    plt.legend()
except Exception:
    pass

plot_path = os.path.join(output_dir, f'{clump_id}_G395H_plot.png')
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
print(f"Saved plot to: {plot_path}")
