import numpy as np
import os
from scipy.optimize import curve_fit

def gaussian(x, amp, cen, wid, bkg):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2)) + bkg

spectra_dir = 'd:/Fudan_University/Research/JWST/extracted_spectra'
clumps = ['Center', 'West', 'North', 'South', 'Southeast']
grating = 'G395H'
z = 6.85 # Redshift
lines = {
    'H_beta': 0.4861, # Rest-frame wavelength in um
    'OIII_4959': 0.4959,
    'OIII_5007': 0.5007,
    'H_alpha': 0.6563
}

print(f"Emission Line Quantification (z={z}):")
print("-" * 60)

for clump in clumps:
    file_path = os.path.join(spectra_dir, f'{clump}_{grating}_spectrum.txt')
    if not os.path.exists(file_path):
        continue
        
    data = np.loadtxt(file_path, skiprows=1)
    wl_obs = data[:, 0]
    flux = data[:, 1]
    
    print(f"--- {clump} Clump ---")
    
    for line_name, wl_rest in lines.items():
        wl_target = wl_rest * (1 + z)
        
        # Select window around the expected line
        window = 0.05 # um
        mask = (wl_obs > wl_target - window) & (wl_obs < wl_target + window)
        
        if not np.any(mask):
            continue
            
        wl_fit = wl_obs[mask]
        flux_fit = flux[mask]
        
        # Initial guess
        amp_guess = np.max(flux_fit) - np.median(flux_fit)
        bkg_guess = np.median(flux_fit)
        p0 = [amp_guess, wl_target, 0.005, bkg_guess]
        
        try:
            popt, _ = curve_fit(gaussian, wl_fit, flux_fit, p0=p0, maxfev=2000)
            amp, cen, wid, bkg = popt
            
            # Integrated flux (approximate)
            area = amp * wid * np.sqrt(2 * np.pi)
            # FWHM in km/s (approximate)
            c = 299792 # km/s
            fwhm_um = 2.355 * wid
            fwhm_kms = (fwhm_um / cen) * c
            
            print(f"  {line_name:10s}: Center={cen:.4f}um, FWHM={fwhm_kms:.0f} km/s, Flux={area:.2e}")
        except Exception as e:
            print(f"  {line_name:10s}: Fit failed - {e}")

print("-" * 60)
