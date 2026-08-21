import numpy as np
import matplotlib.pyplot as plt
import os

spectra_dir = 'd:/Fudan_University/Research/JWST/extracted_spectra'
clumps = ['Center', 'West', 'North', 'South', 'Southeast']
grating = 'G395H'

plt.figure(figsize=(15, 10))

for i, clump in enumerate(clumps):
    file_path = os.path.join(spectra_dir, f'{clump}_{grating}_spectrum.txt')
    
    if os.path.exists(file_path):
        data = np.loadtxt(file_path, skiprows=1)
        wl = data[:, 0]
        flux = data[:, 1]
        
        plt.subplot(3, 2, i + 1)
        plt.plot(wl, flux, 'k-', ds='steps-mid', lw=1)
        plt.title(f'{clump} Clump Spectrum ({grating})')
        plt.xlabel('Observed Wavelength ($\mu$m)')
        plt.ylabel('$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
        plt.xlim(2.8, 5.3)
        plt.grid(True, alpha=0.3)
    else:
        print(f"Warning: {file_path} not found.")

plt.tight_layout()
plt.savefig(os.path.join(spectra_dir, f'all_clumps_{grating}.png'), dpi=300)
print("Plot saved.")
