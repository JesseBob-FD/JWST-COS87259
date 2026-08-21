import numpy as np
import os

fdir = '/mnt/d/Fudan_University/Research/JWST/GalfitS/GalfitS/src/data/filters'
PIXEL_SR = 8.46159499407524e-14
c_A_s = 2.99792458e18

BAND_CENTERS = {
    'F560W': 56000, 'F770W': 77000, 'F1000W': 100000,
    'F1130W': 113000, 'F1280W': 128000, 'F1500W': 150000,
    'F1800W': 180000, 'F2100W': 210000, 'F2550W': 255000
}

bands = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W',
         'F1500W', 'F1800W', 'F2100W', 'F2550W']

for band in bands:
    fname = f'miri_{band.lower()}'
    fpath = os.path.join(fdir, fname)
    
    if os.path.exists(fpath):
        data = np.loadtxt(fpath)
        wl = data[:, 0]
        T = data[:, 1]
        num = np.trapezoid(T * wl, wl)
        den = np.trapezoid(T / wl, wl)
        if den > 0:
            lambda_pivot = np.sqrt(num / den)
        else:
            lambda_pivot = BAND_CENTERS[band]
    else:
        lambda_pivot = BAND_CENTERS[band]
        print(f'{band:8s} (no filter file, using central wavelength)')
        
    factor = (lambda_pivot**2 / c_A_s) / PIXEL_SR * 1e17
    
    print(f'{band:8s} lambda_pivot={lambda_pivot:.1f} A  Ia9={factor:.4e}')