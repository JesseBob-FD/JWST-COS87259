# Comprehensive Guide: FITS Data Structure & `COS87259.ipynb` Code Explanation

## Part 1: Understanding FITS Files in Astronomy

FITS stands for **Flexible Image Transport System**. It is the standard data format used widely in astronomy to store, transmit, and manipulate data (images, spectra, and tables) from observatories like JWST.

### 1.1 The Structure of a FITS File

A FITS file is organized into one or more sequential blocks called **Header Data Units (HDUs)**.

* **Primary HDU**: The very first HDU in the file. It always exists, though sometimes it acts just as an overarching container without direct pixel data.
* **Extensions (HDU 1, 2, ...)**: Additional data blocks. JWST pipeline data is usually separated into multiple extensions reflecting different types of datasets for a single observation.
* **SCI (Science)**: The actual scientific flux/intensity values.
* **ERR (Error)**: The estimated uncertainty (error) standard deviations for each pixel.
* **DQ (Data Quality)**: Flags identifying problematic pixels (like cosmic rays, bad pixels, or edge artifacts).

### 1.2 The Structure of an HDU

Each individual HDU consists of two main components:

1. **The Header**: Analogous to metadata. This contains human-readable keyword-value pairs providing important context about the data block. E.g., observation date, instrument used, exposure time, or **WCS (World Coordinate System)** which defines how pixels map to true sky coordinates (Right Ascension/Declination) and corresponding wavelengths.
2. **The Data Array**: The actual numerical data, stored as a multidimensional array (e.g., a 2D image, a 3D spectral cube, or a binary table).

### 1.3 How to Explore FITS files in Python

You can easily interact with FITS files using the `astropy.io.fits` module.

```python
from astropy.io import fits

# 1. Open the file 
file_path = 'NIRSpec_IFU/COS-87259_NOBG_prism-clear_s3d.fits'
hdul = fits.open(file_path)

# 2. View the overall structure of HDUs in this file
print("File Information:")
hdul.info()
print("-" * 40)


# 3. Access a specific Extension (e.g., 'SCI' for Science)
sci_header = hdul['SCI'].header
sci_data = hdul['SCI'].data

# 4. Read the first 10 lines of the Header to understand metadata
print("First 10 Header Keywords:")
print(list(sci_header.items())[:10])
print("-" * 40)

# 5. Check the shape of the Data array
# For 3D IFU Spectra, it's (Wavelength, Y_Pixels, X_Pixels)
print("Data Shape:", sci_data.shape)

# Close the file when done to save memory
hdul.close()
```

---

## Part 2: Code Explanation for `COS87259.ipynb`

This section provides a line-by-line conceptual breakdown of the logic in the original Notebook used to extract the spectra from the IFU cubes.

### Section 2.1: Initialization and Loading Data

```python
# Import astronomy and processing libraries
import numpy as np
from astropy.io import fits
from astropy import wcs
from scipy.ndimage import binary_dilation
from scipy.signal import savgol_filter
import sep  # Python and C library for Source Extraction and Photometry
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Define which grating file to process (could be 'prism' or 'G395H')
grating = 'prism'

# Open the unprocessed FITS file and extract the Science (sci) and Error (err) cubes 
s3d_file = 'NIRSpec_IFU/COS-87259_NOBG_prism-clear_s3d.fits'
s3d = fits.open(s3d_file)
hdr = s3d['sci'].header
f3d = s3d['sci'].data  # 3D Flux array (Wavelength, Y, X)
e3d = s3d['err'].data  # 3D Error array
```

### Section 2.2: Wavelength Construction

```python
# WCS Coordinate reconstruction for the wavelength axis (NAXIS3)
# CRVAL3 = Starting wavelength coordinate
# CDELT3 = Wavelength step (delta) per pixel layer
# NAXIS3 = Total number of wavelength layers
wl0, dwl, wl_len = hdr['CRVAL3'], hdr['CDELT3'], hdr['NAXIS3']

# Construct an array of exactly what wavelength corresponds to which layer [0, 1, 2...]
wl = wl0 + np.arange(wl_len) * dwl 
```

### Section 2.3: Inverse Variance Weighted Stacking

To see faint structures in 3D data, we collapse the useful wavelength range down into a single 2D image map.

```python
# Identify a safe wavelength range without excessive noise edges
wl_det = (wl>0.95) & (wl<5.29) 

# Coadd (collapse) the 3D cube into a 2D image. 
# It divides the flux (f3d) by the variance (e3d^2) to heavily favor signals with lower error. 
# High error pixels contribute much less to the final image, filtering out noise.
ivw_stack = np.nanmean(f3d[wl_det]/e3d[wl_det]**2, axis=0) / np.nanmean(1./e3d[wl_det]**2, axis=0)

# Create a boolean True/False mask identifying pixels that have NaN (Not a Number) values.
nan_mask = np.isnan(ivw_stack)
# Expands the borders of NaNs marginally to avoid edge artifacts from spilling into calculations
nan_mask = binary_dilation(nan_mask, iterations=2) 
ivw_stack[nan_mask] = np.nan
```

### Section 2.4: Global Background Subtraction

We need to remove the uniform background "glow" from the telescope and sky without removing the targets.

```python
# Calculate the background level of the 2D image using the SExtractor algorithm 
# bw, bh means background is evaluated over 5x5 pixel mesh boxes
bkg = sep.Background(ivw_stack, mask=nan_mask, bw=5, bh=5)

# Calculate a detection threshold to find objects: (2.5 * global noise) + background average
det_thre = 2.5 * bkg.globalrms + bkg.back()

# Find coordinates in the 2D image where the flux is higher than the threshold (i.e. real sources)
obj_pix = np.where((~nan_mask) & (ivw_stack > det_thre))
obj_x, obj_y = obj_pix[1], obj_pix[0]

# Flag these source pixels as 'False' in the background mask. 
# This means "Do not use these pixels to evaluate the true empty background."
for y, x in zip(obj_y, obj_x): bkg_mask[y, x] = False
```

```python
# Loop entirely through the thousands of layers in the 3D cube
# Compute the real median background only using strictly empty space (bkg_mask == True)
bkg_flux = np.zeros(wl_len)
for i in range(wl_len): 
    bkg_flux[i] = np.nanmedian(f3d[i][bkg_mask])

# Smooth out the newly derived 1D background spectrum 
# (Reduces jagged interpolation noise over high resolution data)
if grating == 'G395H': bkg_flux = savgol_filter(bkg_flux, 50, 1)

# Subtract this smoothed background spectrum array from EVERY pixel in the 3D cube
f3d_bkg_sub = np.zeros_like(f3d)
for i in range(wl_len): 
    f3d_bkg_sub[i] = f3d[i] - bkg_flux[i]
```

### Section 2.5: Identifing the Clumps

After flattening the background-subtracted cube down into a clean 2D image, the SExtractor (`sep`) library identifies distinct connected sources.

```python
# Run Source Extraction algorithm to find separated glowing clumps of pixels
# `1.5` represents detecting features that are 1.5 standard deviations above the background globalrms error
objects = sep.extract(stack_bkg_sub, 1.5, err=bkg.globalrms, mask=nan_mask)

# Derive 'Kron Radius' which calculates a scalable elliptical aperture radius 
# capturing the maximum flux distribution around the specific center (x, y) of each found target
r_kron, flag_kron = sep.kron_radius(stack_bkg_sub, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], 6, mask=nan_mask)

# At this point, the original author hardcoded the resulting array numbers for the known 5 clumps
# clump_ids = ['Center', 'West', 'North', 'South', 'Southeast']
# Center points, Semi-major/minor axes (a, b), Kron radius constraint, and rotation tilt (theta).
```

### Section 2.6: Extracting Individual Spectra

Using the defined elliptical geometrical properties for each clump, we loop through all layers of the 3D cube to count pixels belonging to this clump to plot a 1D graph.

```python
for n, clump_id in enumerate(clump_ids): 
    # Create isolated mask bounds for overlapping clumps like Center vs West
    # Specifically setting opposite adjacent pixels to NaN to prevent contamination

    # Initialize empty 1D float arrays for final Output values
    f1d, e1d = np.zeros_like(wl), np.zeros_like(wl)
    
    # Iterate layer by layer throughout the entire 3D array (z-axis)
    for i in range(wl_len):
        
        # Grab the individual 2D image layer (flux and error)
        f2d, e2d = f3d[i], e3d[i]
        
        # The core extraction command: 
        # Calculate the sum of all pixels that fall within the specified mathematical ellipse (x,y,a,b,theta,radius)
        # Handles edge pixel proportions seamlessly
        aper_flux, aper_flux_err, aper_flux_flag = sep.sum_ellipse(f2d, [x[n]], [y[n]], [a[n]], [b[n]], [theta[n]], [r[n]], mask=flux_mask)
        
        # Multiply by PIXAR_SR (pixel area in steradians) to get proper surface brightness flux 
        f1d[i] = aper_flux[0] * fnu_0
        
    # Convert unit formulas from MegaJanskys (per steradian) to Absolute Physical Energy Flux: 
    # Unit -> ergs per second per sq-centimeter per Angstrom (erg/s/cm^2/Angstrom)
    f1d = f1d * 1e-17 * 3e18 / (wl * 1e4)**2
    
    # Plotting code
    plt.plot(wl, f1d, 'k-', ds='steps-mid', label='Kron aperture')
    plt.show()
```
