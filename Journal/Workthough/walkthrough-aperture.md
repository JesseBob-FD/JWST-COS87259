# Second Stage Mission: Advanced Aperture Paradigms

This document outlines the completion of the second stage mission, replacing the hardcoded geometric ellipses with advanced pixel-accurate spatial masks for extracting 1D spectra from the high-resolution G395H IFU mapping.

All code and outputs are entirely self-contained inside your new directory: `d:/Fudan_University/Research/JWST/NewApertureTrials`.

## 1. Paradigm 1: Flux Contour Mapping
**Interactive Notebook:** `Contour_Aperture.ipynb`

As proposed by your original idea, this notebook completely abandons simple boundaries and traces the physical shapes of the clumps across the 2D plane based on their actual flux topology. 

**Workflow:**
1. **Dynamic Core Targeting**: The algorithm searches around predefined anchors to detect the absolute maximum-flux peak pixel of each clump dynamically. 
2. **Standardized Density Drops**: It builds contiguous regional footprint masks for `80%`, `50%`, and `20%` of that central peak. For example, the `50%` mask traces the entire organic footprint up until the light dims to half of its core strength (provided it stays well above the standard $3\sigma$ noise floor).
3. **Collision handling**: If an outer limit like $20\%$ begins to swallow a neighboring clump (e.g., Center touching West), it explicitly isolates only the pixel cluster continuously tied to its *own* peak using the `scipy.ndimage.label` function, protecting the spectral purity.
4. **Extraction**: You will see it saves 1D `.txt` spectra for *every* level of *every* clump in the inner `Contour_Output/` folder, and finishes by automatically fitting the $H\beta + [OIII]$ lines for the extracted arrays.

## 2. Paradigm 2: Watershed AI Segmentation
**Interactive Notebook:** `Watershed_Aperture.ipynb`

As an alternate parallel trial, I implemented a technique widely considered the gold standard for separating intimately touching objects in astrophysics: Topological Watershed Segmentation.

**Workflow:**
1. **Topography generation**: We treat the brightness of the map as geometric depth. Bright pixels are deep valleys, and empty space noise acts as high mountain ranges.
2. **Flooding the basins**: Markers are dropped at the cores of the 5 clumps simultaneously, and the borders are grown outward "uphill" mathematically. 
3. **Ultimate Separation**: When the boundary from the West clump naturally collides with the boundary of the Center clump, the algorithm automatically plots a strict border precisely at the saddle point (the valley ridge where flux is mathematically lowest between them). 
4. **Extraction**: This guarantees every photon belongs to exactly one clump with zero overlap! The notebook saves these definitive 1D spectra inside the `Watershed_Output/` folder, and concludes by fitting the models immediately below.

---

### Verification
Both `.ipynb` files are fully formatted and commented, allowing you to run them cell-by-cell. You will immediately see visual image plots of the mask footprints (so you can explicitly guarantee the contours look correct before doing anything), followed by the extracted models. 
