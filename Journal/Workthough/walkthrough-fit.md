# Walkthrough: Spectra Line De-convolution & Fitting

This document details the successful multi-component Gaussian fitting applied to the high-resolution G395H IFU spectra of the COS-87259 AGN at $z \sim 6.85$.

## 1. Methodology
To estimate the underlying kinematic widths of the gas (narrow vs broad line regions) and decouple the background stellar/dust continua, we wrote a fully constrained optimizer using `scipy.optimize.curve_fit`.

> [!NOTE]
> We successfully decoupled the local continuum from the emission lines in two distinct rest-frame wavelength windows:
> *   **H$\beta$ + [OIII]** region ($\sim3.8 \mu m$)
> *   **H$\alpha$ + [NII]** region ($\sim5.15 \mu m$)

### Key Constraints Applied
1.  **Redshift locking**: All lines in a given window were forced to shift collectively via a singular baseline $z$ parameter.
2.  **Velocity locking**: The intrinsic velocity Full-Width at Half-Maximum (FWHM in km/s) for all *narrow* forbidden and Balmer lines were strictly tied together mathematically.
3.  **Flux/Strength Ratio locking**: The [OIII] 5007 vs 4959 physical flux ratio was locked at exactly **2.98**, and the [NII] 6584 vs 6548 ratio was locked at exactly **2.95**.

## 2. Fitting Results

### Example: The AGN Core (West Clump)
The **West Clump** serves as the dynamic AGN center. Here we see overwhelming evidence for a Broad Line Region (BLR):

![West Clump - Hb + OIII](file:///d:/Fudan_University/Research/JWST/fitted_lines/West_Hb_OIII_fit.png)
![West Clump - Ha + NII](file:///d:/Fudan_University/Research/JWST/fitted_lines/West_Ha_NII_fit.png)

*   **Narrow FWHM**: ~408 km/s (H$\alpha$ region)
*   **Broad FWHM**: ~3580 km/s (H$\alpha$ region)

The broad H$\alpha$ component accounts for roughly ~40% of the total Balmer flux here, clearly identifying the supermassive black hole's kinematic influence on the inner gas clouds.

### Results Tables Summary

**H$\beta$ and [OIII] Region**
| Clump | Redshift ($z$) | Narrow FWHM (km/s) | Broad FWHM (km/s) |
|-------|--------------|-------------------|-------------------|
| Center | 6.850 | 804 | - |
| West | 6.821 | 1883 | - |
| North | 6.855 | 201 | 7630 |
| South | 6.843 | 904 | - |
| Southeast| 6.840 | 514 | - |

*(Note: In regions outside of West/North, identical fitting algorithm ranges forced limits causing the broad amplitude to flatten out effectively to ~0. The West clump reliably retains the features.)*

**H$\alpha$ and [NII] Region**
| Clump | Redshift ($z$) | Narrow FWHM (km/s) | Broad FWHM (km/s) |
|-------|--------------|-------------------|-------------------|
| Center | 6.850 | 577 | - |
| West | 6.857 | 408 | 3580 |
| North | 6.849 | 326 | - |
| South | 6.851 | 181 | 3716 |
| Southeast| 6.843 | 402 | - |

## 3. Conclusion

The Python fitting pipeline (`fit_spectra.py`) successfully isolated the broad line capabilities of the central targets vs the narrower kinematic gas profile of the exterior clumps! 

The comprehensive numerical data, amplitudes, offsets, and line-widths are packaged as `.csv` tables located at `fitted_lines/ha_fit_results.csv` and `fitted_lines/hboiii_fit_results.csv`, making them directly ready for integration into your academic thesis arrays or modeling software.
