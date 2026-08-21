# Spectrum Line Fitting Implementation Plan

The objective is to accurately model and fit the G395H 1D spectra for each clump. Because the clumps exhibit different physical properties—like the West clump showing broad-line AGN signatures—the model must be robust enough to separate overlapping emission lines and underlying background continuum. 

## Proposed Strategy

Rather than fitting the whole 2.8 - 5.3 $\mu$m spectrum simultaneously, we will isolate specific local wavelength windows surrounding the major expected rest-frame optical emission lines. This approach significantly simplifies the continuum background shape to a simple linear line while focusing the solver on the Gaussian fluctuations.

### Phase 1: Sub-setting the Spectrum
We will extract two primary spectral windows based on the redshift ($z=6.85$):
1.  **H$\beta$ and [OIII] Window**: Capturing rest-frame ~4800 Å to 5100 Å.
2.  **H$\alpha$ Window**: Capturing rest-frame ~6500 Å to 6650 Å (shifted to ~5.15 $\mu$m observed).

### Phase 2: Defining the Fit Functions
To get accurate line widths characterizing the kinematics of the gas, we must fit the background and the lines simultaneously. 

1.  **The Background (Local Continuum)**: 
    *   **Function**: Linear Polynomial ($f(\lambda) = m\lambda + c$).
    *   *Why*: Over a narrow window (~0.3 $\mu$m observed), the dust/stellar continuum is effectively a straight sloping line.
2.  **The Narrow Emission Lines**:
    *   **Function**: Gaussian Profiles.
    *   We will fit individual narrow Gaussians for H$\beta$ (4861 Å), [OIII] (4959 Å), [OIII] (5007 Å), and H$\alpha$ (6563 Å).
3.  **The Broad Emission Lines (AGN BLR)**:
    *   **Function**: Broad Gaussian Profile.
    *   For clumps showing broad features (like the West clump), we will include an additional *broad* Gaussian component at the H$\beta$ and H$\alpha$ locations to account for gas moving at extreme velocities purely near the black hole. 

### Phase 3: Applying Physical Constraints
A completely free mathematical fit will often result in unphysical solutions. To ensure the fit reflects actual astronomy, we apply strict constraints to the parameters:
1.  **Wavelength Locking**: The relative distances between the line centers (e.g., [OIII] 4959 vs 5007) must strictly obey laboratory formulas. They shift together linearly based on the same redshift $z$.
2.  **Flux Ratio Locking**: Atomic physics dictates that the [OIII] 5007 Å line is exactly ~2.98 times brighter than the [OIII] 4959 Å line. We will enforce this 1:3 area ratio in the optimizer.
3.  **Velocity Width Tying**: All narrow lines coming from the same cloud of gas will share the exact same intrinsic line width. Thus, the FWHM (Full Width at Half Maximum) in velocity space (km/s) will be tied together for Narrow H$\alpha$, H$\beta$, and the [OIII] doublet.

### Phase 4: Implementation using `lmfit`
*   Instead of basic `scipy.optimize.curve_fit`, we will utilize the **`lmfit`** Python library. 
*   `lmfit` makes it natively easy to declare parameter limits, tie variables together with mathematical expression strings (e.g., `amp_O5007 = 3 * amp_O4959`), and it generates robust statistical error reports on the resulting line widths.

---

> [!IMPORTANT]
> ## User Review Required
> Please check the breakdown above. 
> 
> 1. Do you agree with using `lmfit` to tightly lock the parameters like the 1:3 [OIII] ratio and constant narrow-line velocity FWHMs?
> 2. Do you want the Python script to run automatically for all clumps, or output visual plots of the separate fitted components (the narrow lines vs the broad lines vs the continuum) for manual checking?
> 3. Once you give approval on this plan, we can move strictly into the coding execution phase.
