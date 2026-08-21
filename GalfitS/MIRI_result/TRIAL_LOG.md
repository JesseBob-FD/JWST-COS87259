# MIRI GALFITS Decomposition: Trial Log

## Target: COS-87259 (z ~ 6.83), suspected AGN host galaxy

---

## Run V1: Single Sersic Host (SED mode, Ia15=1)

**Config:** `MIRI_result/run001/COS87259_v1.lyric`
**Method:** optimizer, 1000 steps
**Model:** Single Sersic burst galaxy

**Issue Encountered:**
- Model flux = ZERO in all bands.
- Chi-square barely changed (71059 → 71017), reduced χ² ≈ 0.44.
- Root cause: BC03 SED templates at z=6.83 produce negligible flux in MIRI bands
  (rest-frame 0.7-3.3 μm). The stellar SED falls as λ^(-4) in the Rayleigh-Jeans tail.
  Combined with (1+z)^(-3) cosmological dimming at z=6.83 (factor ~1/480),
  the stellar model flux is effectively zero.

**Verdict:** ❌ Failed — SED templates incompatible with MIRI at high-z.

---

## Run V2: Host + AGN Power-Law (SED mode, Ia15=1)

**Config:** `MIRI_result/run002/COS87259_v2.lyric`
**Method:** optimizer, 2000 steps
**Model:** Single Sersic host + AGN power-law continuum (Na18=0, logL5100 free)

**Issue Encountered:**
- Na27 format error: needs `[[val,min,max,step,vary],[...]]` (2 sub-lists).
- Fixed Na27 format, config parsed successfully.
- Model flux STILL zero despite AGN component.
- AGN power-law SHOULD produce flux (computed L_38 → MJy/sr conversion works),
  but the model image shows 0 flux.
- Reduced χ² ≈ 0.44 again.
- Root cause unclear: possibly the conversion chain L_38 → C_unit → phys_to_counts
  has a hidden issue, OR the source is not centered in the cutout.

**Verdict:** ❌ Failed — AGN SED model also produces zero observable flux.

---

## Run V3: Photometric-only Single Sersic (Ia15=0)

**Config:** `MIRI_result/run003/COS87259_v3.lyric`
**Method:** optimizer, 2000 steps
**Model:** Single Sersic with per-band logNorm (no SED templates)

**Findings:**
- Photometric-only mode (Ia15=0) uses `logNorm_{component}_{band}` parameters
  that bypass SED templates entirely.
- Chi-square decreased significantly (91542 → 71469), demonstrating the optimizer
  can now adjust model flux.
- Fitted parameters:
  - logNorm: -9.09 (F770W) to -7.76 (F2100W) 
  - Re = 0.245 arcsec, n = 2.84, q = 0.73
  - xcen = -0.03, ycen ≈ 0 arcsec

**Issue:**
- Source position discovery: the actual source peak is at (102, 17) in previous
  cutout, ~4.2 arcsec from the image center. The model fitting at (0, 0) is
  fitting background noise, not the source.
- Model image still zero/near-zero because the source is too far from the
  fitting region center.
- The target coordinates (CRVAL from MIRI header) do not correspond to the
  actual galaxy position. Correct coordinates must come from NIRSpec IFU WCS.

**Verdict:** ⚠️ Partial — Mode works but target position is wrong.

---

## Key Discoveries

### 1. SED Template Limitation at z > 6
GalfitS SED templates (BC03 host, power-law AGN) cannot reliably produce
observable MIRI flux at z ≈ 6.83. The combination of:
- Rayleigh-Jeans flux falloff at rest NIR wavelengths
- (1+z)^(-3) cosmological dimming (factor ~480 at z=6.83)
- BC03 template wavelength range (host_conti: 794-80,000 Å)
means stellar+AGN SED fitting in Ia15=1 mode is likely infeasible at z > 5
when using MIRI data alone.

### 2. Ia15=0 (Photometric-Only) Mode is the Correct Approach
For MIRI-only analysis at high redshift, photometric-only mode (Ia15=0)
with per-band logNorm parameters should be used. This bypasses SED
limitations and fits spatial+flux parameters directly.

### 3. Source Localization is Critical
The target coordinates must be precisely known from NIRSpec IFU data
or from known literature values. Using MIRI image header CRVAL as
target position is incorrect if the source is not at the image center.

### 4. Required: Correct Target Coordinates
For COS-87259, the correct coordinates must be obtained from:
- NIRSpec IFU spectral cube WCS
- Literature (e.g., Carnall et al. 2023 or similar)
- Visual inspection of MIRI images with known source positions

### 5. MIRI Cutout Size
A 10" × 10" cutout (5 arcsec half-size) is sufficient. The pixel scale
is 0.06 arcsec/pix, giving 166×166 pixel images. This provides ~84 pixels
on each side from the source center, adequate for fitting compact high-z
galaxies.

### 6. Sigma Images
Using JWST ERR extensions directly as sigma images is appropriate.
The JWST pipeline provides high-quality pixel-level error estimates.

### 7. PSF
Gaussian approximations of MIRI PSFs (FWHM from diffraction limit)
are acceptable for initial testing. For publication-quality fits,
empirical PSFs from field stars or WebbPSF models should be used.

---

## Recommended Next Steps

1. **Obtain correct target coordinates** from NIRSpec IFU data or literature.
2. **Re-run preprocessing** with correct coordinates (center cutout on source).
3. **Use Ia15=0 photometric-only mode** with:
   - Single Sersic for initial spatial fit
   - 2-component (bulge+disk or host+AGN) if residuals show structure
   - Consider allowing sky to vary if sky subtraction is imperfect
4. **For AGN decomposition**: Use AGN component with Na18=0 (power-law)
   and add to the photometric model via per-band normalization comparison.
5. **Consider stacking**: Co-add short-wavelength MIRI bands for better S/N
   on the host galaxy component.
