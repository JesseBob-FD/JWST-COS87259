# MIRI Clump Photometry: COS-87259 (z=6.83)

Following `JWST/MissionLine/Third stage mission.md`, we measure MIRI broadband
fluxes for 5 clumps (Center, West, North, South, Southeast) from NIRCam/NIRSpec
IFU. C1 corresponds to **Center**, C2 to **West** (separation ~0.24").

The image decomposition uses a two-Sérsic model with centers fixed at C1 and C2
(`two_sersic_C1C2/noSED/C1C2_noSED.lyric`, χ²=48,345, BIC=48,709 — best overall).

---

## 1. Correct Results

### 1a. Component Model Images

**Image**: `component_model_images.png`  
**Script**: `plot_component_models.py` (final version)

**Method** (from `plot_resu.ipynb` + `zijian.ipynb`):  
Uses GalfitS's own `cal_model_image()` to regenerate model images with one
component's `logNorm` temporarily zeroed. Each row shows C1 (bulge=Center),
C2 (disk=West), and residual/sigma.

Left column = C1 model, middle = C2 model, right = (data - total)/sigma.  
C1+C2 reproduces GalfitS total model to floating-point precision (max diff ~1e-7).

**Key observations**:
- Both components show PSF broadening from short (F560W) to long (F2550W) wavelengths
- C1 and C2 have different spatial profiles (different Re, n, q) but the PSF
  FWHM (0.20"–0.77") dominates — the two components are blended in all bands
- Residual images are consistent with pure noise — no systematic structure

### 1b. Corrected Component Fluxes (μJy)

**Data**: `clump_fluxes_corrected.csv`  
**Script**: `recompute_fluxes.py`

Fluxes computed by summing the GalfitS-generated per-component model images
over unmasked pixels, converted to μJy via PIXAR_SR × 1e12.

| Band   | Center (C1) | West (C2) | Ratio C1/C2 | C1%  |
|--------|------------|-----------|-------------|------|
| F560W  | 1.78       | 1.00      | 1.78        | 64%  |
| F770W  | 2.96       | 1.14      | 2.60        | 72%  |
| F1000W | 4.85       | 1.81      | 2.68        | 73%  |
| F1130W | 6.35       | 1.68      | 3.78        | 79%  |
| F1280W | 8.39       | 4.29      | 1.96        | 66%  |
| F1500W | 11.9       | 11.6      | 1.03        | 51%  |
| F1800W | 27.8       | 20.1      | 1.38        | 58%  |
| F2100W | 28.3       | 58.3      | 0.49        | 33%  |
| **F2550W** | **59.3** | **122.3** | **0.49**   | **33%** |

**Physical interpretation**:  
C1 (Center) is bluer — it dominates at short MIRI bands.  
C2 (West) is redder — it dominates at F2100W and F2550W.  
Both components have steeply rising SEDs (F_nu ∝ ν^−α with α ~ 2.5–3.0),
consistent with hot dust emission at z=6.83 (rest-frame ~1–3 μm).

### 1c. Corrected SED Plot

**Image**: `clump_seds_corrected.png`  
**Script**: `recompute_fluxes.py`

F_nu vs observed wavelength, with power-law fits to each component.
C1 and C2 clearly diverge at long wavelengths — C2 is redder.

### 1d. Clump SED Overview

**Images**: `clump_seds.png`, `clump_seds_nuFnu.png`  
**Script**: `Journal/plot_clump_seds.py`

All 5 clumps on one plot: Center and West detected, North/South/Southeast
shown as 5σ upper limits (downward triangles).

### 1e. Upper Limits (North, South, Southeast)

**Script**: `Journal/upper_limits_clumps.py`

5σ upper limits from σ-clipped RMS in 10×10 px apertures:

| Band   | North | South | Southeast |
|--------|-------|-------|-----------|
| F560W  | 0.028 | 0.017 | 0.015     |
| F2550W | 0.895 | 0.608 | 0.615     |

All limits are 10–100× below Center/West fluxes — these 3 clumps are undetected in MIRI.

---

## 2. Summary: 5-Clump MIRI Photometry

| Clump      | Status   | F560W  | F770W  | F1500W  | F2550W   |
|------------|----------|--------|--------|---------|----------|
| **Center** | Detected | 1.8 μJy | 3.0 μJy | 11.9 μJy | 59.3 μJy  |
| **West**   | Detected | 1.0 μJy | 1.1 μJy | 11.6 μJy | 122.3 μJy |
| North      | <5σ      | <0.03  | <0.04  | <0.24   | <0.89    |
| South      | <5σ      | <0.02  | <0.02  | <0.06   | <0.61    |
| Southeast  | <5σ      | <0.02  | <0.02  | <0.09   | <0.62    |

---

## 3. Archived: Incorrect Results

The first two attempts used buggy separation methods. These results and their
scripts are **invalid** and kept only for provenance.

### 3a. First incorrect attempt: mass_map without PSF convolution

**Script**: `two_sersic_C1C2/noSED/separate_components.py` (v1)  
**Script**: `plot_component_models.py` (v1, deleted, replaced by v2)

**Bug**: Generated component model images as `mass_map × 10^logNorm × phys_to_counts_rate`
without PSF convolution. The `mass_map` is the raw Sérsic profile — PSF convolution
happens inside `Galaxy.generate_image()`, which was bypassed.

**Symptom**: All 9 bands showed identical spatial profiles. C1/(C1+C2) = 0.511
identically in every band — both components had exactly the same SED shape.

**Saved artifacts** (incorrect):
- `two_sersic_C1C2/noSED/C1_model_{band}.fits` — all zeros (wrong logNorm keys)
- `two_sersic_C1C2/noSED/component_fluxes.csv` — zero fluxes

### 3b. Second incorrect attempt: mass_map with manual PSF convolution

**Script**: `plot_component_models.py` (v2, later overwritten)  
**Bug**: Used `fftconvolve(mass_map × 10^logNorm, PSF) × phys_to_counts_rate`.
The PSF convolution was done AFTER flux scaling, but GalfitS convolves BEFORE
applying `phys_to_counts_rate`. Also, the two subcomponents' mass_maps share
the same PSF convolution in GalfitS (sum first, then convolve), while this
manual approach convolved them separately then summed.

**Invalid images** (overwritten by corrected versions):
- `separation_boundaries.png` — C1/C2 markers incorrectly positioned (ΔRA sign
  not corrected for PC1_1=-1.0 in MIRI WCS); gray_r color saturation created
  false "white holes" at source centers
- `separation_contours.png` — all 9 bands looked identical because the
  un-convolved Sérsic profiles were the same, and `normimg` normalization
  erased the flux amplitude differences
- Early version of `component_model_images.png` — C1+C2 total differed from
  GalfitS total model; residual images showed a blue dip at center indicating
  central flux overestimation
- `clump_seds.png` and `clump_seds_nuFnu.png` (v1) — used equal C1/C2 fractions (~50%)
  from the buggy extraction

### 3c. Correct method (used for final results in Section 1)

**Key insight from `plot_resu.ipynb`**: `cal_model_image()` directly sets
`im.model_image`. Call it once for total, then zero `pardict['logNorm_X']`,
call again to get the other component, subtract for the zeroed component.

**Script**: `plot_component_models.py` (final version)

```python
# 1) Total model
fitter.cal_model_image()
total = im.model_image

# 2) Zero C1 → C2-only
fitter.pardict['logNorm_bulge_band'] = -30
fitter.cal_model_image()
c2_only = im.model_image

# 3) Restore C1, zero C2 → C1-only
fitter.pardict['logNorm_disk_band'] = -30
fitter.cal_model_image()
c1_only = im.model_image
```

This reproduces GalfitS's exact PSF convolution, `phys_to_counts_rate`, and
all internal calibrations. C1+C2 matches total to ~1e-7 MJy/sr.

---

*Last updated: 2026-07-28*
