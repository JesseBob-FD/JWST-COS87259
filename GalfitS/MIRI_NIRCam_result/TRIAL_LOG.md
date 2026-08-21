# MIRI + NIRCam Joint Fitting: Trial Log

## Target

- COS-87259, z = 6.83, RA = 149.74276239°, Dec = 1.6555373°
- C1 (Center) = [0.124, 0.139] arcsec; C2 (West) = [−0.116, 0.169] arcsec (clumps from F410M)

## Data & Preprocessing

- **MIRI**: 9 bands (F560W–F2550W), 2" half-size cutouts, 2nd-order polynomial sky subtraction (σ=3 clipping), Gaussian PSF (FWHM = λ/D), sigma from JWST ERR extension. Pixel scale 0.06"/px. Unit MJy/sr. Source: `MIRI_cutout_v3/`
- **NIRCam**: 3 bands (F115W, F200W, F410M), 2" half-size cutouts, σ-clipped median sky subtraction, σ-clipped std as sigma, Gaussian PSF. Pixel scale 0.03"/px. Unit MJy/sr (converted from nJy via PHOTMJSR). Source: `NIRCam_cutout/`

## Fit Settings (all runs)

- Method: optimizer, 3000 steps, lr = 0.001 (CPU JAX)
- Atlases: `jwst_miri` (9 bands) + `jwst_nircam` (3 bands)
- Modes: `noSED` (Ia15=0, per-band logNorm per component); `SED` (Ia15=1, BC03 host + AGN SED)
- **NP AGN** (Na18=4): AGN flux set by 12 per-band free `logL` parameters. The other AGN parameters (logM, logLedd, spin, Av, logL5100, slope, torus, hot dust) stay at their initial values in the best fit (they receive no gradient from the NP branch).
- AGN per-band `logL` initial values interpolated from the Elvis+94 radio-quiet quasar template.
- **Fixed center** means `vary=0` on xcen/ycen in the config (5th element of the value array).

## Run Summary (chronological by fit completion)

| Date | ID | Model | Mode | χ² | red. χ² | BIC | Config (current path) |
|------|----|-------|------|-----|---------|-----|-----------------------|
| 2026-07-18 | R001 | single Sérsic | noSED | 53,508.65 | 0.6135 | 53,713.42 | `single_sersic/noSED/COS87259_R001_noSED.lyric` |
| 2026-07-18 | R002 | single Sérsic | SED | 55,748.22 | 0.6392 | 55,873.36 | `single_sersic/SED/COS87259_R001_SED.lyric` |
| 2026-07-18 | R005 | bulge + disk | noSED | 53,508.66 | 0.6137 | 53,918.20 | `bulge_disk/noSED/COS87259_R003_noSED.lyric` |
| 2026-07-18 | R006 | bulge + disk | SED | 55,653.25 | 0.6381 | 55,903.52 | `bulge_disk/SED/COS87259_R003_SED.lyric` |
| 2026-07-18 | R007 | Sérsic + free sky | noSED | 50,611.37 | 0.5804 | 50,952.65 | `free_sky_single_sersic/noSED/COS87259_R004_noSED.lyric` |
| 2026-07-18 | R008 | Sérsic + free sky | SED | 53,996.36 | 0.6192 | 54,258.01 | `free_sky_single_sersic/SED/COS87259_R004_SED.lyric` |
| 2026-07-18 | R003 | host + power-law AGN | noSED | 50,862.51 | 0.5832 | 51,067.28 | `host_agn_powerlaw/noSED/COS87259_R002_noSED.lyric` |
| 2026-07-18 | R004 | host + power-law AGN | SED | 51,909.28 | 0.5952 | 52,125.42 | `host_agn_powerlaw/SED/COS87259_R002_SED.lyric` |
| 2026-07-25 | R009 | host + NP AGN | SED | 99,007.55 | 1.1354 | 99,348.83 | `host_freeAGN/SED/COS87259_R009_NPSED.lyric` |
| 2026-07-26 | R010 | host + NP AGN | noSED | 53,442.40 | 0.6129 | 53,783.68 | `host_freeAGN/noSED/COS87259_R010_NPnoSED.lyric` |
| 2026-07-26 | — | host(C1) + NP AGN(C2), centers fixed | noSED | 49,113.79 | 0.5632 | 49,432.31 | `fixed_center_C1host_C2AGN/noSED/C1host_C2AGN_noSED.lyric` |
| 2026-07-26 | — | host(C1) + NP AGN(C2), centers fixed | SED | 74,585.15 | 0.8553 | 74,880.92 | `fixed_center_C1host_C2AGN/SED/C1host_C2AGN_SED.lyric` |
| 2026-07-26 | — | host(C2) + NP AGN(C1), centers fixed | noSED | 52,951.86 | 0.6072 | 53,270.38 | `fixed_center_C2host_C1AGN/noSED/C2host_C1AGN_noSED.lyric` |
| 2026-07-26 | — | host(C2) + NP AGN(C1), centers fixed | SED | 57,639.18 | 0.6609 | 57,934.95 | `fixed_center_C2host_C1AGN/SED/C2host_C1AGN_SED.lyric` |
| 2026-07-27 | — | two Sérsic at C1, C2 (no AGN) | noSED | 48,345.35 | 0.5544 | 48,709.38 | `two_sersic_C1C2/noSED/C1C2_noSED.lyric` |
| 2026-07-27 | — | two Sérsic at C1, C2 (no AGN) | SED | 56,784.54 | 0.6511 | 56,989.31 | `two_sersic_C1C2/SED/C1C2_SED.lyric` |
| 2026-07-31 | R011 | dual NP AGN, centers free | noSED | 57,982.44 | 0.6650 | 58,460.23 | `dual_AGN/noSED/dual_AGN_C1C2_noSED.lyric` |
| 2026-07-31 | R012 | host + dual NP AGN, all centers free | noSED | 47,341.38 | 0.5431 | 48,023.91 | `host_dualAGN/noSED/host_dualAGN_C1C2_noSED.lyric` |
| 2026-08-02 | R013 | 2 Sérsic (centers fixed) + 2 NP AGN (centers free) | noSED | 47,567.80 | 0.5458 | 48,409.59 | `two_sersic_two_AGN/noSED/two_sersic_two_AGN_C1C2_noSED.lyric` |
| 2026-08-17 | — | host + dual NP AGN, AGN centers fixed | noSED | 48,492.45 | 0.5563 | 49,129.49 | `host_dualAGN_C1C2/noSED/host_dualAGN_C1C2_noSED.lyric` |
| 2026-08-17 | — | dual NP AGN, centers fixed | noSED | 62,550.41 | 0.7174 | 62,982.70 | `dual_AGN_C1C2/noSED/dual_AGN_C1C2_noSED.lyric` |
| 2026-08-17 | — | 2 Sérsic (centers fixed) + 2 NP AGN (centers fixed) | noSED | 49,109.95 | 0.5634 | 49,906.24 | `two_sersic_two_AGN_C1C2/noSED/two_sersic_two_AGN_C1C2_noSED.lyric` |

---

## Run Details

### 2026-07-18 — baseline series (R001–R008)

Single Sérsic / bulge+disk / free-sky / host+power-law-AGN setups, each in noSED and SED mode. Execution order that day (from file timestamps): R001, R002, R005, R006, R007, R008, R003, R004 — R-numbers do not match execution order.

| Run | Model | Mode | χ² | Δχ² vs noSED pair |
|-----|-------|------|-----|-------------------|
| R001 / R002 | single Sérsic | noSED / SED | 53,508.65 / 55,748.22 | +2,239.6 |
| R005 / R006 | bulge + disk | noSED / SED | 53,508.66 / 55,653.25 | +2,144.6 |
| R007 / R008 | Sérsic + free sky | noSED / SED | 50,611.37 / 53,996.36 | +3,385.0 |
| R003 / R004 | host + PL AGN | noSED / SED | 50,862.51 / 51,909.28 | +1,046.8 |

- For every paired setup, the noSED fit has lower χ² than the SED fit.
- R007 (free sky) fitted sky values per band: 0.0034, 0.0041, 0.0079, 0.0041, 0.0011, 0.0048, −0.0054, −0.0097, 0.0791 (MIRI F560W→F2550W); 0.0008, 0.0027, 0.0106 (NIRCam) MJy/sr. All ≤ 0.011 MJy/sr except F2550W.
- Lowest BIC on this date: R007 (50,952.65).

### 2026-07-25 / 07-26 — NP AGN runs (R009, R010) and fixed-center host+AGN runs

| Run | Model | Mode | χ² | red. χ² | BIC |
|-----|-------|------|-----|---------|-----|
| R009 | host + NP AGN | SED | 99,007.55 | 1.1354 | 99,348.83 |
| R010 | host + NP AGN | noSED | 53,442.40 | 0.6129 | 53,783.68 |
| C1host_C2AGN | host at C1 + NP AGN at C2 | noSED | 49,113.79 | 0.5632 | 49,432.31 |
| C1host_C2AGN | host at C1 + NP AGN at C2 | SED | 74,585.15 | 0.8553 | 74,880.92 |
| C2host_C1AGN | host at C2 + NP AGN at C1 | noSED | 52,951.86 | 0.6072 | 53,270.38 |
| C2host_C1AGN | host at C2 + NP AGN at C1 | SED | 57,639.18 | 0.6609 | 57,934.95 |

- R009 (NP SED) has the largest χ² and BIC of all runs; its red. χ² = 1.135 is the closest to 1 of all runs.
- In the fixed-center host+AGN runs, all AGN parameters are fixed except the 12 per-band `logL` (free).
- The noSED C1host_C2AGN fit (49,113.79) is better than its C2host_C1AGN counterpart (52,951.86) by Δχ² = 3,838.

### 2026-07-27 — two Sérsic at C1/C2 (no AGN)

| Mode | χ² | red. χ² | BIC |
|------|-----|---------|-----|
| noSED | 48,345.35 | 0.5544 | 48,709.38 |
| SED | 56,784.54 | 0.6511 | 56,989.31 |

Both Sérsic centers fixed at C1 and C2. This was the best model before the dual-AGN runs of 2026-07-31.

### 2026-07-31 — dual AGN (R011) and host + dual AGN (R012), free-center variants

Configs set all three (R012) / both (R011) centers free (`vary=1`). These folders were renamed on 2026-08-17: `dual_AGN_C1C2` → `dual_AGN`, `host_dualAGN_C1C2` → `host_dualAGN` (see Folder Notes). The fixed-AGN-center versions were re-run on 2026-08-17 under the `*_C1C2` names.

| Run | Model | χ² | red. χ² | BIC |
|-----|-------|-----|---------|-----|
| R011 | AGN_C1 + AGN_C2 NP, no host, centers free | 57,982.44 | 0.6650 | 58,460.23 |
| R012 | host Sérsic (free) + AGN_C1 + AGN_C2 NP, centers free | 47,341.38 | 0.5431 | 48,023.91 |

R012 best-fit parameters:
- host: xcen = 0.101, ycen = 0.099, Re = 0.641", n = 2.075, ang = 42.0°, axrat = 0.873
- AGN_C1 center: (0.154, 0.107) — drifted 0.044" from C1
- AGN_C2 center: (−0.101, 0.151) — 0.024" from C2

R012 component decomposition (μJy, `host_dualAGN/noSED/three_component_fluxes.csv`):

| Band | Host | AGN_C1 | AGN_C2 | Host% | C1% | C2% |
|------|------|--------|--------|-------|-----|-----|
| F560W | 3.32 | 0.10 | 0.16 | 92.7 | 2.8 | 4.4 |
| F1000W | 7.63 | 0.08 | 0.60 | 91.8 | 1.0 | 7.2 |
| F1500W | 16.92 | 0.48 | 6.43 | 71.0 | 2.0 | 27.0 |
| F2100W | 22.25 | 11.84 | 37.67 | 31.0 | 16.5 | 52.5 |
| F2550W | 77.69 | 1.18 | 90.81 | 45.8 | 0.7 | 53.5 |
| F410M | 14.22 | 0.20 | 0.08 | 98.1 | 1.4 | 0.6 |

### 2026-08-02 — two Sérsic + two NP AGN (R013), AGN centers free variant

Config: Sérsic centers fixed at C1/C2; both AGN centers free. Folder renamed on 2026-08-17: `two_sersic_two_AGN_C1C2` → `two_sersic_two_AGN`.

**χ²:** 47,567.80 | **red. χ²:** 0.5458 | **BIC:** 48,409.59

Best-fit parameters:
- sersic_C1: Re = 0.709", n = 2.114, ang = 15.5°, axrat = 0.885 (center fixed at C1)
- sersic_C2: Re = 0.271", n = 1.201, ang = 14.2°, axrat = 0.728 (center fixed at C2)
- AGN_C1 center: (0.136, 0.094) — drifted 0.047" from C1
- AGN_C2 center: (−0.100, 0.150) — 0.025" from C2

Component decomposition (μJy, `two_sersic_two_AGN/noSED/four_component_fluxes.csv`):

| Band | Sérsic_C1 | Sérsic_C2 | AGN_C1 | AGN_C2 | C1 group% | C2 group% |
|------|-----------|-----------|--------|--------|-----------|-----------|
| F560W | 2.39 | 0.72 | 0.15 | 0.09 | 75.7 | 24.4 |
| F1000W | 6.58 | 0.82 | 0.33 | 0.38 | 85.1 | 14.9 |
| F1500W | 10.39 | 3.26 | 2.63 | 5.00 | 61.2 | 38.8 |
| F2100W | 8.44 | 16.92 | 17.02 | 26.59 | 36.9 | 63.1 |
| F2550W | 48.28 | 31.00 | 16.82 | 67.82 | 39.7 | 60.3 |
| F410M | 15.32 | 0.44 | 0.29 | 0.09 | 96.7 | 3.3 |

(C1 group = sersic_C1 + AGN_C1; C2 group = sersic_C2 + AGN_C2; see `two_group_fluxes.csv`.)

### 2026-08-17 — fixed-AGN-center re-runs

Re-runs with only the AGN centers fixed at C1 and C2 (`vary=0`); all other settings identical to the original configs. Host Sérsic center remains free in the host+dualAGN config; Sérsic centers remain fixed in the two-Sérsic config.

| Run | χ² | red. χ² | BIC | Δχ² vs free-center pair |
|-----|-----|---------|-----|-------------------------|
| host_dualAGN_C1C2 (vs R012) | 48,492.45 | 0.5563 | 49,129.49 | +1,151.1 |
| dual_AGN_C1C2 (vs R011) | 62,550.41 | 0.7174 | 62,982.70 | +4,568.0 |
| two_sersic_two_AGN_C1C2 (vs R013) | 49,109.95 | 0.5634 | 49,906.24 | +1,542.2 |

**host_dualAGN_C1C2** best-fit parameters:
- host: xcen = 0.118, ycen = 0.094, Re = 0.615", n = 2.537, ang = 49.1°, axrat = 0.860
- AGN_C1 center fixed at (0.124, 0.139); AGN_C2 center fixed at (−0.116, 0.169)
- AGN_C1 logL: 37.9 (F560W) → 40.2 (F2100W); AGN_C2 logL: 38.3 (F560W) → 41.0 (F2550W)

Component decomposition (μJy, `three_component_fluxes.csv`):

| Band | Host | AGN_C1 | AGN_C2 | Host% | C1% | C2% |
|------|------|--------|--------|-------|-----|-----|
| F560W | 3.05 | 0.07 | 0.17 | 92.7 | 2.2 | 5.1 |
| F1000W | 7.28 | 0.05 | 0.57 | 92.1 | 0.7 | 7.2 |
| F1500W | 17.90 | 0.24 | 5.85 | 74.6 | 1.0 | 24.4 |
| F2100W | 25.37 | 13.28 | 34.07 | 34.9 | 18.3 | 46.9 |
| F2550W | 87.39 | 1.27 | 83.88 | 50.6 | 0.7 | 48.6 |
| F410M | 13.75 | 0.06 | 0.04 | 99.3 | 0.4 | 0.3 |

**dual_AGN_C1C2** component decomposition (μJy, `two_component_fluxes.csv`):

| Band | AGN_C1 | AGN_C2 | C1% | C2% |
|------|--------|--------|-----|-----|
| F560W | 0.38 | 0.33 | 53.3 | 46.7 |
| F1000W | 1.62 | 1.11 | 59.3 | 40.7 |
| F1500W | 5.55 | 7.15 | 43.7 | 56.3 |
| F2100W | 22.97 | 36.22 | 38.8 | 61.2 |
| F2550W | 41.49 | 91.09 | 31.3 | 68.7 |
| F410M | 0.54 | 0.12 | 82.2 | 17.8 |

**two_sersic_two_AGN_C1C2** best-fit parameters:
- sersic_C1: Re = 0.647", n = 2.614, ang = 22.0°, axrat = 0.898 (center fixed at C1)
- sersic_C2: Re = 0.263", n = 1.224, ang = 9.7°, axrat = 0.724 (center fixed at C2)
- AGN centers fixed at C1 and C2

Component decomposition (μJy, `four_component_fluxes.csv`):

| Band | Sérsic_C1 | Sérsic_C2 | AGN_C1 | AGN_C2 | C1 group% | C2 group% |
|------|-----------|-----------|--------|--------|-----------|-----------|
| F560W | 2.33 | 0.65 | 0.09 | 0.09 | 76.5 | 23.5 |
| F1000W | 6.69 | 1.00 | 0.08 | 0.17 | 85.2 | 14.8 |
| F1500W | 11.10 | 5.10 | 2.12 | 3.42 | 60.8 | 39.2 |
| F2100W | 9.58 | 20.80 | 18.04 | 21.37 | 39.6 | 60.4 |
| F2550W | 58.54 | 38.90 | 13.17 | 56.91 | 42.8 | 57.2 |
| F410M | 15.02 | 0.55 | 0.14 | 0.00 | 96.5 | 3.5 |

---

## Rankings

Computed with `compute_miri_stats.py` from the per-band chisq/dof in each `.gssummary`.
BIC_MIRI is derived from the full-fit BIC formula: BIC_full = χ²_full + k·ln(dof_full + k) is solved for k, then BIC_MIRI = χ²_MIRI + k·ln(dof_MIRI + k).

### Overall Ranking (noSED, by full BIC)

| Rank | Model | χ² | red. χ² | BIC | χ²_MIRI | BIC_MIRI | k_free |
|------|-------|-----|---------|-----|---------|----------|--------|
| 1 | R012 host + dual AGN (free centers) | 47,341.4 | 0.5427 | 48,023.9 | 19,083.5 | 19,715.3 | 60 |
| 2 | R013 2 Sérsic + 2 AGN (free AGN centers) | 47,567.8 | 0.5453 | 48,409.6 | 19,127.9 | 19,907.1 | 74 |
| 3 | C1C2 two-Sérsic | 48,345.4 | 0.5542 | 48,709.4 | 19,730.2 | 20,067.1 | 32 |
| 4 | host + dual AGN, AGN centers fixed (2026-08-17) | 48,492.5 | 0.5559 | 49,129.5 | 19,259.9 | 19,849.6 | 56 |
| 5 | C1host_C2AGN NP | 49,113.8 | 0.5630 | 49,432.3 | 19,526.3 | 19,821.1 | 28 |
| 6 | 2 Sérsic + 2 AGN, AGN centers fixed (2026-08-17) | 49,110.0 | 0.5630 | 49,906.2 | 19,438.1 | 20,175.2 | 70 |
| 7 | R007 Sérsic + free sky | 50,611.4 | 0.5802 | 50,952.6 | 20,175.7 | 20,491.5 | 30 |
| 8 | R003 host + power-law AGN | 50,862.5 | 0.5831 | 51,067.3 | 20,230.1 | 20,419.6 | 18 |
| 9 | C2host_C1AGN NP | 52,951.8 | 0.6070 | 53,270.4 | 20,092.4 | 20,387.2 | 28 |
| 10 | R001 single Sérsic | 53,508.6 | 0.6134 | 53,713.4 | 20,197.9 | 20,387.4 | 18 |
| 11 | R010 host + NP AGN | 53,442.4 | 0.6126 | 53,783.7 | 19,566.4 | 19,882.2 | 30 |
| 12 | R005 bulge + disk | 53,508.7 | 0.6134 | 53,918.2 | 20,197.9 | 20,576.9 | 36 |
| 13 | R011 dual AGN no host (free centers) | 57,982.4 | 0.6647 | 58,460.2 | 20,377.2 | 20,819.5 | 42 |
| 14 | dual AGN, centers fixed (2026-08-17) | 62,550.4 | 0.7170 | 62,982.7 | 21,296.6 | 21,696.7 | 38 |

SED runs (different parameter space, listed by χ², not ranked with noSED):

| Model | χ² | red. χ² | BIC |
|-------|-----|---------|-----|
| R004 host + PL AGN | 51,909.3 | 0.5951 | 52,125.4 |
| R008 Sérsic + free sky | 53,996.4 | 0.6190 | 54,258.0 |
| R006 bulge + disk | 55,653.2 | 0.6380 | 55,903.5 |
| R002 single Sérsic | 55,748.2 | 0.6391 | 55,873.4 |
| C1C2 two-Sérsic | 56,784.5 | 0.6510 | 56,989.3 |
| C2host_C1AGN NP | 57,639.2 | 0.6607 | 57,935.0 |
| C1host_C2AGN NP | 74,585.1 | 0.8550 | 74,880.9 |
| R009 host + NP AGN | 99,007.5 | 1.1350 | 99,348.8 |

### MIRI-Only Ranking (noSED, by BIC_MIRI)

| Rank | Model | χ²_MIRI | dof_MIRI | red. χ²_MIRI | BIC_MIRI | χ²_full | BIC_full |
|------|-------|---------|----------|--------------|----------|---------|----------|
| 1 | R012 host + dual AGN (free centers) | 19,083.5 | 37,394 | 0.5103 | 19,715.3 | 47,341.4 | 48,023.9 |
| 2 | C1host_C2AGN NP | 19,526.3 | 37,394 | 0.5222 | 19,821.1 | 49,113.8 | 49,432.3 |
| 3 | host + dual AGN, AGN centers fixed (2026-08-17) | 19,259.9 | 37,394 | 0.5151 | 19,849.6 | 48,492.5 | 49,129.5 |
| 4 | R010 host + NP AGN | 19,566.4 | 37,394 | 0.5232 | 19,882.2 | 53,442.4 | 53,783.7 |
| 5 | R013 2 Sérsic + 2 AGN (free AGN centers) | 19,127.9 | 37,394 | 0.5115 | 19,907.1 | 47,567.8 | 48,409.6 |
| 6 | C1C2 two-Sérsic | 19,730.2 | 37,394 | 0.5276 | 20,067.1 | 48,345.4 | 48,709.4 |
| 7 | 2 Sérsic + 2 AGN, AGN centers fixed (2026-08-17) | 19,438.1 | 37,394 | 0.5198 | 20,175.2 | 49,110.0 | 49,906.2 |
| 8 | C2host_C1AGN NP | 20,092.4 | 37,394 | 0.5373 | 20,387.2 | 52,951.8 | 53,270.4 |
| 9 | R001 single Sérsic | 20,197.9 | 37,394 | 0.5401 | 20,387.4 | 53,508.6 | 53,713.4 |
| 10 | R003 host + PL AGN | 20,230.1 | 37,394 | 0.5410 | 20,419.6 | 50,862.5 | 51,067.3 |
| 11 | R007 Sérsic + free sky | 20,175.7 | 37,394 | 0.5395 | 20,491.5 | 50,611.4 | 50,952.6 |
| 12 | R005 bulge + disk | 20,197.9 | 37,394 | 0.5401 | 20,576.9 | 53,508.7 | 53,918.2 |
| 13 | R011 dual AGN no host (free centers) | 20,377.2 | 37,394 | 0.5449 | 20,819.5 | 57,982.4 | 58,460.2 |
| 14 | dual AGN, centers fixed (2026-08-17) | 21,296.6 | 37,394 | 0.5695 | 21,696.7 | 62,550.4 | 62,982.7 |

---

## Observations

1. **noSED vs SED**: in every paired setup on 2026-07-18 and 07-26/27, the noSED fit has lower χ² and lower BIC than the SED fit. R009 (NP SED) is the only run with red. χ² ≈ 1 (1.135); all other runs have red. χ² ≈ 0.54–0.72.
2. **Per-band fit quality**: NIRCam per-band red. χ² (≈0.53–0.61) is higher than that of most MIRI bands (≈0.41–0.54); among MIRI bands F2550W has the highest per-band red. χ² (0.98 in the host+dualAGN fits).
3. **NP AGN behavior**: only the 12 per-band `logL` parameters move in the fit; logL5100, logM, logLedd, spin, Av, torus and hot-dust parameters stay at their initial values.
4. **Free vs fixed AGN centers**: fixing the AGN centers at C1/C2 increases χ² relative to the free-center counterparts by +1,151 (host+dualAGN), +1,542 (2Sérsic+2AGN), +4,568 (dual AGN only).
5. **Free-center fits drift**: with centers free, AGN_C1 drifts ≈0.045" from C1 in both R012 and R013, while AGN_C2 stays within ≈0.03" of C2.
6. **Component fluxes with fixed AGN centers**: in the fixed-center runs, both AGN components carry flux across the MIRI bands (e.g. host_dualAGN_C1C2: AGN_C1 = 18% and AGN_C2 = 47% at F2100W; dual_AGN_C1C2: C1/C2 ≈ 53/47 at F560W falling to 31/69 at F2550W).
7. **Clump groups**: in the 2Sérsic+2AGN runs, the C1 group (Sérsic+AGN at C1) dominates at λ ≤ 5.6 μm (≥76%), the C2 group dominates at F2100W–F2550W (57–63%).

## Decomposition Outputs

Each `noSED` folder contains the component separation script and outputs:
- per-band component FITS: `{Component}_model_{band}.fits`
- flux table: `*_fluxes.csv` (full 12-band data)
- images: `*_model_images.png`, `*_model_images_MIRI.png`
- SED plots: `*_SED.png` (F_ν and νF_ν)

Reconstruction check (component sum vs total model) is exact to floating-point precision (~10⁻⁸) in all runs.

## Folder Notes

- **2026-08-17 renames** (free-center variant results kept, folder names dropped the C1C2 suffix):
  - `host_dualAGN_C1C2` → `host_dualAGN`
  - `dual_AGN_C1C2` → `dual_AGN`
  - `two_sersic_two_AGN_C1C2` → `two_sersic_two_AGN`
  - `two_sersic_C1C2` unchanged (no AGN components; Sérsic centers fixed).
- The new fixed-AGN-center runs (2026-08-17) live under the `*_C1C2` names.
- Earlier runs were moved from `run001/`–`run010/` folders into descriptive folders (`single_sersic/`, `host_agn_powerlaw/`, `bulge_disk/`, `free_sky_single_sersic/`, `host_freeAGN/`); config paths listed in this log are the current locations. The `.gssummary` headers of those early runs still record the old `runXXX/` paths.
- `host_freeAGN/SED/COS87259_R009_NPnoSED.lyric` is a leftover config with no fit output (no `.gssummary`).

---

*Last updated: 2026-08-17*
