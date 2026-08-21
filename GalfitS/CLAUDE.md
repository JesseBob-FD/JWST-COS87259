# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GalfitS is a galaxy imaging spectra fitting tool for multi-band bulge-disk SED decomposition. It jointly fits morphological parameters (Sérsic index, Re, axis ratio, PA) and SED physical parameters (age, metallicity, dust, stellar mass) in a unified pixel-level χ² framework, using JAX for GPU-accelerated automatic differentiation.

Key capabilities:
- **Joint morpho-SED fitting** across 29+ instruments (JWST NIRCam/MIRI, HST, SDSS, GALEX, 2MASS, WISE, Pan-STARRS, CSST, and more)
- **Multiple fitting methods**: gradient descent (Optax), CMA-ES (evosax), nested sampling (Dynesty, JNesty), MCMC (FlowMC), chi-square (lmfit/scipy)
- **Full SED physics**: BC03 stellar population synthesis (continuous + burst SFH), CLOUDY nebular emission, Draine & Li 2007 dust emission, AGN accretion disk and torus models
- **Mode auto-detection** from config: `images - photometry`, `images - SED`, `images + spectra - SED`, `spectrum - SED`, `grism - SED`

Reference: [Online Documentation](https://ruancunli.github.io/GalfitS/) | [Discord](https://discord.gg/SSCpB73DaF)

## Environment Setup

### Requirements

- **Python 3.11.x** (strict — 3.12+ is incompatible with JAX CUDA plugins)
- **CUDA 12.4** for GPU support; NVIDIA Driver 550.54.14+
- **`GS_DATA_PATH`** environment variable must point to the SED template data directory

### Installation

```bash
# From repo root
cd GalfitS
conda create -n galfits python=3.11 -y
conda activate galfits
pip install -r requirement.txt
pip install -e .

# GPU JAX:
pip install -r requirements-jax-cuda.txt

# CPU-only JAX:
pip install -r requirements-jax-cpu.txt
```

Set the template data path (persist in `~/.bashrc` or `~/.zshrc`):

```bash
export GS_DATA_PATH=/path/to/GalfitS/data
```

SED templates (`.npz` files) must be downloaded separately from [Google Drive](https://drive.google.com/drive/folders/1pWSl7po42iiv01HcjEA1_phlUX5jmqRW) and placed under `$GS_DATA_PATH/templates/`.

### Verify Installation

```bash
python -c "
import jax; print(f'JAX: {jax.__version__}, devices: {len(jax.devices())}, backend: {jax.default_backend()}')
import galfits; print('GalfitS imported successfully')
"
```

## Running GalfitS

```bash
galfits <config.lyric> --workplace <output_dir> [options]
```

### Commonly Used Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | required | Path to `.lyric` configuration file |
| `--workplace` / `--work` | `./` | Output directory |
| `--fit_method` | `optimizer` | `optimizer`, `ES`, `dynesty`, `jnesty`, `flowmc`, `chisq` |
| `--notfit` | false | Skip fitting, only generate `.params` and `.constrain` files |
| `--savefull_results` | false | Save full pickled results (`.gsresu`) |
| `--saveimgs` | false | Save model/output/residual images |
| `--savelog` | false | Tee stdout/stderr to a log file |
| `--num_steps` | 3000 | Optimizer iterations |
| `--learning_rate` | 0.0008 | Optimizer learning rate |
| `--multistart` | false | Multi-start optimizer across GPUs |
| `--nlive` | 80 | Live points for nested sampling |
| `--maxiters` | 100000 | Max iterations for nested sampling |
| `--dlogz` | 0.02 | Evidence tolerance for nested sampling |
| `--nchains` | 30 | Chains for FlowMC |
| `--num_generations` | 5000 | Generations for CMA-ES |
| `--popsize` | 10 | Population size for CMA-ES |
| `--baysian` | false | Use Bayesian priors in optimizer |
| `--cal_sigma` | false | Compute Hessian uncertainties |
| `--readpar` | None | Restart from previous `.params` file |
| `--parconstrain` | None | Apply constraint file |
| `--priorpath` | None | Path to astrophysical prior file |
| `--weight_spec` | 1.0 | Weight of spectrum in joint fitting |

### Fitting Methods

| Method | Flag | Best For |
|--------|------|----------|
| Gradient Descent | `optimizer` | Fast optimization, multi-start across GPUs |
| CMA-ES | `ES` | Global optimization, rugged landscapes |
| Dynamic Nested Sampling | `dynesty` | Bayesian evidence, posteriors |
| JAX Nested Sampling | `jnesty` | GPU-accelerated nested sampling |
| FlowMC | `flowmc` | MCMC posterior sampling |
| Chi-square | `chisq` | Traditional LM fitting (GALFIT-like) |

### Example Commands

```bash
# Quickstart: SDSS + 2MASS + GALEX multi-band fit
galfits data/quickstart.lyric --work data/result

# Gradient descent with multi-start
galfits config.lyric --fit_method optimizer --multistart --nstart 20 --num_steps 5000

# Nested sampling (JAX-native)
galfits config.lyric --fit_method jnesty --nlive 2000 --dlogz 0.01 --saveimgs

# First pass: generate parameter file, then edit and re-run
galfits config.lyric --notfit
# edit target.params, then:
galfits config.lyric --readpar target.params --parconstrain target.constrain

# With astrophysical priors
galfits config.lyric --fit_method dynesty --priorpath astro.prior
```

## Architecture

The codebase has four layers. The source lives in `GalfitS/src/galfits/`.

```
Application:  galfitS.py (CLI, argparse)  →  gsutils.py (config parsing, GSData container)
Fitting:      gsfit.py (gsfitter class — 6+ fit methods, χ², uncertainty estimation)
Model:        galaxy.py (Galaxy/AGN/Star)  +  profiles.py (2D profiles)
              sed_interp.py (SED templates)  +  emission_lines.py (lines, ISM)
Data:         images.py (FITS I/O, WCS, PSF, cutouts, sigma, source detection)
              disperser.py (grism spectroscopy)
```

### Key Source Modules

| Module | Purpose |
|--------|---------|
| `galfitS.py` | CLI entry point — `galfits` console command, argument parsing, dispatches to fit method |
| `gsfit.py` | Core fitting engine (~4760 lines) — `gsfitter` class: `optimizer()`, `optimizer_multi()`, `evolution_strategies()`, `nested_sampling()`, `jnesty_nested_sampling()`, `flowmc()`, `minimize()`, MCMC and Fisher uncertainty estimation |
| `gsutils.py` | Configuration loading (`read_config_file` wires up everything from `.lyric`), parameter initialization, constraint generation, photometric calibration tables, visualization |
| `images.py` | `image` class (FITS loading, WCS, cutout, PSF handling, sigma maps, sky subtraction, source detection with SEP), `image_atlas` (groups bands + spectra), `GSData` top-level container |
| `galaxy.py` | `Galaxy` class — multi-component galaxy (bulge/disk/bar) with SED gradients, nebular emission, dust. Generates model images from combined morphological + SED parameters |
| `galaxy_3D.py` | `Galaxy3D` class — IFU/3D spectroscopy with velocity broadening |
| `profiles.py` | 2D profile functions (all JAX-compatible): `sersic2D`, `Ferrer`, `EdgeonDisk`, `GaussianRing`, Fourier modes (`sersic2D_fourier`, `brokenfourier`, `ringfourier`), broken/ring Sérsic variants |
| `sed_interp.py` | SED template interpolation — loads BC03 stellar, AGN, Draine & Li dust, and nebular templates from `$GS_DATA_PATH/templates/`. Cosmology (luminosity distance, kpc/arcsec, cosmic age) |
| `emission_lines.py` | Emission line catalog (OVI, Lyα, CIV, MgII, OII, OIII, Hβ, Hα, SII, etc.) with rest wavelengths and ionization types. `ISM` class for nebular continuum + line modeling |
| `disperser.py` | `GSDisperser` — slitless spectroscopy forward model (HST G102/G141, JWST NIRCam grism, CSST GI/GV/GU) |
| `plot.py` | Model comparison plots, IFU interactive visualization |
| `mathfunc.py` | Aperture photometry, elliptical masks, polynomial fitting, cross-matching |
| `external.py` | SExtractor configuration generation |
| `constant.py` | Physical constants (degree_to_rad, twopi, ckm, FWHM-to-sigma, etc.) |

### Data Flow

1. `galfitS.py:main()` reads CLI args → calls `gsutils.read_config_file(config, workplace)`
2. `read_config_file` parses `.lyric` → creates `image` objects (loads FITS, PSFs, masks), `image_atlas` groups, `Galaxy`/`AGN`/`Star` model objects, and a `gsfitter` instance
3. `gsfitter` initializes parameters from config → builds a unified parameter vector θ (morphology + SED)
4. During fitting, `gsfitter.cal_residual(θ)` calls `Galaxy.generate_image(band)` which combines:
   - **SED**: `sed_interp.get_host_SED(logM, age, Z, Av, ...)` → flux per band via filter integration
   - **Morphology**: `profiles.sersic2D(Re, n, axrat, PA, ...)` → spatial light distribution
   - Result: model image = mass_map × flux_per_band, convolved with PSF
5. χ² = Σ[(data − model)/σ]² across all bands and optionally spectra
6. The chosen fit method optimizes χ² w.r.t. θ using JAX autodiff

## Configuration Files (.lyric)

Config files use a hierarchical key-value format with prefix letters:

| Prefix | Component | Key Contents |
|--------|-----------|-------------|
| `R1`–`R3` | Region | Target name, RA/Dec, redshift |
| `I{x}1`–`I{x}15` | Image | Data FITS, band name, sigma, PSF, mask, unit, sky model, pixel shifts |
| `S{x}1`–`S{x}4` | Spectrum | Data file, flux conversion, wavelength range, hi-res template flag |
| `A{x}1`–`A{x}7` | Atlas | Image list, spectrum list, shift linking, aperture definition |
| `P{x}1`–`P{x}32` | Profile | Type (sersic/ferrer/edgeondisk/GauRing/const), position, Re, n, PA, q, SED params (sSFR, age, Z, Av, logM, SFH, logU) |
| `G{x}1`–`G{x}7` | Galaxy | Profile list, redshift, Galactic extinction |
| `N{x}1`–`N{x}27` | AGN | BH mass, Eddington ratio, spin, line profiles, torus params |
| `F{x}1`–`F{x}6` | Foreground Star | Position, Teff, logL, logg, metallicity |

Parameter format: `[value, min, max, step, vary]` — `vary` is 1 (free) or 0 (fixed).

### Typical Workflow

1. Write `.lyric` config with data (R, I, S, A) and model (P, G) definitions
2. Run with `--notfit` → generates `target.params` and `target.constrain`
3. Edit `.params` to adjust bounds or `.constrain` for parameter linking
4. Run fitting: `galfits config.lyric --readpar target.params --parconstrain target.constrain`

### Parameter Constraints

- **Expression column** in `.params`: e.g., `1*host_xcen` to link parameters
- **`.constrain` file**: Python function `Update_Constraints(pardictlc)` for complex relations
- **`.prior` file**: Astrophysical priors — MSR (mass-size), MMR (mass-metallicity), SFH, AGN scaling relations, Gaussian priors, energy balance

### Config Examples

Example `.lyric` files are in `GalfitS/examples/`:
- `quickstart.lyric` — Multi-band SDSS+2MASS+GALEX bulge-disk SED decomposition
- `galfit.lyric` — Single-band pure photometry (traditional GALFIT-like)
- `grism.lyric` — HST WFC3 G102 grism with rotation curve
- `spectrum.lyric` — SDSS spectrum with AGN+host decomposition
- `combine_fit.lyric` — Joint imaging + spectroscopy with AGN
- `other_profiles.lyric` — Non-Sérsic profiles (Ferrer, edge-on disk, Gaussian ring, broken/ring Sérsic)
- `arbitraySED.lyric` — Custom table SED models
- `data_example/obj29489_s1.lyric` — Full JWST NIRCam 7-band example with AGN

## Key Directories

| Path | Purpose |
|------|---------|
| `GalfitS/src/galfits/` | Python source package |
| `GalfitS/src/data/filters/` | ~140 instrument filter transmission curves |
| `GalfitS/src/data/grism/` | Grism config (`.conf`) and sensitivity (`.fits`) files |
| `GalfitS/examples/` | Example `.lyric` config files and NIRCam demo data |
| `GalfitS/docs/` | Sphinx documentation (RST + built HTML) |
| `GalfitS/dev/` | Development workspace — task folders and tracking files |
| `data/` | Quickstart data: SDSS+GALEX+2MASS cutouts, PSFs, and SED templates |
| `tutorial/` | Tutorials: config file format, data preparation, running GalfitS |
| `astroskills/` | Claude Code skill definitions for GalfitS pipeline |
| `libprofit/` | Vendored C++ library for fast 2D profile evaluation |
| `MIRI/`, `NIRCam/` | Raw JWST imaging data for target COS-87259 |
| `MIRI_result*/`, `MIRI_analysis/` | MIRI fitting results and photometry |

## Development Workflow

The existing `GalfitS/CLAUDE.md` defines a structured development workflow:

- All tasks live in `GalfitS/dev/task_XXX_name/` folders (scripts, notes, test_results)
- Three tracked files: `dev/develop_log.md` (daily log), `dev/plan.md` (todo list), `dev/problems.md` (issue database)
- Individual task folders are NOT tracked by git
- Source code in `src/` is modified only when necessary

When starting work, read `GalfitS/dev/plan.md` for current status.

## Important Notes

- **Python 3.11 is strict** — JAX 0.8.0 CUDA plugins are incompatible with Python 3.12+
- **XLA memory**: `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set in `galfitS.py` to avoid GPU memory preallocation
- **Float32 only**: `jax.config.update("jax_enable_x64", False)` — all computation is float32
- **No formal tests** exist for the GalfitS package; `libprofit/` has its own C++ tests
- **SED templates** (`.npz` files) are downloaded separately from Google Drive; they are NOT in the repository
- **Filter files** in `GalfitS/src/data/filters/` are plain text (two-column: wavelength Å, throughput); missing bands can be added by creating new files following the naming convention
- The `jnesty` nested sampler replaced JAXNS (May 2026) due to MultiEllipsoidalSampler inconsistencies
- **evosax must be exactly 0.2.0** (pinned in `requirement.txt`)
- Logging: when `--savelog` is set, both stdout and stderr are tee'd to `workplace/target.galfitS.log`
