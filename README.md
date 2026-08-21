# JWST COS-87259 analysis

Private research repository for the COS-87259 JWST analysis. It contains the
analysis code, notebooks, configuration files, research notes, and selected
small derived products used across the NIRSpec, MIRI, NIRCam, and GalfitS
workflows.

The detailed project map, data-flow diagrams, current scientific picture, and
known caveats are maintained in [INDEX.md](INDEX.md). GalfitS-specific work is
indexed in [GalfitS/GalfitS_INDEX.md](GalfitS/GalfitS_INDEX.md).

## Repository scope

Tracked here:

- Python scripts and Jupyter notebooks
- Markdown research logs and workflow documentation
- GalfitS configuration and batch scripts
- Small CSV/TXT tables and diagnostic figures
- Records and patches for locally modified third-party dependencies

Not tracked here:

- The local Python virtual environment
- Raw or reduced FITS data and large template arrays
- Download archives, reference-paper PDFs, and office documents
- Compiled binaries, caches, logs, and temporary outputs

These exclusions keep Git history reviewable and avoid treating GitHub as the
primary archive for multi-gigabyte science data. The ignored files remain in
the local working directory.

## Environment

The working environment currently uses Python 3.11. GalfitS additionally
requires CUDA 12.4 and JAX. Third-party repositories and their pinned commits
are documented in [DEPENDENCIES.md](DEPENDENCIES.md); local GalfitS changes are
preserved under `patches/`.

## Research status

This repository contains ongoing, unpublished research. Scientific results and
interpretations should be treated as provisional and checked against the latest
entries in `INDEX.md`, `Journal/`, and `GalfitS/MIRI_NIRCam_result/TRIAL_LOG.md`.

