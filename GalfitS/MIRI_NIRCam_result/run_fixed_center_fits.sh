#!/bin/bash
# Run the 3 fixed-AGN-center noSED fits sequentially (2026-08-17)
# Fit settings match the original free-center runs: optimizer, 3000 steps, lr=0.001
source ~/miniconda3/etc/profile.d/conda.sh
conda activate galfits
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BASE=/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result
for d in host_dualAGN_C1C2 dual_AGN_C1C2 two_sersic_two_AGN_C1C2; do
  cd "$BASE/$d/noSED" || exit 1
  echo "=== FIT START $d $(date +%F_%T) ==="
  galfits ${d}_noSED.lyric --work ./ --fit_method optimizer --num_steps 3000 --learning_rate 0.001 --savelog > fit_stdout.log 2>&1
  ec=$?
  echo "=== FIT END $d $(date +%F_%T) exit=$ec ==="
done
echo "=== ALL FITS DONE ==="
