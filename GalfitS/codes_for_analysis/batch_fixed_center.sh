#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate galfits
cd /mnt/d/Fudan_University/Research/JWST/GalfitS

runs=(
  "fixed_center_C1host_C2AGN/noSED:C1host_C2AGN_noSED"
  "fixed_center_C1host_C2AGN/SED:C1host_C2AGN_SED"
  "fixed_center_C2host_C1AGN/noSED:C2host_C1AGN_noSED"
  "fixed_center_C2host_C1AGN/SED:C2host_C1AGN_SED"
)

for entry in "${runs[@]}"; do
  subdir="${entry%%:*}"
  tag="${entry##*:}"
  dir="MIRI_NIRCam_result/$subdir"
  echo "===== $subdir ====="
  galfits "$dir/${tag}.lyric" --work "$dir" --fit_method optimizer --num_steps 3000 --learning_rate 0.001 --saveimgs --savelog 2>&1 | grep -E 'Using imagefitter|^# chisq|^# BIC|Error'
  echo ""
done
echo "===== DONE ====="
