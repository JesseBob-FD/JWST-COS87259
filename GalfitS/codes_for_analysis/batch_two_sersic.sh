#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate galfits
cd /mnt/d/Fudan_University/Research/JWST/GalfitS
for mode in noSED SED; do
  dir="MIRI_NIRCam_result/two_sersic_C1C2/$mode"
  tag="C1C2_${mode}"
  echo "===== $mode ====="
  galfits "$dir/${tag}.lyric" --work "$dir" --fit_method optimizer --num_steps 3000 --learning_rate 0.001 --saveimgs --savelog 2>&1 | grep -E 'Using imagefitter|^# chisq|^# BIC|Error'
  echo ""
done
echo "DONE"
