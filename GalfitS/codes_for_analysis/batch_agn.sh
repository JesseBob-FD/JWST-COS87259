#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate galfits
cd /mnt/d/Fudan_University/Research/JWST/GalfitS
for run in 003 004; do
  echo "===== Run $run ====="
  dir="MIRI_NIRCam_result/run$run"
  tag=$(ls $dir/*.lyric | head -1 | xargs basename | sed 's/.lyric//')
  galfits "$dir/$tag.lyric" --work "$dir" --fit_method optimizer --num_steps 3000 --learning_rate 0.001 --saveimgs --savelog 2>&1 | grep -E 'Using imagefitter|^# chisq|^# BIC|Error|Traceback'
done
echo "===== DONE ====="
