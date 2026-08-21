#!/bin/bash
for mode in noSED SED; do
  dir="/mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/two_sersic_C1C2/$mode"
  f=$(ls $dir/*.gssummary 2>/dev/null)
  echo "=== $mode ==="
  head -8 "$f"
  echo ""
done
