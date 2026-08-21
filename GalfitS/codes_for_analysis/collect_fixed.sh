#!/bin/bash
for dir in fixed_center_C1host_C2AGN fixed_center_C2host_C1AGN; do
  for mode in noSED SED; do
    f=$(ls /mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/$dir/$mode/*.gssummary 2>/dev/null)
    if [ -n "$f" ]; then
      echo "=== $dir/$mode ==="
      head -8 "$f"
      echo ""
    fi
  done
done
