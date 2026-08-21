#!/bin/bash
for f in /mnt/d/Fudan_University/Research/JWST/GalfitS/MIRI_NIRCam_result/run00*/*.gssummary; do
  echo "=== $(basename $f .gssummary) ==="
  head -8 "$f"
  echo ""
done
