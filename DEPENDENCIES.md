# Third-party source dependencies

The local workspace contains cloned third-party repositories. Their embedded
`.git` directories are intentionally not nested inside the main research
repository. Instead, this file records the exact upstream revisions needed to
reconstruct the working environment.

| Local path | Upstream repository | Revision |
| --- | --- | --- |
| `GalfitS/GalfitS` | `https://github.com/RuancunLi/GalfitS` | `53e7636a66eec799fa481822e43765292fceffd1` |
| `GalfitS/astroskills` | `https://github.com/RuancunLi/astroskills` | `f72e7061206ab2ed6af184d64dfb0e7cc8645e2c` |
| `GalfitS/libprofit` | `https://github.com/ICRAR/libprofit` | `927228194c047644367a6a8ff8374df810239ba7` |
| `GalfitS_test/GalfitS` | `https://github.com/RuancunLi/GalfitS` | `53e7636a66eec799fa481822e43765292fceffd1` |
| `GalfitS_test/astroskills` | `https://github.com/RuancunLi/astroskills` | `f72e7061206ab2ed6af184d64dfb0e7cc8645e2c` |
| `GalfitS_test/libprofit` | `https://github.com/ICRAR/libprofit` | `927228194c047644367a6a8ff8374df810239ba7` |

The active `GalfitS/GalfitS` checkout has local changes on top of the recorded
revision. Apply `patches/GalfitS-local.patch` after checking out that revision,
then restore the files under `patches/GalfitS-untracked/` to their corresponding
paths in the cloned repository.
