# FixingLog

Errors and warnings encountered during GalfitS testing/debugging.

## Error #1: AttributeError - numpy.ndarray has no attribute 'newbyteorder'
**Time**: 2026-06-04
**Traceback**:
```
File "galfitS.py", line 24, in _safe_jax_array
    object = object.byteswap().newbyteorder()
AttributeError: 'numpy.ndarray' object has no attribute 'newbyteorder'
```
**Root Cause**: `numpy.ndarray.newbyteorder()` was removed in numpy >= 2.0. The `_safe_jax_array` helper (added for a prior fix) uses this deprecated method.
**Fix**: Replace `object.byteswap().newbyteorder()` with `object.byteswap().view(object.dtype.newbyteorder('='))`.

