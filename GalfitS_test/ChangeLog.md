# ChangeLog

Code changes made during GalfitS debugging.

## Change #1: Fix numpy.newbyteorder() removal
**File**: `GalfitS/src/galfits/galfitS.py:24`
**Reason**: `numpy.ndarray.newbyteorder()` removed in numpy >= 2.0.
**Old**:
```python
object = object.byteswap().newbyteorder()
```
**New**:
```python
object = object.byteswap().view(object.dtype.newbyteorder('='))
```

