# NumPy: Reshaping

## Learning objectives

After this chapter you will be able to:

* Explain the difference between shape, strides, and memory layout (C vs Fortran order).
* Use `reshape`, `resize`, `ravel`, `flatten`, `transpose`, `swapaxes`, `moveaxis`, `expand_dims`, and `squeeze` appropriately.
* Understand when operations return views vs copies and why that matters for performance and memory.
* Use `-1` in `reshape` and know its constraints.
* Convert arrays between C-contiguous and Fortran-contiguous layouts and when to prefer either.
* Recognize common pitfalls (in-place reshape failures, non-contiguous arrays, ambiguous reshape) and how to resolve them.



## Why reshape matters

Reshaping changes the *view* of array dimensions without necessarily altering the underlying data. Many numerical algorithms expect specific shapes (e.g., matrix ops, CNN inputs), so reshaping is a frequent and important task. Efficient reshaping avoids copying memory and relies on adjusting shape and strides.



## Core API: `reshape`

```python
b = a.reshape(new_shape)
```

* `np.reshape(a, newshape)` or `a.reshape(newshape)` returns an array with the requested shape if possible.
* If possible, NumPy will return a **view** (no data copied); otherwise it will return a **copy**.
* `newshape` may include one `-1` which tells NumPy to infer that dimension.

Examples:

```python
import numpy as np
x = np.arange(12)
x.shape        # (12,)
X = x.reshape((3,4))   # shape (3,4), no copy for contiguous x
Y = x.reshape((2,2,-1)) # -1 inferred -> (2,2,3)
```

Constraints when using `-1`:

* Only one `-1` allowed.
* The inferred dimension must be an integer (the total size must be divisible by the product of specified dimensions).

Ambiguous reshape cases will raise an error if the total sizes mismatch.



## `reshape` vs `resize`

* `np.reshape` / `ndarray.reshape` returns a new view or copy and does not change `a`.
* `ndarray.resize(new_shape)` **modifies the array in-place** and can change the total size; if the new size is larger, the array is filled with zeros. `np.resize(a, new_shape)` returns a new array (copy) with repeated data if needed.

Examples:

```python
a = np.arange(6)
b = a.reshape((2,3))   # a unchanged

a.resize((3,2))       # a itself changes shape (in-place)
```

Be careful with `resize` as it mutates the original.



## Flattening: `ravel` vs `flatten`

* `ravel()` returns a **view** whenever possible (no copy) — it’s the preferred fast way to flatten when you don’t need an independent copy.
* `flatten()` always returns a **copy**.

```python
x = np.arange(6).reshape(2,3)
flat_view = x.ravel()
flat_copy = x.flatten()
```

If you modify `flat_view`, the change may reflect in `x` (if ravel returned a view).



## Transpose and axis manipulation

### `transpose`

Reorders axes. Shorthand: `a.T` for reversing axes of a 2D array.

```python
A = np.arange(12).reshape(3,4)
A.T.shape  # (4,3)
```

`transpose` generally returns a view by adjusting strides; no copy for contiguous arrays.

### `swapaxes` and `moveaxis`

* `swapaxes(a, i, j)` swaps two axes.
* `moveaxis(a, source, destination)` moves axes to new positions while preserving order of others.

```python
np.swapaxes(A, 0, 1)        # same as A.T for 2D
np.moveaxis(A, 0, -1)       # move axis 0 to last position
```



## Using `np.newaxis` / `expand_dims` / `squeeze`

* `np.newaxis` (or `None`) adds a dimension:

```python
v = np.array([1,2,3])      # shape (3,)
v[:, None].shape          # (3,1)
```

* `np.expand_dims(a, axis)` is an explicit function to add axes.
* `np.squeeze(a)` removes axes of length 1.

These are commonly used to prepare inputs for broadcasting or functions that expect particular dimensionality.



## Memory layout: C-order vs Fortran-order and strides

* C-order (row-major): last index changes fastest. `np.arange(6).reshape((2,3), order='C')`.
* Fortran-order (column-major): first index changes fastest. `order='F'`.

`a.flags` tells you about contiguity:

* `a.flags['C_CONTIGUOUS']` (or `a.flags.c_contiguous`)
* `a.flags['F_CONTIGUOUS']`

Reshaping may fail to produce a view if the array is not contiguous in the required order. Use `np.ascontiguousarray(a)` or `np.asfortranarray(a)` to get contiguous copies when needed.

Example where reshape requires copy:

```python
A = np.arange(6).reshape(2,3)
B = A.T            # B is non-contiguous in C-order
B.reshape(6)       # may return a copy if strides incompatible
```

Understanding `strides` helps reason about views and reshape compatibility. `a.strides` gives a tuple of bytes to step in each dimension.



## Order argument in reshape/ravel/flatten

Most reshape-like functions accept an `order` argument: `'C'`, `'F'`, `'A'`, `'K'`.

* `'C'`: C-order (row-major)
* `'F'`: Fortran-order (column-major)
* `'A'`: Fortran if input is Fortran contiguous, C otherwise
* `'K'`: Keep order (attempt to preserve memory layout)

```python
x.reshape((3,4), order='F')
```

`ravel(order='F')` will flatten in column-major order.



## Reshape with broadcasting and views

Reshaping to add singleton dimensions and using broadcasting can achieve many transformations without copying data.

```python
x = np.arange(6)              # shape (6,)
x = x.reshape((2,3))          # shape (2,3)
# add singleton dimension for broadcasting
x[:, :, None]                 # shape (2,3,1)
```

When you broadcast, NumPy behaves as if smaller arrays are repeated across axes, but it does not actually copy data.



## Common pitfalls and how to fix them

* **Error**: cannot reshape array of size N into shape M — check that product of dimensions matches.
* **Unexpected copy**: reshape returned a copy because array is non-contiguous — use `np.ascontiguousarray` before reshaping or use appropriate `order`.
* **Memory blowup**: creating many copies with `reshape` + operations can double memory usage. Use views when possible and delete intermediates.



## Practical recipes

* Ensure C-contiguous before reshaping without copy: `a = np.ascontiguousarray(a)`.
* Flatten safely (copy): `flat = a.flatten()`.
* Flatten view (fast, may alias): `flat_view = a.ravel()`.
* Convert between layouts: `a_c = np.ascontiguousarray(a)`; `a_f = np.asfortranarray(a)`.
* Insert/remove axes: `np.expand_dims(a, axis)` and `np.squeeze(a)`.
