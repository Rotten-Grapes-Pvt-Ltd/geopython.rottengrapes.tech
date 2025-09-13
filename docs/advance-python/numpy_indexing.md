# NumPy: Indexing

## Learning objectives

By the end of this chapter you will be able to:

* Use basic slicing and indexing to select subarrays.
* Distinguish between views and copies produced by different indexing methods.
* Apply advanced (fancy) indexing with integer arrays and boolean masks.
* Use `np.newaxis`, `...` (ellipsis), and `np.ix_` to reshape and select across axes.
* Understand indexing performance implications and when to prefer different approaches.



## Indexing fundamentals

NumPy arrays use zero-based indexing and support multi-dimensional indexing via a comma-separated tuple.

```python
import numpy as np
A = np.arange(24).reshape(4,6)
# A is shape (4,6)
# element at row 1, column 2
x = A[1, 2]
# equivalent using nested brackets
y = A[1][2]
```

**Tip:** `A[1, 2]` is preferred because `A[1]` returns a view of row 1 and then `[2]` indexes that row — the end result is same value but `A[1, 2]` is clearer and slightly more efficient.



## Slicing (basic and multi-axis)

Slicing follows `start:stop:step` semantics like Python lists. A slice returns a view (no copy) when possible.

```python
arr = np.arange(10)
arr[2:8:2]    # elements at indices 2,4,6

# 2D slicing
A = np.arange(20).reshape(4,5)
A[1:3, 2:5]   # rows 1-2 and columns 2-4 -> shape (2,3)
```

A slice preserves the original array's memory when possible, so modifying the slice can modify the original array.

```python
s = A[0:2, 0:2]
s[0,0] = 99
# A[0,0] is now 99
```

Use `.copy()` if you need an independent array.



## Negative indices and steps

Negative indices count from the end. A negative step can reverse an axis.

```python
arr = np.arange(6)
arr[-1]     # last element (5)
arr[::-1]   # reversed array

# 2D reverse rows
A[::-1, :]  # rows reversed
```



## Boolean indexing (masking)

Boolean indexing selects elements where a boolean array is `True`. The boolean index can be the same shape as the array (elementwise) or broadcastable.

```python
x = np.arange(10)
mask = x % 2 == 0
x[mask]           # even numbers: array([0,2,4,6,8])

# use mask to assign
x[mask] = -1
```

Boolean indexing returns a copy (a 1D array) of matching elements, not a view. Assignments using boolean indexing affect the original array where the mask is `True`.



##  Integer (fancy) indexing

Integer arrays allow selecting arbitrary elements or subarrays — and they *always* return copies.

```python
A = np.arange(12).reshape(3,4)
rows = np.array([0,2])
cols = np.array([1,3])
A[rows]         # selects rows 0 and 2 -> shape (2,4)
A[:, cols]      # selects columns 1 and 3 -> shape (3,2)

# elementwise selection (paired indices)
A[rows, cols]   # picks elements (0,1) and (2,3) -> shape (2,)
```

Fancy indexing with multiple index arrays follows broadcasting rules: if you pass two 2D index arrays of the same shape, you get an array of that shape where each entry is the selected element.



##  Mixing integer and slice indexing

When mixing slices and integer arrays, result shape is determined by the integer arrays and the remaining slices.

```python
A = np.arange(24).reshape(4,6)
A[[0,2], 1:4]  # select rows 0 and 2, and columns 1..3 -> shape (2,3)
```

**Important:** fancy indexing always makes a copy; slicing returns a view. This affects both memory and whether in-place assignment modifies the source.



##  Using `np.newaxis` / `None` and `...` (ellipsis)

`np.newaxis` (or `None`) adds a new axis to turn a 1D array into 2D or to align shapes for broadcasting.

```python
x = np.array([1,2,3])        # shape (3,)
x[:, None]                   # shape (3,1)
x[None, :]                   # shape (1,3)
```

Ellipsis `...` is a convenient shorthand to represent ":`:` until the last axis".

```python
A = np.zeros((2,3,4,5))
A[1, ..., 2]   # A[1, :, :, 2]
```



##  `np.ix_` for cross product indexing

`np.ix_` builds open meshes to select a cross-product of indices.

```python
A = np.arange(12).reshape(3,4)
rows = [0,2]
cols = [1,3]
A[np.ix_(rows, cols)]   # shape (2,2) with elements at row/col cross-product
```

This is useful when you want the full outer selection, not elementwise pairing.



##  Advanced helpers: `take`, `put`, `choose`

* `np.take(a, indices, axis=None)` extracts elements along an axis (similar to fancy indexing but often faster and clearer).
* `np.put(a, indices, values, axis=None)` places values into array positions.
* `np.choose` picks elements from multiple choices along an index array.

```python
np.take(A, [0,2], axis=0)
```



##  Indexing and broadcasting: selecting with broadcastable masks

Boolean masks can be broadcast to match an axis.

```python
A = np.arange(12).reshape(3,4)
mask = np.array([True, False, True])[:, None]  # shape (3,1)
A[mask]   # flattened array of selected rows (rows 0 and 2)
```



##  Views vs copies — summary and examples

* Slicing (`start:stop`) → **view** (no copy) when possible.
* Fancy indexing (integer arrays) → **copy**.
* Boolean indexing → **copy** of matching elements (1D) but assignment uses mask to modify original.
* `np.take` often returns a copy.

When in doubt, test with `is` or `np.shares_memory`.

```python
s = A[0:2]
np.shares_memory(A, s)   # True -> view
f = A[[0,1]]
np.shares_memory(A, f)   # False -> copy
```



##  Performance considerations

* Prefer slices and views for memory efficiency and speed when operating on contiguous subarrays.
* Fancy and boolean indexing create copies — avoid in tight loops if you can restructure to use slicing or vectorized masking.
* Use `np.take` with `mode` parameter for safety and sometimes better performance.



##  Common pitfalls and debugging tips

* Unexpected `dtype=object` after indexing often means you created an array of Python objects earlier.
* Using chained indexing like `A[0][1:3] = ...` can be harder to reason about; prefer `A[0, 1:3] = ...`.
* After fancy indexing remember you have a copy — assignments to the result won't affect the source.



## Exercises

1. Given `A = np.arange(30).reshape(5,6)`: select rows 1,3 and columns 2,4 as a (2,2) array using `np.ix_`.
2. Use boolean indexing to set all negative values in an array to zero.
3. Demonstrate that slicing returns a view (modify slice and show original changed) and that fancy indexing returns a copy.
4. Use `np.newaxis` to convert a shape `(10,)` vector to `(10,1)` and `(1,10)` then compute their outer product.
5. Time selecting a large contiguous block using slicing vs using fancy indexing with an integer array. Observe memory usage.

