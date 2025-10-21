# NumPy: Split

## Learning objectives

By the end of this chapter you will be able to:

* Split arrays along any axis using `np.split`, `np.array_split`, and convenience functions (`hsplit`, `vsplit`, `dsplit`).
* Use integer indices and sections to control where splits occur.
* Understand when splits return views vs copies and why that matters.
* Reconstruct arrays after splitting using `np.concatenate`, `np.stack`, and `np.vstack`/`hstack`.
* Apply splitting in practical workflows (batching, cross-validation, tiling images).



## Overview

Splitting an array means dividing it into multiple sub-arrays along a specified axis. NumPy provides several functions for controlled splitting. The basic functions are:

* `np.split(ary, indices_or_sections, axis=0)`
* `np.array_split(ary, indices_or_sections, axis=0)` (handles uneven splits)
* `np.hsplit`, `np.vsplit`, `np.dsplit` — convenience wrappers for splitting along common axes for 1D/2D/3D arrays.

All return a list of sub-arrays.



## `np.split` (equal sections required)

`np.split` divides an array into *equal* sized sub-arrays when `indices_or_sections` is an integer. If the array cannot be evenly divided, it raises `ValueError`.

```python
import numpy as np
x = np.arange(12)
parts = np.split(x, 3)  # -> [array([0,1,2,3]), array([4,5,6,7]), array([8,9,10,11])]

# 2D split along rows (axis=0)
A = np.arange(24).reshape(6,4)
rows = np.split(A, 3, axis=0)  # returns 3 arrays of shape (2,4)
```

You can also pass a list of indices where the array will be split. These indices indicate the *start* of the next section.

```python
x = np.arange(10)
np.split(x, [3,7])  # -> [x[:3], x[3:7], x[7:]]
```



## `np.array_split` (uneven splits allowed)

`np.array_split` performs like `np.split` but allows uneven divisions when an integer is provided. The earlier sections will be one element larger when the size isn't divisible.

```python
x = np.arange(10)
np.array_split(x, 3)  # -> [array([0,1,2,3]), array([4,5,6]), array([7,8,9])]  # sizes 4,3,3
```

Use `array_split` when you expect possible uneven splits (e.g., batching data into N chunks).



## Convenience wrappers: `hsplit`, `vsplit`, `dsplit`

These are shorthand functions for common split axes:

* `np.hsplit(ary, sections)` — split horizontally (axis=1 for 2D arrays). For 1D input it behaves like `split`.
* `np.vsplit(ary, sections)` — split vertically (axis=0). Only meaningful for arrays with `ndim >= 2`.
* `np.dsplit(ary, sections)` — split along the 3rd axis (axis=2), used for 3D arrays like images with channel dimension.

Examples:

```python
A = np.arange(12).reshape(3,4)
np.hsplit(A, 2)  # two arrays each with shape (3,2)
np.vsplit(A, 3)  # three arrays each with shape (1,4)
```



## Axis argument and negative axes

All split functions accept an `axis` argument. A negative axis counts from the end, e.g. `axis=-1` for the last axis.

```python
B = np.arange(24).reshape(2,3,4)
# split along last axis into 2 parts
np.split(B, 2, axis=-1)  # splits axis 2 into two arrays of shape (2,3,2)
```



## Views vs copies

Splitting often returns **views** when the requested sub-arrays correspond to contiguous memory regions in the original array. However, depending on strides and array contiguity, NumPy may return copies.

* For simple contiguous arrays and standard splits along major axes, results are usually views (no data copy).
* When using fancy indexing or splitting non-contiguous arrays, the result may be a copy.

You can check with `np.shares_memory(a, sub)` to confirm whether a sub-array shares memory with the original.

```python
A = np.arange(12).reshape(6,2)
parts = np.split(A, 3, axis=0)
np.shares_memory(A, parts[0])  # True -> view

# after transpose, splits may produce copies
B = A.T
parts = np.split(B, 2, axis=1)
np.shares_memory(B, parts[0])  # may be False
```

Because views are lightweight, using `np.split` for tiling or batching large arrays is memory-friendly when it returns views.



## Reconstructing arrays after splitting

To combine the pieces back into a single array, use concatenation functions:

* `np.concatenate(seq, axis=...)`
* `np.stack(seq, axis=...)` (adds a new axis)
* Convenience: `np.vstack`, `np.hstack`, `np.dstack`

```python
parts = np.array_split(np.arange(10), 3)
np.concatenate(parts)    # reconstructs original order
```

Be mindful of shapes: concatenation requires matching dimensions except along the concatenation axis.



## Practical uses

* **Batching**: split data into minibatches for SGD: `batches = np.array_split(dataset, n_batches)`
* **Cross-validation folds**: split indices into k folds for cross-validation.
* **Image tiling**: split large images into tiles for processing, and then recombine.
* **Parallel processing**: divide work into sections and process in parallel, then join results.



## Common pitfalls and tips

* `np.split` with integer sections requires exact divisibility; prefer `np.array_split` for robustness.
* When splitting along axis=1 for 1D arrays, you may get unexpected behavior; ensure correct dimensionality first.
* Always check shapes of parts before concatenation; shape mismatches cause errors.
* Use `np.shares_memory` to verify whether splits create views if you plan to modify sub-arrays in-place.
