# NumPy: Creating Arrays (from lists)

## Learning objectives

After this chapter you will be able to:

* Create NumPy `ndarray` objects from Python lists, tuples and other iterable sources.
* Understand `np.array` vs `np.asarray` vs `np.asanyarray` and when copies are made.
* Create multi-dimensional arrays from nested Python sequences and handle ragged input.
* Use NumPy convenience constructors (`arange`, `linspace`, `zeros`, `ones`, `full`, `empty`, `eye`, `identity`, `frombuffer`, `fromfile`) and know when to prefer each.
* Specify and convert `dtype` at creation time and understand implicit upcasting rules.
* Detect and avoid common pitfalls (object-dtype arrays, unexpected shapes, memory/layout concerns).



## Basic creation from Python lists and tuples

The simplest way to create an array is from a Python list or tuple using `np.array`.

```python
import numpy as np

# 1D array from list
arr1 = np.array([1, 2, 3])        # dtype inferred, usually int64 or int32

# 2D array from nested lists (list of rows)
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

# from tuple
arr3 = np.array((7, 8, 9))
```

Key points:

* `np.array` will try to infer the best common dtype. If the list mixes `int` and `float`, the resulting dtype will be a float (upcasting).
* If nested lists have equal-length sublists, a regular 2D `ndarray` will be produced. If lengths are unequal, NumPy will create a 1D array with `dtype=object` (a ragged array).

```python
# ragged example
ragged = np.array([[1, 2], [3]])
print(ragged.dtype)   # object
print(ragged.shape)   # (2,)
```

If you want a 2D array with missing values, provide explicit padding (e.g. `np.nan`) and a float dtype, or construct arrays with `np.full` then fill values.



##  Choosing dtype explicitly

You can request a dtype at creation time:

```python
arr = np.array([1, 2, 3], dtype=np.float32)
```

Why set dtype explicitly?

* Control memory footprint (e.g., `float32` vs `float64`).
* Avoid accidental downcasts or upcasts later.
* Ensure compatibility with other libraries (some expect specific dtypes).

Converting dtype after creation uses `astype` which creates a copy:

```python
arr_f64 = arr.astype(np.float64)
```



##  `np.asarray` vs `np.array` vs `np.asanyarray`

* `np.array(x)`: Always tries to create a new `ndarray`. It will copy data if necessary (for example when `dtype` conversion is needed or the input is not an `ndarray`).
* `np.asarray(x)`: Avoids copying when the input is already an `ndarray` with the required dtype. Use this when you want to wrap inputs without unnecessary copies.
* `np.asanyarray(x)`: Similar to `asarray` but will preserve subclasses of `ndarray` (like `np.matrix`).

Examples:

```python
lst = [1, 2, 3]
arr = np.asarray(lst)   # creates an ndarray

# if you call with an array, asarray returns the same object (no copy) unless dtype change required
a = np.array([1,2,3], dtype=np.int32)
b = np.asarray(a)
print(a is b)  # True -> no copy

# np.array would copy unless you pass copy=False explicitly (numpy sometimes still copies)
c = np.array(a, copy=False)
```

Use `np.asarray` in functions that accept array-like input to minimize copies.



## Convenience constructors

NumPy provides many factory functions for common arrays.

### `arange`

Like Python `range`, but returns an array:

```python
np.arange(0, 10, 2)     # array([0,2,4,6,8])
```

Be cautious with floating step sizes (like `0.1`) because of floating point rounding: prefer `linspace` when you want a specific number of evenly spaced values.

### `linspace`

Generates `num` evenly spaced samples between `start` and `stop` (inclusive by default):

```python
np.linspace(0, 1, 5)    # array([0. ,0.25,0.5 ,0.75,1. ])
```

### `zeros`, `ones`, `full`, `empty`

```python
np.zeros((2,3))    # zeros with float dtype by default
np.ones(5)         # ones
np.full((2,2), 7)  # filled with 7
np.empty((3,3))    # uninitialized memory — fast but contains garbage
```

`np.empty` is useful for performance when you will fill the entries later; do not read uninitialized entries.

### Identity and diagonal

```python
np.eye(3)          # 3x3 identity (float dtype)
np.identity(3)     # same as eye
np.diag([1,2,3])   # diagonal matrix from vector
```

### `frombuffer` and `fromfile`

Used to create arrays from raw binary data — advanced and often faster for large binary files. Example:

```python
# from a bytes-like object
b = b"\x01\x00\x02\x00"
np.frombuffer(b, dtype=np.int16)
```

`fromfile` reads directly from a binary file but is platform-dependent and less flexible than using Python's `open` + `np.frombuffer` on `mmap` or using `np.load`/`np.save` for NumPy binary files (`.npy`).



##  Creating arrays with a specific shape

Often you want an array of a certain shape and dtype. Use `reshape` on an existing array or create directly:

```python
# create 12 values then reshape
a = np.arange(12)
a2 = a.reshape((3,4))   # shape (3,4)

# direct creation
b = np.zeros((3,4))
```

When reshaping, ensure the total size matches: product of shape dims must equal number of elements.



##  Multi-dimensional arrays from nested lists

When creating arrays from nested sequences, NumPy expects consistent nesting depths and lengths for regular arrays.

```python
regular = np.array([[1,2,3], [4,5,6]])   # shape (2,3)

# inconsistent lengths -> object array
bad = np.array([[1,2], [3,4,5]])
```

If you need to create an array from nested lists but force a shape or dtype, consider:

* Pad shorter rows with `np.nan` and use a floating dtype.
* Build an empty array with `np.empty((nrows, ncols))` and fill row-by-row.



##  Performance considerations at creation time

* Creating arrays from Python lists has overhead because NumPy must iterate the Python objects and convert them; creating arrays from already-binary sources (binary files, memoryviews, `np.frombuffer`) is much faster.
* If you repeatedly build large arrays by appending Python lists, prefer collecting into a list and calling `np.array` once, or use functions like `np.concatenate` on a list of arrays.

Example (bad):

```python
arr = np.array([])
for i in range(1000):
    arr = np.append(arr, i)   # repeatedly reallocates — slow
```

Better:

```python
lst = [i for i in range(1000)]
arr = np.array(lst)
```



##  Reading arrays from text and binary files

* `np.loadtxt` / `np.genfromtxt` — load numerical data from text files (CSV-like). `genfromtxt` handles missing values but is slower.
* `np.load` / `np.save` — read and write NumPy `.npy` and `.npz` binary formats (fast, recommended for NumPy-only workflows).

Examples:

```python
arr = np.loadtxt('data.csv', delimiter=',')
np.save('arr.npy', arr)
arr2 = np.load('arr.npy')
```



##  Common pitfalls and debugging tips

* If you see `dtype=object`, inspect your source lists for inconsistent shapes or mixed types.
* Use `np.asarray` if you want to avoid copies when the input might already be an `ndarray`.
* For reproducible behavior across platforms, prefer explicit dtypes like `np.int32` or `np.float64`.
* When using `fromfile`, be mindful of endianness and struct alignment — prefer `np.load` for portability.

