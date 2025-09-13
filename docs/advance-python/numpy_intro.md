# NumPy: Introduction

## Learning objectives

By the end of this chapter you will be able to:

* Explain what NumPy is and why it is the foundational numerical library in Python.
* Install and import NumPy properly in different environments.
* Create and inspect `ndarray` objects (NumPy arrays) and understand their core attributes.
* Perform simple, idiomatic array operations and understand why NumPy is faster than plain Python lists.
* Learn common terminology used in later chapters (dtype, shape, ndim, broadcasting, ufuncs).



## What is NumPy?

NumPy (Numerical Python) is a fundamental package for scientific computing in Python. It provides:

* A powerful N-dimensional array object, `ndarray`, which stores elements of the same data type in contiguous memory.
* Efficient, vectorized operations implemented in C, which are much faster than equivalent pure-Python loops.
* A large collection of mathematical functions (universal functions, or *ufuncs*), random number capabilities, linear algebra helpers, FFTs, and tools for memory-efficient data handling.

NumPy is the basis for most scientific Python libraries (Pandas, SciPy, scikit-learn, TensorFlow, etc.), so understanding it unlocks a large part of the scientific Python ecosystem.



##  Why use NumPy instead of Python lists?

**Performance**

* NumPy arrays store elements in contiguous memory and use a fixed data type. This enables low-level optimizations and vectorized operations (operations applied elementwise in native machine code).

**Memory efficiency**

* `ndarray` uses a fixed-size data type (`dtype`) per array, so memory is compact and predictable.

**Convenience**

* Built-in broadcasting rules, slicing, boolean indexing, reduction operations, and a large collection of numerics utilities make code concise and expressive.

**Example comparison** (conceptual):

* Adding two lists element-wise with Python loops: `O(n)` Python-level operations with interpreter overhead for each element.
* Adding two NumPy arrays: one fast C loop, minimal Python overhead.



## Installation and import

### Using pip

```bash
pip install numpy
```

### Using conda

```bash
conda install numpy
```

### Import convention

```python
import numpy as np
```

This is the standard alias used in most code and documentation.



## The core object: `ndarray`

An `ndarray` (N-dimensional array) is the main data structure in NumPy. Key characteristics:

* **Homogeneous**: every element has the same `dtype` (data type).
* **Contiguous memory** (usually): efficient for low-level operations.
* **Shape**: tuple describing length along each axis, e.g. `(3, 4)` for 2D.

### Creating arrays

```python
import numpy as np

# from a Python list
arr = np.array([1, 2, 3])

# explicit dtype
arr_f = np.array([1, 2, 3], dtype=np.float64)

# common convenience constructors
zeros = np.zeros((2, 3))          # 2x3 array filled with 0.0
ones = np.ones(5)                 # 1D array with five 1.0s
arange = np.arange(0, 10, 2)      # like range(), returns array([0,2,4,6,8])
lin = np.linspace(0, 1, 5)        # 5 values spaced between 0 and 1
eye = np.eye(3)                   # 3x3 identity matrix
rand = np.random.default_rng().random((2, 2))  # random numbers (preferred RNG API)
```

### Inspecting arrays

```python
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
arr.shape      # (2, 3)
arr.ndim       # 2 (number of dimensions)
arr.size       # 6 (total number of elements)
arr.dtype      # dtype('int32')
arr.itemsize   # bytes per element (4 for int32)
arr.nbytes     # total bytes (size * itemsize)
```



##  Data types (`dtype`)

NumPy uses `dtype` objects to describe the type of the array’s elements (e.g., `int32`, `float64`, `bool`, `complex128`).

* You can request a dtype at array creation. If not provided, NumPy infers from the input.
* Converting dtypes: `arr.astype(np.float32)` (creates a copy with new dtype).
* Common pitfalls: mixing Python `int` and `float` in a list will upcast to float when creating an array.

```python
mixed = np.array([1, 2.5])
print(mixed.dtype)  # float64
```



##  Vectorized operations & ufuncs (universal functions)

NumPy performs elementwise arithmetic without Python-level loops using *ufuncs*. This is the major source of speed.

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

# elementwise
c = a + b      # array([11,22,33])
d = a * 2      # array([2,4,6])

# math functions are vectorized
np.sqrt(a)     # array([1.0, 1.41421356, 1.732...])
np.exp(a)      # elementwise e**x

# reductions
a.sum()        # 6
b.mean()       # 20.0
```

Because these operations are executed in optimized C loops, they are significantly faster for large arrays than Python loops.



##  Broadcasting (brief introduction)

Broadcasting is a set of rules NumPy follows to perform arithmetic between arrays of different shapes. If the shapes are compatible, NumPy "stretches" the smaller array across the larger one without copying data.

Example:

```python
# a is shape (3,)
# b is shape (3,)
# they broadcast naturally

row = np.array([1, 2, 3])      # shape (3,)
col = np.array([[10], [20]])   # shape (2,1)

# adding row to col: result shape (2,3)
result = col + row
```

We will cover broadcasting in depth in later chapters because it’s crucial for writing compact and efficient numerical code.



##  Memory view vs copy

Some NumPy operations return views (no data copied) and some return copies (independent memory). Knowing which operations create views helps avoid unintended side-effects and excessive memory use.

```python
a = np.arange(10)
b = a[2:5]      # b is a view on a
b[0] = 99
print(a[2])     # 99 -> original changed

# to force a copy
c = a[2:5].copy()
```

Slicing always creates a view when possible; operations that change shape in most cases create a copy.



## Performance tip — avoid Python loops

Prefer vectorized operations and ufuncs. When you must loop, try using `np.nditer` or write the hot loop in C/Cython, Numba, or use libraries built on top of NumPy.

Quick benchmark (conceptual):

```python
# not to run here — conceptual example
# Using list comprehension
lst = [i * 2 for i in range(1_000_000)]

# Using numpy
arr = np.arange(1_000_000)
arr2 = arr * 2
```

On large arrays, the NumPy version will be orders of magnitude faster.



##  Common gotchas

* Mixing dtypes leads to upcasting (e.g., int + float -> float).
* Use `.copy()` if you need an independent array, since slices are views.
* Be mindful of memory when creating many large intermediate arrays.
* Floating point precision: `.dtype` matters for numerical stability and memory usage.



##  Suggested exercises

1. Install NumPy and print its version: `np.__version__`.
2. Create a 1D array from a Python list and print its `.shape`, `.dtype`, `.nbytes`.
3. Create a 3x3 identity matrix using `np.eye` and multiply it by a vector.
4. Time adding two lists using Python vs adding two NumPy arrays (use `timeit` or `%%timeit` in Jupyter).
5. Create a large array and practice slicing and making copies vs views. Observe when the original changes.



##  Further reading

* Official NumPy documentation: [https://numpy.org/doc](https://numpy.org/doc)
* "NumPy Beginner's Guide" and online tutorials
* Practice problems on Kaggle and local datasets

