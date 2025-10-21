# Pandas Essentials & Data Model

## Learning objectives

* Understand what Pandas is and why it complements NumPy for tabular data tasks.
* Know the two core data structures: `Series` and `DataFrame`, and when to use each.
* Create `Series` and `DataFrame` from lists, NumPy arrays, and dictionaries.
* Inspect and manipulate `index`, `columns`, and `dtypes` (including `Categorical` and datetime types).
* Understand alignment, broadcasting-like behavior, and basic memory considerations.



## What is Pandas and when to use it

* Pandas is a high-level library for working with structured/tabular data in Python.
* It builds on NumPy but provides labelled axes (`index`, `columns`), richer IO, and convenient data manipulation primitives.
* Use Pandas for data cleaning, transformation, exploratory data analysis, basic stats, and preparing data for machine learning.


> Checkout official documentation at [https://pandas.pydata.org/docs/index.html](https://pandas.pydata.org/docs/index.html)


## Core data structures

  
- `Series`
  
    * 1D labeled array that holds a sequence of values and an associated `index`.
    * Useful for time series, a single column of data, or results of computations.

- `DataFrame`

    * 2D labeled tabular data structure with rows (`index`) and columns (`columns`).
    * Each column is effectively a `Series` with a shared index.

- `Index` objects

    * Immutable labels for axes. There are specialised index types (`RangeIndex`, `DatetimeIndex`, `CategoricalIndex`).
    * Indexes enable fast alignment and lookups.



## Import convention

```python
import pandas as pd
```

* `pd` is the standard alias. Make sure Pandas is installed (`pip install pandas` or `conda install pandas`).



## Creating Series and DataFrames (examples)

* From a Python list or NumPy array:

```python
import numpy as np
import pandas as pd

s = pd.Series([10, 20, 30])                 # index defaults to RangeIndex(0, 3)
arr = np.array([1.0, 2.0, 3.0])
s2 = pd.Series(arr, index=['a','b','c'])
```

* From a dictionary (keys → index or columns):

```python
data = {'name': ['alice', 'bob'], 'age': [25, 30]}
df = pd.DataFrame(data)
# or create a Series from mapping index->value
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})
```

* From a list of dicts (orient rows):

```python
df2 = pd.DataFrame([{'x':1,'y':2}, {'x':3,'y':4}])
```

* From NumPy structured arrays or other tabular sources (CSV, SQL, Excel):

```python
df_csv = pd.read_csv('file.csv')
```



## Inspecting data

* Attributes and quick look:

    * `df.head()`, `df.tail()` — peek at rows.
    * `df.shape` — (rows, columns).
    * `df.dtypes` — dtype per column.
    * `df.info()` — compact summary including memory usage and non-null counts.
    * `df.describe()` — summary statistics for numeric columns.

* Accessing columns and rows:

    * `df['col']` or `df.col` (dot notation works for simple names) → returns a `Series`.
    * `df[['a','b']]` → returns a `DataFrame` with selected columns.
    * `df.loc[label]` and `df.iloc[position]` — label- and integer-location-based indexing (expanded in later chapters).



## Indexes and alignment

* Index labels align automatically in arithmetic and joins. This means operations between two `Series` or `DataFrame` objects match rows/columns by label, not by position.

```python
s1 = pd.Series([1,2], index=['a','b'])
s2 = pd.Series([10,20], index=['b','c'])
s1 + s2  # result index -> union of labels: a,b,c (NaN where missing)
```

* Reindexing:

    * `s.reindex(new_index)` changes the labels and inserts `NaN` for missing entries.
    * `df.set_index('col')` and `df.reset_index()` convert columns to/from index.

* Index types matter:

    * `RangeIndex` is compact and efficient for default integer indexes.
    * `DatetimeIndex` enables fast date-based slicing and resampling.
    * `Categorical` dtype saves memory and speeds up groupby/joins when there are repeated values.



## Dtypes and memory considerations

* Column-wise dtypes are crucial for performance and memory:

    * `int64`, `float64`, `bool`, `object` (python objects / strings), `category`, `datetime64[ns]`.
* Avoid `object` columns for repeated string values — use `category` if the set of unique values is small relative to rows.
* Convert numeric columns to smaller integer types (`int32` / `float32`) when appropriate to save memory.
* `df.memory_usage(deep=True)` shows memory used per column (deep inspects object arrays).



## Missing data (`NaN`, `pd.NA`) and null handling

* Pandas uses `NaN` for floats and `pd.NA` (nullable dtypes) for newer nullable integer/boolean/string types.

* Common operations:

    * `df.isna()`, `df.notna()` — boolean masks for missingness.
    * `df.dropna()` — drop rows/columns with missing values.
    * `df.fillna(value)` — fill missing values.

* Best practices:

    * Inspect missingness early with `df.info()` and `df.isna().sum()`.
    * Prefer dtype-aware nullable integers (`Int64`) if you need integer columns with missing values.



## Column operations and vectorized behavior

* Column arithmetic is vectorized (leverages NumPy under the hood):

```python
df['total'] = df['price'] * df['quantity']
```

* Broadcasting-like behavior applies across aligned indexes. Mixing mismatched indexes yields `NaN` where labels are missing.

* Adding a scalar to a column is fast and does not change the index:

```python
df['price'] += 1.5
```



## Views vs copies (important subtlety)

* Slicing a `DataFrame` or selecting a single column often returns a view *or* a copy depending on context — pandas may warn with a `SettingWithCopyWarning` when chained assignment could fail.
* Avoid chained assignment like `df[df['x']>0]['y'] = 0` — prefer `loc` assignment:

```python
# fragile (may raise SettingWithCopyWarning)
df[df['x']>0]['y'] = 0
# preferred
df.loc[df['x']>0, 'y'] = 0
```

* When in doubt, use `df.copy()` to get an explicit copy you can mutate safely.



## Basic I/O quick recipes (teaser for Chapter 2)

* Read CSV:

```python
df = pd.read_csv('data.csv', parse_dates=['date_col'])
```

* Write to CSV/Excel/Parquet:

```python
df.to_csv('out.csv', index=False)
df.to_parquet('out.parquet')
```

* Use `dtype=` and `parse_dates=` options on read functions to control memory and parsing.



## Quick practical examples

* Creating a small DataFrame and computing group totals:

```python
df = pd.DataFrame({'city': ['A','A','B'], 'sales': [100,150,200]})
grouped = df.groupby('city')['sales'].sum()
```

* Merging two DataFrames (teaser for later):

```python
left = pd.DataFrame({'id':[1,2], 'x':[10,20]})
right = pd.DataFrame({'id':[2,3], 'y':[30,40]})
merged = left.merge(right, on='id', how='outer')
```



## Exercises

* Create a `DataFrame` from a list of dictionaries and inspect `dtypes`, `shape`, and `memory_usage`.
* Convert a string column with repeated values into `category` dtype and observe memory usage before and after.
* Demonstrate automatic alignment by adding two `Series` with different indexes and explain the result.
* Practice safe assignment using `loc` instead of chained assignment; show a case that triggers `SettingWithCopyWarning` and fix it.
