# Pandas: Indexing, Advanced Selection & Reshaping


## Learning objectives

* Know the difference between label‑based and position‑based indexing and when to use each.
* Use `loc`, `iloc`, `at`, `iat`, boolean masks, and `query` for efficient selection.
* Work safely with assignment to subsets of a DataFrame and avoid `SettingWithCopyWarning`.
* Use hierarchical indexes (`MultiIndex`) for multi‑level selection and learn helpful utilities like `xs` and `swaplevel`.
* Reshape tables using `stack`/`unstack`, `melt`/`pivot`, `pivot_table`, and combine with `groupby` for tidy workflows.
* Apply advanced selection patterns: partial string indexing for `DatetimeIndex`, `between_time`, `truncate`, and `merge_asof` for ordered joins.



## Label vs position — the foundational distinction

* Labels refer to the index or column names (what you see printed). Positions are integer positions (0, 1, 2...).
* `loc` is label‑based: `df.loc[row_label, col_label]`.
* `iloc` is integer position‑based: `df.iloc[row_pos, col_pos]`.
* Use `at` and `iat` for fast scalar access:

```python
val = df.at['row123', 'amount']   # label-based scalar access (fast)
val2 = df.iat[5, 2]               # integer-based scalar access (fast)
```

* Why it matters:

  * Using the wrong method can silently select unexpected rows/columns (e.g., integer labels that look like positions).
  * For clarity, use `loc` when you think in terms of labels and `iloc` when you want positional slicing.



## Basic selection patterns

* Column access:

  * `df['col']` returns a `Series` for a column.
  * `df[['a','b']]` returns a `DataFrame` with selected columns.
  
* Row selection:

  * `df.loc['label']` or `df.iloc[0]` for a single row.
  * Slicing by label with `loc` is inclusive of the stop label; slicing by position with `iloc` follows Python semantics (stop exclusive).

```python
# label slice includes endpoint
df.loc['2020-01-01':'2020-01-31']
# positional slice excludes endpoint
df.iloc[0:10]
```

* Selecting rows and columns together:

```python
sub = df.loc['id_1':'id_4', ['name','amount']]
```



## Boolean indexing (masking)

* Create boolean masks and apply them to select rows:

```python
mask = df['amount'] > 100
df_large = df[mask]
```

* Combine conditions with `&` / `|` remembering to parenthesize each condition:

```python
df[(df['amount'] > 100) & (df['country'] == 'USA')]
```

* Use `.loc` for assignment with boolean masks to avoid ambiguity:

```python
# preferred
df.loc[df['amount'] < 0, 'amount'] = 0
```

* Note: boolean indexing returns a copy of the matching rows (1‑D for Series, 2‑D for DataFrame). Assigning to that copy will not modify the original unless you use `.loc` on the original object.



## Fancy/advanced indexing with integer arrays

* Use integer arrays to select arbitrary rows/columns by position. This is *fancy indexing* and returns a copy:

```python
rows = [0, 2, 5]
cols = [1, 3]
subset = df.iloc[rows, cols]
```

* Remember that fancy indexing always makes a copy — changes to `subset` will not change `df`.



## `at` / `iat` for fast scalar access

* When you need to read or write a single value, prefer `at` (label) or `iat` (position) because they are optimized:

```python
# read
v = df.at['rowA','colX']
# set
df.at['rowA','colX'] = 42
```

* These are much faster than `loc`/`iloc` for single‑element access in tight loops.



## `query` and `eval` for readable boolean selections

* `df.query('amount > 100 and country == "USA"')` offers readable expressions using column names directly.
* `eval` can evaluate expressions efficiently and with lower memory overhead for complex column operations.
* Use `@` to reference Python variables inside `query`.

```python
threshold = 100
df.query('amount > @threshold and country == "USA"')
```

* `query` is convenient and often clearer for non‑programmers, but be cautious with column names that have spaces or conflict with Python keywords.



## Setting values safely — avoid `SettingWithCopyWarning`

* Pandas sometimes cannot tell whether you are working on a view or a copy; this raises `SettingWithCopyWarning` when assigning through chained indexing like `df[cond]['col'] = val`.
* Always use `.loc` to be explicit about modifying the original DataFrame:

```python
# fragile (may warn and may not modify original)
df[df['x'] > 0]['y'] = 0
# safe
df.loc[df['x'] > 0, 'y'] = 0
```

* If you intentionally want an independent copy to mutate, call `.copy()` first:

```python
sub = df[df['x']>0].copy()
sub['y'] = 0
```



## Partial string indexing with `DatetimeIndex` and `Index` convenience

* If your DataFrame has a `DatetimeIndex`, you can slice by partial dates:

```python
# df indexed by datetime
df.loc['2020-01']   # all rows in January 2020
```

* Use `between_time` to select rows by time of day and `truncate` to cut at endpoints:

```python
df.between_time('09:00','17:00')
df.truncate(before='2020-01-01', after='2020-03-31')
```

* These operations are concise and fast when the index is sorted and a `DatetimeIndex`.



## MultiIndex (hierarchical index) — powerful but needs care

* `MultiIndex` lets you index on multiple levels (e.g., `country`, `city`, `store`). Create from columns or during grouping/pivoting.

```python
mi = df.set_index(['country','city'])
```

* Helpful selection methods:

  * `mi.loc[('USA','Seattle')]` — select a specific tuple.
  * `mi.xs('Seattle', level='city')` — cross‑section to select all entries for a sublevel.
  * `mi.swaplevel(0,1)` and `mi.sort_index()` for reorganizing and ensuring efficient selection.

* Use `pd.IndexSlice` for slicing across multiple levels elegantly:

```python
idx = pd.IndexSlice
mi.loc[idx['USA':'UK', 'London':'York'], :]  # slice across levels
```

* Pitfalls:

  * Some operations return `Series` when selecting single columns; be explicit with `mi.reset_index()` if needed.
  * MultiIndex can make joins and merges more complex; prefer a flat index for simpler pipelines unless hierarchical indexing provides clear benefits.



## Reshaping: `stack` and `unstack`

* `stack` pivots columns into the index (wide → long). `unstack` reverses the operation.

```python
wide = pd.DataFrame({('sales','A'):[1,2], ('sales','B'):[3,4]}, index=['d1','d2'])
wide.columns = pd.MultiIndex.from_tuples(wide.columns)
long = wide.stack(level=0)    # move outer column level into index
```

* Use `stack`/`unstack` for grouped time series or multi‑level pivoting.



## Reshaping: `melt`, `pivot`, and `pivot_table`

* `melt` converts wide tables to long (tidy) format — useful prior to groupby/aggregation or plotting.
* `pivot` rearranges long data back to wide; `pivot_table` aggregates when there are duplicates.

```python
long = df.melt(id_vars=['id','date'], value_vars=['sales_A','sales_B'], var_name='store', value_name='sales')
wide = long.pivot(index=['id','date'], columns='store', values='sales')
summary = long.pivot_table(index='id', columns='store', values='sales', aggfunc='sum')
```

* `pivot_table` is more robust than `pivot` because it can aggregate duplicates instead of raising an error.



## Combining selection and reshaping in common workflows

* Common pattern: read data → `melt` to long → `groupby` + `agg` → `pivot_table` or `unstack` for a report.
* Example: sales by store and month

```python
long = df.melt(id_vars=['date','store'], value_vars=['sales'])
monthly = (long
           .assign(month=lambda d: d['date'].dt.to_period('M'))
           .groupby(['month','store'])['sales'].sum()
           .unstack('store'))
```



## Joining and alignment with indexing in mind

* Indexes are used for alignment during arithmetic and for `join`/`merge` operations.
* `merge` is SQL‑like and works on columns; `join` is index‑based by default.

```python
# column-based join
out = left.merge(right, on='id', how='left')
# index-based join
out = left.join(right, how='left')
```

* `merge_asof` performs ordered (time) joins that match nearest keys — useful for event alignment or combining time series with different timestamps.

```python
pd.merge_asof(left.sort_values('time'), right.sort_values('time'), on='time', direction='backward')
```



## Performance and memory considerations for indexing/reshaping

* Prefer selecting columns before filtering rows to reduce memory footprint: `df[['col1','col2']].loc[mask]` instead of selecting all columns then slicing.
* Ensure indexes used for frequent lookups are appropriate (`set_index('id')`) — lookups by index are much faster than by column scan.
* Use `inplace=False` operations (default) and assign the result; chained `inplace=True` is discouraged and is often removed from APIs.
* Sorting an index (`sort_index()`) before repeated slicing on labels improves performance.



## Common pitfalls and debugging tips

* `SettingWithCopyWarning`: avoid chained assignment; use `.loc` or `.copy()`.
* Off-by-one errors: remember `loc` includes end label, `iloc` excludes end position.
* Ambiguous integer index: if your index is integers that look like positions, prefer `loc` with labels or convert index to `RangeIndex` for positional semantics.
* Unexpected dtype changes after unstack/pivot: check `dtypes` and coerce explicitly if needed.



## Mini recipes (copy & run)

* Select rows by label range and specific columns

```python
report = df.loc['2021-01-01':'2021-01-31', ['store','amount']]
```

* Filter, assign safely, and save

```python
mask = (df['amount'] > 100) & (df['country']=='USA')
df.loc[mask, 'flag_high'] = True
df.to_parquet('high_sales.parquet')
```

* Create MultiIndex from columns and select a cross‑section

```python
mi = df.set_index(['country','city'])
mi.xs('Mumbai', level='city')
```

* Melt wide sales table and compute monthly totals

```python
long = wide.reset_index().melt(id_vars=['date'], value_vars=['sales_A','sales_B'], var_name='store', value_name='sales')
monthly = long.groupby([long['date'].dt.to_period('M'),'store'])['sales'].sum().unstack('store')
```



## Exercises (practical)

* Using a time‑indexed sales DataFrame, select all rows for March 2020 and compute daily totals.
* Given a DataFrame with columns `['id','region','Q1','Q2','Q3','Q4']`, reshape to long format with `melt` and compute total quarterly sales per region.
* Create a `MultiIndex` on `['country','city','store']` and practice selecting different levels using `loc`, `xs`, and `IndexSlice`.
* Demonstrate `merge_asof` by aligning trade events (orders) with the latest price quote before each order.


