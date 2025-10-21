# Pandas: Grouping, Aggregation & Joins


## Learning objectives

* Understand the purpose of grouping and aggregation: turning row‑level data into meaningful summaries.
* Use `groupby` patterns: `agg`, `transform`, `filter`, and `apply` and know when to use each.
* Compute multiple aggregations in one pass and produce tidy results ready for analysis or plotting.
* Join (merge) DataFrames with `merge`, `join`, and use `concat` for stacking tables; understand `how` and `on` semantics.
* Use specialized joins: `merge_asof`, `merge_ordered`, and database-style joins for real-world workflows.
* Understand performance and memory tradeoffs; use categorical keys and indexes to speed groupby and joins.



## Why grouping and joins matter

* Grouping converts detailed records into summaries (totals, averages, counts) that are easier to understand and visualize.
* Joins combine data from different tables (e.g., customer info + transactions) to enrich analysis.
* Real-world analysis often alternates between grouping and joining to assemble final datasets for reporting and modeling.



## The `groupby` concept — intuitive explanation

* `groupby` follows a three‑step mental model:

    * split the data into groups by one or more keys,
    * apply an aggregation or transformation to each group,
    * combine the results back into a DataFrame or Series.

* Example intuition: "group transactions by customer, sum the amounts".

```python
grouped = df.groupby('customer_id')['amount'].sum()
```

* `groupby` does not modify the original DataFrame; it returns a `DataFrameGroupBy`/`SeriesGroupBy` object that you then aggregate/transform.



## Common `groupby` operations and when to use them

* `agg` (aggregate)

    * Use to compute summary statistics that reduce each group to a single value per column (sum, mean, count, min, max, custom functions).

```python
# multiple aggregations in one call
summary = df.groupby('store').agg(
    total_sales=('amount','sum'),
    avg_order=('amount','mean'),
    orders=('order_id','nunique')
)
```

* `transform`

    * Use when you want to produce an output aligned to the original index (same shape as input) — useful for group‑wise normalization or filling.

```python
# subtract group mean from each row (demean)
df['amount_demeaned'] = df.groupby('store')['amount'].transform(lambda x: x - x.mean())
```

* `filter`

    * Use to drop or keep entire groups based on group‑level criteria (e.g., keep stores with >100 orders).

```python
big_stores = df.groupby('store').filter(lambda g: len(g) > 100)
```

* `apply`

    * Powerful but slower: runs a function on each group and concatenates results. Use when `agg`/`transform` cannot express the logic.
    * Prefer `agg`/`transform` for performance when possible.



## Multiple aggregations and reshaping results

* Use named aggregations (Pandas >= 0.25) to produce tidy columns with meaningful names. This avoids awkward column tuples.

```python
agg = df.groupby(['region','product']).agg(
    revenue=('amount','sum'),
    avg_price=('price','mean'),
    orders=('order_id','nunique')
).reset_index()
```

* If you use a list of aggregations without names, Pandas may produce MultiIndex columns. Use `reset_index()` and `rename` for clarity.



## Grouped time series and resampling

* For time series data, `resample` works like `groupby` but for fixed time bins (requires a `DatetimeIndex`).

```python
# daily totals from minute-level data
daily = df.resample('D')['amount'].sum()

# combine resample with groupby: daily sales per store
daily_store = df.set_index('timestamp').groupby('store').resample('D')['amount'].sum().unstack(0)
```

* Use `rolling` for moving-window statistics (e.g., 7‑day moving average) and `expanding` for cumulative calculations.



## Joins and merges — the basics

* `merge` is the most flexible and common way to join two tables. Think in SQL terms: `inner`, `left`, `right`, `outer`.

```python
# add customer details to transactions
tx = transactions.merge(customers, on='customer_id', how='left')
```

* `how` semantics:

    * `inner`: keep rows with keys in both tables.
    * `left`: keep all rows from left table, add matching rows from right or NaN.
    * `right`: symmetric to left.
    * `outer`: union of keys from both tables.

* `on`, `left_on`, `right_on`

    * Use `on` when the key column has the same name in both tables. Use `left_on/right_on` for different names.



## Index vs column joins and `join`

* `merge` works on columns; `join` is a convenience for index‑based joins.

```python
left.set_index('id').join(right.set_index('id'), how='left')
```

* When joining large tables repeatedly, setting an index on the join key can speed operations. Remember to `.reset_index()` if you need the key back as a column afterwards.



## Ordered joins: `merge_asof` and `merge_ordered`

* `merge_asof` matches each row in the left table to the last row in the right table whose key is less than or equal to the left key — useful for time series alignment (e.g., match trades to most recent quote).

```python
pd.merge_asof(buys.sort_values('time'), quotes.sort_values('time'), on='time', by='symbol', direction='backward')
```

* `merge_ordered` performs a merge that preserves ordering and can forward/backward fill keys; useful for longitudinal tables.



## Concatenating and appending tables

* `pd.concat` stacks tables vertically (`axis=0`) or horizontally (`axis=1`). Use when you have partitioned outputs (e.g., monthly files) to combine into one table.

```python
combined = pd.concat([jan, feb, mar], ignore_index=True)
```

* Use `ignore_index=True` if you do not want to keep original row indices.



## Performance tips for grouping and joins

* Convert join/group keys to `category` dtype when there are many repeated values; this can reduce memory and speed up operations.
* For very large datasets, sort by the key and use `merge_asof` or database systems (SQL) to push joins into the database.
* Avoid expensive Python-level functions inside `groupby.apply`; prefer built-in aggregations or `transform`.
* Use `split-apply-combine` carefully: avoid creating many small DataFrames inside loops.
* When merging many small tables into a big one, consider building a dictionary keyed by join column and mapping values, or use database bulk operations.



## Common pitfalls and how to diagnose them

* Unexpected duplicates after join: inspect the join keys to see if they are unique in the right table. Use `right.duplicated(subset=['key']).any()` to test.
* Memory blow-up during `merge`: ensure you `usecols` when reading data and convert dtypes to be compact before merging.
* Losing rows unintentionally: check `how` and key alignment; use `indicator=True` in `merge` to see where rows came from.

```python
merged = left.merge(right, on='id', how='left', indicator=True)
merged['_merge'].value_counts()
```

* Slow joins: try setting the join key as an index on both tables and use `join`, or perform join in a database engine.



## Practical examples and step-by-step recipes

* Summarize sales by store and category with multiple aggregations

```python
summary = (
    sales
    .groupby(['store','category'])
    .agg(total_revenue=('amount','sum'),
         avg_price=('price','mean'),
         orders=('order_id','nunique'))
    .reset_index()
)
```

* Add customer demographic info to transactions and compute average basket value by segment

```python
tx = transactions.merge(customers[['customer_id','segment']], on='customer_id', how='left')
avg_basket = tx.groupby('segment')['amount'].mean()
```

* Align trades to most recent price quote using `merge_asof`

```python
quotes = quotes.sort_values(['symbol','time'])
buys = buys.sort_values(['symbol','time'])
matched = pd.merge_asof(buys, quotes, on='time', by='symbol', direction='backward')
```

* Combine monthly parquet files into a single DataFrame

```python
import pathlib
files = list(pathlib.Path('data/').glob('sales_2025-*.parquet'))
monthly = [pd.read_parquet(f) for f in files]
all_sales = pd.concat(monthly, ignore_index=True)
```



## Exercises (hands-on)

* Compute the top 5 products by revenue per region and present a tidy table with region, product, revenue.
* Given customer and transaction tables, merge them and find customers with transactions but missing demographics — count them and inspect sample rows.
* Simulate quote and trade data, then use `merge_asof` to attach the latest quote to each trade and compute slippage (trade price vs quote price).
* Read a directory of CSVs for each month, concatenate them, and compute year‑to‑date metrics using `groupby`.



## Quick checklist for safe joins and group operations

* Inspect keys for uniqueness before joining (`.duplicated()`)
* Convert keys to categorical if repeated many times
* Use `indicator=True` in debug runs to validate join behavior
* Reduce memory before merge by selecting required columns and compact dtypes
* Use `reset_index()` after groupby if you want keys back as columns
