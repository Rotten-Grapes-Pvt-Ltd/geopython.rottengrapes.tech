# Pandas: Time Series, Window Functions & Performance Engineering


## Learning objectives

* Understand Pandas' time series types and `DatetimeIndex` basics.
* Index and slice data by time efficiently using date strings and partial indexing.
* Use `resample`, `rolling`, `expanding`, and `ewm` (exponential windows) for common time‑based summaries.
* Align and shift time series with `shift`, `tshift` (legacy), and `asof` patterns.
* Handle irregular timestamps: interpolation, reindexing, forward/backward fill, and up/down sampling.
* Apply common analyses: moving averages, rolling correlations, cumulative sums, and lag features for modeling.
* Learn practical performance engineering: dtype choice, memory profiling, chunked processing, Parquet, Dask, vectorization, and avoiding common slow patterns.
* Know how to productionize pipelines: checkpointing, monitoring, and basic parallel patterns.



## Time series basics and `DatetimeIndex`

* Time-aware dtype:

    * Pandas uses `datetime64[ns]` for timestamps and `Timedelta` for durations. Use `pd.to_datetime()` to convert strings.

* Creating and setting a `DatetimeIndex`:

```python
df['ts'] = pd.to_datetime(df['timestamp'])
df = df.set_index('ts').sort_index()
```

* Partial date indexing is convenient and readable:

```python
# selects all rows in January 2021
df.loc['2021-01']
# selects a full day
df.loc['2021-01-15']
```

* Ensure the index is sorted for many operations (`df.sort_index()`); some functions rely on monotonic time.

* Inspect frequency and gaps:

```python
df.index.freq          # may be None for irregular data
pd.infer_freq(df.index)
```



## Resampling — changing the frequency

* `resample` groups by fixed time bins (like `groupby` but for time):

```python
# convert minute-level to hourly sums
hourly = df.resample('H')['value'].sum()

# resample with custom aggregation and fill
daily = df.resample('D').agg({'value':'sum', 'temperature':'mean'})
daily = daily.fillna(method='ffill')
```

* Upsampling vs downsampling:

    * Downsampling aggregates many rows into fewer bins (e.g., minute → hour) — use `sum`, `mean`, etc.
    * Upsampling creates higher frequency rows (e.g., daily → hourly) — you must decide how to fill new rows (`ffill`, `bfill`, `interpolate`).

* Use `label` and `closed` arguments to control which bin edge is used for labeling and inclusion.



## Rolling, expanding and exponentially weighted windows

* `rolling(window='7D')` or `rolling(window=7)` computes statistics over a sliding window.

```python
# 7-day moving average (on daily index)
df['ma7'] = df['value'].rolling(window=7, min_periods=1).mean()

# rolling std and correlation
df['std7'] = df['value'].rolling(7).std()
df['corr7'] = df['value'].rolling(7).corr(df['other'])
```

* `expanding()` computes cumulative statistics from the start:

```python
df['cummean'] = df['value'].expanding().mean()
```

* Exponential moving windows (`ewm`) weight recent observations more:

```python
df['ewm12'] = df['value'].ewm(span=12, adjust=False).mean()
```

* Use `min_periods` to control how many observations are required before yielding a result.

* For time‑aware rolling windows use `rolling('7D')` on a `DatetimeIndex` which counts time span rather than number of rows.



## Shifts, differences and lag features

* `shift()` moves data forward/backward by periods — useful for lag features and differences.

```python
df['lag1'] = df['value'].shift(1)
df['diff1'] = df['value'] - df['lag1']
```

* For time‑based shifts use `shift(freq='1D')` to shift the index.

* Beware of missing values introduced by shift — handle by filling or dropping depending on modeling needs.



## Aligning irregular time series, interpolation and filling

* Reindex to a desired frequency and fill:

```python
idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
df_hourly = df.reindex(idx)
df_hourly['value'] = df_hourly['value'].interpolate(method='time')
```

* Interpolation methods:

    * `method='time'` (time-aware linear interpolation), `'linear'`, `'polynomial'`, `'nearest'`.
    * Use `limit` to control maximum consecutive NaNs to fill.

* Forward/backward fill:

```python
df.ffill(limit=2)
```

* When working with measurements, prefer interpolation that respects domain knowledge (e.g., piecewise linear for sensor readings, not cubic splines without reason).



## Time zone handling and DST

* Localize naive timestamps and convert between zones:

```python
# localize naive index as UTC then convert to a zone
df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
```

* Be careful when resampling across DST transitions — prefer UTC for storage and convert to local timezone for display.



## Seasonal decomposition and basic time series diagnostics

* Simple decomposition using `statsmodels` (optional dependency):

```python
from statsmodels.tsa.seasonal import seasonal_decompose
res = seasonal_decompose(df['value'].asfreq('D'), model='additive')
res.plot()
```

* Autocorrelation and partial autocorrelation (ACF/PACF) help identify lags to include in models; use `statsmodels.graphics.tsaplots.plot_acf`.



## Feature engineering for time series models

* Create lag features, rolling statistics, time-of-day and day-of-week indicators, and holiday flags.

```python
df['hour'] = df.index.hour
df['dow'] = df.index.dayofweek
df['lag24'] = df['value'].shift(24)
df['rolling7_mean'] = df['value'].rolling(7).mean()
```

* Beware of look‑ahead bias: build features only using information that would have been available at prediction time.

* For categorical cyclical features (hour of day, day of week) consider encoding as sine/cosine pairs for many models.



## Working with irregular event series: `merge_asof` and event alignment

* Use `merge_asof` to align events to the most recent prior state (e.g., attach last price quote to each trade). Sort both tables by time and use `by` for per‑symbol matching.

```python
pd.merge_asof(trades.sort_values('time'), quotes.sort_values('time'), on='time', by='symbol')
```

* For nearest neighbor matching in time, `direction='nearest'` is available.



## Performance engineering — practical tactics

* Inspect memory usage:

```python
df.memory_usage(deep=True)
```

* Choose efficient dtypes:

    * Use `int32` / `float32` if precision allows.
    * Use `category` for repeated strings (IDs, symbols).
    * Use nullable dtypes (`Int64`, `boolean`) when you need to preserve missing data without `object` dtype.

* Avoid `object` dtype where possible: it stores generic Python objects and is slow to process.

* Read and write using efficient binary formats (Parquet, Feather) for repeated analysis; they preserve dtypes and are faster than CSV.

* Vectorize heavy computations using NumPy operations instead of Python loops.

* Use `eval()` and `query()` for some expressions — they can be faster and use less memory by avoiding intermediate Python objects.

* Profile where the time is spent:

    * Use `%timeit` in notebooks for quick timings.
    * Use `cProfile` or `line_profiler` for deeper investigation.



## Scaling beyond a single machine

* If data does not fit in memory, consider:

    * Dask DataFrame — familiar Pandas-like API but operates out-of-core and supports distributed execution.
    * Incremental/streaming approaches using `chunksize` or databases.
    * Spark/PySpark if you have a cluster and need wide ecosystem integration.

* Typical pattern with Dask:

```python
import dask.dataframe as dd
ddf = dd.read_parquet('s3://bucket/path/*.parquet')
# apply rolling/window operations carefully — some operations require known partitioning and indexing
```

* For simple parallelism, use `concurrent.futures` to process independent time partitions (e.g., per symbol) and then concatenate results. Beware of memory duplication.



## Checkpointing and reproducibility in pipelines

* Save intermediate transforms (Parquet) frequently so failed runs can resume from a checkpoint instead of restarting from raw data.
* Record the index frequency, timezone, and any resampling choices in pipeline metadata.
* Validate invariants after each major step (row counts, ranges, no negative values where impossible).



## Recipes: common tasks (copy & run)

* Compute 7-day rolling average on daily data and save to Parquet

```python
df = df.set_index('date').sort_index()
df['ma7'] = df['value'].rolling(window=7, min_periods=1).mean()
df.to_parquet('daily_with_ma7.parquet')
```

* Upsample daily to hourly and interpolate values

```python
hr_idx = pd.date_range(df.index.min(), df.index.max(), freq='H')
df_hr = df.reindex(hr_idx)
df_hr['value'] = df_hr['value'].interpolate(method='time')
```

* Compute lag features for modeling (lags 1..3)

```python
for lag in range(1,4):
    df[f'lag_{lag}'] = df['value'].shift(lag)
```

* Chunked processing: compute monthly aggregates on a large CSV using `chunksize`

```python
import pandas as pd
from collections import defaultdict
sums = defaultdict(float)
for chunk in pd.read_csv('big_events.csv', parse_dates=['time'], chunksize=200_000):
    chunk = chunk.set_index('time').resample('M')['value'].sum()
    for idx, val in chunk.items():
        sums[idx.to_timestamp()] += val
monthly = pd.Series(sums).sort_index()
```



## Monitoring and alerting for production jobs

* Log progress and anomaly counts (missing data, imputed rows, dropped rows) during ETL.
* Emit metrics (rows processed per second, time per chunk) to monitoring systems.
* Add simple health checks: ensure no data skew, check for new/unexpected categories in categorical columns.



## Exercises (practical)

* Given a minute‑level sensor dataset, compute hourly means, 1‑hour rolling std, and create lag features for the previous 24 hours. Evaluate how many NaNs are introduced and propose a handling strategy.
* Simulate trade and quote streams and use `merge_asof` to attach quotes to trades. Compute average slippage by symbol and time of day.
* Timezone exercise: convert a UTC index to a local timezone, resample to daily and compare results near DST transitions.
* Benchmark: read a 5 GB Parquet file, compute a small aggregation (groupby + sum) and time it. Experiment with casting to `category` for the group key and measure the speedup.
