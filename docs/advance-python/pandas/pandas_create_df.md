# Reading & Writing Data; Parsing Strategies

## Learning objectives

* Understand the common file formats used for tabular data and their trade-offs.
* Load data from CSV, Excel, JSON, SQL, Parquet, compressed files, and web APIs into Pandas safely and efficiently.
* Parse dates, numeric formats, encodings, and missing values correctly.
* Read very large files without running out of memory (chunking, iterators, streaming).
* Write DataFrames back to disk or databases using sensible defaults with attention to reproducibility and performance.
* Diagnose and fix common parsing errors and messy input files.



## Quick format guide — when to use what

* **CSV / TSV**

    * Human‑readable, very common. Good for small→medium files and easy sharing.
    * Slower to parse and larger on disk than binary formats; fields may require careful parsing (quotes, separators).
  
* **Excel (`.xls` / `.xlsx`)**

    * Business friendly (multiple sheets, formatting). Good for small→medium ad‑hoc datasets.
    * Slower to parse programmatically and may include merged cells or headers that need cleanup.
    
* **JSON / NDJSON**

    * Flexible for nested or hierarchical data and APIs. Use newline‑delimited JSON (NDJSON) for large streams.
* **Parquet / Feather**

    * Columnar binary formats — fast I/O, compression, and efficient for analytics. Prefer for large datasets and repeated reads.
 
* **SQL databases**

    * Best for data that needs indexing, concurrent access, or partial queries. Pull only what you need with SQL.



## Reading CSV files (`pd.read_csv`) — the essential knobs

* **Basic usage**

```python
import pandas as pd

df = pd.read_csv('data.csv')
```

* **Important options and why they matter**

    * `sep` / `delimiter` — set the column separator (`,`, `\t`, `;`). Wrong `sep` means all data ends up in one column.
    * `header` / `names` — control column names when your file has no header or has extra header rows.
    * `dtype` — pass a dict like `{'id': 'int32', 'amount': 'float32'}` to avoid slow type inference and reduce memory.
    * `parse_dates` — parse date columns into `datetime64[ns]` so you can slice, resample, and extract date parts.
    * `usecols` — only read needed columns to save memory and time.
    * `na_values` — tell Pandas which strings should be treated as missing (e.g., `['', 'NA', 'n/a', '-']`).
    * `encoding` / `encoding_errors` — fix Unicode problems like `UnicodeDecodeError` (try `latin-1` or `cp1252`).
    * `engine='c'` or `engine='python'` — the C engine is much faster; use `python` for complex parsing options like unusual quoting or regex separators.
    * `chunksize` — read in row‑based chunks (returns an iterator of DataFrames) for streaming large files.
    * `compression` — `'gzip'`, `'bz2'`, `'xz'`, or `'zip'` when reading compressed files.
    * `low_memory` — if True (default) Pandas may infer dtypes in chunks and result in mixed dtypes. Set `low_memory=False` when you plan to supply `dtype` or want single‑pass inference.

* **Practical examples**

```python
# read only useful columns, parse a date, and force types
df = pd.read_csv(
    'sales.csv',
    usecols=['order_id','order_date','amount','country'],
    dtype={'order_id': 'int64', 'amount': 'float32'},
    parse_dates=['order_date']
)

# read a tab separated file
df = pd.read_csv('data.tsv', sep='\t')

# read a gzipped CSV directly
df = pd.read_csv('data.csv.gz', compression='gzip')
```

* **Tips for international number formats**

```python
# numbers like "1.234,56" (dot thousands, comma decimal)
df = pd.read_csv('euro.csv', thousands='.', decimal=',')
```

* **Dealing with malformed rows**

    * Use `on_bad_lines='skip'` (pandas >= 1.3) to skip bad rows and inspect later.
    * Use `engine='python'` and `error_bad_lines=False` for older pandas versions (deprecated keywords).



## Practical date parsing strategies

* **Why parse dates**

    * Converting strings to datetimes unlocks time‑aware operations: slicing by date, `resample`, `.dt` accessors, timezone handling.

* **Common approaches**

    * Simple column: `parse_dates=['date_col']`.
    * Multiple columns to combine: `parse_dates={'date': ['year','month','day']}` or combine after load with `pd.to_datetime`.
    * Use `pd.to_datetime(df['col'], format='%Y-%m-%d')` when you know the exact format — it is much faster and more reliable.

* **Ambiguous formats**

    * Use `dayfirst=True` or `yearfirst=True` for ambiguous day/month parsing (e.g., `05/06/2020`).

* **Large datasets**

    * Read date columns as strings first (`dtype={'date_col': 'string'}`), then convert with `pd.to_datetime(..., format=...)` in a second pass when you can guarantee format and speed.



## Encodings, BOMs, and stray characters

* **Symptoms of encoding issues**

    * `UnicodeDecodeError` when reading.
    * Strange characters like `Ã©` instead of `é`.

* **Fixes**

    * Try `encoding='latin-1'` or `encoding='cp1252'` if `utf-8` fails.
    * Use `encoding_errors='replace'` (pandas >= 1.4) to replace invalid bytes.
    * If file has a Byte Order Mark (BOM), `pd.read_csv` usually handles it; otherwise use `encoding='utf-8-sig'`.



## Reading Excel files (`read_excel`) — practical notes

* Use `sheet_name` (`None` for all sheets) and `usecols` / `skiprows` to extract the table area you need.
* Excel files may contain merged cells and headers in multiple rows; inspect visually and use `skiprows` / `header` to pick the right row for column names.
* Excel reading is slower — for repeatable automated workflows, prefer converting to CSV or Parquet beforehand.

```python
# read all sheets into a dict of DataFrames
all_sheets = pd.read_excel('workbook.xlsx', sheet_name=None)
```



## JSON and nested data: `read_json` and `json_normalize`

* **NDJSON** (newline delimited) is easy to stream: `pd.read_json('file.ndjson', lines=True)`.
* For nested JSON, use `pd.json_normalize` to flatten records:

```python
from pandas import json_normalize
import json

with open('nested.json') as f:
    data = json.load(f)
flat = json_normalize(data, record_path='items', meta=['order_id','customer'])
```

* When consuming APIs, use `requests` to fetch JSON and normalize into a DataFrame.



## Parquet and Feather — fast columnar formats

* **Why use them**

    * Faster read/write than CSV, smaller disk footprint due to compression, and better for analytics because they are columnar (you can read just the columns you need).

* **Usage**

```python
df.to_parquet('data.parquet', index=False)
df = pd.read_parquet('data.parquet')
```

* **Partitioning**

    * When writing large datasets, partitioning by a column (e.g., `year=2020/ month=01/...`) can speed filters in downstream tools.



## SQL databases and `read_sql` patterns

* Use `sqlalchemy.create_engine` for a connection and let the database filter/aggregate data when possible.
* For very large tables, read with `chunksize` and process each chunk, or push heavy aggregation into SQL to avoid transferring massive raw tables.

```python
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host/db')
for chunk in pd.read_sql_query('SELECT * FROM big_table', engine, chunksize=10000):
    process(chunk)
```



## Reading from web APIs

* Fetch JSON with `requests`, handle pagination, and normalize responses into rows.
* Respect API rate limits and authentication.

```python
import requests
r = requests.get('https://api.example.com/data', params={'page': 1})
rows = r.json()['results']
df = pd.json_normalize(rows)
```



## Working with very large files — streaming patterns

* **Chunked processing**

    * Use `chunksize` in `read_csv` to iterate over the file and update aggregates, write partial results to disk, or populate a database. Never build a giant list of chunks in memory.

```python
agg = {}
for chunk in pd.read_csv('big.csv', chunksize=200_000, usecols=['country','amount']):
    s = chunk.groupby('country')['amount'].sum()
    for k,v in s.items():
        agg[k] = agg.get(k, 0) + v
```

* **Memory tricks**

    * Use `usecols` to drop unnecessary columns.
    * Supply `dtype` to prevent mixed types and `object` columns.
    * Convert repeated string columns to `category` immediately: `chunk['col'] = chunk['col'].astype('category')`.
    * If you need random access or complex analytics on huge datasets, consider using a database, Dask, or reading Parquet with partitioned layout instead of a single giant CSV.



## Writing data (`to_csv`, `to_parquet`, `to_sql`) — good defaults

* **CSV**

    * `df.to_csv('out.csv', index=False)` — avoid writing the index unless you need it.
    * Use `compression='gzip'` to save space: `df.to_csv('out.csv.gz', compression='gzip', index=False)`.
* **Parquet**

    * `df.to_parquet('out.parquet', index=False)` — preserve dtypes and use compression.
* **SQL**

    * Use `df.to_sql('table', engine, if_exists='append', index=False)` to append; for large writes consider batching and disabling indexes during load for speed.



## Common parsing errors, symptoms and fixes

* **ParserError / tokenizing errors**

  * Symptom: `Error tokenizing data` or `EOF inside string`.
    * Cause: inconsistent quoting, embedded newlines, bad separators.
    * Fixes: try `engine='python'` with `sep` and `quoting=csv.QUOTE_NONE`, or open file and inspect bad lines; use `on_bad_lines='skip'` to skip and log problematic rows.

* **UnicodeDecodeError**

    * Symptom: decoding error when reading file.
    * Fix: try `encoding='latin-1'` or `encoding='utf-8-sig'` (handles BOM); set `encoding_errors='replace'` to avoid failures.

* **Mixed dtypes in column**

    * Symptom: column shows `object` dtype or warnings about mixed types.
    * Fix: pass `dtype` or `converters` to coerce types at read time, or sanitize values after reading and convert types explicitly.

* **Dates read as strings or `NaT`**

    * Symptom: dates are strings or missing after read.
    * Fix: supply `parse_dates` or convert post-read with `pd.to_datetime(..., format=...)`.

* **Large memory usage**

    * Symptom: your machine runs out of RAM or swaps.
    * Fix: use `usecols`, `dtype` optimization, read with `chunksize`, or switch to Parquet / database / Dask.



## Step-by-step recipes (copy & run)

* **Inspect the first few lines before loading**

```python
# fast preview using Python (no Pandas)
with open('messy.csv', 'r', encoding='utf-8', errors='replace') as f:
    for _ in range(20):
        print(f.readline().rstrip('\n'))
```

* **Load a messy CSV with tuned options**

```python
df = pd.read_csv(
    'messy.csv',
    sep=';',
    decimal=',',
    thousands='.',
    encoding='latin-1',
    parse_dates=['order_date'],
    dayfirst=True,
    na_values=['', 'NA', 'n/a', '-']
)
```

* **Stream process and write partial aggregates**

```python
out_path = 'totals_by_region.csv'
first_write = True
for chunk in pd.read_csv('big_sales.csv', chunksize=100_000, usecols=['region','amount'], dtype={'region':'category','amount':'float32'}):
    sums = chunk.groupby('region')['amount'].sum().reset_index()
    sums.to_csv(out_path, mode='a', header=first_write, index=False)
    first_write = False
```



## Exercises (practical, friendly to non-programmers)

* Open a sample CSV in a plain text editor and identify: the delimiter, header row, and examples of missing values.
* Load a CSV with `pd.read_csv('file.csv', nrows=100)` to inspect types, then reload with `dtype` and `usecols` to reduce memory. Compare `df.memory_usage(deep=True)` before and after.
* Read an Excel workbook with multiple sheets using `sheet_name=None` and explore the dictionary of DataFrames.
* Read a large CSV using `chunksize=50000` and compute the total of a numeric column using chunked aggregation.



## Troubleshooting checklist (keep this handy)

* Inspect the top 20 lines of the file manually when parsing fails.
* Try `encoding='latin-1'` or `encoding='utf-8-sig'` for Unicode errors.
* Use `usecols` to isolate problematic columns and save memory.
* If parser fails, try `engine='python'` and `on_bad_lines='skip'` to continue while you inspect bad lines.
* For performance: sample the file with `nrows=1000`, infer good `dtype` mappings, then reload full data with `dtype` specified.

