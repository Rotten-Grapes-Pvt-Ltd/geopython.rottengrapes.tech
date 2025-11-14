# Pandas: Cleaning & Transforming Data


## Learning objectives

* Recognize common real‑world data problems and decide principled ways to fix them.
* Clean strings, numbers, dates and categorical data with Pandas’ vectorized APIs.
* Choose safe casting & nullable dtypes to preserve missing data.
* Reshape awkward tables and expand nested values for tidy analysis.
* Prepare features for machine learning: handling categories, encoding, scaling, and avoiding leakage.
* Build reproducible, testable pipelines and save intermediate results with provenance.



## Quick overview (what good cleaning achieves)

* Accurate downstream analysis and models.
* Reproducible steps so colleagues can re‑run your work.
* Reasonable memory usage and runtime for data at scale.
* Clear documentation and saved intermediate artifacts.



## Inspecting data — a practical checklist

* Preview the data structure

    * `df.head()`, `df.tail()`, `df.sample(5)`
    * `df.shape` and `df.columns`
    * `df.info()` with `memory_usage='deep'`
* Check for missingness and uniques

    * `df.isna().sum()`
    * `df.nunique()` and `df['col'].value_counts(dropna=False).head()`
* Quick stats

    * `df.describe(include='all')` — notice `count` vs `len(df)` to spot missing rows
* Spot-check problem rows

    * `df[df['col'].isna()].sample(10)` and `df[df['col'].astype(str).str.len()>100]`
    * `df[df.isna().any(axis=1)]` get all NA values 

**Rule of thumb**: never clean blindly — inspect, hypothesize why a problem exists, then apply a fix and re‑check.



## Missing data — strategies, trade‑offs and examples

* Why it matters

    * Missing values change aggregates, breaks ML algorithms that expect numeric input, and may bias results.

* Common strategies (pros/cons):

    * **Drop rows**: simple, may bias sample if missingness is informative.
    * **Drop columns**: useful when a column is mostly empty (>90%) and not useful.
    * **Fill with constant**: `0` or `'Unknown'` — simple but can distort distributions.
    * **Impute with mean/median/mode**: preserves quantity of data but hides uncertainty.
    * **Model‑based imputation**: uses other columns to predict missing values; more accurate but complex and may leak information.
    * **Flag missingness**: create an indicator column — lets models learn from the fact that something was missing.

* Recipes

```python
# drop rows missing 'age'
df = df.dropna(subset=['age'])

# fill numeric with median
df['income'] = df['income'].fillna(df['income'].median())

# group-wise fill (e.g., fill salary within each department)
df['salary'] = df.groupby('dept')['salary'].transform(lambda x: x.fillna(x.median()))

# indicator column for missing phone
df['phone_missing'] = df['phone'].isna().astype('int8')
```

* When to use imputation vs dropping

    * If the proportion of missing is small and not systematically related to outcome, impute.
    * If missing is too frequent or the column is unreliable, prefer dropping or collecting better data.



## Strings and text cleaning (step‑by‑step)

* Why vectorized string ops

    * `df['col'].str.*` methods are implemented in C and much faster than Python loops.

* Basic pipeline for a name column

    * Trim whitespace, normalize spacing, fix encoding, standardize case, remove punctuation, correct common typos.

```python
s = df['name'].astype('string')
s = s.str.strip()
s = s.str.replace('\s+', ' ', regex=True)
s = s.str.normalize('NFKC')  # fix unicode normalization issues
s = s.str.title()
# correct common misspellings
s = s.str.replace('Jonh', 'John')
df['name'] = s
```

* Extract pieces

```python
# phone area code
df['area_code'] = df['phone'].str.extract(r'\(?([0-9]{3})\)?', expand=False)
```

* Performance tip: convert high‑repetition text to `category` after cleaning for memory savings.



## Categorical data — deeper dive

* What happens under the hood

    * Pandas stores categories as integer codes + category index. This reduces memory and makes comparisons fast.

* When to use

    * Columns with many repeating values (country, product\_id, status)
    * Keys used in `groupby` or `merge` operations

* Creating and ordering categories

```python
df['color'] = df['color'].astype('category')
df['size'] = pd.Categorical(df['size'], categories=['S','M','L','XL'], ordered=True)
```

* Encoding for models

    * `cat.codes` gives ordinal codes (watch `-1` for missing)
    * `pd.get_dummies` or `OneHotEncoder` for many models

* Pitfalls

    * Changing categories later can keep stale categories — use `df['col'] = df['col'].cat.remove_unused_categories()` or reset categories explicitly.



## Safe type conversion (practical rules)

* Prefer `pd.to_numeric` and `pd.to_datetime` with `errors='coerce'` to convert safely and then handle `NaN`.

```python
# numbers with stray characters
df['price'] = pd.to_numeric(df['price'].str.replace('[^0-9\.-]', '', regex=True), errors='coerce')

# safely convert dates
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce', dayfirst=False)
```

* Use nullable integer dtype (`Int64`) if integers may contain NA:

```python
df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
```

* Always inspect `df.dtypes` after conversions.



## Duplicates and record reconciliation

* Detect duplicates

```python
# exact duplicate rows
df[df.duplicated(keep=False)]

# duplicated by id
df[df.duplicated(subset=['id'], keep=False)]
```

* Strategies to reconcile

    * Keep first/last (`keep='first'`), merge information (use `groupby.agg` to aggregate fields), or flag for manual review.

```python
# keep most recent record by timestamp
df = df.sort_values('updated_at').drop_duplicates(subset=['id'], keep='last')
```



## Reshaping tools with cleaning examples

* `melt` (wide→long) helps when multiple measurement columns share semantics.

```python
long = df.melt(id_vars=['id','date'], value_vars=['sales_A','sales_B'], var_name='store', value_name='sales')
```

* `pivot_table` to aggregate and reshape

```python
summary = (long
           .pivot_table(index=['id','date'], columns='store', values='sales', aggfunc='sum')
           .reset_index())
```

* `explode` for list-like cells

```python
# tags column like 'a,b,c'
df['tags_list'] = df['tags'].str.split(',')
df = df.explode('tags_list')
```



## Mapping, replacing and conditional fixes

* Use `replace` for exact substitutions and `map` for mapping keys to values (map produces NaN for missing keys).

```python
# multiple replacements
df['country'] = df['country'].replace({'U.S.': 'USA', 'United states': 'USA'})

# mapping to codes
codes = {'USA':'NA','India':'AS'}
df['region'] = df['country'].map(codes)
```

* Conditional updates with `.loc`

```python
mask = df['email'].str.contains('@example.com', na=False)
df.loc[mask, 'is_example'] = True
```



## When to use `apply` / `map` / `iterrows`

* Avoid row-wise Python loops for large data. Prefer vectorized solutions.
* Use `.apply` for complex row logic when vectorization is infeasible, but test performance on a sample first.

```python
# slow: avoid if large
df['foo'] = df.apply(lambda r: expensive_calc(r['a'], r['b']), axis=1)
```

* Benchmark with `%timeit` in notebooks; consider `swifter` or `dask` if apply is a bottleneck.



## Units, currencies and consistent measures

* Always store a canonical unit column (e.g., `weight_kg`) and keep original unit metadata if needed.
* For currency conversion, get reliable exchange rates and apply reproducibly.

```python
# convert grams to kg
df.loc[df['unit']=='g', 'weight_kg'] = df.loc[df['unit']=='g', 'weight'] / 1000
```



## Outlier detection: methods and trade‑offs

* IQR rule (simple and robust): flag values outside `Q1 - 1.5*IQR` … `Q3 + 1.5*IQR`.
* Z‑score (assumes approximate normality) — use only when distribution roughly normal.
* Robust methods: median absolute deviation (MAD).

```python
# IQR example
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
mask = (df['amount'] < Q1 - 1.5*IQR) | (df['amount'] > Q3 + 1.5*IQR)
df_out = df[mask]
```

* Decide per business rules: sometimes outliers are important signals, not errors.



## Preparing features for modeling: encoding & scaling

* Encoding strategies

    * Low-cardinality nominal → one‑hot (`pd.get_dummies`)
    * Ordered categories → ordinal encoding (`cat.codes`)
    * High-cardinality → target encoding, hashing, or embeddings (careful with leakage)

* Scaling

    * `StandardScaler` for mean‑centered algorithms
    * `MinMaxScaler` for bounded scaling
    * Fit on training set only, apply same transform to validation/test

* Example workflow with scikit‑learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

num_cols = ['age','income']
cat_cols = ['color','region']

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

pipe = Pipeline([('pre', pre), ('clf', SomeEstimator())])
```



## Pipelines, provenance and testing

* Encapsulate cleaning steps into functions and unit‑test them (example: `clean_names`, `impute_income`).
* Use `pipe()` to chain transformations in a readable way:

```python
cleaned = (raw
           .pipe(clean_names)
           .pipe(fill_missing)
           .pipe(encode_features))
```

* Save intermediate artifacts (Parquet) with clear filenames and timestamps. Keep a small README describing transformations.
* Add simple assertions in pipelines to verify invariants (`assert df['id'].is_unique`) and raise early if expectations fail.



## Logging and monitoring cleaning jobs

* For long runs, log progress (rows processed, chunks, anomalies found).
* Example: log counts of missing values and number of rows written per chunk.
* In production, emit metrics (counts of dropped rows, imputed values) to monitoring dashboards.



## Real-world mini case study: cleaning a customer dataset (walkthrough)

* Problem: CSV from marketing with columns: `cust_id, name, email, phone, country, signup_date, spend_usd` with various issues:

    * stray whitespace and bad capitalization in `name`
    * missing emails and inconsistent phone formats
    * `country` values contain typos
    * `signup_date` in `dd/mm/yyyy` format
    * `spend_usd` stored as strings with `$` and `,`

* Steps (apply, check, repeat):

    * Inspect sample lines in the file (first 20 rows)
    * Read with `dtype={'cust_id': 'string'}` and `parse_dates=['signup_date']` using `dayfirst=True`
    * Clean `name`: strip/normalize/title-case and remove punctuation
    * Clean `email`: `str.lower()` and `str.contains('@')` to flag invalids
    * Normalize `phone`: remove non-digit chars, extract country code
    * Standardize `country` using a mapping dict and convert to `category`
    * Convert `spend_usd` with `str.replace('[^0-9\.-]','')` then `pd.to_numeric`
    * Save cleaned dataset to `cleaned_customers.parquet`

This stepwise approach (inspect → parse → transform → validate → save) is reusable across datasets.



## Tests and validation checks to add to pipelines

* Row‑count sanity: input vs output (allow for expected drops).
* Unique key check: `assert df['cust_id'].is_unique`.
* Range checks for numeric columns: `assert df['age'].between(0,120).all()`.
* Null checks for required columns: `assert df['email'].notna().sum() >= threshold` or log the shortfall.



## Exercises (expanded and guided)

* Guided: clean a toy customer CSV (provided or self‑created). Include a step where you intentionally introduce a few bad rows to test `on_bad_lines='skip'` and then recover them for manual inspection.
* Exploratory: take a messy dataset, write three small cleaning functions (`normalize_names`, `fill_missing_values`, `encode_categorical`), and unit test them on small examples.
* Performance: compare runtime of `df.apply(func, axis=1)` vs vectorized operations on a 1‑million row sample (use `%timeit`).



## Quick checklist to copy into your pipeline scripts

* [ ] Inspect first 100 rows and top 10 problematic rows
* [ ] Log initial `df.shape`, `df.info()` and `df.isna().sum()`
* [ ] Standardize strings and strip whitespace
* [ ] Convert datetimes and numeric columns with `errors='coerce'`
* [ ] Handle missing values and add indicators where appropriate
* [ ] Convert repeated strings to `category`
* [ ] Validate primary keys and critical ranges
* [ ] Save cleaned artifact (Parquet) and a short transformation README



## Further reading and tools

* Pandas documentation on IO and indexes
* "Feature Engineering for Machine Learning" — recipes for transformations and encoding
* Dask for scaling Pandas workflows out‑of‑core
* `great_expectations` for data validation in pipelines