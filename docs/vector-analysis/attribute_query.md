# Attribute Operations in GeoPandas

**GeoPandas** allows users to manipulate spatial and non-spatial data efficiently — just like **pandas**, but with the power of geometry operations. In this chapter, we’ll explore how to work with attributes, filter data, and combine spatial datasets.

---

## Filtering and Querying Features

Filtering is essential to extract specific features based on attribute values or conditions.

### Example: Filter by attribute value
```python
import geopandas as gpd

gdf = gpd.read_file("data/countries.gpkg")

# Filter countries with population greater than 10 million
large_countries = gdf[gdf["population"] > 10_000_000]
```

### Using `query()` for readable conditions
```python
asian_countries = gdf.query("continent == 'Asia' and population > 5000000")
```

### Filter by multiple conditions
```python
filtered = gdf[(gdf["area"] > 100000) & (gdf["gdp_per_capita"] > 10000)]
```

---

## Creating and Updating Columns

GeoPandas allows easy creation and modification of attribute columns.

### Example: Create new column
```python
gdf["population_density"] = gdf["population"] / gdf["area"]
```

### Example: Update column values
```python
gdf.loc[gdf["continent"] == "Asia", "region"] = "Eastern Hemisphere"
```

### Example: Conditional column creation
```python
gdf["category"] = gdf["population"].apply(lambda x: "High" if x > 10_000_000 else "Low")
```

### Example: Rename or drop columns
```python
gdf = gdf.rename(columns={"gdp": "GDP"})
gdf = gdf.drop(columns=["old_column"])
```

---

## Merging Tabular and Spatial Data

You can combine spatial datasets or attach non-spatial attributes using `merge()` or `join()`.

### Example: Merge GeoDataFrame with tabular data
```python
import pandas as pd

stats = pd.read_csv("data/country_stats.csv")

merged = gdf.merge(stats, on="country_code")
```

### Spatial join (combine based on geometry relationship)
```python
points = gpd.read_file("data/cities.geojson")
polygons = gpd.read_file("data/states.geojson")

# Join points to polygons to identify which state each city belongs to
joined = gpd.sjoin(points, polygons, how="left", predicate="within")
```

### Merge multiple layers
```python
gdf_combined = gdf1.merge(gdf2, on="region_id", how="inner")
```

---

## GroupBy, Aggregation, and Statistical Summaries

Like pandas, GeoPandas supports powerful group and aggregation operations.

### Example: Group by region and sum population
```python
region_stats = gdf.groupby("region")["population"].sum().reset_index()
```

### Example: Multiple aggregations
```python
summary = gdf.groupby("continent").agg({
    "population": ["sum", "mean"],
    "area": ["mean"]
}).reset_index()
```

### Example: Join results back to spatial data
```python
gdf_summary = gdf.merge(region_stats, on="region")
```

### Example: Group by geometry type
```python
geom_summary = gdf.groupby(gdf.geom_type).size()
print(geom_summary)
```

---

## Practical Tips

- Always verify the join key when merging — mismatched values can lead to missing data.
- Use `.copy()` before modifying columns to avoid warnings.
- For large datasets, filter first before performing expensive spatial joins.
- For performance, prefer vectorized operations (`apply()` or arithmetic) over row-by-row loops.

---

## Summary

This chapter covered how to manage, query, and manipulate attributes in **GeoPandas**, including filtering data, creating and updating columns, merging tables, and performing aggregations. Mastering these operations is key to effective spatial data analysis.
