# Spatial Operations in GeoPandas

## Introduction

Spatial operations are the heart of vector-based GIS analysis. GeoPandas provides high-level access to these operations through its integration with **Shapely** and **PyGEOS** under the hood. With these tools, you can measure spatial relationships, combine geometries, and derive new spatial datasets.

---

## Core Spatial Relationships

- **Intersects** — Check if geometries overlap or touch.  
  ```python
  gdf['intersects'] = gdf.geometry.intersects(other_gdf.unary_union)
  ```

- **Within / Contains** — Identify features inside or containing others.  
  ```python
  gdf['within_boundary'] = gdf.geometry.within(boundary.geometry.iloc[0])
  gdf['contains_point'] = gdf.geometry.contains(point)
  ```

- **Touches / Crosses / Overlaps** — Detect specific spatial relationships between lines and polygons.  
  ```python
  roads['touches_boundary'] = roads.touches(city_boundary.geometry.iloc[0])
  rivers['crosses_road'] = rivers.crosses(roads.unary_union)
  ```

- **Distance-based relations** — Compute minimum distance between features.  
  ```python
  gdf['dist_to_road'] = gdf.distance(roads.unary_union)
  ```

---

## Geometric Set Operations

- **Union** — Merge multiple geometries into one.  
  ```python
  merged = gdf.unary_union
  ```

- **Intersection** — Extract the common area between two geometries.  
  ```python
  common_area = gpd.overlay(gdf1, gdf2, how='intersection')
  ```

- **Difference** — Subtract one geometry from another.  
  ```python
  remaining = gpd.overlay(gdf1, gdf2, how='difference')
  ```

- **Symmetric Difference** — Areas belonging to one or the other, but not both.  
  ```python
  unique_areas = gpd.overlay(gdf1, gdf2, how='symmetric_difference')
  ```

---

## Buffering and Proximity Analysis

Buffers are used to create zones around features — essential in proximity and impact analysis.

```python
# Create 500-meter buffer around roads
roads['buffer_500m'] = roads.geometry.buffer(500)

# Find all schools within 500m of a road
schools_near_roads = schools[schools.geometry.intersects(roads.unary_union.buffer(500))]
```

Tips:
- Always confirm the **CRS is in meters** before buffering (e.g., `EPSG:32643`).
- Use `buffer(distance, cap_style=2)` for square edges.

---

## Spatial Joins

Spatial joins link features based on spatial relationships — like finding which district a school belongs to.

```python
schools_with_districts = gpd.sjoin(schools, districts, how='left', predicate='within')
```

Options for `predicate`:
- `'intersects'`
- `'within'`
- `'contains'`

After a spatial join, your attributes are merged, enabling both geometric and tabular analysis.

---

## Clipping and Masking

Clip a layer to a defined boundary (common for administrative region analysis).

```python
subset = gpd.clip(landcover, region)
```

This reduces data size and isolates the area of interest for further analysis.

---

## Dissolving and Aggregation

Combine geometries sharing the same attribute, such as merging polygons of the same state.

```python
states = districts.dissolve(by='state_name', aggfunc='sum')
```

- `aggfunc` can summarize numeric columns (`sum`, `mean`, `max`, etc.).
- Dissolving simplifies layers for cleaner visualization or reporting.

---

## Centroids and Bounding Boxes

Useful for representing complex geometries with simpler spatial representations.

```python
# Compute centroid of each polygon
gdf['centroid'] = gdf.geometry.centroid

# Extract bounding boxes
gdf['bbox'] = gdf.geometry.envelope
```

---

## Real-World Use Cases

- **Environmental Analysis** — Overlay land cover with administrative boundaries to estimate forest cover by region.  
- **Urban Planning** — Buffer roads or rivers to identify restricted zones.  
- **Accessibility Studies** — Use distance and intersection functions to find service areas around facilities.  
- **Infrastructure Risk Assessment** — Clip hazard zones (floods, earthquakes) with asset locations.

---

## Best Practices

- Always **verify CRS alignment** before performing spatial operations. Use `gdf.to_crs()` if needed.  
- When working with large datasets, prefer **vectorized** operations instead of row-by-row loops.  
- Validate geometries before processing using `gdf.is_valid`. Repair with `gdf.buffer(0)` if needed.  
- Use **simplified geometries** for faster performance in large-scale spatial joins.

---

## Further Reading

- GeoPandas documentation — [https://geopandas.org/](https://geopandas.org/)  
- Shapely documentation for geometry operations — [https://shapely.readthedocs.io/](https://shapely.readthedocs.io/)  
- GDAL/OGR command-line utilities for preprocessing large spatial files.

---
