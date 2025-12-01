
# Advanced Spatial Operations with GeoPandas

## Overview
Advanced spatial operations are essential for complex geospatial analysis tasks such as urban planning, environmental monitoring, transportation analysis, and demographic studies. This guide covers sophisticated techniques that go beyond basic spatial queries.

## When to Use Advanced Operations
- **Urban Planning**: Analyzing accessibility, service coverage, and land use patterns
- **Environmental Analysis**: Studying habitat connectivity, pollution dispersion, watershed analysis
- **Transportation**: Route optimization, service area analysis, network connectivity
- **Demographics**: Population distribution analysis, catchment area studies
- **Emergency Response**: Risk assessment, evacuation planning, resource allocation

## 1️⃣ Setup

```python
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

## 2️⃣ Load Natural Earth Data

```python
# Countries (polygons)
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Populated places (points)
pop_places = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

# Roads (lines) - from Natural Earth
roads = gpd.read_file("https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_roads.zip")
```

## 3️⃣ Inspect & Visualize

```python
fig, ax = plt.subplots(figsize=(12,8))
countries.plot(ax=ax, color='lightgray', edgecolor='black')
roads.plot(ax=ax, color='blue')
pop_places.plot(ax=ax, color='red', markersize=20)
plt.show()
```

## 4️⃣ CRS Transformation

```python
# Project to metric CRS for distance-based operations
countries = countries.to_crs(epsg=3857)
pop_places = pop_places.to_crs(epsg=3857)
roads = roads.to_crs(epsg=3857)
```

## 5️⃣ Buffering - Real-World Applications

### Use Case 1: School Catchment Areas
```python
# Create 1km walking distance buffer around schools
schools = pop_places.sample(10)  # Sample some cities as schools
schools['catchment_1km'] = schools.geometry.buffer(1000)

# Visualize catchment areas
fig, ax = plt.subplots(figsize=(12,8))
countries.plot(ax=ax, color='lightgray')
schools['catchment_1km'].plot(ax=ax, facecolor='lightblue', alpha=0.5)
schools.plot(ax=ax, color='red', markersize=50)
plt.title('School Catchment Areas (1km radius)')
plt.show()
```

### Use Case 2: Environmental Impact Assessment
```python
# 50km buffer around cities for pollution analysis
pop_places['buffer_50km'] = pop_places.geometry.buffer(50000)

# Multi-level risk zones
pop_places['high_risk'] = pop_places.geometry.buffer(20000)
pop_places['medium_risk'] = pop_places.geometry.buffer(35000)
pop_places['low_risk'] = pop_places.geometry.buffer(50000)

# Plot multi-level buffers
fig, ax = plt.subplots(figsize=(12,8))
countries.plot(ax=ax, color='lightgray')
pop_places['low_risk'].plot(ax=ax, facecolor='yellow', alpha=0.3, label='Low Risk')
pop_places['medium_risk'].plot(ax=ax, facecolor='orange', alpha=0.4, label='Medium Risk')
pop_places['high_risk'].plot(ax=ax, facecolor='red', alpha=0.5, label='High Risk')
pop_places.plot(ax=ax, color='darkred', markersize=20)
plt.legend()
plt.title('Multi-level Risk Assessment Zones')
plt.show()
```

## 6️⃣ Spatial Joins - Administrative & Proximity Analysis

```python
# Find which country each city belongs to
city_country = gpd.sjoin(pop_places, countries, how='left', predicate='within')

# Find roads within a buffer of 50km of cities
roads_near_cities = gpd.sjoin(roads, pop_places[['buffer_50km']], how='inner', predicate='intersects')

# Count features per administrative unit
cities_per_country = city_country.groupby('name_right').size().reset_index(name='city_count')
print(f"Countries with most cities: {cities_per_country.nlargest(5, 'city_count')}")

# Service coverage analysis
serviced_areas = gpd.sjoin(countries, pop_places[['buffer_50km']], predicate='intersects')
coverage_stats = serviced_areas.groupby('name_left').agg({
    'pop_est': 'first',
    'name_right': 'count'
}).rename(columns={'name_right': 'service_points'})
```

## 7️⃣ Nearest Feature Example

```python
# Find nearest country for each city (if outside any country)
pop_places['nearest_country'] = pop_places.geometry.apply(lambda x: countries.distance(x).idxmin())
```

## 8️⃣ Overlay Operations

```python
# Intersection: parts of countries covered by city buffers
intersect_gdf = gpd.overlay(countries, pop_places[['buffer_50km']], how='intersection')

# Difference: country area not covered by buffers
diff_gdf = gpd.overlay(countries, pop_places[['buffer_50km']], how='difference')

# Symmetric difference
sym_diff_gdf = gpd.overlay(countries, pop_places[['buffer_50km']], how='symmetric_difference')

# Union: combine all geometries into one
union_gdf = gpd.overlay(countries, pop_places[['buffer_50km']], how='union')
```

## 9️⃣ Clipping

```python
# Clip roads to country boundaries
roads_clipped = gpd.clip(roads, countries)
```

## 🔟 Dissolve & Aggregation

```python
# Dissolve countries by continent
continent_gdf = countries.dissolve(by='continent', aggfunc='sum')
```

## 1️⃣1️⃣ Centroids & Bounding Boxes

```python
countries['centroid'] = countries.centroid
countries['bbox'] = countries.envelope
```

## 1️⃣2️⃣ Distance Calculations - Accessibility Analysis

```python
# Distance from each city to the nearest road
pop_places['dist_to_road'] = pop_places.geometry.apply(lambda x: roads.distance(x).min())

# Accessibility classification
pop_places['accessibility'] = pd.cut(pop_places['dist_to_road'], 
                                   bins=[0, 5000, 15000, 50000, float('inf')],
                                   labels=['Excellent', 'Good', 'Fair', 'Poor'])

# Distance matrix between cities
from scipy.spatial.distance import pdist, squareform
coords = np.array(list(zip(pop_places.geometry.x, pop_places.geometry.y)))
dist_matrix = squareform(pdist(coords))

# Find nearest neighbors
nearest_city_idx = np.argsort(dist_matrix, axis=1)[:, 1]  # Second closest (first is self)
pop_places['nearest_city_dist'] = dist_matrix[np.arange(len(pop_places)), nearest_city_idx]

# Visualize accessibility
fig, ax = plt.subplots(figsize=(12, 8))
countries.plot(ax=ax, color='lightgray')
pop_places.plot(ax=ax, column='accessibility', categorical=True, 
               legend=True, markersize=50, cmap='RdYlGn')
plt.title('City Accessibility to Road Network')
plt.show()
```

## 1️⃣3️⃣ Advanced Workflows

### Workflow 1: Accessibility Analysis
```python
def accessibility_analysis(facilities, population, buffer_distance=1000):
    # Create service areas around facilities
    facilities['service_area'] = facilities.geometry.buffer(buffer_distance)
    
    # Calculate population served
    pop_served = gpd.sjoin(population, facilities[['service_area']], predicate='within')
    
    # Identify underserved areas
    all_service_areas = facilities['service_area'].unary_union
    underserved = population[~population.geometry.within(all_service_areas)]
    
    return pop_served, underserved

# Apply accessibility analysis
served_cities, underserved_cities = accessibility_analysis(pop_places.sample(20), pop_places)
print(f"Cities served: {len(served_cities)}, Underserved: {len(underserved_cities)}")
```

### Workflow 2: Land Use Conflict Analysis
```python
def land_use_conflict_analysis(industrial, residential, buffer_distance=1000):
    # Buffer industrial areas for impact analysis
    industrial['impact_zone'] = industrial.geometry.buffer(buffer_distance)
    
    # Find conflicts
    conflicts = gpd.overlay(residential, industrial[['impact_zone']], how='intersection')
    conflicts['conflict_area'] = conflicts.geometry.area
    
    return conflicts

# Simulate industrial and residential areas
industrial_sites = pop_places.sample(10).copy()
residential_areas = countries.sample(5).copy()

conflicts = land_use_conflict_analysis(industrial_sites, residential_areas)
print(f"Found {len(conflicts)} potential land use conflicts")
```

### Workflow 3: Multi-Criteria Site Selection
```python
def site_selection_analysis(candidates, criteria_layers, weights):
    results = candidates.copy()
    
    for i, (layer, weight) in enumerate(zip(criteria_layers, weights)):
        # Calculate distance to each criteria layer
        distances = candidates.geometry.apply(lambda x: layer.distance(x).min())
        # Normalize and apply weight
        normalized = (distances.max() - distances) / (distances.max() - distances.min())
        results[f'criteria_{i}'] = normalized * weight
    
    # Calculate total score
    criteria_cols = [col for col in results.columns if col.startswith('criteria_')]
    results['total_score'] = results[criteria_cols].sum(axis=1)
    
    return results.sort_values('total_score', ascending=False)

# Example: Select best locations for new facilities
candidate_sites = countries.centroid.to_frame('geometry')
candidate_sites = gpd.GeoDataFrame(candidate_sites, geometry='geometry', crs=countries.crs)

criteria = [pop_places, roads]  # Near cities and roads
weights = [0.6, 0.4]  # Cities more important than roads

best_sites = site_selection_analysis(candidate_sites, criteria, weights)
print("Top 5 candidate sites:")
print(best_sites.head()[['total_score']])
```

## 🚀 Performance Optimization

### Spatial Indexing for Large Datasets
```python
# Create spatial index for faster operations
countries_sindex = countries.sindex

# Use spatial index for faster intersection queries
def fast_spatial_join(points, polygons):
    results = []
    for idx, point in points.iterrows():
        possible_matches_index = list(polygons.sindex.intersection(point.geometry.bounds))
        possible_matches = polygons.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(point.geometry)]
        if not precise_matches.empty:
            results.append((idx, precise_matches.index[0]))
    return results
```

### Memory-Efficient Processing
```python
# Process large datasets in chunks
def chunked_spatial_operation(large_gdf, reference_gdf, chunk_size=1000):
    results = []
    for i in range(0, len(large_gdf), chunk_size):
        chunk = large_gdf.iloc[i:i+chunk_size]
        processed = gpd.sjoin(chunk, reference_gdf, predicate='intersects')
        results.append(processed)
    return pd.concat(results, ignore_index=True)
```

## ⚠️ Data Quality & Error Handling

```python
# Geometry validation and repair
invalid_geoms = ~countries.geometry.is_valid
if invalid_geoms.any():
    print(f"Found {invalid_geoms.sum()} invalid geometries")
    countries.loc[invalid_geoms, 'geometry'] = countries.loc[invalid_geoms, 'geometry'].buffer(0)

# CRS compatibility check
def check_crs_compatibility(*gdfs):
    crs_list = [gdf.crs for gdf in gdfs]
    if len(set(crs_list)) > 1:
        print(f"Warning: Mixed CRS detected: {crs_list}")
        return False
    return True

# Safe spatial operations with error handling
def safe_spatial_join(left_gdf, right_gdf, **kwargs):
    try:
        result = gpd.sjoin(left_gdf, right_gdf, **kwargs)
        if result.empty:
            print("Warning: Spatial join returned no results")
        return result
    except Exception as e:
        print(f"Spatial join failed: {e}")
        return gpd.GeoDataFrame()
```

## 📊 Advanced Visualization

```python
# Multi-layer visualization with transparency
fig, ax = plt.subplots(figsize=(15, 10))

# Base layer
countries.plot(ax=ax, color='lightgray', edgecolor='white', alpha=0.7)

# Overlay layers with different transparency
if 'buffer_50km' in pop_places.columns:
    pop_places['buffer_50km'].plot(ax=ax, facecolor='blue', alpha=0.3, edgecolor='blue')
roads.plot(ax=ax, color='red', alpha=0.6, linewidth=0.5)
pop_places.plot(ax=ax, color='darkred', markersize=30, alpha=0.8)

# Styling
ax.set_title('Multi-layer Spatial Analysis', fontsize=16, fontweight='bold')
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

## ✅ Summary: Key Operations Covered

- **Data Loading & Inspection**: Natural Earth datasets, visualization
- **CRS Management**: Projection for accurate distance calculations
- **Buffering**: Service areas, risk zones, catchment analysis
- **Spatial Joins**: Administrative assignment, proximity analysis
- **Overlay Operations**: Coverage analysis, land use conflicts
- **Distance Analysis**: Accessibility metrics, nearest neighbor
- **Advanced Workflows**: Multi-criteria analysis, site selection
- **Performance**: Spatial indexing, chunked processing
- **Quality Control**: Geometry validation, error handling

## 🔗 Integration Opportunities

- **Rasterio**: Combine with raster analysis for elevation, land cover
- **OSMnx**: Network analysis for routing and accessibility
- **Folium**: Interactive web mapping for stakeholder engagement
- **Plotly**: Interactive dashboards for spatial analytics
- **NetworkX**: Graph analysis for connectivity studies
