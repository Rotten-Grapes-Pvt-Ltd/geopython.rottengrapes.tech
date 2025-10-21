
# Advanced Spatial Operations with GeoPandas

## 1️⃣ Setup

```python
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
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

## 5️⃣ Buffering

```python
# 50km buffer around cities
pop_places['buffer_50km'] = pop_places.geometry.buffer(50000)

# Plot
fig, ax = plt.subplots(figsize=(12,8))
countries.plot(ax=ax, color='lightgray')
pop_places['buffer_50km'].plot(ax=ax, facecolor='none', edgecolor='green')
pop_places.plot(ax=ax, color='red', markersize=20)
plt.show()
```

## 6️⃣ Spatial Joins

```python
# Find which country each city belongs to
city_country = gpd.sjoin(pop_places, countries, how='left', predicate='within')

# Find roads within a buffer of 50km of cities
roads_near_cities = gpd.sjoin(roads, pop_places[['buffer_50km']], how='inner', predicate='intersects')
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

## 1️⃣2️⃣ Distance Calculations

```python
# Distance from each city to the nearest road
pop_places['dist_to_road'] = pop_places.geometry.apply(lambda x: roads.distance(x).min())
```

## 1️⃣3️⃣ Advanced Examples

- Find cities within 50km of any road:  

```python
cities_near_roads = gpd.sjoin(pop_places, roads.buffer(50000).rename('geometry').to_frame(), how='inner', predicate='intersects')
```

- Count number of cities per country:  

```python
city_count = city_country.groupby('name_right').size().reset_index(name='city_count')
```

- Plot country centroids with city counts:

```python
fig, ax = plt.subplots(figsize=(12,8))
countries.plot(ax=ax, color='lightgray')
for idx, row in countries.iterrows():
    plt.text(row['centroid'].x, row['centroid'].y, str(city_count.loc[city_count['name_right']==row['name'], 'city_count'].values[0] if not city_count.loc[city_count['name_right']==row['name']].empty else 0), fontsize=8)
```

## ✅ All Key Spatial Operations Covered

- Inspect & plot  
- CRS transformation  
- Buffer & proximity  
- Spatial join & nearest  
- Overlay: intersection, difference, union, symmetric difference  
- Clip & mask  
- Dissolve & aggregation  
- Centroid & bounding box  
- Distance calculation  
