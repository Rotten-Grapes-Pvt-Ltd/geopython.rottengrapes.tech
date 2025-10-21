# Spatial Operations in GeoPandas

Spatial operations allow us to combine, compare, and analyze geographic
features based on their geometry.\
In this tutorial, we'll use **Natural Earth** datasets and explore how
to perform key spatial operations in **GeoPandas**.

------------------------------------------------------------------------

## 1️⃣ Setup and Import

``` python
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
```

------------------------------------------------------------------------

## 2️⃣ Download and Load Natural Earth Data

GeoPandas has a convenient function to load sample datasets from the
**Natural Earth** collection.

``` python
# Load world boundaries
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Load cities (requires geopandas datasets plugin)
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

world.head()
```

------------------------------------------------------------------------

## 3️⃣ Inspecting and Visualizing

``` python
# Quick visualization
fig, ax = plt.subplots(figsize=(10, 6))
world.plot(ax=ax, color='lightgray', edgecolor='black')
cities.plot(ax=ax, color='red', markersize=10)
plt.show()
```

------------------------------------------------------------------------

## 4️⃣ Reprojecting to a Metric CRS

Always use a **projected CRS** for distance or area calculations.

``` python
world = world.to_crs(epsg=3857)
cities = cities.to_crs(epsg=3857)
```

------------------------------------------------------------------------

## 5️⃣ Buffer Operation (Example: 200 km around cities)

``` python
cities['buffer_200km'] = cities.buffer(200000)  # meters since EPSG:3857
buffer_gdf = gpd.GeoDataFrame(cities[['name']], geometry=cities['buffer_200km'])
```

``` python
# Plot buffers
fig, ax = plt.subplots(figsize=(10, 6))
world.plot(ax=ax, color='lightgray')
buffer_gdf.plot(ax=ax, facecolor='none', edgecolor='blue')
cities.plot(ax=ax, color='red', markersize=20)
plt.show()
```

------------------------------------------------------------------------

## 6️⃣ Spatial Join --- Find Which Country Each City Belongs To

``` python
city_country = gpd.sjoin(cities, world, how='left', predicate='within')
city_country[['name', 'name_right']].head()
```

This gives the country name for each city.

------------------------------------------------------------------------

## 7️⃣ Overlay Operations

Overlay operations combine two spatial datasets based on geometry ---
like **intersection**, **union**, **difference**, and **symmetric
difference**.

Example: Find part of countries within a 200 km radius of any major
city.

``` python
country_near_city = gpd.overlay(world, buffer_gdf, how='intersection')
country_near_city.plot(color='orange', edgecolor='black')
```

------------------------------------------------------------------------

## 8️⃣ Measuring Area and Distance

``` python
# Area in square kilometers
world['area_km2'] = world.geometry.area / 10**6

# Example: Distance between first two cities
distance = cities.distance(cities.geometry.iloc[0])
distance.head()
```

------------------------------------------------------------------------

## 9️⃣ Centroid Extraction

``` python
world['centroid'] = world.centroid
centroids = gpd.GeoDataFrame(geometry=world['centroid'])
```

``` python
# Plot centroids
fig, ax = plt.subplots(figsize=(10, 6))
world.plot(ax=ax, color='lightgreen', edgecolor='black')
centroids.plot(ax=ax, color='red', markersize=15)
plt.show()
```

------------------------------------------------------------------------

## 🔟 Summary of Spatial Operations Covered

  -----------------------------------------------------------------------------
  Operation               Function              Description
  ----------------------- --------------------- -------------------------------
  Buffer                  `buffer()`            Creates zones around geometries

  Spatial Join            `gpd.sjoin()`         Combine layers based on spatial
                                                relation

  Overlay                 `gpd.overlay()`       Perform
                                                intersection/union/difference

  Reprojection            `to_crs()`            Convert CRS for metric accuracy

  Area/Distance           `.area`,              Compute measurements
                          `.distance()`         

  Centroid                `.centroid`           Get geometric center
  -----------------------------------------------------------------------------

------------------------------------------------------------------------

## ✅ Conclusion

You've learned how to: - Load real-world datasets (Natural Earth) -
Perform spatial joins, overlays, and measurements - Visualize and
manipulate geometry data in GeoPandas

Next, try combining these tools for your **own spatial analysis
workflows** --- such as proximity analysis, regional summaries, or
environmental impact assessments.
