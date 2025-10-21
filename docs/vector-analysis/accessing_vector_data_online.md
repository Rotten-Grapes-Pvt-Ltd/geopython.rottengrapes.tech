
# Accessing Vector Data from Online Sources — Full Working Example

## 1️⃣ Setup

```python
import geopandas as gpd
import osmnx as ox
import requests
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt
```

---

## 2️⃣ Download Vector Data via HTTP

```python
# Download a zipped shapefile from a URL
url = "https://github.com/gregoiredavid/france-geojson/raw/master/departements.zip"
r = requests.get(url)
with zipfile.ZipFile(BytesIO(r.content)) as z:
    z.extractall("data_zip/")

# Load the shapefile
gdf_zip = gpd.read_file("data_zip/departements.shp")
print(gdf_zip.head())
```

---

## 3️⃣ Reading from OpenStreetMap using OSMnx

```python
# Download the road network for Pune, India
G = ox.graph_from_place("Pune, India", network_type="drive")

# Convert the graph to GeoDataFrames
edges = ox.graph_to_gdfs(G, nodes=False)
nodes = ox.graph_to_gdfs(G, edges=False)

print(edges.head())
```

---

## 4️⃣ Using Overpass API for Custom Queries

```python
# Define a bounding box: (south, west, north, east)
bbox = (18.45, 73.80, 18.55, 73.95)

# Query all parks inside bounding box
tags = {"leisure": "park"}
gdf_parks = ox.geometries_from_bbox(*bbox, tags=tags)

print(gdf_parks.head())
```

---

## 5️⃣ Web-hosted GeoJSON Example

```python
# Directly read GeoJSON from a URL
url_geojson = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
gdf_geojson = gpd.read_file(url_geojson)
print(gdf_geojson.head())
```

---

## 6️⃣ Combine All Sources: Full Workflow

```python
fig, ax = plt.subplots(figsize=(12,8))

# Plot HTTP shapefile (France departments) as background
gdf_zip.to_crs(epsg=3857).plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)

# Plot Pune road network
edges.to_crs(epsg=3857).plot(ax=ax, color='blue', linewidth=1, alpha=0.7)

# Plot parks from Overpass API
gdf_parks.to_crs(epsg=3857).plot(ax=ax, color='green', markersize=20)

plt.title("Combining Vector Data: HTTP + OSMnx + Overpass")
plt.show()
```

---

## 7️⃣ Best Practices

- Check CRS with `gdf.crs` and convert if needed.
- Filter large datasets using bounding boxes or tags.
- Inspect using `.head()`, `.info()`, and `.plot()`.
- Save cleaned or downloaded data locally:

```python
gdf_parks.to_file("parks_pune.geojson", driver="GeoJSON")
```

---

## ✅ Summary

- HTTP downloads allow accessing generic shapefiles and GeoJSONs.
- OSMnx provides street networks and building footprints.
- Overpass API is for flexible, tag-based custom queries.
- Combining multiple sources is easy with GeoPandas and consistent CRS.
