
# Using OSMnx — Methods and Workflows

## 1️⃣ Setup

```python
import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
```
- Ensure latest OSMnx version installed: `pip install osmnx --upgrade`
- OSMnx uses OpenStreetMap data for street networks, building footprints, POIs, etc.

---

## 2️⃣ Basic Network Download

```python
G = ox.graph_from_place("Pune, India", network_type="drive")
G_walk = ox.graph_from_place("Pune, India", network_type="walk")
```
- `network_type` options: `"drive"`, `"walk"`, `"bike"`, `"all"`, `"all_private"`
- Graph is a NetworkX MultiDiGraph

---

## 3️⃣ Convert Graph to GeoDataFrames

```python
edges = ox.graph_to_gdfs(G, nodes=False)
nodes = ox.graph_to_gdfs(G, edges=False)
```
- `edges` contain street segments
- `nodes` contain intersections

---

## 4️⃣ Plotting Networks

```python
ox.plot_graph(G, node_size=10, edge_color='blue', bgcolor='white')
```
- Simple graph plotting
- Can also use Matplotlib for custom plots with `edges.plot()` and `nodes.plot()`

---

## 5️⃣ Geometries from Place or Polygon

```python
buildings = ox.geometries_from_place("Pune, India", tags={"building": True})
cafes = ox.geometries_from_place("Pune, India", tags={"amenity": "cafe"})
```
- Returns GeoDataFrame with geometries and attributes
- Can filter by tags like `building`, `amenity`, `highway`, `landuse`, etc.

---

## 6️⃣ Using Bounding Boxes

```python
bbox = (18.45, 73.80, 18.55, 73.95)
highways = ox.geometries_from_bbox(*bbox, tags={"highway": True})
```
- Useful for sub-city areas
- Works with buildings, streets, and other features

---

## 7️⃣ Using Polygons

```python
from shapely.geometry import Polygon

poly = Polygon([(73.80,18.45),(73.80,18.55),(73.95,18.55),(73.95,18.45)])
roads_poly = ox.geometries_from_polygon(poly, tags={"highway": True})
```
- Polygon-based queries allow precise custom regions

---

## 8️⃣ Graph Utilities

```python
stats = ox.basic_stats(G)
print(stats)

G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
```
- Adds realistic attributes for routing

---

## 9️⃣ Routing Examples

```python
import networkx as nx

orig_node = list(G.nodes)[0]
dest_node = list(G.nodes)[50]

route = nx.shortest_path(G, orig_node, dest_node, weight='length')

fig, ax = ox.plot_graph_route(G, route, node_size=0, bgcolor='white')
```
- Can use `length`, `travel_time`, or custom weights

---

## 10️⃣ Saving and Loading

```python
ox.save_graphml(G, "pune_drive.graphml")
G2 = ox.load_graphml("pune_drive.graphml")
edges.to_file("pune_edges.shp")
```
- OSMnx integrates with GeoPandas for saving and analysis

---

## 11️⃣ Additional Utilities

- Simplify graphs: `ox.simplify_graph(G)`  
- Project graphs: `G_proj = ox.project_graph(G)` (to metric CRS)  
- Calculate network stats: `ox.extended_stats(G_proj)`  
- Nearest nodes: `ox.distance.nearest_nodes(G, X, Y)`  
- Shortest path with travel times, lengths, or custom attributes

---

## 12️⃣ Full Hands-On Workflow: Roads, Buildings, and POIs

```python
# Define a polygon for a small Pune area
from shapely.geometry import Polygon

poly = Polygon([(73.80,18.50),(73.80,18.52),(73.83,18.52),(73.83,18.50)])

# Get road network within polygon
G_sub = ox.graph_from_polygon(poly, network_type="drive")

# Convert to GeoDataFrames
edges_sub = ox.graph_to_gdfs(G_sub, nodes=False)
nodes_sub = ox.graph_to_gdfs(G_sub, edges=False)

# Get buildings and cafes inside polygon
buildings_sub = ox.geometries_from_polygon(poly, tags={"building": True})
cafes_sub = ox.geometries_from_polygon(poly, tags={"amenity": "cafe"})

# Plot combined layers
fig, ax = plt.subplots(figsize=(10,10))

# Plot roads
edges_sub.plot(ax=ax, linewidth=1, edgecolor='blue')

# Plot buildings
buildings_sub.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)

# Plot cafes
cafes_sub.plot(ax=ax, color='red', markersize=20, label='Cafe')

plt.title("Pune Subset: Roads, Buildings, Cafes")
plt.legend()
plt.show()
```
- Combines network, building footprints, and points of interest
- CRS handled automatically for plotting
- Can extend workflow to routing or spatial analysis

---

## ✅ Summary

- OSMnx allows querying networks, buildings, and POIs by place, bounding box, or polygon
- Provides GeoDataFrames for easy manipulation in GeoPandas
- Full workflows can combine multiple layers for visualization and analysis
- Supports routing, metrics, and saving/loading for reuse
