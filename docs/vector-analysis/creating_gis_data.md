# Creating GIS Data in Python from Scratch

This chapter focuses on creating and manipulating GIS data entirely in Python, with an emphasis on **Shapely**, the core library for geometric operations.  
By the end of this chapter, you’ll be able to create, transform, analyze, and export geometry data using **Shapely** and **GeoPandas**.

---

## 1. Introduction to Shapely

[Shapely](https://shapely.readthedocs.io/) is a Python library for **creating and manipulating planar geometric objects**.  
It is the foundation for spatial data operations in Python and integrates seamlessly with **GeoPandas**.

Shapely allows you to create and analyze geometries like:
- Points (for locations)
- LineStrings (for linear features like roads or rivers)
- Polygons (for areas like lakes, parks, or buildings)

---

## 2. Creating Geometries

### Importing Shapely Geometry Classes
```python
from shapely.geometry import Point, LineString, Polygon
```

### Creating a Point
```python
p = Point(77.5946, 12.9716)  # Bengaluru
print(p.x, p.y)
```
You can also create **3D points**:
```python
p3d = Point(77.5946, 12.9716, 920)
print(p3d.z)
```

### Creating a LineString
```python
line = LineString([(77.5, 12.9), (77.6, 13.0), (77.7, 13.2)])
print(line.length)
print(line.coords[:])
```

### Creating a Polygon
```python
poly = Polygon([(77.5, 12.9), (77.7, 13.1), (77.9, 13.0), (77.5, 12.9)])
print(poly.area)
print(poly.bounds)  # (minx, miny, maxx, maxy)
```

---

## 3. Common Shapely Attributes and Methods

### Point Methods
| Method | Description |
|---------|--------------|
| `.x`, `.y`, `.z` | Access coordinate values |
| `.buffer(distance)` | Creates a circular buffer polygon around the point |
| `.distance(other)` | Calculates Euclidean distance to another geometry |
| `.within(other)` | Checks if point lies within another geometry |
| `.intersects(other)` | Checks if it touches or overlaps another geometry |

### Example: Point Buffer
```python
p = Point(0, 0)
buffered = p.buffer(10)
print(buffered.area)  # Area of circle with radius 10
```

### LineString Methods
| Method | Description |
|---------|--------------|
| `.length` | Total length of line |
| `.coords` | Returns list of coordinates |
| `.interpolate(distance)` | Returns a point at given distance along line |
| `.simplify(tolerance)` | Simplifies line by reducing vertices |

```python
line = LineString([(0, 0), (1, 1), (2, 2)])
print(line.interpolate(1))  # Point at distance 1 along line
```

### Polygon Methods
| Method | Description |
|---------|--------------|
| `.area` | Polygon area |
| `.centroid` | Returns the geometric center |
| `.bounds` | Bounding box (minx, miny, maxx, maxy) |
| `.exterior` | Outer boundary as LineString |
| `.interiors` | List of holes within polygon |
| `.contains(other)` | True if polygon fully contains another geometry |
| `.touches(other)` | True if boundaries touch |
| `.difference(other)` | Returns area difference |
| `.union(other)` | Merges two polygons |

---

## 4. Geometry Relationships and Boolean Operations

Shapely supports powerful geometric predicates and operations.

### Boolean Operations
| Operation | Example | Description |
|------------|----------|--------------|
| `intersects` | `poly1.intersects(poly2)` | Checks if they touch/overlap |
| `contains` | `poly1.contains(p)` | True if poly1 fully contains p |
| `within` | `p.within(poly1)` | True if p inside poly1 |
| `equals` | `geom1.equals(geom2)` | Checks geometric equality |

### Set-like Operations
| Operation | Example | Description |
|------------|----------|--------------|
| `union` | `poly1.union(poly2)` | Combines both areas |
| `intersection` | `poly1.intersection(poly2)` | Common overlapping region |
| `difference` | `poly1.difference(poly2)` | Removes overlapping part |
| `symmetric_difference` | `poly1.symmetric_difference(poly2)` | Keeps non-overlapping parts |

Example:
```python
from shapely.geometry import Polygon

poly1 = Polygon([(0,0), (2,0), (2,2), (0,2)])
poly2 = Polygon([(1,1), (3,1), (3,3), (1,3)])

intersection = poly1.intersection(poly2)
union = poly1.union(poly2)
difference = poly1.difference(poly2)
```

---

## 5. Geometric Manipulation

### Buffering
```python
p = Point(0, 0)
circle = p.buffer(5)
circle.boundary.length  # Circumference
```

### Simplification
```python
poly = Polygon([(0,0), (1,0.1), (2,0.2), (3,0), (3,1), (0,1)])
simple_poly = poly.simplify(0.2)
```

### Convex Hull
Creates the smallest convex polygon that encloses all points.
```python
from shapely.geometry import MultiPoint

points = MultiPoint([(0,0), (1,2), (2,1), (2,2)])
hull = points.convex_hull
```

### Centroid
```python
poly = Polygon([(0,0), (4,0), (4,4), (0,4)])
print(poly.centroid)
```

### Bounds and Envelope
```python
print(poly.bounds)     # (minx, miny, maxx, maxy)
print(poly.envelope)   # Minimum bounding rectangle
```

---

## 6. Combining Shapely with GeoPandas

After creating geometries, we can integrate them into **GeoDataFrames** for storage and export.

```python
import geopandas as gpd
from shapely.geometry import Point, Polygon

cities = ['Delhi', 'Mumbai', 'Chennai']
geometry = [Point(77.1, 28.7), Point(72.8, 19.0), Point(80.2, 13.0)]

gdf = gpd.GeoDataFrame({'city': cities}, geometry=geometry, crs='EPSG:4326')
gdf.plot(markersize=80, color='crimson')
```

### Export Formats
```python
gdf.to_file('cities.geojson', driver='GeoJSON')
gdf.to_file('cities.gpkg', layer='cities', driver='GPKG')
```

---

## 7. Practical GIS Examples Using Shapely

### 7.1 Creating Buffer Zones Around Locations
```python
schools = [Point(77.5, 12.9), Point(77.6, 13.0)]
buffers = [s.buffer(0.05) for s in schools]
```

### 7.2 Checking Spatial Relationships
```python
city_boundary = Polygon([(77.4,12.8),(77.8,12.8),(77.8,13.2),(77.4,13.2)])
school = Point(77.6, 12.9)

print(school.within(city_boundary))
```

### 7.3 Clipping a Line by a Polygon
```python
line = LineString([(77.3, 12.9), (77.9, 13.1)])
clipped = line.intersection(city_boundary)
```

### 7.4 Calculating Area and Length
```python
print(city_boundary.area)
print(line.length)
```

---

## 8. Summary of Common Shapely Functions

| Category | Function | Description |
|-----------|-----------|-------------|
| **Creation** | `Point`, `LineString`, `Polygon`, `MultiPoint`, `MultiPolygon` | Create geometries |
| **Analysis** | `area`, `length`, `centroid`, `bounds`, `envelope` | Measure properties |
| **Relationships** | `contains`, `within`, `intersects`, `touches`, `equals` | Spatial relationships |
| **Operations** | `union`, `difference`, `intersection`, `buffer`, `simplify` | Modify shapes |
| **Transformation** | `translate`, `rotate`, `scale` | Move or reshape geometries |

---

## 9. Real-World Use Cases

- **Urban Planning:** Create 500m buffers around schools to check overlapping noise zones.  
- **Environmental Studies:** Calculate intersection area of flood zones and forest cover.  
- **Transportation:** Simplify road networks for visualization and performance.  
- **Boundary Management:** Merge administrative polygons to create regional boundaries.  

---

## 10. Practice Exercises

1. Create 10 random points and compute a convex hull around them.  
2. Create buffers around these points and calculate overlap area.  
3. Clip a polygon with another and visualize using GeoPandas.  
4. Create a custom function that calculates distance between multiple city pairs.  

---

*End of Chapter — “Creating GIS Data in Python from Scratch”*
