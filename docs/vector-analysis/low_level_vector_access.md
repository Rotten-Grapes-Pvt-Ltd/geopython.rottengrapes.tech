# Low-Level Vector File Access with Fiona

Fiona is a Python library that provides a **Pythonic interface to GDAL/OGR**, designed for reading and writing vector geospatial data formats such as Shapefile, GeoPackage, and GeoJSON.  
It is a **low-level** library, meaning it gives you **fine-grained control** over how files are created, read, and modified.

---

## Introduction

- Fiona focuses on **files and records** rather than large in-memory data structures like GeoPandas.
- It is perfect for scenarios where you need to:
  - Inspect metadata and schema
  - Write or append to existing vector datasets
  - Manage large files without loading them entirely into memory

---

## Schema Inspection and Custom Field Types

Every vector dataset (e.g., a Shapefile) has a **schema** — a definition of its geometry type and attributes.

### Reading a Schema



```python
import fiona

with fiona.open("populated_places/ne_10m_populated_places_simple.shp", "r") as src:
    print(src.schema)
    print(src.crs)
    print(src.driver)
```
Output might look like:
```python
{'geometry': 'LineString',
 'properties': {'road_name': 'str:80', 'length': 'float'}}
```

### Custom Schemas

When writing new data, you define your own schema:

```python
schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int', 'name': 'str', 'area': 'float'}
}
```

### Supported Data Types

Common property types:
- `'int'` or `'int:10'`
- `'float'` or `'float:24.15'`
- `'str'` or `'str:254'`
- `'date'`, `'datetime'`, `'time'`

---

## Writing Complex Geometry with Fiona

Fiona writes geometries using **GeoJSON-style dictionaries**.

Example of writing multiple geometries:

```python
import fiona
from shapely.geometry import mapping, Point, Polygon

schema = {
    'geometry': 'Polygon',
    'properties': {'name': 'str', 'area': 'float'}
}

with fiona.open(
    "output/zones.shp",
    mode="w",
    driver="ESRI Shapefile",
    crs="EPSG:4326",
    schema=schema
) as layer:
    poly1 = Polygon([(0,0), (1,0), (1,1), (0,1)])
    poly2 = Polygon([(2,2), (3,2), (3,3), (2,3)])

    layer.write({
        'geometry': mapping(poly1),
        'properties': {'name': 'Zone A', 'area': poly1.area}
    })
    layer.write({
        'geometry': mapping(poly2),
        'properties': {'name': 'Zone B', 'area': poly2.area}
    })
```

### Writing Multiple Layers to a GeoPackage

```python
with fiona.open(
    "output/mylayers.gpkg",
    layer="buildings",
    driver="GPKG",
    mode="w",
    crs="EPSG:4326",
    schema={'geometry': 'Polygon', 'properties': {'id': 'int', 'height': 'float'}}
) as layer:
    # Write feature
    pass
```

### Appending to an Existing Dataset

```python
with fiona.open("output/zones.shp", "a") as dst:
    new_poly = Polygon([(4,4), (5,4), (5,5), (4,5)])
    dst.write({
        'geometry': mapping(new_poly),
        'properties': {'name': 'Zone C', 'area': new_poly.area}
    })
```

---

## GDAL/OGR CLI Basics for Preprocessing

Before writing with Fiona, you may need to preprocess data using **GDAL/OGR command-line utilities**.

### Inspecting Vector Data

```bash
ogrinfo data/roads.shp -so -al
```

Displays summary information, schema, and field types.

### Reprojecting Vector Data

```bash
ogr2ogr -t_srs EPSG:4326 output/roads_wgs84.shp data/roads.shp
```

Converts the coordinate system to WGS84.

### Filtering Features by Attribute

```bash
ogr2ogr -where "road_type='highway'" output/highways.shp data/roads.shp
```

### Converting Between Formats

```bash
ogr2ogr -f "GeoJSON" output/roads.json data/roads.shp
ogr2ogr -f "GPKG" output/roads.gpkg data/roads.shp
```

---

## Combining Fiona with Shapely for Geometry Processing

Fiona and Shapely integrate well for editing and manipulating geometries before saving back to disk.

```python
import fiona
from shapely.geometry import shape, mapping

with fiona.open("data/roads.shp") as src:
    schema = src.schema.copy()
    with fiona.open("output/roads_buffered.shp", "w", driver=src.driver, crs=src.crs, schema=schema) as dst:
        for feature in src:
            geom = shape(feature["geometry"])
            buffered = geom.buffer(0.01)
            feature["geometry"] = mapping(buffered)
            dst.write(feature)
```

---

## Best Practices

- Always close datasets (use `with fiona.open()` context managers).
- Define schemas explicitly to avoid type mismatches.
- Validate geometries with Shapely before writing.
- Use GDAL CLI tools for heavy preprocessing (e.g., reprojection, format conversion).
- Fiona is **I/O-focused**, not analytical — use GeoPandas for analysis.

---

## Summary

Fiona gives precise control over geospatial vector file input/output, making it ideal for data engineers and GIS developers working with raw data pipelines.  
By combining Fiona, Shapely, and GDAL/OGR, you can efficiently handle everything from geometry creation to schema management and data transformation.

