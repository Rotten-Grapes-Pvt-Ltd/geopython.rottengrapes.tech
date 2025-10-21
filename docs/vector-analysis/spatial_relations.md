

## Introduction

- Spatial relationships define how spatial features relate to one another in space.
- Geometry operations allow us to transform, analyze, and manipulate spatial shapes.
- Together, these are foundational in spatial data analysis.

---

## Types of Spatial Relationships


---

## 📌 Disjoint

- **Explanation**: No common point between geometries.
- **Use Case**: Find urban areas not impacted by flooding zones.


![Disjoint](/assets/images/spatial/Disjoint.png)

---

## 📌 Touch

- **Explanation**: Geometries touch at boundaries but do not overlap.
- **Use Case**: Identify villages that touch forest boundaries.

![Touch](/assets/images/spatial/touching.png)

---

## 📌 Intersect

- **Explanation**: Any spatial overlap between features.
- **Use Case**: Roads that intersect flood zones.

![Intersect](/assets/images/spatial/intersect.png)
---

## 📌 Within / Contains

- **Explanation**:
  - A within B: A lies completely inside B.
  - B contains A: Same as above, reversed.
- **Use Case**: Lakes within administrative boundaries.
  
![Within](/assets/images/spatial/Within.png)


---

## 📌 Overlaps

- **Explanation**: Features partially overlap but neither completely contains the other.
- **Use Case**: Identify conflict zones between mining and wildlife areas.

![Overlaps](/assets/images/spatial/Overlaps.png)

---

## 📌 Crosses

- **Explanation**: Features cross each other like an 'X'.
- **Use Case**: Rivers crossing roads or railways.

![Crosses](/assets/images/spatial/Crosses.png)

---

## 📌 Equals

- **Explanation**: Geometries are identical in shape and position.
- **Use Case**: Validate duplicate parcel entries.


![Equals](/assets/images/spatial/Equals.png)



---

## Geometry Operations

---

## 📍 Buffer

- **Explanation**: Creates zones at a specified distance from features.
- **Use Case**: Identify schools within 500m of highways.


![Buffer](/assets/images/spatial/Buffer.png)

---

## 📍 Intersection

- **Explanation**: Extracts shared area between layers.
- **Use Case**: Areas where agriculture and flood zones overlap.

![Intersection](/assets/images/spatial/Intersection.png)

---

## 📍 Union

- **Explanation**: Merges geometries into one combined shape.
- **Use Case**: Combine two adjacent conservation zones.

![Union](/assets/images/spatial/Union.png)

---

## 📍 Difference

- **Explanation**: Removes the area of B from A.
- **Use Case**: Exclude roads from construction zones.

![Difference](/assets/images/spatial/Difference.png)

---

## 📍 Symmetric Difference

- **Explanation**: Returns areas unique to each geometry.
- **Use Case**: Change detection between two land-use years.

![Symmetric Difference](/assets/images/spatial/SymmetricDifference.png)

---

## 📍 Centroid

- **Explanation**: Finds the geometric center of a shape.
- **Use Case**: Label location of administrative zones.

![Centroid](/assets/images/spatial/Centroid.png)


---

## 📍 Convex Hull

- **Explanation**: Smallest convex polygon enclosing a geometry.
- **Use Case**: Estimate range of animal movement from GPS points.

![Convex Hull](/assets/images/spatial/ConvexHull.png)

---

## 📍 Envelope (Bounding Box)

- **Explanation**: Minimum bounding rectangle that encloses a feature.
- **Use Case**: Indexing features for faster processing.

![Envelope](/assets/images/spatial/Envelope.png)

---

## 📍 Simplify

- **Explanation**: Reduces the number of vertices in complex geometries.
- **Use Case**: Create simplified boundaries for web maps.


![Simplify](/assets/images/spatial/Simplify.png)

---

## 📍 Densify

- **Explanation**: Adds vertices at regular intervals along geometry.
- **Use Case**: Improve accuracy for projection transformations.

![Densify](/assets/images/spatial/Densify.png)

---

## 📍 Split

- **Explanation**: Divides geometry using a line or another polygon.
- **Use Case**: Split a district into zones using a river.

![Split](/assets/images/spatial/Split.png)

---

## 📍 Clip

- **Explanation**: Cuts a geometry using another as a mask.
- **Use Case**: Clip land-use data to a city boundary.

![Clip](/assets/images/spatial/Clip.png)

---

## 📍 Distance

- **Explanation**: Shortest path between features.
- **Use Case**: Distance from a village to nearest river.

---

## 📍 Spatial Join

- **Explanation**: Attribute data is transferred based on spatial relation.
- **Use Case**: Attach population data to districts.
