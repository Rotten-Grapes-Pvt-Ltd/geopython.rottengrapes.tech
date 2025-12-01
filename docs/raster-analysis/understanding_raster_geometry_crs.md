# Understanding Raster Geometry and CRS

## Overview
Raster data represents spatial information as a grid of pixels, each containing values. Understanding raster geometry and coordinate reference systems (CRS) is crucial for accurate spatial analysis, especially when combining raster and vector data or working with multiple raster datasets.

## Why Raster Geometry Matters
- **Pixel Alignment**: Ensures accurate overlay analysis between datasets
- **Spatial Accuracy**: Proper CRS handling prevents distortion in measurements
- **Data Integration**: Enables combining satellite imagery, DEMs, and climate data
- **Analysis Precision**: Critical for environmental modeling, urban planning, and agriculture

---

## 1️⃣ Affine Transform Basics

### What is an Affine Transform?
An affine transform defines how pixel coordinates map to real-world coordinates. It's a 6-parameter transformation that handles translation, scaling, rotation, and shearing.

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show

# Load sample raster data
with rasterio.open('landsat_sample.tif') as src:
    # Access the affine transform
    transform = src.transform
    print(f"Affine Transform: {transform}")
    print(f"Pixel size (X, Y): {transform.a}, {-transform.e}")
    print(f"Upper-left corner: {transform.c}, {transform.f}")
```

### Understanding Transform Parameters
```python
# Affine transform components
# | a  b  c |   | x |
# | d  e  f | × | y |
# | 0  0  1 |   | 1 |

# Where:
# a = pixel width (x-direction)
# b = row rotation (usually 0)
# c = x-coordinate of upper-left corner
# d = column rotation (usually 0) 
# e = pixel height (y-direction, negative)
# f = y-coordinate of upper-left corner

def pixel_to_world(col, row, transform):
    """Convert pixel coordinates to world coordinates"""
    x = transform.a * col + transform.b * row + transform.c
    y = transform.d * col + transform.e * row + transform.f
    return x, y

# Example: Convert pixel (100, 50) to world coordinates
world_x, world_y = pixel_to_world(100, 50, transform)
print(f"Pixel (100, 50) → World ({world_x:.2f}, {world_y:.2f})")
```

### Creating Custom Affine Transforms
```python
from rasterio.transform import from_bounds, from_origin

# Method 1: From bounding box
west, south, east, north = -120.5, 35.0, -119.5, 36.0
width, height = 1000, 1000
transform_bounds = from_bounds(west, south, east, north, width, height)

# Method 2: From origin and pixel size
transform_origin = from_origin(-120.5, 36.0, 0.001, 0.001)  # 0.001° pixels

print(f"From bounds: {transform_bounds}")
print(f"From origin: {transform_origin}")
```

---

## 2️⃣ Raster CRS vs Vector CRS

### Key Differences
```python
import geopandas as gpd
import rioxarray as rxr

# Load raster with CRS
raster = rxr.open_rasterio('elevation.tif')
print(f"Raster CRS: {raster.rio.crs}")
print(f"Raster shape: {raster.shape}")
print(f"Raster bounds: {raster.rio.bounds()}")

# Load vector with CRS
vector = gpd.read_file('boundaries.shp')
print(f"Vector CRS: {vector.crs}")
print(f"Vector bounds: {vector.total_bounds}")
```

### CRS Impact on Analysis
```python
# Geographic vs Projected CRS comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Geographic CRS (WGS84) - degrees
raster_geo = raster.rio.reproject('EPSG:4326')
raster_geo.plot(ax=axes[0], cmap='terrain')
axes[0].set_title('Geographic CRS (EPSG:4326)\nUnits: Degrees')

# Projected CRS (UTM) - meters
raster_proj = raster.rio.reproject('EPSG:32633')  # UTM Zone 33N
raster_proj.plot(ax=axes[1], cmap='terrain')
axes[1].set_title('Projected CRS (UTM 33N)\nUnits: Meters')

plt.tight_layout()
plt.show()

# Calculate pixel areas
geo_pixel_area = abs(raster_geo.rio.transform().a * raster_geo.rio.transform().e)
proj_pixel_area = abs(raster_proj.rio.transform().a * raster_proj.rio.transform().e)

print(f"Geographic pixel area: {geo_pixel_area:.8f} degrees²")
print(f"Projected pixel area: {proj_pixel_area:.2f} meters²")
```

---

## 3️⃣ Reprojection with rioxarray and rasterio

### Basic Reprojection with rioxarray
```python
import rioxarray as rxr

# Load and reproject raster
original = rxr.open_rasterio('satellite_image.tif')
print(f"Original CRS: {original.rio.crs}")

# Reproject to different CRS
reprojected = original.rio.reproject('EPSG:3857')  # Web Mercator
print(f"Reprojected CRS: {reprojected.rio.crs}")

# Save reprojected raster
reprojected.rio.to_raster('satellite_image_3857.tif')
```

### Advanced Reprojection with Custom Parameters
```python
# Reproject with specific resolution and resampling
target_crs = 'EPSG:32633'
target_resolution = 30  # 30 meter pixels

reprojected_custom = original.rio.reproject(
    target_crs,
    resolution=target_resolution,
    resampling=rasterio.enums.Resampling.bilinear
)

print(f"New resolution: {reprojected_custom.rio.resolution()}")
print(f"New shape: {reprojected_custom.shape}")
```

### Reprojection with rasterio.warp
```python
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_raster(src_path, dst_path, dst_crs):
    """Reproject raster using rasterio.warp"""
    with rasterio.open(src_path) as src:
        # Calculate transform and dimensions for target CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Perform reprojection
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )

# Usage
reproject_raster('input.tif', 'output_utm.tif', 'EPSG:32633')
```

---

## 4️⃣ Aligning Multiple Rasters

### Checking Raster Alignment
```python
def check_raster_alignment(raster1, raster2):
    """Check if two rasters are aligned"""
    # Check CRS
    crs_match = raster1.rio.crs == raster2.rio.crs
    
    # Check transform (pixel size and origin)
    transform_match = raster1.rio.transform() == raster2.rio.transform()
    
    # Check shape
    shape_match = raster1.shape == raster2.shape
    
    print(f"CRS match: {crs_match}")
    print(f"Transform match: {transform_match}")
    print(f"Shape match: {shape_match}")
    
    return crs_match and transform_match and shape_match

# Load multiple rasters
dem = rxr.open_rasterio('elevation.tif')
landcover = rxr.open_rasterio('landcover.tif')

aligned = check_raster_alignment(dem, landcover)
print(f"Rasters aligned: {aligned}")
```

### Aligning Rasters to a Reference
```python
def align_raster_to_reference(target_raster, reference_raster):
    """Align target raster to reference raster geometry"""
    
    # Reproject and resample to match reference
    aligned = target_raster.rio.reproject_match(
        reference_raster,
        resampling=rasterio.enums.Resampling.bilinear
    )
    
    return aligned

# Align landcover to DEM
landcover_aligned = align_raster_to_reference(landcover, dem)

# Verify alignment
print("After alignment:")
check_raster_alignment(dem, landcover_aligned)
```

### Creating a Common Grid
```python
def create_common_grid(*rasters, target_crs='EPSG:4326', resolution=0.001):
    """Create a common grid for multiple rasters"""
    
    # Calculate combined bounds
    all_bounds = []
    for raster in rasters:
        # Reproject bounds to target CRS if needed
        if raster.rio.crs != target_crs:
            temp = raster.rio.reproject(target_crs)
            bounds = temp.rio.bounds()
        else:
            bounds = raster.rio.bounds()
        all_bounds.append(bounds)
    
    # Find overall extent
    min_x = min(bounds[0] for bounds in all_bounds)
    min_y = min(bounds[1] for bounds in all_bounds)
    max_x = max(bounds[2] for bounds in all_bounds)
    max_y = max(bounds[3] for bounds in all_bounds)
    
    # Create target transform
    from rasterio.transform import from_bounds
    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)
    
    target_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Align all rasters to common grid
    aligned_rasters = []
    for raster in rasters:
        aligned = raster.rio.reproject(
            target_crs,
            transform=target_transform,
            shape=(height, width),
            resampling=rasterio.enums.Resampling.bilinear
        )
        aligned_rasters.append(aligned)
    
    return aligned_rasters

# Usage
dem_aligned, landcover_aligned, temperature_aligned = create_common_grid(
    dem, landcover, temperature, 
    target_crs='EPSG:4326', 
    resolution=0.001
)
```

---

## 5️⃣ Practical Examples & Data Sources

### Example 1: Multi-temporal Analysis
```python
# Align time series of satellite images
import glob

# Load all Landsat images from a directory
image_files = glob.glob('landsat_*.tif')
images = [rxr.open_rasterio(f) for f in image_files]

# Align all to first image
reference = images[0]
aligned_images = []

for img in images[1:]:
    aligned = img.rio.reproject_match(reference)
    aligned_images.append(aligned)

# Stack for time series analysis
import xarray as xr
time_series = xr.concat([reference] + aligned_images, dim='time')
```

### Example 2: Raster-Vector Integration
```python
# Align raster to vector extent
import geopandas as gpd

# Load vector boundary
boundary = gpd.read_file('study_area.shp')

# Get boundary in raster CRS
boundary_proj = boundary.to_crs(dem.rio.crs)

# Clip and align raster to boundary
dem_clipped = dem.rio.clip(boundary_proj.geometry, boundary_proj.crs)

# Ensure consistent pixel alignment
dem_aligned = dem_clipped.rio.reproject(
    dem_clipped.rio.crs,
    resolution=30,  # 30m pixels
    resampling=rasterio.enums.Resampling.bilinear
)
```

---

## 📊 Visualization and Validation

### Visualizing Alignment Issues
```python
def visualize_alignment(raster1, raster2, title1="Raster 1", title2="Raster 2"):
    """Visualize two rasters to check alignment"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot individual rasters
    raster1.plot(ax=axes[0,0], cmap='viridis')
    axes[0,0].set_title(title1)
    
    raster2.plot(ax=axes[0,1], cmap='plasma')
    axes[0,1].set_title(title2)
    
    # Plot overlay (if aligned)
    try:
        if raster1.shape == raster2.shape:
            # Create RGB composite
            overlay = np.stack([
                raster1.values[0] / raster1.values[0].max(),
                raster2.values[0] / raster2.values[0].max(),
                np.zeros_like(raster1.values[0])
            ], axis=-1)
            
            axes[1,0].imshow(overlay)
            axes[1,0].set_title('Overlay (Red: R1, Green: R2)')
        else:
            axes[1,0].text(0.5, 0.5, 'Shapes do not match', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
    except:
        axes[1,0].text(0.5, 0.5, 'Cannot overlay', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
    
    # Plot difference (if aligned)
    try:
        if raster1.shape == raster2.shape:
            diff = raster1 - raster2
            diff.plot(ax=axes[1,1], cmap='RdBu_r')
            axes[1,1].set_title('Difference (R1 - R2)')
    except:
        axes[1,1].text(0.5, 0.5, 'Cannot compute difference', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.show()

# Usage
visualize_alignment(dem, landcover_aligned, "DEM", "Land Cover")
```

---

## 🔗 Data Sources

### Free Raster Data Sources
- **Landsat**: [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- **Sentinel**: [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **SRTM DEM**: [NASA Earthdata](https://earthdata.nasa.gov/)
- **MODIS**: [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/)
- **Climate Data**: [WorldClim](https://worldclim.org/)

### Sample Data Download
```python
# Download sample data using Python
import requests
import zipfile

def download_sample_data():
    """Download sample raster data for practice"""
    
    # Example: Download SRTM tile
    url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm_mosaic_30m.zip"
    
    response = requests.get(url)
    with open('srtm_sample.zip', 'wb') as f:
        f.write(response.content)
    
    # Extract
    with zipfile.ZipFile('srtm_sample.zip', 'r') as zip_ref:
        zip_ref.extractall('sample_data/')
    
    print("Sample data downloaded to 'sample_data/' directory")

# Uncomment to download
# download_sample_data()
```

---

## ⚠️ Common Pitfalls & Solutions

### Pitfall 1: Mixed CRS Operations
```python
# ❌ Wrong: Operating on rasters with different CRS
# result = raster1 + raster2  # May give incorrect results

# ✅ Correct: Align CRS first
raster2_aligned = raster2.rio.reproject_match(raster1)
result = raster1 + raster2_aligned
```

### Pitfall 2: Ignoring Pixel Alignment
```python
# ❌ Wrong: Assuming same extent means alignment
# if raster1.rio.bounds() == raster2.rio.bounds():
#     result = raster1 + raster2

# ✅ Correct: Check complete alignment
def rasters_aligned(r1, r2):
    return (r1.rio.crs == r2.rio.crs and 
            r1.rio.transform() == r2.rio.transform() and
            r1.shape == r2.shape)

if rasters_aligned(raster1, raster2):
    result = raster1 + raster2
else:
    raster2_aligned = raster2.rio.reproject_match(raster1)
    result = raster1 + raster2_aligned
```

### Pitfall 3: Inappropriate Resampling
```python
# Choose appropriate resampling method
resampling_methods = {
    'categorical': rasterio.enums.Resampling.nearest,      # Land cover, classes
    'continuous': rasterio.enums.Resampling.bilinear,     # Temperature, elevation
    'aggregation': rasterio.enums.Resampling.average      # Downsampling
}

# Example: Reproject land cover (categorical)
landcover_reprojected = landcover.rio.reproject(
    'EPSG:32633',
    resampling=rasterio.enums.Resampling.nearest  # Preserve class values
)
```

---

## 🎯 Key Takeaways

1. **Affine Transform**: Defines pixel-to-world coordinate mapping
2. **CRS Consistency**: Always check and align CRS before analysis
3. **Pixel Alignment**: Ensure identical transforms for accurate overlay
4. **Resampling Choice**: Use appropriate method for data type
5. **Validation**: Always verify alignment before proceeding with analysis

This foundation enables accurate multi-raster analysis, raster-vector integration, and reliable spatial modeling workflows.