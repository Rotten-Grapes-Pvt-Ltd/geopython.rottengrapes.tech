# Clipping and Masking Rasters

## Overview
Clipping and masking are fundamental operations in raster analysis that allow you to extract specific regions of interest from larger datasets. Whether you need to focus on a study area, remove unwanted data, or combine raster and vector information, these techniques are essential for efficient spatial analysis and data management.

## Why Clip and Mask Rasters?
- **Focus Analysis**: Extract specific study areas from large datasets
- **Data Management**: Reduce file sizes and processing time
- **Quality Control**: Remove invalid or unwanted data regions
- **Integration**: Combine raster data with vector boundaries
- **Visualization**: Create clean, focused maps and visualizations

---

## 1️⃣ Masking with Shapefiles (Vector Mask)

### Basic Vector Masking
Vector masking uses polygon boundaries to extract raster data within specific geographic areas.

```python
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box

# Create sample raster data
def create_sample_raster():
    """Create sample raster for demonstration"""
    height, width = 200, 200
    x = np.linspace(-10, 10, width)
    y = np.linspace(-10, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Create elevation-like data
    data = 1000 + 500 * np.exp(-((X-2)**2 + (Y+1)**2) / 8) + \
           200 * np.sin(X) * np.cos(Y) + \
           100 * np.random.random((height, width))
    
    return data, x, y

# Create sample vector boundary
def create_sample_boundary():
    """Create sample polygon boundary"""
    # Create irregular polygon
    coords = [(-5, -5), (5, -3), (7, 4), (2, 8), (-6, 6), (-8, 2)]
    polygon = Polygon(coords)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs='EPSG:4326')
    return gdf

# Demonstrate basic masking
raster_data, x_coords, y_coords = create_sample_raster()
boundary_gdf = create_sample_boundary()

# Convert to xarray with proper coordinates
import xarray as xr
raster_xr = xr.DataArray(
    raster_data,
    coords={'y': y_coords, 'x': x_coords},
    dims=['y', 'x']
)
raster_xr.rio.write_crs('EPSG:4326', inplace=True)

# Mask raster with vector boundary
masked_raster = raster_xr.rio.clip(boundary_gdf.geometry, boundary_gdf.crs)

# Visualize masking result
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original raster
raster_xr.plot(ax=axes[0], cmap='terrain')
boundary_gdf.boundary.plot(ax=axes[0], color='red', linewidth=2)
axes[0].set_title('Original Raster + Boundary')

# Boundary only
boundary_gdf.plot(ax=axes[1], facecolor='lightblue', edgecolor='red', alpha=0.7)
axes[1].set_title('Vector Boundary')
axes[1].set_xlim(-10, 10)
axes[1].set_ylim(-10, 10)

# Masked result
masked_raster.plot(ax=axes[2], cmap='terrain')
axes[2].set_title('Masked Raster')

plt.tight_layout()
plt.show()

print(f"Original raster shape: {raster_xr.shape}")
print(f"Masked raster shape: {masked_raster.shape}")
print(f"Original data range: {raster_xr.min().values:.1f} - {raster_xr.max().values:.1f}")
print(f"Masked data range: {masked_raster.min().values:.1f} - {masked_raster.max().values:.1f}")
```

### Advanced Vector Masking Techniques
```python
def advanced_vector_masking():
    """Demonstrate advanced vector masking techniques"""
    
    # Load real-world example (simulated)
    raster = rxr.open_rasterio('satellite_image.tif') if False else raster_xr
    
    # Multiple polygon masking
    polygons = [
        Polygon([(-3, -3), (0, -3), (0, 0), (-3, 0)]),  # Southwest quadrant
        Polygon([(2, 2), (6, 2), (6, 6), (2, 6)]),      # Northeast area
        Polygon([(-6, 4), (-2, 4), (-2, 8), (-6, 8)])   # Northwest area
    ]
    
    multi_gdf = gpd.GeoDataFrame(
        {'id': [1, 2, 3], 'name': ['Area_A', 'Area_B', 'Area_C']},
        geometry=polygons,
        crs='EPSG:4326'
    )
    
    # Mask with multiple polygons
    multi_masked = raster.rio.clip(multi_gdf.geometry, multi_gdf.crs)
    
    # Mask with buffer around polygons
    buffered_gdf = multi_gdf.copy()
    buffered_gdf.geometry = buffered_gdf.geometry.buffer(1.0)
    buffered_masked = raster.rio.clip(buffered_gdf.geometry, buffered_gdf.crs)
    
    # Inverse masking (mask out the polygons)
    # Create bounding box of entire raster
    bounds = raster.rio.bounds()
    bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
    
    # Subtract polygons from bounding box
    from shapely.ops import unary_union
    union_polygons = unary_union(multi_gdf.geometry)
    inverse_geom = bbox.difference(union_polygons)
    
    inverse_gdf = gpd.GeoDataFrame([1], geometry=[inverse_geom], crs='EPSG:4326')
    inverse_masked = raster.rio.clip(inverse_gdf.geometry, inverse_gdf.crs)
    
    # Visualize different masking approaches
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original with polygons
    raster.plot(ax=axes[0,0], cmap='terrain')
    multi_gdf.boundary.plot(ax=axes[0,0], color='red', linewidth=2)
    axes[0,0].set_title('Original + Multiple Polygons')
    
    # Multi-polygon mask
    multi_masked.plot(ax=axes[0,1], cmap='terrain')
    axes[0,1].set_title('Multi-Polygon Mask')
    
    # Buffered mask
    buffered_masked.plot(ax=axes[1,0], cmap='terrain')
    buffered_gdf.boundary.plot(ax=axes[1,0], color='blue', linewidth=1)
    axes[1,0].set_title('Buffered Polygon Mask')
    
    # Inverse mask
    inverse_masked.plot(ax=axes[1,1], cmap='terrain')
    axes[1,1].set_title('Inverse Mask (Exclude Polygons)')
    
    plt.tight_layout()
    plt.show()
    
    return multi_masked, buffered_masked, inverse_masked

# Demonstrate advanced masking
multi_result, buffered_result, inverse_result = advanced_vector_masking()
```

### Masking with Attribute-Based Selection
```python
def attribute_based_masking():
    """Mask raster based on vector attributes"""
    
    # Create administrative boundaries with attributes
    admin_polygons = [
        Polygon([(-8, -8), (-2, -8), (-2, -2), (-8, -2)]),  # Urban
        Polygon([(2, -6), (8, -6), (8, 0), (2, 0)]),        # Agricultural
        Polygon([(-6, 2), (0, 2), (0, 8), (-6, 8)]),        # Forest
        Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])           # Protected
    ]
    
    admin_gdf = gpd.GeoDataFrame({
        'zone_id': [1, 2, 3, 4],
        'zone_type': ['Urban', 'Agricultural', 'Forest', 'Protected'],
        'priority': ['High', 'Medium', 'High', 'Critical'],
        'area_km2': [36, 36, 36, 36]
    }, geometry=admin_polygons, crs='EPSG:4326')
    
    # Mask by zone type
    forest_zones = admin_gdf[admin_gdf['zone_type'] == 'Forest']
    forest_masked = raster_xr.rio.clip(forest_zones.geometry, forest_zones.crs)
    
    # Mask by priority level
    high_priority = admin_gdf[admin_gdf['priority'].isin(['High', 'Critical'])]
    priority_masked = raster_xr.rio.clip(high_priority.geometry, high_priority.crs)
    
    # Visualize attribute-based masking
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # All zones
    raster_xr.plot(ax=axes[0,0], cmap='terrain', alpha=0.7)
    admin_gdf.plot(ax=axes[0,0], facecolor='none', edgecolor='red', linewidth=2)
    for idx, row in admin_gdf.iterrows():
        centroid = row.geometry.centroid
        axes[0,0].text(centroid.x, centroid.y, row['zone_type'], 
                      ha='center', va='center', fontweight='bold')
    axes[0,0].set_title('All Administrative Zones')
    
    # Zone types legend
    admin_gdf.plot(ax=axes[0,1], column='zone_type', categorical=True, 
                   legend=True, cmap='Set3', alpha=0.7)
    axes[0,1].set_title('Zones by Type')
    
    # Forest zones only
    forest_masked.plot(ax=axes[1,0], cmap='terrain')
    axes[1,0].set_title('Forest Zones Only')
    
    # High priority zones
    priority_masked.plot(ax=axes[1,1], cmap='terrain')
    axes[1,1].set_title('High Priority Zones')
    
    plt.tight_layout()
    plt.show()
    
    # Statistics by zone
    print("ZONE STATISTICS")
    print("=" * 20)
    for idx, row in admin_gdf.iterrows():
        zone_masked = raster_xr.rio.clip([row.geometry], admin_gdf.crs)
        if zone_masked.size > 0:
            print(f"{row['zone_type']} ({row['priority']} priority):")
            print(f"  Mean elevation: {zone_masked.mean().values:.1f}")
            print(f"  Min elevation: {zone_masked.min().values:.1f}")
            print(f"  Max elevation: {zone_masked.max().values:.1f}")
            print(f"  Valid pixels: {(~np.isnan(zone_masked)).sum().values}")
    
    return admin_gdf, forest_masked, priority_masked

# Demonstrate attribute-based masking
admin_data, forest_result, priority_result = attribute_based_masking()
```

---

## 2️⃣ Cropping with Bounding Box

### Basic Bounding Box Cropping
```python
def bounding_box_cropping():
    """Demonstrate bounding box cropping techniques"""
    
    # Define bounding boxes
    bbox_coords = {
        'Small Area': (-3, -3, 3, 3),      # (minx, miny, maxx, maxy)
        'Western Half': (-10, -10, 0, 10),
        'Central Strip': (-2, -10, 2, 10),
        'Northeast': (0, 0, 10, 10)
    }
    
    cropped_results = {}
    
    # Crop raster with different bounding boxes
    for name, (minx, miny, maxx, maxy) in bbox_coords.items():
        # Method 1: Using rio.clip_box
        cropped = raster_xr.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        cropped_results[name] = cropped
    
    # Visualize cropping results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original raster
    raster_xr.plot(ax=axes[0], cmap='terrain')
    axes[0].set_title('Original Raster')
    
    # Draw all bounding boxes on original
    for name, (minx, miny, maxx, maxy) in bbox_coords.items():
        rect = plt.Rectangle((minx, miny), maxx-minx, maxy-miny, 
                           fill=False, edgecolor='red', linewidth=2, alpha=0.7)
        axes[0].add_patch(rect)
        axes[0].text((minx+maxx)/2, (miny+maxy)/2, name, 
                    ha='center', va='center', fontweight='bold', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Show cropped results
    for i, (name, cropped) in enumerate(cropped_results.items(), 1):
        cropped.plot(ax=axes[i], cmap='terrain')
        axes[i].set_title(f'{name}\nShape: {cropped.shape}')
    
    # Hide unused subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print cropping statistics
    print("BOUNDING BOX CROPPING RESULTS")
    print("=" * 35)
    print(f"{'Region':<15} {'Original':<10} {'Cropped':<10} {'Reduction':<10}")
    print("-" * 50)
    
    original_size = raster_xr.size
    for name, cropped in cropped_results.items():
        cropped_size = cropped.size
        reduction = (1 - cropped_size/original_size) * 100
        print(f"{name:<15} {original_size:<10} {cropped_size:<10} {reduction:<9.1f}%")
    
    return cropped_results

# Demonstrate bounding box cropping
bbox_results = bounding_box_cropping()
```

### Interactive Bounding Box Selection
```python
def interactive_bbox_selection():
    """Demonstrate programmatic bounding box selection"""
    
    # Method 1: Based on data values
    def crop_by_elevation_threshold(raster, min_elevation=1200):
        """Crop to areas above elevation threshold"""
        # Find pixels above threshold
        high_elevation = raster > min_elevation
        
        # Get bounding box of high elevation areas
        y_indices, x_indices = np.where(high_elevation.values)
        
        if len(y_indices) > 0:
            min_y_idx, max_y_idx = y_indices.min(), y_indices.max()
            min_x_idx, max_x_idx = x_indices.min(), x_indices.max()
            
            # Convert indices to coordinates
            y_coords = raster.y.values
            x_coords = raster.x.values
            
            minx = x_coords[min_x_idx]
            maxx = x_coords[max_x_idx]
            miny = y_coords[max_y_idx]  # Note: y is flipped
            maxy = y_coords[min_y_idx]
            
            # Crop to bounding box
            cropped = raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            return cropped, (minx, miny, maxx, maxy)
        else:
            return raster, None
    
    # Method 2: Based on statistical outliers
    def crop_to_interesting_areas(raster, std_threshold=1.5):
        """Crop to areas with high variability"""
        # Calculate local standard deviation
        from scipy import ndimage
        local_std = ndimage.generic_filter(raster.values, np.std, size=5)
        
        # Find areas with high variability
        interesting = local_std > (local_std.mean() + std_threshold * local_std.std())
        
        # Get bounding box
        y_indices, x_indices = np.where(interesting)
        
        if len(y_indices) > 0:
            min_y_idx, max_y_idx = y_indices.min(), y_indices.max()
            min_x_idx, max_x_idx = x_indices.min(), x_indices.max()
            
            y_coords = raster.y.values
            x_coords = raster.x.values
            
            minx = x_coords[min_x_idx]
            maxx = x_coords[max_x_idx]
            miny = y_coords[max_y_idx]
            maxy = y_coords[min_y_idx]
            
            cropped = raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
            return cropped, interesting
        else:
            return raster, interesting
    
    # Apply different cropping methods
    elevation_cropped, elev_bbox = crop_by_elevation_threshold(raster_xr, 1200)
    variability_cropped, variability_mask = crop_to_interesting_areas(raster_xr, 1.0)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    raster_xr.plot(ax=axes[0,0], cmap='terrain')
    axes[0,0].set_title('Original Raster')
    
    # High elevation areas
    high_elev_mask = raster_xr > 1200
    raster_xr.where(high_elev_mask).plot(ax=axes[0,1], cmap='terrain')
    axes[0,1].set_title('High Elevation Areas (>1200)')
    
    # Cropped to high elevation
    elevation_cropped.plot(ax=axes[0,2], cmap='terrain')
    axes[0,2].set_title('Cropped to High Elevation')
    
    # Variability mask
    axes[1,0].imshow(variability_mask, cmap='Reds', alpha=0.7, 
                     extent=[raster_xr.x.min(), raster_xr.x.max(), 
                            raster_xr.y.min(), raster_xr.y.max()])
    axes[1,0].set_title('High Variability Areas')
    
    # Original with variability overlay
    raster_xr.plot(ax=axes[1,1], cmap='terrain', alpha=0.7)
    axes[1,1].imshow(variability_mask, cmap='Reds', alpha=0.5,
                     extent=[raster_xr.x.min(), raster_xr.x.max(), 
                            raster_xr.y.min(), raster_xr.y.max()])
    axes[1,1].set_title('Original + Variability Overlay')
    
    # Cropped to high variability
    variability_cropped.plot(ax=axes[1,2], cmap='terrain')
    axes[1,2].set_title('Cropped to High Variability')
    
    plt.tight_layout()
    plt.show()
    
    return elevation_cropped, variability_cropped

# Demonstrate interactive selection
elev_crop, var_crop = interactive_bbox_selection()
```

---

## 3️⃣ Difference Between mask() and window()

### Understanding mask() vs window()
```python
def compare_mask_vs_window():
    """Compare masking and windowing approaches"""
    
    # Create test polygon
    test_polygon = Polygon([(-4, -4), (4, -4), (4, 4), (-4, 4)])
    test_gdf = gpd.GeoDataFrame([1], geometry=[test_polygon], crs='EPSG:4326')
    
    # Method 1: mask() - Sets values outside polygon to NoData
    masked_result = raster_xr.rio.clip(test_gdf.geometry, test_gdf.crs, drop=False)
    
    # Method 2: window() - Crops to bounding box, preserves all values
    bounds = test_gdf.total_bounds  # minx, miny, maxx, maxy
    windowed_result = raster_xr.rio.clip_box(
        minx=bounds[0], miny=bounds[1], 
        maxx=bounds[2], maxy=bounds[3]
    )
    
    # Method 3: Combined approach - Window then mask
    combined_result = windowed_result.rio.clip(test_gdf.geometry, test_gdf.crs)
    
    # Demonstrate with rasterio for low-level control
    import rasterio
    from rasterio.mask import mask
    from rasterio.windows import from_bounds
    
    # Simulate rasterio dataset (normally you'd open a real file)
    print("COMPARISON: mask() vs window()")
    print("=" * 35)
    
    print("mask() characteristics:")
    print("  - Preserves original extent")
    print("  - Sets outside values to NoData")
    print("  - Maintains spatial reference")
    print("  - Memory usage: Same as original")
    
    print("\nwindow() characteristics:")
    print("  - Reduces extent to bounding box")
    print("  - Preserves all pixel values")
    print("  - Updates spatial reference")
    print("  - Memory usage: Reduced")
    
    print("\nCombined approach:")
    print("  - Reduces extent AND sets NoData")
    print("  - Most memory efficient")
    print("  - Best for analysis workflows")
    
    # Visualize differences
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original with polygon
    raster_xr.plot(ax=axes[0,0], cmap='terrain')
    test_gdf.boundary.plot(ax=axes[0,0], color='red', linewidth=3)
    axes[0,0].set_title('Original + Polygon Boundary')
    
    # Masked (clip with drop=False)
    masked_result.plot(ax=axes[0,1], cmap='terrain')
    axes[0,1].set_title(f'Masked (clip)\nShape: {masked_result.shape}')
    
    # Windowed (bounding box crop)
    windowed_result.plot(ax=axes[1,0], cmap='terrain')
    test_gdf.boundary.plot(ax=axes[1,0], color='red', linewidth=2)
    axes[1,0].set_title(f'Windowed (clip_box)\nShape: {windowed_result.shape}')
    
    # Combined (window + mask)
    combined_result.plot(ax=axes[1,1], cmap='terrain')
    axes[1,1].set_title(f'Combined (window + mask)\nShape: {combined_result.shape}')
    
    plt.tight_layout()
    plt.show()
    
    # Performance comparison
    print("\nPERFORMANCE COMPARISON")
    print("=" * 25)
    
    original_pixels = raster_xr.size
    masked_pixels = (~np.isnan(masked_result)).sum().values
    windowed_pixels = windowed_result.size
    combined_pixels = (~np.isnan(combined_result)).sum().values
    
    print(f"Original pixels: {original_pixels:,}")
    print(f"Masked valid pixels: {masked_pixels:,} ({masked_pixels/original_pixels*100:.1f}%)")
    print(f"Windowed pixels: {windowed_pixels:,} ({windowed_pixels/original_pixels*100:.1f}%)")
    print(f"Combined valid pixels: {combined_pixels:,} ({combined_pixels/original_pixels*100:.1f}%)")
    
    return masked_result, windowed_result, combined_result

# Compare masking approaches
mask_result, window_result, combined_result = compare_mask_vs_window()
```

### Performance Optimization Strategies
```python
def optimization_strategies():
    """Demonstrate performance optimization for clipping operations"""
    
    import time
    
    # Create larger test dataset
    large_height, large_width = 1000, 1000
    x_large = np.linspace(-50, 50, large_width)
    y_large = np.linspace(-50, 50, large_height)
    X_large, Y_large = np.meshgrid(x_large, y_large)
    
    large_data = (1000 + 500 * np.exp(-((X_large-10)**2 + (Y_large+5)**2) / 100) + 
                  200 * np.sin(X_large/5) * np.cos(Y_large/8) + 
                  100 * np.random.random((large_height, large_width)))
    
    large_raster = xr.DataArray(
        large_data,
        coords={'y': y_large, 'x': x_large},
        dims=['y', 'x']
    )
    large_raster.rio.write_crs('EPSG:4326', inplace=True)
    
    # Test polygon
    test_poly = Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)])
    test_gdf = gpd.GeoDataFrame([1], geometry=[test_poly], crs='EPSG:4326')
    
    # Strategy 1: Direct clipping (baseline)
    start_time = time.time()
    direct_clip = large_raster.rio.clip(test_gdf.geometry, test_gdf.crs)
    direct_time = time.time() - start_time
    
    # Strategy 2: Window first, then clip
    start_time = time.time()
    bounds = test_gdf.total_bounds
    windowed = large_raster.rio.clip_box(
        minx=bounds[0], miny=bounds[1], 
        maxx=bounds[2], maxy=bounds[3]
    )
    windowed_clip = windowed.rio.clip(test_gdf.geometry, test_gdf.crs)
    optimized_time = time.time() - start_time
    
    # Strategy 3: Chunked processing for very large datasets
    start_time = time.time()
    # Simulate chunked processing
    chunk_size = 500
    chunked_results = []
    
    for i in range(0, large_height, chunk_size):
        for j in range(0, large_width, chunk_size):
            # Get chunk bounds
            y_start = max(0, i)
            y_end = min(large_height, i + chunk_size)
            x_start = max(0, j)
            x_end = min(large_width, j + chunk_size)
            
            # Extract chunk
            chunk = large_raster.isel(y=slice(y_start, y_end), x=slice(x_start, x_end))
            
            # Check if chunk intersects with polygon
            chunk_bounds = chunk.rio.bounds()
            chunk_box = box(chunk_bounds[0], chunk_bounds[1], 
                           chunk_bounds[2], chunk_bounds[3])
            
            if chunk_box.intersects(test_poly):
                try:
                    chunk_clipped = chunk.rio.clip(test_gdf.geometry, test_gdf.crs)
                    if chunk_clipped.size > 0:
                        chunked_results.append(chunk_clipped)
                except:
                    pass  # Skip chunks that don't intersect
    
    chunked_time = time.time() - start_time
    
    # Performance results
    print("PERFORMANCE OPTIMIZATION RESULTS")
    print("=" * 35)
    print(f"Dataset size: {large_height} x {large_width} = {large_height*large_width:,} pixels")
    print(f"Memory usage: ~{(large_height*large_width*8)/(1024**2):.1f} MB (float64)")
    print()
    print(f"Strategy 1 - Direct clipping: {direct_time:.3f} seconds")
    print(f"Strategy 2 - Window + clip: {optimized_time:.3f} seconds")
    print(f"Strategy 3 - Chunked processing: {chunked_time:.3f} seconds")
    print()
    print(f"Speedup (Strategy 2): {direct_time/optimized_time:.1f}x faster")
    
    # Memory usage comparison
    print("\nMEMORY USAGE COMPARISON")
    print("=" * 25)
    print(f"Original: {large_raster.nbytes/(1024**2):.1f} MB")
    print(f"Direct clip: {direct_clip.nbytes/(1024**2):.1f} MB")
    print(f"Optimized: {windowed_clip.nbytes/(1024**2):.1f} MB")
    
    return direct_clip, windowed_clip

# Demonstrate optimization strategies
direct_result, optimized_result = optimization_strategies()
```

---

## 4️⃣ Exporting Cropped/Masked Rasters

### Basic Export Operations
```python
def export_raster_results():
    """Demonstrate various export options for processed rasters"""
    
    # Use previously created results
    sample_raster = raster_xr
    
    # Create different processed versions
    # 1. Simple crop
    cropped = sample_raster.rio.clip_box(minx=-5, miny=-5, maxx=5, maxy=5)
    
    # 2. Masked version
    circle_coords = [(5*np.cos(t), 5*np.sin(t)) for t in np.linspace(0, 2*np.pi, 50)]
    circle_poly = Polygon(circle_coords)
    circle_gdf = gpd.GeoDataFrame([1], geometry=[circle_poly], crs='EPSG:4326')
    masked = sample_raster.rio.clip(circle_gdf.geometry, circle_gdf.crs)
    
    # Export options
    export_formats = {
        'GeoTIFF (uncompressed)': {
            'filename': 'cropped_uncompressed.tif',
            'options': {}
        },
        'GeoTIFF (LZW compressed)': {
            'filename': 'cropped_lzw.tif',
            'options': {'compress': 'lzw'}
        },
        'GeoTIFF (JPEG compressed)': {
            'filename': 'cropped_jpeg.tif',
            'options': {'compress': 'jpeg', 'jpeg_quality': 85}
        },
        'Cloud Optimized GeoTIFF': {
            'filename': 'cropped_cog.tif',
            'options': {
                'compress': 'deflate',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'BIGTIFF': 'IF_SAFER'
            }
        }
    }
    
    print("EXPORT FORMAT COMPARISON")
    print("=" * 30)
    
    # Export with different formats and measure file sizes
    for format_name, config in export_formats.items():
        try:
            # Export cropped raster
            cropped.rio.to_raster(
                f"output/{config['filename']}", 
                **config['options']
            )
            
            # Simulate file size calculation
            base_size = cropped.nbytes
            if 'compress' in config['options']:
                if config['options']['compress'] == 'lzw':
                    estimated_size = base_size * 0.6  # ~40% compression
                elif config['options']['compress'] == 'jpeg':
                    estimated_size = base_size * 0.3  # ~70% compression
                elif config['options']['compress'] == 'deflate':
                    estimated_size = base_size * 0.5  # ~50% compression
                else:
                    estimated_size = base_size
            else:
                estimated_size = base_size
            
            print(f"{format_name}:")
            print(f"  Estimated size: {estimated_size/(1024**2):.2f} MB")
            print(f"  Compression ratio: {base_size/estimated_size:.1f}:1")
            
        except Exception as e:
            print(f"{format_name}: Export failed - {e}")
    
    return cropped, masked

# Demonstrate export operations
cropped_export, masked_export = export_raster_results()
```

### Advanced Export with Metadata
```python
def export_with_metadata():
    """Export rasters with comprehensive metadata"""
    
    # Prepare raster with metadata
    processed_raster = raster_xr.copy()
    
    # Add attributes
    processed_raster.attrs = {
        'title': 'Processed Elevation Data',
        'description': 'Clipped and masked elevation raster for study area',
        'source': 'Synthetic data for demonstration',
        'processing_date': '2024-01-01',
        'processing_software': 'Python rioxarray',
        'units': 'meters',
        'vertical_datum': 'WGS84 ellipsoid',
        'nodata_value': -9999
    }
    
    # Set NoData value
    processed_raster = processed_raster.rio.write_nodata(-9999)
    
    # Export with different metadata options
    def export_with_tags(raster, filename, custom_tags=None):
        """Export raster with custom tags"""
        
        # Prepare tags
        tags = {
            'AREA_OR_POINT': 'Area',
            'PROCESSING_LEVEL': 'L2',
            'CREATION_DATE': '2024-01-01T00:00:00Z',
            'SOFTWARE': 'Python rioxarray'
        }
        
        if custom_tags:
            tags.update(custom_tags)
        
        # Export (in real scenario, you'd use rasterio for more control)
        raster.rio.to_raster(filename, tags=tags)
        
        return filename
    
    # Export with different metadata configurations
    metadata_configs = {
        'Basic': {},
        'Scientific': {
            'CITATION': 'Doe, J. (2024). Elevation Analysis Study',
            'METHODOLOGY': 'Digital elevation model processing',
            'ACCURACY': '+/- 5 meters vertical'
        },
        'Processing': {
            'PROCESSING_STEPS': 'Clip,Mask,Reproject',
            'INPUT_FILES': 'original_dem.tif,study_area.shp',
            'PARAMETERS': 'buffer=100m,resampling=bilinear'
        }
    }
    
    exported_files = {}
    
    for config_name, tags in metadata_configs.items():
        filename = f"output/elevation_{config_name.lower()}.tif"
        try:
            export_with_tags(processed_raster, filename, tags)
            exported_files[config_name] = filename
            print(f"Exported: {filename}")
        except Exception as e:
            print(f"Failed to export {config_name}: {e}")
    
    return exported_files

# Demonstrate metadata export
metadata_exports = export_with_metadata()
```

### Batch Export Operations
```python
def batch_export_operations():
    """Demonstrate batch processing and export"""
    
    # Create multiple study areas
    study_areas = {
        'North': Polygon([(-8, 2), (0, 2), (0, 10), (-8, 10)]),
        'South': Polygon([(-8, -10), (0, -10), (0, -2), (-8, -2)]),
        'East': Polygon([(2, -8), (10, -8), (10, 8), (2, 8)]),
        'West': Polygon([(-10, -8), (-2, -8), (-2, 8), (-10, 8)]),
        'Center': Polygon([(-3, -3), (3, -3), (3, 3), (-3, 3)])
    }
    
    # Process and export each area
    export_summary = []
    
    for area_name, polygon in study_areas.items():
        try:
            # Create GeoDataFrame
            area_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs='EPSG:4326')
            
            # Process raster
            clipped = raster_xr.rio.clip(area_gdf.geometry, area_gdf.crs)
            
            # Calculate statistics
            stats = {
                'area': area_name,
                'pixels': clipped.size,
                'valid_pixels': (~np.isnan(clipped)).sum().values,
                'min_value': clipped.min().values,
                'max_value': clipped.max().values,
                'mean_value': clipped.mean().values,
                'std_value': clipped.std().values
            }
            
            # Export
            output_filename = f"output/study_area_{area_name.lower()}.tif"
            clipped.rio.to_raster(output_filename)
            stats['filename'] = output_filename
            
            export_summary.append(stats)
            
        except Exception as e:
            print(f"Failed to process {area_name}: {e}")
    
    # Create summary report
    print("BATCH EXPORT SUMMARY")
    print("=" * 25)
    print(f"{'Area':<8} {'Pixels':<8} {'Valid':<8} {'Min':<8} {'Max':<8} {'Mean':<8}")
    print("-" * 55)
    
    for stats in export_summary:
        print(f"{stats['area']:<8} {stats['pixels']:<8} {stats['valid_pixels']:<8} "
              f"{stats['min_value']:<8.1f} {stats['max_value']:<8.1f} {stats['mean_value']:<8.1f}")
    
    # Export summary as CSV
    import pandas as pd
    summary_df = pd.DataFrame(export_summary)
    summary_df.to_csv('output/export_summary.csv', index=False)
    
    print(f"\nExported {len(export_summary)} study areas")
    print("Summary saved to: output/export_summary.csv")
    
    return export_summary

# Demonstrate batch operations
batch_summary = batch_export_operations()
```

---

## 🔧 Best Practices and Tips

### Memory Management
```python
def memory_management_tips():
    """Best practices for memory-efficient clipping"""
    
    tips = {
        'Use Chunking': {
            'description': 'Process large rasters in chunks',
            'code': '''
# For very large rasters
large_raster = rxr.open_rasterio('huge_file.tif', chunks={'x': 1024, 'y': 1024})
clipped = large_raster.rio.clip(polygons.geometry, polygons.crs)
'''
        },
        'Window First': {
            'description': 'Crop to bounding box before masking',
            'code': '''
# More efficient for complex polygons
bounds = polygons.total_bounds
windowed = raster.rio.clip_box(minx=bounds[0], miny=bounds[1], 
                               maxx=bounds[2], maxy=bounds[3])
masked = windowed.rio.clip(polygons.geometry, polygons.crs)
'''
        },
        'Simplify Geometries': {
            'description': 'Simplify complex polygons before clipping',
            'code': '''
# Simplify complex boundaries
simplified_polygons = polygons.copy()
simplified_polygons.geometry = simplified_polygons.geometry.simplify(0.01)
clipped = raster.rio.clip(simplified_polygons.geometry, simplified_polygons.crs)
'''
        },
        'Use Appropriate Data Types': {
            'description': 'Convert to smaller data types when possible',
            'code': '''
# Convert to smaller data type if range allows
if raster.min() >= 0 and raster.max() <= 65535:
    raster_uint16 = raster.astype(np.uint16)
    clipped = raster_uint16.rio.clip(polygons.geometry, polygons.crs)
'''
        }
    }
    
    print("MEMORY MANAGEMENT BEST PRACTICES")
    print("=" * 35)
    
    for tip_name, tip_info in tips.items():
        print(f"\n{tip_name}:")
        print(f"  {tip_info['description']}")
        print(f"  Code example:")
        for line in tip_info['code'].strip().split('\n'):
            print(f"    {line}")
    
    return tips

# Display best practices
best_practices = memory_management_tips()
```

### Quality Control
```python
def quality_control_checks():
    """Implement quality control for clipping operations"""
    
    def validate_clipping_result(original, clipped, operation_name):
        """Validate clipping operation results"""
        
        checks = {
            'Non-empty result': clipped.size > 0,
            'Valid CRS': clipped.rio.crs is not None,
            'Reasonable extent': (clipped.rio.bounds()[2] > clipped.rio.bounds()[0] and 
                                clipped.rio.bounds()[3] > clipped.rio.bounds()[1]),
            'Some valid data': (~np.isnan(clipped)).sum() > 0,
            'Extent within original': (
                clipped.rio.bounds()[0] >= original.rio.bounds()[0] and
                clipped.rio.bounds()[1] >= original.rio.bounds()[1] and
                clipped.rio.bounds()[2] <= original.rio.bounds()[2] and
                clipped.rio.bounds()[3] <= original.rio.bounds()[3]
            )
        }
        
        print(f"\nQUALITY CONTROL: {operation_name}")
        print("-" * (15 + len(operation_name)))
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{check_name:<20}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("Overall: ✓ ALL CHECKS PASSED")
        else:
            print("Overall: ✗ SOME CHECKS FAILED")
        
        return all_passed
    
    # Test with sample operations
    test_polygon = Polygon([(-3, -3), (3, -3), (3, 3), (-3, 3)])
    test_gdf = gpd.GeoDataFrame([1], geometry=[test_polygon], crs='EPSG:4326')
    
    # Valid clipping
    valid_clip = raster_xr.rio.clip(test_gdf.geometry, test_gdf.crs)
    validate_clipping_result(raster_xr, valid_clip, "Valid Clipping")
    
    # Empty result (polygon outside raster)
    outside_polygon = Polygon([(50, 50), (60, 50), (60, 60), (50, 60)])
    outside_gdf = gpd.GeoDataFrame([1], geometry=[outside_polygon], crs='EPSG:4326')
    
    try:
        empty_clip = raster_xr.rio.clip(outside_gdf.geometry, outside_gdf.crs)
        validate_clipping_result(raster_xr, empty_clip, "Empty Result")
    except Exception as e:
        print(f"\nQUALITY CONTROL: Empty Result")
        print("-" * 25)
        print(f"Operation failed: {e}")

# Run quality control checks
quality_control_checks()
```

---

## 🎯 Key Takeaways

1. **Vector Masking**: Use `rio.clip()` for precise boundary extraction with shapefiles
2. **Bounding Box Cropping**: Use `rio.clip_box()` for rectangular regions and performance optimization
3. **Method Selection**: Choose `mask()` for NoData setting, `window()` for extent reduction
4. **Performance**: Window first, then mask for complex polygons and large datasets
5. **Export Options**: Consider compression, metadata, and format based on use case
6. **Quality Control**: Always validate results and implement error handling

These techniques form the foundation for efficient raster data processing, enabling focused analysis on specific regions while managing memory and processing time effectively.