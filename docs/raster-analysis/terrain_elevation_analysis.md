# Terrain and Elevation Analysis

## Overview
Terrain analysis is fundamental to understanding Earth's surface processes, hydrology, and geomorphology. Digital Elevation Models (DEMs) provide the foundation for calculating slope, aspect, flow direction, and other terrain derivatives that are essential for environmental modeling, hazard assessment, and landscape planning.

## Why Terrain Analysis Matters
- **Hydrology**: Understanding water flow patterns and watershed boundaries
- **Geomorphology**: Analyzing landform characteristics and processes
- **Hazard Assessment**: Identifying areas prone to landslides, flooding, and erosion
- **Agriculture**: Optimizing irrigation and crop placement based on topography
- **Urban Planning**: Site suitability analysis and infrastructure planning

---

## 1️⃣ Working with DEMs

### Loading and Exploring DEM Data
```python
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import xarray as xr
from matplotlib.colors import LightSource

# Create realistic DEM for demonstration
def create_realistic_dem():
    """Create a realistic Digital Elevation Model"""
    
    # Create coordinate system (UTM-like, in meters)
    width, height = 500, 500
    x = np.linspace(0, 25000, width)  # 25km x 25km area
    y = np.linspace(0, 25000, height)
    X, Y = np.meshgrid(x, y)
    
    # Create complex terrain with multiple features
    elevation = (
        800 +  # Base elevation
        600 * np.exp(-((X-8000)**2 + (Y-18000)**2) / 10000000) +  # Major peak
        400 * np.exp(-((X-18000)**2 + (Y-8000)**2) / 8000000) +   # Secondary peak
        200 * np.exp(-((X-12000)**2 + (Y-12000)**2) / 5000000) +  # Hill
        -150 * np.exp(-((X-20000)**2 + (Y-20000)**2) / 6000000) + # Valley
        100 * np.sin(X/3000) * np.cos(Y/4000) +  # Rolling terrain
        50 * np.random.normal(0, 1, X.shape)  # Natural variation
    )
    
    # Add river valley
    river_mask = np.abs(Y - 12500 - 2000*np.sin(X/5000)) < 500
    elevation[river_mask] -= 50
    
    # Ensure realistic elevation range
    elevation = np.clip(elevation, 200, 1800)
    
    # Create xarray DEM
    dem = xr.DataArray(
        elevation,
        coords={'y': y, 'x': x},
        dims=['y', 'x'],
        attrs={
            'long_name': 'Digital Elevation Model',
            'units': 'meters',
            'vertical_datum': 'WGS84 ellipsoid',
            'horizontal_datum': 'WGS84',
            'resolution': '50m'
        }
    )
    
    # Set CRS (UTM Zone 33N as example)
    dem.rio.write_crs('EPSG:32633', inplace=True)
    dem.rio.write_nodata(-9999, inplace=True)
    
    return dem

# Load and explore DEM
def explore_dem_properties(dem):
    """Explore DEM properties and characteristics"""
    
    print("DEM PROPERTIES")
    print("=" * 15)
    print(f"Shape: {dem.shape}")
    print(f"CRS: {dem.rio.crs}")
    print(f"Resolution: {dem.rio.resolution()}")
    print(f"Bounds: {dem.rio.bounds()}")
    print(f"Elevation range: {dem.min().values:.1f} - {dem.max().values:.1f} m")
    print(f"Mean elevation: {dem.mean().values:.1f} m")
    print(f"Standard deviation: {dem.std().values:.1f} m")
    
    # Calculate basic statistics
    elevation_stats = {
        'Min': dem.min().values,
        'Max': dem.max().values,
        'Mean': dem.mean().values,
        'Median': dem.median().values,
        'Std': dem.std().values,
        'Range': dem.max().values - dem.min().values
    }
    
    # Visualize DEM with different representations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Basic elevation map
    dem.plot(ax=axes[0,0], cmap='terrain')
    axes[0,0].set_title('Elevation Map')
    axes[0,0].set_xlabel('Easting (m)')
    axes[0,0].set_ylabel('Northing (m)')
    
    # Elevation histogram
    axes[0,1].hist(dem.values.flatten(), bins=50, alpha=0.7, color='brown', edgecolor='black')
    axes[0,1].set_xlabel('Elevation (m)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Elevation Distribution')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3D surface (simplified)
    from mpl_toolkits.mplot3d import Axes3D
    ax3d = fig.add_subplot(223, projection='3d')
    
    # Downsample for 3D visualization
    step = 10
    X_3d = dem.x.values[::step]
    Y_3d = dem.y.values[::step]
    Z_3d = dem.values[::step, ::step]
    X_mesh, Y_mesh = np.meshgrid(X_3d, Y_3d)
    
    surf = ax3d.plot_surface(X_mesh, Y_mesh, Z_3d, cmap='terrain', alpha=0.8)
    ax3d.set_title('3D Terrain Surface')
    ax3d.set_xlabel('Easting (m)')
    ax3d.set_ylabel('Northing (m)')
    ax3d.set_zlabel('Elevation (m)')
    
    # Contour map
    contour_levels = np.arange(dem.min().values, dem.max().values, 50)
    cs = axes[1,1].contour(dem.x, dem.y, dem.values, levels=contour_levels, colors='black', alpha=0.6)
    axes[1,1].contourf(dem.x, dem.y, dem.values, levels=contour_levels, cmap='terrain', alpha=0.8)
    axes[1,1].clabel(cs, inline=True, fontsize=8)
    axes[1,1].set_title('Contour Map (50m intervals)')
    axes[1,1].set_xlabel('Easting (m)')
    axes[1,1].set_ylabel('Northing (m)')
    
    plt.tight_layout()
    plt.show()
    
    return elevation_stats

# Create and explore DEM
sample_dem = create_realistic_dem()
dem_stats = explore_dem_properties(sample_dem)
```

### DEM Quality Assessment
```python
def assess_dem_quality(dem):
    """Assess DEM quality and identify potential issues"""
    
    # Check for common DEM issues
    quality_checks = {}
    
    # 1. NoData values
    nodata_count = np.isnan(dem).sum().values if dem.rio.nodata is None else (dem == dem.rio.nodata).sum().values
    quality_checks['NoData pixels'] = nodata_count
    
    # 2. Flat areas (potential artifacts)
    flat_threshold = 0.1  # meters
    gy, gx = np.gradient(dem.values)
    slope_magnitude = np.sqrt(gx**2 + gy**2)
    flat_areas = (slope_magnitude < flat_threshold).sum()
    quality_checks['Flat areas'] = flat_areas
    
    # 3. Extreme slopes (potential errors)
    pixel_size = abs(dem.rio.resolution()[0])
    slope_degrees = np.arctan(slope_magnitude / pixel_size) * 180 / np.pi
    extreme_slopes = (slope_degrees > 60).sum()  # Slopes > 60 degrees
    quality_checks['Extreme slopes (>60°)'] = extreme_slopes
    
    # 4. Elevation spikes (outliers)
    elevation_median = np.median(dem.values)
    elevation_mad = np.median(np.abs(dem.values - elevation_median))
    outlier_threshold = elevation_median + 5 * elevation_mad
    spikes = (dem.values > outlier_threshold).sum()
    quality_checks['Elevation spikes'] = spikes
    
    # 5. Data gaps or voids
    # Simulate checking for rectangular NoData areas
    kernel = np.ones((5, 5))
    if dem.rio.nodata is not None:
        nodata_mask = (dem == dem.rio.nodata).values
        dilated = ndimage.binary_dilation(nodata_mask, kernel)
        data_gaps = dilated.sum() - nodata_mask.sum()
        quality_checks['Potential data gaps'] = data_gaps
    
    # Visualize quality issues
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original DEM
    dem.plot(ax=axes[0,0], cmap='terrain')
    axes[0,0].set_title('Original DEM')
    
    # Slope magnitude
    slope_da = xr.DataArray(slope_magnitude, coords=dem.coords, dims=dem.dims)
    slope_da.plot(ax=axes[0,1], cmap='Reds')
    axes[0,1].set_title('Slope Magnitude (m/m)')
    
    # Slope in degrees
    slope_deg_da = xr.DataArray(slope_degrees, coords=dem.coords, dims=dem.dims)
    slope_deg_da.plot(ax=axes[0,2], cmap='YlOrRd')
    axes[0,2].set_title('Slope (degrees)')
    
    # Flat areas
    flat_mask = slope_magnitude < flat_threshold
    axes[1,0].imshow(flat_mask, cmap='Reds', extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,0].set_title(f'Flat Areas (<{flat_threshold}m/m)')
    
    # Extreme slopes
    extreme_mask = slope_degrees > 60
    axes[1,1].imshow(extreme_mask, cmap='Reds', extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,1].set_title('Extreme Slopes (>60°)')
    
    # Elevation histogram with outliers marked
    axes[1,2].hist(dem.values.flatten(), bins=50, alpha=0.7, color='brown', edgecolor='black')
    axes[1,2].axvline(outlier_threshold, color='red', linestyle='--', label=f'Outlier threshold: {outlier_threshold:.1f}m')
    axes[1,2].set_xlabel('Elevation (m)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Elevation Distribution')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print quality assessment
    print("DEM QUALITY ASSESSMENT")
    print("=" * 25)
    total_pixels = dem.size
    
    for check_name, count in quality_checks.items():
        percentage = (count / total_pixels) * 100
        print(f"{check_name}: {count:,} pixels ({percentage:.2f}%)")
    
    # Quality recommendations
    print("\nQUALITY RECOMMENDATIONS:")
    if quality_checks['NoData pixels'] > total_pixels * 0.05:
        print("⚠️  High NoData content - consider gap filling")
    if quality_checks['Flat areas'] > total_pixels * 0.1:
        print("⚠️  Many flat areas detected - check for artifacts")
    if quality_checks['Extreme slopes (>60°)'] > 0:
        print("⚠️  Extreme slopes detected - validate or smooth")
    if quality_checks['Elevation spikes'] > 0:
        print("⚠️  Elevation spikes detected - consider outlier removal")
    
    return quality_checks, slope_magnitude, slope_degrees

# Assess DEM quality
quality_results, slope_mag, slope_deg = assess_dem_quality(sample_dem)
```

---

## 2️⃣ Generating Slope, Aspect, and Hillshade

### Slope Calculation
```python
def calculate_slope(dem, method='horn'):
    """Calculate slope from DEM using different methods"""
    
    # Get pixel size
    pixel_size_x = abs(dem.rio.resolution()[0])
    pixel_size_y = abs(dem.rio.resolution()[1])
    
    if method == 'horn':
        # Horn's method (3x3 kernel) - most common
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * pixel_size_x)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * pixel_size_y)
        
        dz_dx = ndimage.convolve(dem.values, kernel_x)
        dz_dy = ndimage.convolve(dem.values, kernel_y)
        
    elif method == 'simple':
        # Simple gradient method
        dz_dy, dz_dx = np.gradient(dem.values, pixel_size_y, pixel_size_x)
    
    else:
        raise ValueError("Method must be 'horn' or 'simple'")
    
    # Calculate slope magnitude and convert to degrees
    slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_degrees = np.degrees(slope_radians)
    slope_percent = np.tan(slope_radians) * 100
    
    # Create xarray objects
    slope_deg = xr.DataArray(
        slope_degrees,
        coords=dem.coords,
        dims=dem.dims,
        attrs={'long_name': 'Slope', 'units': 'degrees', 'method': method}
    )
    
    slope_pct = xr.DataArray(
        slope_percent,
        coords=dem.coords,
        dims=dem.dims,
        attrs={'long_name': 'Slope', 'units': 'percent', 'method': method}
    )
    
    return slope_deg, slope_pct, dz_dx, dz_dy

def calculate_aspect(dz_dx, dz_dy):
    """Calculate aspect from slope gradients"""
    
    # Calculate aspect in radians
    aspect_radians = np.arctan2(-dz_dx, dz_dy)
    
    # Convert to degrees (0-360, where 0 = North)
    aspect_degrees = np.degrees(aspect_radians)
    aspect_degrees = (aspect_degrees + 360) % 360
    
    # Handle flat areas (undefined aspect)
    slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    flat_threshold = 0.001
    aspect_degrees[slope_magnitude < flat_threshold] = -1  # -1 for flat areas
    
    return aspect_degrees

def calculate_hillshade(dem, azimuth=315, altitude=45):
    """Calculate hillshade for terrain visualization"""
    
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    
    # Calculate gradients
    pixel_size_x = abs(dem.rio.resolution()[0])
    pixel_size_y = abs(dem.rio.resolution()[1])
    
    dz_dy, dz_dx = np.gradient(dem.values, pixel_size_y, pixel_size_x)
    
    # Calculate slope and aspect
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    
    # Calculate hillshade
    hillshade = (
        np.sin(altitude_rad) * np.sin(slope_rad) +
        np.cos(altitude_rad) * np.cos(slope_rad) *
        np.cos(azimuth_rad - aspect_rad)
    )
    
    # Normalize to 0-255 range
    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)
    
    return hillshade

# Calculate terrain derivatives
slope_degrees, slope_percent, dx, dy = calculate_slope(sample_dem, method='horn')
aspect_degrees = calculate_aspect(dx, dy)
hillshade = calculate_hillshade(sample_dem, azimuth=315, altitude=45)

# Create xarray objects for visualization
aspect_da = xr.DataArray(
    aspect_degrees,
    coords=sample_dem.coords,
    dims=sample_dem.dims,
    attrs={'long_name': 'Aspect', 'units': 'degrees'}
)

hillshade_da = xr.DataArray(
    hillshade,
    coords=sample_dem.coords,
    dims=sample_dem.dims,
    attrs={'long_name': 'Hillshade', 'units': 'DN'}
)

# Visualize terrain derivatives
def visualize_terrain_derivatives(dem, slope, aspect, hillshade):
    """Visualize all terrain derivatives"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original DEM
    dem.plot(ax=axes[0,0], cmap='terrain')
    axes[0,0].set_title('Digital Elevation Model')
    
    # Slope
    slope.plot(ax=axes[0,1], cmap='YlOrRd')
    axes[0,1].set_title('Slope (degrees)')
    
    # Aspect
    aspect_masked = np.ma.masked_where(aspect.values == -1, aspect.values)
    im = axes[0,2].imshow(aspect_masked, cmap='hsv', vmin=0, vmax=360,
                         extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[0,2].set_title('Aspect (degrees)')
    plt.colorbar(im, ax=axes[0,2], label='Degrees from North')
    
    # Hillshade
    axes[1,0].imshow(hillshade.values, cmap='gray',
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,0].set_title('Hillshade')
    
    # Combined: DEM + Hillshade
    axes[1,1].imshow(hillshade.values, cmap='gray', alpha=0.6,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    dem.plot(ax=axes[1,1], cmap='terrain', alpha=0.7, add_colorbar=False)
    axes[1,1].set_title('DEM + Hillshade Overlay')
    
    # Slope classification
    slope_classes = np.digitize(slope.values, bins=[0, 5, 15, 30, 45, 90])
    class_labels = ['Flat (0-5°)', 'Gentle (5-15°)', 'Moderate (15-30°)', 
                   'Steep (30-45°)', 'Very Steep (>45°)']
    
    from matplotlib.colors import ListedColormap
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    cmap_classes = ListedColormap(colors)
    
    im_class = axes[1,2].imshow(slope_classes, cmap=cmap_classes, vmin=1, vmax=5,
                               extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,2].set_title('Slope Classification')
    
    # Add legend for slope classes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=class_labels[i]) for i in range(5)]
    axes[1,2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("TERRAIN DERIVATIVES STATISTICS")
    print("=" * 35)
    print(f"Slope - Mean: {slope.mean().values:.1f}°, Max: {slope.max().values:.1f}°")
    print(f"Aspect - Flat areas: {(aspect.values == -1).sum()} pixels")
    
    # Slope class distribution
    print("\nSlope Classification Distribution:")
    for i, label in enumerate(class_labels):
        count = (slope_classes == i+1).sum()
        percentage = (count / slope_classes.size) * 100
        print(f"  {label}: {count:,} pixels ({percentage:.1f}%)")

# Visualize all derivatives
visualize_terrain_derivatives(sample_dem, slope_degrees, aspect_da, hillshade_da)
```

---

## 3️⃣ Watershed and Flow Direction (Basic Introduction)

### Flow Direction Calculation
```python
def calculate_flow_direction(dem, method='d8'):
    """Calculate flow direction using D8 algorithm"""
    
    if method != 'd8':
        raise ValueError("Only D8 method implemented in this example")
    
    # D8 flow direction codes
    # 32  64  128
    # 16   0    1
    #  8   4    2
    
    flow_codes = np.array([[32, 64, 128],
                          [16,  0,   1],
                          [ 8,  4,   2]])
    
    # Initialize flow direction array
    flow_dir = np.zeros_like(dem.values, dtype=np.uint8)
    
    # Calculate flow direction for each pixel
    rows, cols = dem.shape
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            center_elev = dem.values[i, j]
            
            # Check all 8 neighbors
            max_slope = -1
            flow_direction = 0
            
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    
                    neighbor_elev = dem.values[i+di, j+dj]
                    
                    # Calculate slope (drop per unit distance)
                    if di == 0 or dj == 0:  # Cardinal directions
                        distance = abs(dem.rio.resolution()[0])
                    else:  # Diagonal directions
                        distance = abs(dem.rio.resolution()[0]) * np.sqrt(2)
                    
                    slope = (center_elev - neighbor_elev) / distance
                    
                    if slope > max_slope:
                        max_slope = slope
                        flow_direction = flow_codes[di+1, dj+1]
            
            flow_dir[i, j] = flow_direction
    
    return flow_dir

def calculate_flow_accumulation(flow_dir, dem):
    """Calculate flow accumulation (simplified version)"""
    
    # Initialize flow accumulation
    flow_acc = np.ones_like(dem.values, dtype=np.float32)
    
    # Create elevation-sorted indices (process from high to low)
    flat_indices = np.unravel_index(np.argsort(-dem.values.ravel()), dem.shape)
    
    # Flow direction lookup
    flow_lookup = {
        1: (0, 1),    # East
        2: (1, 1),    # Southeast  
        4: (1, 0),    # South
        8: (1, -1),   # Southwest
        16: (0, -1),  # West
        32: (-1, -1), # Northwest
        64: (-1, 0),  # North
        128: (-1, 1)  # Northeast
    }
    
    rows, cols = dem.shape
    
    # Process pixels from highest to lowest elevation
    for idx in range(len(flat_indices[0])):
        i, j = flat_indices[0][idx], flat_indices[1][idx]
        
        # Skip boundary pixels
        if i == 0 or i == rows-1 or j == 0 or j == cols-1:
            continue
        
        flow_code = flow_dir[i, j]
        
        if flow_code in flow_lookup:
            di, dj = flow_lookup[flow_code]
            ni, nj = i + di, j + dj
            
            # Check bounds
            if 0 <= ni < rows and 0 <= nj < cols:
                flow_acc[ni, nj] += flow_acc[i, j]
    
    return flow_acc

def delineate_watersheds(flow_dir, flow_acc, outlets):
    """Delineate watersheds from outlet points (simplified)"""
    
    watersheds = np.zeros_like(flow_dir, dtype=np.int32)
    
    # Reverse flow direction lookup
    reverse_flow = {
        1: (0, -1),   # From East
        2: (-1, -1),  # From Southeast
        4: (-1, 0),   # From South
        8: (-1, 1),   # From Southwest
        16: (0, 1),   # From West
        32: (1, 1),   # From Northwest
        64: (1, 0),   # From North
        128: (1, -1)  # From Northeast
    }
    
    rows, cols = flow_dir.shape
    
    for outlet_id, (outlet_row, outlet_col) in enumerate(outlets, 1):
        # Trace upstream from outlet
        to_process = [(outlet_row, outlet_col)]
        processed = set()
        
        while to_process:
            current_row, current_col = to_process.pop()
            
            if (current_row, current_col) in processed:
                continue
            
            processed.add((current_row, current_col))
            watersheds[current_row, current_col] = outlet_id
            
            # Find all pixels that flow to current pixel
            for i in range(max(0, current_row-1), min(rows, current_row+2)):
                for j in range(max(0, current_col-1), min(cols, current_col+2)):
                    if i == current_row and j == current_col:
                        continue
                    
                    flow_code = flow_dir[i, j]
                    if flow_code in reverse_flow:
                        di, dj = reverse_flow[flow_code]
                        target_i, target_j = i + di, j + dj
                        
                        if target_i == current_row and target_j == current_col:
                            to_process.append((i, j))
    
    return watersheds

# Calculate flow direction and accumulation
print("Calculating flow direction...")
flow_direction = calculate_flow_direction(sample_dem)

print("Calculating flow accumulation...")
flow_accumulation = calculate_flow_accumulation(flow_direction, sample_dem)

# Define outlet points (manually selected for demonstration)
outlet_points = [(400, 200), (300, 350), (150, 100)]  # (row, col) coordinates

print("Delineating watersheds...")
watersheds = delineate_watersheds(flow_direction, flow_accumulation, outlet_points)

# Visualize hydrological analysis
def visualize_hydrology(dem, flow_dir, flow_acc, watersheds, outlets):
    """Visualize hydrological analysis results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original DEM
    dem.plot(ax=axes[0,0], cmap='terrain')
    axes[0,0].set_title('Digital Elevation Model')
    
    # Flow direction
    flow_dir_masked = np.ma.masked_where(flow_dir == 0, flow_dir)
    im1 = axes[0,1].imshow(flow_dir_masked, cmap='tab10',
                          extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[0,1].set_title('Flow Direction (D8)')
    
    # Flow accumulation (log scale for better visualization)
    flow_acc_log = np.log10(flow_acc + 1)
    im2 = axes[0,2].imshow(flow_acc_log, cmap='Blues',
                          extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[0,2].set_title('Flow Accumulation (log scale)')
    plt.colorbar(im2, ax=axes[0,2])
    
    # Stream network (high flow accumulation)
    stream_threshold = np.percentile(flow_acc, 95)
    streams = flow_acc > stream_threshold
    
    axes[1,0].imshow(dem.values, cmap='terrain', alpha=0.7,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,0].imshow(streams, cmap='Blues', alpha=0.8,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,0].set_title('Stream Network')
    
    # Watersheds
    watersheds_masked = np.ma.masked_where(watersheds == 0, watersheds)
    im3 = axes[1,1].imshow(watersheds_masked, cmap='Set3', alpha=0.7,
                          extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    dem.plot(ax=axes[1,1], cmap='terrain', alpha=0.3, add_colorbar=False)
    
    # Mark outlet points
    for i, (row, col) in enumerate(outlets):
        x_coord = dem.x.values[col]
        y_coord = dem.y.values[row]
        axes[1,1].plot(x_coord, y_coord, 'ro', markersize=8, label=f'Outlet {i+1}')
    
    axes[1,1].set_title('Watersheds')
    axes[1,1].legend()
    
    # Combined analysis
    axes[1,2].imshow(dem.values, cmap='terrain', alpha=0.5,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,2].imshow(streams, cmap='Blues', alpha=0.8,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,2].imshow(watersheds_masked, cmap='Set3', alpha=0.4,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    axes[1,2].set_title('Combined Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Print watershed statistics
    print("WATERSHED ANALYSIS RESULTS")
    print("=" * 30)
    
    for i in range(1, watersheds.max() + 1):
        watershed_mask = watersheds == i
        watershed_area = watershed_mask.sum() * (abs(dem.rio.resolution()[0]) ** 2) / 1000000  # km²
        mean_elevation = dem.values[watershed_mask].mean()
        
        print(f"Watershed {i}:")
        print(f"  Area: {watershed_area:.2f} km²")
        print(f"  Mean elevation: {mean_elevation:.1f} m")
        print(f"  Pixels: {watershed_mask.sum():,}")

# Visualize hydrological analysis
visualize_hydrology(sample_dem, flow_direction, flow_accumulation, watersheds, outlet_points)
```

---

## 4️⃣ Creating Contour Lines from Elevation Data

### Contour Generation
```python
def generate_contours(dem, interval=50, smooth=True):
    """Generate contour lines from DEM"""
    
    from matplotlib import pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from scipy.ndimage import gaussian_filter
    
    # Smooth DEM if requested
    if smooth:
        smoothed_dem = gaussian_filter(dem.values, sigma=1.0)
    else:
        smoothed_dem = dem.values
    
    # Define contour levels
    min_elev = np.floor(dem.min().values / interval) * interval
    max_elev = np.ceil(dem.max().values / interval) * interval
    contour_levels = np.arange(min_elev, max_elev + interval, interval)
    
    # Generate contours using matplotlib
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create contour lines
    cs_lines = ax.contour(dem.x, dem.y, smoothed_dem, 
                         levels=contour_levels, colors='brown', linewidths=0.8)
    
    # Create filled contours for visualization
    cs_filled = ax.contourf(dem.x, dem.y, smoothed_dem, 
                           levels=contour_levels, cmap='terrain', alpha=0.7)
    
    # Add contour labels
    ax.clabel(cs_lines, inline=True, fontsize=8, fmt='%d m')
    
    # Add colorbar
    plt.colorbar(cs_filled, ax=ax, label='Elevation (m)')
    
    ax.set_title(f'Contour Map ({interval}m intervals)')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return cs_lines, contour_levels

def create_contour_geodataframe(dem, interval=50):
    """Create GeoDataFrame of contour lines for export"""
    
    import geopandas as gpd
    from shapely.geometry import LineString
    from matplotlib import pyplot as plt
    
    # Generate contours
    min_elev = np.floor(dem.min().values / interval) * interval
    max_elev = np.ceil(dem.max().values / interval) * interval
    contour_levels = np.arange(min_elev, max_elev + interval, interval)
    
    # Create temporary plot to extract contour paths
    fig, ax = plt.subplots()
    cs = ax.contour(dem.x, dem.y, dem.values, levels=contour_levels)
    plt.close(fig)  # Close the temporary figure
    
    # Extract contour lines
    contour_lines = []
    elevations = []
    
    for i, level in enumerate(contour_levels):
        try:
            # Get paths for this contour level
            paths = cs.collections[i].get_paths()
            
            for path in paths:
                # Convert path to coordinates
                vertices = path.vertices
                
                if len(vertices) > 1:
                    # Create LineString
                    line = LineString(vertices)
                    contour_lines.append(line)
                    elevations.append(level)
        except IndexError:
            continue
    
    # Create GeoDataFrame
    if contour_lines:
        contours_gdf = gpd.GeoDataFrame({
            'elevation': elevations,
            'interval': interval
        }, geometry=contour_lines, crs=dem.rio.crs)
        
        return contours_gdf
    else:
        return None

def advanced_contour_analysis(dem, major_interval=100, minor_interval=20):
    """Advanced contour analysis with major and minor contours"""
    
    # Generate major and minor contours
    min_elev = np.floor(dem.min().values / minor_interval) * minor_interval
    max_elev = np.ceil(dem.max().values / minor_interval) * minor_interval
    
    minor_levels = np.arange(min_elev, max_elev + minor_interval, minor_interval)
    major_levels = np.arange(min_elev, max_elev + major_interval, major_interval)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Basic contour map
    cs1 = axes[0,0].contour(dem.x, dem.y, dem.values, levels=minor_levels, 
                           colors='brown', linewidths=0.5, alpha=0.7)
    cs1_major = axes[0,0].contour(dem.x, dem.y, dem.values, levels=major_levels, 
                                 colors='darkred', linewidths=1.5)
    axes[0,0].clabel(cs1_major, inline=True, fontsize=8, fmt='%d m')
    axes[0,0].set_title('Major & Minor Contours')
    
    # Filled contours with hillshade
    hillshade_norm = hillshade_da.values / 255.0
    axes[0,1].imshow(hillshade_norm, cmap='gray', alpha=0.8,
                    extent=[dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()])
    cs2 = axes[0,1].contour(dem.x, dem.y, dem.values, levels=major_levels, 
                           colors='white', linewidths=1.0, alpha=0.9)
    axes[0,1].clabel(cs2, inline=True, fontsize=8, fmt='%d m', 
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7))
    axes[0,1].set_title('Contours on Hillshade')
    
    # Slope with contours
    slope_degrees.plot(ax=axes[1,0], cmap='YlOrRd', alpha=0.8)
    cs3 = axes[1,0].contour(dem.x, dem.y, dem.values, levels=major_levels, 
                           colors='black', linewidths=1.0, alpha=0.8)
    axes[1,0].set_title('Contours on Slope Map')
    
    # 3D contour visualization
    ax3d = fig.add_subplot(224, projection='3d')
    
    # Downsample for 3D performance
    step = 5
    X_3d = dem.x.values[::step]
    Y_3d = dem.y.values[::step]
    Z_3d = dem.values[::step, ::step]
    X_mesh, Y_mesh = np.meshgrid(X_3d, Y_3d)
    
    # Surface plot
    surf = ax3d.plot_surface(X_mesh, Y_mesh, Z_3d, cmap='terrain', alpha=0.6)
    
    # Add 3D contours at different elevations
    for level in major_levels[::2]:  # Every other major contour
        cs_3d = ax3d.contour(dem.x, dem.y, dem.values, levels=[level], 
                            colors='red', linewidths=2, alpha=0.8)
    
    ax3d.set_title('3D Terrain with Contours')
    ax3d.set_xlabel('Easting (m)')
    ax3d.set_ylabel('Northing (m)')
    ax3d.set_zlabel('Elevation (m)')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate contour statistics
    contour_stats = {
        'Total elevation range': max_elev - min_elev,
        'Number of major contours': len(major_levels),
        'Number of minor contours': len(minor_levels),
        'Average slope between contours': (max_elev - min_elev) / len(major_levels)
    }
    
    print("CONTOUR ANALYSIS STATISTICS")
    print("=" * 30)
    for stat_name, value in contour_stats.items():
        if isinstance(value, float):
            print(f"{stat_name}: {value:.1f}")
        else:
            print(f"{stat_name}: {value}")
    
    return major_levels, minor_levels, contour_stats

# Generate contours
print("Generating contour lines...")
contour_lines, levels = generate_contours(sample_dem, interval=50)

print("Creating contour GeoDataFrame...")
contours_gdf = create_contour_geodataframe(sample_dem, interval=50)

if contours_gdf is not None:
    print(f"Created {len(contours_gdf)} contour lines")
    print(f"Elevation range: {contours_gdf['elevation'].min():.0f} - {contours_gdf['elevation'].max():.0f} m")

print("Performing advanced contour analysis...")
major_contours, minor_contours, stats = advanced_contour_analysis(sample_dem)
```

### Contour Export and Applications
```python
def export_contours_multiple_formats(contours_gdf, dem, base_filename="contours"):
    """Export contours in multiple formats"""
    
    if contours_gdf is None:
        print("No contours to export")
        return
    
    export_formats = {
        'Shapefile': f'{base_filename}.shp',
        'GeoJSON': f'{base_filename}.geojson',
        'GeoPackage': f'{base_filename}.gpkg',
        'KML': f'{base_filename}.kml'
    }
    
    print("EXPORTING CONTOURS")
    print("=" * 20)
    
    for format_name, filename in export_formats.items():
        try:
            if format_name == 'Shapefile':
                contours_gdf.to_file(filename, driver='ESRI Shapefile')
            elif format_name == 'GeoJSON':
                contours_gdf.to_file(filename, driver='GeoJSON')
            elif format_name == 'GeoPackage':
                contours_gdf.to_file(filename, driver='GPKG')
            elif format_name == 'KML':
                # Convert to WGS84 for KML
                contours_wgs84 = contours_gdf.to_crs('EPSG:4326')
                contours_wgs84.to_file(filename, driver='KML')
            
            print(f"✓ {format_name}: {filename}")
            
        except Exception as e:
            print(f"✗ {format_name}: Failed - {e}")
    
    # Create metadata file
    metadata = f"""
CONTOUR LINES METADATA
======================

Source DEM Information:
- CRS: {dem.rio.crs}
- Resolution: {dem.rio.resolution()}
- Extent: {dem.rio.bounds()}
- Elevation Range: {dem.min().values:.1f} - {dem.max().values:.1f} m

Contour Information:
- Number of contours: {len(contours_gdf)}
- Contour interval: {contours_gdf['interval'].iloc[0]} m
- Elevation range: {contours_gdf['elevation'].min():.0f} - {contours_gdf['elevation'].max():.0f} m

Processing Information:
- Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- Software: Python rioxarray + matplotlib
- Smoothing: Applied (Gaussian filter, sigma=1.0)

Quality Notes:
- Contours generated from smoothed DEM
- Suitable for visualization and general analysis
- For high-precision applications, consider original DEM resolution
"""
    
    with open(f'{base_filename}_metadata.txt', 'w') as f:
        f.write(metadata)
    
    print(f"✓ Metadata: {base_filename}_metadata.txt")

# Export contours if available
if contours_gdf is not None:
    import pandas as pd
    export_contours_multiple_formats(contours_gdf, sample_dem)
```

---

## 🔧 Best Practices and Applications

### Terrain Analysis Best Practices
```python
def terrain_analysis_best_practices():
    """Best practices for terrain analysis"""
    
    best_practices = {
        'DEM Preprocessing': {
            'Quality Assessment': 'Check for voids, spikes, and artifacts',
            'Smoothing': 'Apply appropriate filtering for noise reduction',
            'Resolution': 'Match resolution to analysis requirements',
            'CRS': 'Use projected coordinate system for accurate calculations'
        },
        'Slope Calculation': {
            'Method Selection': 'Horn method for most applications',
            'Units': 'Choose degrees, percent, or radians based on use case',
            'Edge Effects': 'Handle boundary pixels appropriately',
            'Validation': 'Compare with field measurements when possible'
        },
        'Flow Analysis': {
            'Algorithm Choice': 'D8 for simplicity, D-infinity for accuracy',
            'Preprocessing': 'Fill sinks before flow direction calculation',
            'Validation': 'Compare with known drainage patterns',
            'Scale': 'Consider appropriate DEM resolution for catchment size'
        },
        'Contour Generation': {
            'Interval Selection': 'Choose based on terrain roughness and map scale',
            'Smoothing': 'Apply to reduce noise-induced artifacts',
            'Labeling': 'Include elevation values for interpretation',
            'Export': 'Use appropriate format for intended application'
        }
    }
    
    print("TERRAIN ANALYSIS BEST PRACTICES")
    print("=" * 35)
    
    for category, practices in best_practices.items():
        print(f"\n{category.upper()}:")
        print("-" * len(category))
        for practice, description in practices.items():
            print(f"  • {practice}: {description}")
    
    # Common applications
    applications = {
        'Hydrology': ['Watershed delineation', 'Stream network extraction', 'Flood modeling'],
        'Geomorphology': ['Landform classification', 'Erosion assessment', 'Terrain roughness'],
        'Engineering': ['Site suitability', 'Cut-fill analysis', 'Visibility analysis'],
        'Agriculture': ['Precision farming', 'Irrigation planning', 'Soil erosion risk'],
        'Forestry': ['Timber harvesting', 'Fire modeling', 'Habitat analysis'],
        'Urban Planning': ['Slope stability', 'Drainage design', 'View shed analysis']
    }
    
    print("\n\nCOMMON APPLICATIONS")
    print("=" * 20)
    
    for field, uses in applications.items():
        print(f"\n{field}:")
        for use in uses:
            print(f"  - {use}")

# Display best practices
terrain_analysis_best_practices()
```

---

## 🎯 Key Takeaways

1. **DEM Quality**: Always assess and preprocess DEMs before analysis
2. **Method Selection**: Choose appropriate algorithms based on application requirements
3. **Scale Considerations**: Match DEM resolution to analysis scale and objectives
4. **Validation**: Compare results with field data and known patterns when possible
5. **Integration**: Combine multiple terrain derivatives for comprehensive analysis
6. **Export Standards**: Use appropriate formats and include comprehensive metadata

Terrain analysis provides the foundation for understanding Earth surface processes and supports decision-making across numerous disciplines from hydrology to urban planning.