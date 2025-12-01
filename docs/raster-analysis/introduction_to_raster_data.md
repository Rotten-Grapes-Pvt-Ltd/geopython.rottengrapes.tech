# Introduction to Raster Data

## Overview
Raster data is the foundation of digital spatial analysis, representing the world as a grid of pixels where each cell contains a value. From satellite imagery capturing Earth's surface to elevation models mapping terrain, raster data enables us to analyze, visualize, and understand spatial patterns across diverse fields including remote sensing, environmental science, agriculture, and urban planning.

## Why Raster Data Matters
- **Global Coverage**: Satellite imagery provides consistent, repeatable observations of Earth's surface
- **Quantitative Analysis**: Each pixel contains measurable values for mathematical operations
- **Temporal Analysis**: Time-series data reveals changes and trends over time
- **Multi-spectral Information**: Different wavelengths capture various surface properties
- **Modeling Foundation**: Essential for environmental modeling, climate studies, and predictive analysis

---

## 1️⃣ What is Raster Data?

### Core Concept
Raster data represents spatial information as a regular grid of cells (pixels), where each cell contains a value representing a measurement or classification at that location.

```python
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# Simple raster concept demonstration
def create_simple_raster():
    """Create a simple raster to demonstrate the concept"""
    
    # Create a 10x10 grid with elevation values
    elevation_grid = np.array([
        [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
        [102, 107, 112, 117, 122, 127, 132, 137, 142, 147],
        [104, 109, 114, 119, 124, 129, 134, 139, 144, 149],
        [106, 111, 116, 121, 126, 131, 136, 141, 146, 151],
        [108, 113, 118, 123, 128, 133, 138, 143, 148, 153],
        [110, 115, 120, 125, 130, 135, 140, 145, 150, 155],
        [112, 117, 122, 127, 132, 137, 142, 147, 152, 157],
        [114, 119, 124, 129, 134, 139, 144, 149, 154, 159],
        [116, 121, 126, 131, 136, 141, 146, 151, 156, 161],
        [118, 123, 128, 133, 138, 143, 148, 153, 158, 163]
    ])
    
    # Visualize the raster
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show as image
    im1 = ax1.imshow(elevation_grid, cmap='terrain')
    ax1.set_title('Raster as Image')
    ax1.set_xlabel('Column (X)')
    ax1.set_ylabel('Row (Y)')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    
    # Show as 3D surface
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    x, y = np.meshgrid(range(10), range(10))
    ax2.plot_surface(x, y, elevation_grid, cmap='terrain')
    ax2.set_title('Raster as 3D Surface')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Elevation (m)')
    
    plt.tight_layout()
    plt.show()
    
    return elevation_grid

# Demonstrate raster concept
sample_raster = create_simple_raster()
print(f"Raster shape: {sample_raster.shape}")
print(f"Min elevation: {sample_raster.min()} m")
print(f"Max elevation: {sample_raster.max()} m")
```

### Types of Raster Data

#### 1. Imagery (Optical Sensors)
```python
# Satellite imagery characteristics
imagery_types = {
    'Landsat 8': {
        'bands': 11,
        'resolution': '15-100m',
        'spectral_range': '0.43-12.51 μm',
        'revisit_time': '16 days',
        'applications': ['Land cover', 'Agriculture', 'Urban planning']
    },
    'Sentinel-2': {
        'bands': 13,
        'resolution': '10-60m', 
        'spectral_range': '0.44-2.19 μm',
        'revisit_time': '5 days',
        'applications': ['Vegetation monitoring', 'Water quality', 'Disaster response']
    },
    'MODIS': {
        'bands': 36,
        'resolution': '250m-1km',
        'spectral_range': '0.4-14.4 μm',
        'revisit_time': '1-2 days',
        'applications': ['Climate monitoring', 'Fire detection', 'Ocean color']
    }
}

for sensor, specs in imagery_types.items():
    print(f"\n{sensor}:")
    for key, value in specs.items():
        print(f"  {key}: {value}")
```

#### 2. Grids (Continuous Surfaces)
```python
def demonstrate_grid_types():
    """Show different types of grid data"""
    
    # Temperature grid (continuous values)
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    temperature = 20 + 5 * np.exp(-(X**2 + Y**2) / 20)
    
    # Precipitation grid (continuous values)
    precipitation = 100 + 50 * np.sin(X/2) * np.cos(Y/2)
    
    # Population density grid (continuous values)
    population_density = 1000 * np.exp(-((X-2)**2 + (Y+1)**2) / 10)
    
    # Visualize different grid types
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    grids = [
        (temperature, 'Temperature (°C)', 'coolwarm'),
        (precipitation, 'Precipitation (mm)', 'Blues'),
        (population_density, 'Population Density', 'Reds')
    ]
    
    for i, (grid, title, cmap) in enumerate(grids):
        im = axes[i].imshow(grid, cmap=cmap, extent=[-10, 10, -10, 10])
        axes[i].set_title(title)
        axes[i].set_xlabel('X (km)')
        axes[i].set_ylabel('Y (km)')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.show()
    
    return temperature, precipitation, population_density

# Demonstrate grid types
temp_grid, precip_grid, pop_grid = demonstrate_grid_types()
```

#### 3. Digital Elevation Models (DEMs)
```python
def create_realistic_dem():
    """Create a realistic-looking DEM"""
    
    # Create coordinate grids
    x = np.linspace(0, 100, 100)  # 100km x 100km area
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create elevation with multiple peaks and valleys
    elevation = (
        500 +  # Base elevation
        200 * np.exp(-((X-30)**2 + (Y-70)**2) / 200) +  # Mountain peak 1
        150 * np.exp(-((X-70)**2 + (Y-30)**2) / 150) +  # Mountain peak 2
        -100 * np.exp(-((X-50)**2 + (Y-50)**2) / 100) + # Valley
        50 * np.sin(X/10) * np.cos(Y/15) +  # Rolling hills
        np.random.normal(0, 10, X.shape)  # Natural variation
    )
    
    # Visualize DEM
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 2D elevation map
    im1 = axes[0].imshow(elevation, cmap='terrain', extent=[0, 100, 0, 100])
    axes[0].set_title('Digital Elevation Model')
    axes[0].set_xlabel('Distance (km)')
    axes[0].set_ylabel('Distance (km)')
    plt.colorbar(im1, ax=axes[0], label='Elevation (m)')
    
    # Contour map
    contours = axes[1].contour(X, Y, elevation, levels=15, colors='black', alpha=0.6)
    axes[1].contourf(X, Y, elevation, levels=15, cmap='terrain', alpha=0.8)
    axes[1].clabel(contours, inline=True, fontsize=8)
    axes[1].set_title('Contour Map')
    axes[1].set_xlabel('Distance (km)')
    axes[1].set_ylabel('Distance (km)')
    
    # 3D surface
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(X[::5, ::5], Y[::5, ::5], elevation[::5, ::5], 
                           cmap='terrain', alpha=0.8)
    ax3.set_title('3D Terrain')
    ax3.set_xlabel('X (km)')
    ax3.set_ylabel('Y (km)')
    ax3.set_zlabel('Elevation (m)')
    
    plt.tight_layout()
    plt.show()
    
    return elevation, X, Y

# Create and visualize DEM
dem_data, x_coords, y_coords = create_realistic_dem()
print(f"DEM statistics:")
print(f"  Min elevation: {dem_data.min():.1f} m")
print(f"  Max elevation: {dem_data.max():.1f} m")
print(f"  Mean elevation: {dem_data.mean():.1f} m")
print(f"  Elevation range: {dem_data.max() - dem_data.min():.1f} m")
```

---

## 2️⃣ Common Raster Formats

### Format Comparison and Use Cases
```python
import os
from pathlib import Path

def compare_raster_formats():
    """Compare different raster formats and their characteristics"""
    
    formats = {
        'GeoTIFF (.tif/.tiff)': {
            'description': 'Most common geospatial raster format',
            'compression': 'LZW, JPEG, DEFLATE, PackBits',
            'metadata': 'Extensive (GeoKeys, tags)',
            'multi_band': 'Yes',
            'pyramids': 'Yes (overviews)',
            'streaming': 'Excellent',
            'use_cases': ['Satellite imagery', 'DEMs', 'Analysis results'],
            'pros': ['Universal support', 'Rich metadata', 'Efficient'],
            'cons': ['Large file sizes without compression']
        },
        'ERDAS IMAGINE (.img)': {
            'description': 'ERDAS native format',
            'compression': 'RLE, JPEG',
            'metadata': 'Good (statistics, pyramids)',
            'multi_band': 'Yes',
            'pyramids': 'Yes (built-in)',
            'streaming': 'Good',
            'use_cases': ['Remote sensing', 'Image processing'],
            'pros': ['Fast access', 'Built-in pyramids', 'Statistics'],
            'cons': ['Proprietary', 'Limited software support']
        },
        'NetCDF (.nc)': {
            'description': 'Network Common Data Form',
            'compression': 'Various (zlib, szip)',
            'metadata': 'Excellent (self-describing)',
            'multi_band': 'Yes (as variables)',
            'pyramids': 'No (but chunking)',
            'streaming': 'Good (with chunking)',
            'use_cases': ['Climate data', 'Oceanography', 'Atmospheric data'],
            'pros': ['Self-describing', 'Multi-dimensional', 'Standards-based'],
            'cons': ['Complex for simple imagery']
        },
        'HDF5 (.h5/.hdf5)': {
            'description': 'Hierarchical Data Format',
            'compression': 'Multiple algorithms',
            'metadata': 'Excellent (hierarchical)',
            'multi_band': 'Yes (as datasets)',
            'pyramids': 'Custom implementation',
            'streaming': 'Excellent (chunking)',
            'use_cases': ['Scientific data', 'Large datasets', 'MODIS/VIIRS'],
            'pros': ['Very flexible', 'Efficient', 'Hierarchical'],
            'cons': ['Complex structure', 'Learning curve']
        },
        'Cloud Optimized GeoTIFF (COG)': {
            'description': 'GeoTIFF optimized for cloud access',
            'compression': 'JPEG, DEFLATE, LZW, WEBP',
            'metadata': 'Same as GeoTIFF',
            'multi_band': 'Yes',
            'pyramids': 'Required (internal)',
            'streaming': 'Excellent (HTTP range requests)',
            'use_cases': ['Cloud platforms', 'Web mapping', 'Large datasets'],
            'pros': ['Cloud-native', 'Fast streaming', 'Standard format'],
            'cons': ['Specific structure requirements']
        }
    }
    
    print("RASTER FORMAT COMPARISON")
    print("=" * 50)
    
    for format_name, specs in formats.items():
        print(f"\n{format_name}")
        print("-" * len(format_name))
        print(f"Description: {specs['description']}")
        print(f"Compression: {specs['compression']}")
        print(f"Multi-band: {specs['multi_band']}")
        print(f"Use cases: {', '.join(specs['use_cases'])}")
        print(f"Pros: {', '.join(specs['pros'])}")
        print(f"Cons: {', '.join(specs['cons'])}")

# Display format comparison
compare_raster_formats()
```

### Format-Specific Reading Examples
```python
import rioxarray as rxr
import xarray as xr

def read_different_formats():
    """Demonstrate reading different raster formats"""
    
    # GeoTIFF - Most common
    try:
        geotiff = rxr.open_rasterio('landsat_scene.tif')
        print("GeoTIFF loaded successfully")
        print(f"  Shape: {geotiff.shape}")
        print(f"  CRS: {geotiff.rio.crs}")
    except FileNotFoundError:
        print("GeoTIFF example file not found")
    
    # NetCDF - Climate data
    try:
        netcdf = xr.open_dataset('temperature_data.nc')
        print("\nNetCDF loaded successfully")
        print(f"  Variables: {list(netcdf.data_vars)}")
        print(f"  Dimensions: {dict(netcdf.dims)}")
    except FileNotFoundError:
        print("NetCDF example file not found")
    
    # HDF5 - MODIS data
    try:
        hdf5 = rxr.open_rasterio('MOD11A1.hdf')
        print("\nHDF5 loaded successfully")
        print(f"  Shape: {hdf5.shape}")
    except FileNotFoundError:
        print("HDF5 example file not found")
    
    # ERDAS IMAGINE
    try:
        img = rxr.open_rasterio('satellite_image.img')
        print("\nERDAS IMG loaded successfully")
        print(f"  Shape: {img.shape}")
    except FileNotFoundError:
        print("ERDAS IMG example file not found")

# Demonstrate format reading
read_different_formats()
```

---

## 3️⃣ Structure of Raster Files

### Bands: Multi-spectral Information
```python
def analyze_band_structure():
    """Analyze the band structure of raster data"""
    
    # Simulate multi-band satellite data
    height, width = 1000, 1000
    
    # Create synthetic Landsat-8 bands
    bands_info = {
        'Band 1 - Coastal': {'wavelength': '0.43-0.45 μm', 'resolution': '30m'},
        'Band 2 - Blue': {'wavelength': '0.45-0.51 μm', 'resolution': '30m'},
        'Band 3 - Green': {'wavelength': '0.53-0.59 μm', 'resolution': '30m'},
        'Band 4 - Red': {'wavelength': '0.64-0.67 μm', 'resolution': '30m'},
        'Band 5 - NIR': {'wavelength': '0.85-0.88 μm', 'resolution': '30m'},
        'Band 6 - SWIR1': {'wavelength': '1.57-1.65 μm', 'resolution': '30m'},
        'Band 7 - SWIR2': {'wavelength': '2.11-2.29 μm', 'resolution': '30m'},
        'Band 8 - Pan': {'wavelength': '0.50-0.68 μm', 'resolution': '15m'},
        'Band 9 - Cirrus': {'wavelength': '1.36-1.38 μm', 'resolution': '30m'},
        'Band 10 - TIRS1': {'wavelength': '10.6-11.19 μm', 'resolution': '100m'},
        'Band 11 - TIRS2': {'wavelength': '11.5-12.51 μm', 'resolution': '100m'}
    }
    
    print("LANDSAT-8 BAND STRUCTURE")
    print("=" * 40)
    
    for i, (band_name, info) in enumerate(bands_info.items(), 1):
        print(f"Band {i:2d}: {band_name}")
        print(f"         Wavelength: {info['wavelength']}")
        print(f"         Resolution: {info['resolution']}")
        print()
    
    # Demonstrate band combinations
    band_combinations = {
        'True Color (RGB)': [4, 3, 2],  # Red, Green, Blue
        'False Color (CIR)': [5, 4, 3],  # NIR, Red, Green
        'Agriculture': [6, 5, 2],  # SWIR1, NIR, Blue
        'Geology': [7, 6, 4],  # SWIR2, SWIR1, Red
        'Bathymetry': [4, 3, 1],  # Red, Green, Coastal
        'Vegetation Analysis': [5, 6, 4]  # NIR, SWIR1, Red
    }
    
    print("COMMON BAND COMBINATIONS")
    print("=" * 30)
    for combo_name, bands in band_combinations.items():
        print(f"{combo_name}: Bands {bands}")
    
    return bands_info, band_combinations

# Analyze band structure
band_info, combinations = analyze_band_structure()
```

### Resolution: Spatial Detail
```python
def demonstrate_resolution_concepts():
    """Demonstrate different resolution concepts"""
    
    # Create base image at different resolutions
    original_size = 1000
    
    resolutions = {
        'High Resolution (1m)': 1000,
        'Medium Resolution (10m)': 100, 
        'Low Resolution (100m)': 10,
        'Very Low Resolution (1km)': 1
    }
    
    # Create a synthetic landscape
    x = np.linspace(0, 10, original_size)
    y = np.linspace(0, 10, original_size)
    X, Y = np.meshgrid(x, y)
    
    # Complex landscape with multiple features
    landscape = (
        np.sin(X) * np.cos(Y) +  # Rolling terrain
        2 * np.exp(-((X-3)**2 + (Y-7)**2)) +  # Mountain
        -1.5 * np.exp(-((X-7)**2 + (Y-3)**2)) +  # Valley
        0.5 * np.random.random((original_size, original_size))  # Noise
    )
    
    # Visualize different resolutions
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (res_name, size) in enumerate(resolutions.items()):
        # Downsample to simulate different resolutions
        step = original_size // size
        downsampled = landscape[::step, ::step]
        
        im = axes[i].imshow(downsampled, cmap='terrain', extent=[0, 10, 0, 10])
        axes[i].set_title(f'{res_name}\nArray size: {downsampled.shape}')
        axes[i].set_xlabel('Distance (km)')
        axes[i].set_ylabel('Distance (km)')
        
        # Add pixel grid for very low resolution
        if size <= 10:
            for x_line in range(size + 1):
                axes[i].axvline(x_line * 10/size, color='white', alpha=0.5, linewidth=0.5)
            for y_line in range(size + 1):
                axes[i].axhline(y_line * 10/size, color='white', alpha=0.5, linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Resolution impact on file size
    print("RESOLUTION IMPACT ON DATA")
    print("=" * 30)
    for res_name, size in resolutions.items():
        pixels = size * size
        # Assume 4 bytes per pixel (float32)
        size_mb = (pixels * 4) / (1024 * 1024)
        print(f"{res_name}:")
        print(f"  Pixels: {pixels:,}")
        print(f"  File size (single band): {size_mb:.3f} MB")
        print(f"  File size (11 bands): {size_mb * 11:.1f} MB")
        print()

# Demonstrate resolution concepts
demonstrate_resolution_concepts()
```

### NoData: Handling Missing Values
```python
def demonstrate_nodata_handling():
    """Demonstrate NoData concepts and handling"""
    
    # Create sample data with various NoData scenarios
    height, width = 100, 100
    data = np.random.rand(height, width) * 100
    
    # Introduce different types of NoData
    scenarios = {
        'Ocean/Water Bodies': (data < 20),  # Water areas
        'Cloud Cover': ((data > 40) & (data < 60)),  # Cloudy areas
        'Sensor Malfunction': (np.random.random((height, width)) < 0.05),  # Random failures
        'Outside Scene': ((np.arange(height)[:, None] < 10) | 
                         (np.arange(width)[None, :] < 10))  # Scene edges
    }
    
    # Common NoData values by data type
    nodata_values = {
        'uint8': 255,
        'int16': -32768,
        'uint16': 65535,
        'float32': np.nan,
        'float64': np.nan
    }
    
    print("COMMON NODATA VALUES BY DATA TYPE")
    print("=" * 40)
    for dtype, nodata in nodata_values.items():
        print(f"{dtype:8s}: {nodata}")
    
    # Visualize NoData scenarios
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original data
    im0 = axes[0].imshow(data, cmap='viridis')
    axes[0].set_title('Original Data')
    plt.colorbar(im0, ax=axes[0])
    
    # Apply NoData scenarios
    for i, (scenario_name, mask) in enumerate(scenarios.items(), 1):
        data_with_nodata = data.copy()
        data_with_nodata[mask] = np.nan
        
        im = axes[i].imshow(data_with_nodata, cmap='viridis')
        axes[i].set_title(f'{scenario_name}\nNoData: {mask.sum()} pixels')
        plt.colorbar(im, ax=axes[i])
    
    # Combined NoData
    combined_mask = np.zeros_like(data, dtype=bool)
    for mask in scenarios.values():
        combined_mask |= mask
    
    data_combined = data.copy()
    data_combined[combined_mask] = np.nan
    
    im5 = axes[5].imshow(data_combined, cmap='viridis')
    axes[5].set_title(f'All NoData Combined\nValid: {(~combined_mask).sum()} pixels')
    plt.colorbar(im5, ax=axes[5])
    
    plt.tight_layout()
    plt.show()
    
    # NoData statistics
    print("\nNODATA IMPACT ANALYSIS")
    print("=" * 25)
    total_pixels = height * width
    valid_pixels = (~combined_mask).sum()
    nodata_pixels = combined_mask.sum()
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
    print(f"NoData pixels: {nodata_pixels:,} ({nodata_pixels/total_pixels*100:.1f}%)")
    
    return data_combined, combined_mask

# Demonstrate NoData handling
processed_data, nodata_mask = demonstrate_nodata_handling()
```

---

## 4️⃣ Real-World Examples

### Land Cover Classification
```python
def create_land_cover_example():
    """Create and analyze a land cover classification raster"""
    
    # Define land cover classes
    land_cover_classes = {
        0: {'name': 'No Data', 'color': '#000000', 'description': 'No data available'},
        1: {'name': 'Water', 'color': '#0066CC', 'description': 'Rivers, lakes, oceans'},
        2: {'name': 'Urban', 'color': '#CC0000', 'description': 'Built-up areas, roads'},
        3: {'name': 'Forest', 'color': '#006600', 'description': 'Dense tree cover'},
        4: {'name': 'Grassland', 'color': '#66CC00', 'description': 'Grass, pasture'},
        5: {'name': 'Cropland', 'color': '#FFCC00', 'description': 'Agricultural fields'},
        6: {'name': 'Barren', 'color': '#CC9900', 'description': 'Bare soil, rock'},
        7: {'name': 'Wetland', 'color': '#0099CC', 'description': 'Marshes, swamps'}
    }
    
    # Create synthetic land cover map
    height, width = 200, 200
    
    # Generate realistic land cover patterns
    x = np.linspace(0, 20, width)
    y = np.linspace(0, 20, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize with grassland
    land_cover = np.full((height, width), 4, dtype=np.uint8)
    
    # Add water bodies (rivers and lakes)
    river_mask = (np.abs(Y - 10 - 2*np.sin(X/2)) < 0.5)
    lake_mask = ((X-15)**2 + (Y-5)**2) < 4
    land_cover[river_mask | lake_mask] = 1
    
    # Add urban areas
    urban_mask = ((X-5)**2 + (Y-15)**2) < 9
    land_cover[urban_mask] = 2
    
    # Add forest
    forest_mask = ((X > 12) & (Y > 12)) | ((X < 8) & (Y < 8))
    land_cover[forest_mask] = 3
    
    # Add cropland
    crop_mask = ((X > 8) & (X < 15) & (Y > 8) & (Y < 12))
    land_cover[crop_mask] = 5
    
    # Add barren areas
    barren_mask = ((X > 15) & (Y < 5))
    land_cover[barren_mask] = 6
    
    # Add wetlands near water
    wetland_mask = ((X-15)**2 + (Y-5)**2 > 4) & ((X-15)**2 + (Y-5)**2 < 9)
    land_cover[wetland_mask] = 7
    
    # Create custom colormap
    colors = [land_cover_classes[i]['color'] for i in range(8)]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Visualize land cover
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Land cover map
    im1 = ax1.imshow(land_cover, cmap=cmap, vmin=0, vmax=7)
    ax1.set_title('Land Cover Classification')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=land_cover_classes[i]['color'], 
                           label=land_cover_classes[i]['name']) 
                      for i in range(1, 8)]  # Skip NoData
    ax1.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Statistics
    unique, counts = np.unique(land_cover, return_counts=True)
    total_pixels = height * width
    
    stats_data = []
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        stats_data.append([
            land_cover_classes[class_id]['name'],
            count,
            f"{percentage:.1f}%"
        ])
    
    # Plot statistics
    class_names = [row[0] for row in stats_data]
    percentages = [float(row[2].rstrip('%')) for row in stats_data]
    
    ax2.pie(percentages, labels=class_names, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Land Cover Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("LAND COVER STATISTICS")
    print("=" * 30)
    print(f"{'Class':<12} {'Pixels':<8} {'Percentage':<10} {'Area (km²)':<10}")
    print("-" * 45)
    
    pixel_area = (20/width) * (20/height)  # km² per pixel
    
    for class_id, count in zip(unique, counts):
        name = land_cover_classes[class_id]['name']
        percentage = (count / total_pixels) * 100
        area_km2 = count * pixel_area
        print(f"{name:<12} {count:<8} {percentage:<9.1f}% {area_km2:<9.1f}")
    
    return land_cover, land_cover_classes

# Create land cover example
lc_data, lc_classes = create_land_cover_example()
```

### NDVI (Vegetation Index)
```python
def create_ndvi_example():
    """Create and analyze NDVI data"""
    
    # Simulate Red and Near-Infrared bands
    height, width = 150, 150
    
    # Create realistic vegetation patterns
    x = np.linspace(0, 15, width)
    y = np.linspace(0, 15, height)
    X, Y = np.meshgrid(x, y)
    
    # Simulate different vegetation types
    # Dense forest (high NDVI)
    forest_mask = ((X-4)**2 + (Y-11)**2) < 9
    
    # Agricultural fields (medium-high NDVI)
    crop_mask = ((X > 8) & (X < 13) & (Y > 2) & (Y < 8))
    
    # Grassland (medium NDVI)
    grass_mask = ((X < 6) & (Y < 6)) | ((X > 10) & (Y > 10))
    
    # Urban/bare soil (low NDVI)
    urban_mask = ((X-12)**2 + (Y-12)**2) < 4
    
    # Water (negative NDVI)
    water_mask = (np.abs(Y - 7.5 - np.sin(X)) < 0.8)
    
    # Create Red and NIR bands
    # Red band (higher values for vegetation, lower for water)
    red_band = np.full((height, width), 0.3)  # Base reflectance
    red_band[forest_mask] = 0.1  # Low red reflectance for dense vegetation
    red_band[crop_mask] = 0.15
    red_band[grass_mask] = 0.2
    red_band[urban_mask] = 0.25
    red_band[water_mask] = 0.05
    
    # NIR band (much higher for vegetation)
    nir_band = np.full((height, width), 0.3)  # Base reflectance
    nir_band[forest_mask] = 0.8  # High NIR for dense vegetation
    nir_band[crop_mask] = 0.6
    nir_band[grass_mask] = 0.45
    nir_band[urban_mask] = 0.3
    nir_band[water_mask] = 0.02  # Very low NIR for water
    
    # Add realistic noise
    red_band += np.random.normal(0, 0.02, (height, width))
    nir_band += np.random.normal(0, 0.02, (height, width))
    
    # Calculate NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    
    # Handle division by zero
    ndvi = np.where((nir_band + red_band) == 0, 0, ndvi)
    
    # Visualize bands and NDVI
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Red band
    im1 = axes[0,0].imshow(red_band, cmap='Reds', vmin=0, vmax=0.4)
    axes[0,0].set_title('Red Band Reflectance')
    plt.colorbar(im1, ax=axes[0,0])
    
    # NIR band
    im2 = axes[0,1].imshow(nir_band, cmap='RdYlBu_r', vmin=0, vmax=0.8)
    axes[0,1].set_title('Near-Infrared Band Reflectance')
    plt.colorbar(im2, ax=axes[0,1])
    
    # NDVI
    im3 = axes[1,0].imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    axes[1,0].set_title('NDVI (Normalized Difference Vegetation Index)')
    plt.colorbar(im3, ax=axes[1,0])
    
    # NDVI histogram
    axes[1,1].hist(ndvi.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].set_xlabel('NDVI Value')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('NDVI Distribution')
    axes[1,1].axvline(0, color='red', linestyle='--', label='Water threshold')
    axes[1,1].axvline(0.2, color='orange', linestyle='--', label='Sparse vegetation')
    axes[1,1].axvline(0.5, color='green', linestyle='--', label='Dense vegetation')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # NDVI interpretation
    ndvi_classes = {
        'Water/Snow': (ndvi < 0),
        'Bare Soil/Rock': ((ndvi >= 0) & (ndvi < 0.2)),
        'Sparse Vegetation': ((ndvi >= 0.2) & (ndvi < 0.4)),
        'Moderate Vegetation': ((ndvi >= 0.4) & (ndvi < 0.6)),
        'Dense Vegetation': (ndvi >= 0.6)
    }
    
    print("NDVI ANALYSIS RESULTS")
    print("=" * 25)
    print(f"{'Class':<20} {'Pixels':<8} {'Percentage':<10} {'NDVI Range':<12}")
    print("-" * 55)
    
    total_pixels = height * width
    
    for class_name, mask in ndvi_classes.items():
        count = mask.sum()
        percentage = (count / total_pixels) * 100
        if count > 0:
            ndvi_range = f"{ndvi[mask].min():.2f} - {ndvi[mask].max():.2f}"
        else:
            ndvi_range = "N/A"
        print(f"{class_name:<20} {count:<8} {percentage:<9.1f}% {ndvi_range:<12}")
    
    print(f"\nOverall NDVI Statistics:")
    print(f"  Mean: {ndvi.mean():.3f}")
    print(f"  Std:  {ndvi.std():.3f}")
    print(f"  Min:  {ndvi.min():.3f}")
    print(f"  Max:  {ndvi.max():.3f}")
    
    return red_band, nir_band, ndvi

# Create NDVI example
red, nir, ndvi_data = create_ndvi_example()
```

### Elevation Analysis
```python
def create_elevation_analysis():
    """Create comprehensive elevation analysis"""
    
    # Create realistic topographic data
    height, width = 200, 200
    x = np.linspace(0, 50, width)  # 50km x 50km area
    y = np.linspace(0, 50, height)
    X, Y = np.meshgrid(x, y)
    
    # Create complex terrain
    elevation = (
        1000 +  # Base elevation (1000m)
        800 * np.exp(-((X-15)**2 + (Y-35)**2) / 50) +  # Major peak
        400 * np.exp(-((X-35)**2 + (Y-15)**2) / 30) +  # Secondary peak
        200 * np.exp(-((X-25)**2 + (Y-25)**2) / 20) +  # Hill
        -300 * np.exp(-((X-40)**2 + (Y-40)**2) / 25) + # Valley
        100 * np.sin(X/5) * np.cos(Y/8) +  # Rolling terrain
        50 * np.random.normal(0, 1, (height, width))  # Natural variation
    )
    
    # Ensure realistic elevation range
    elevation = np.clip(elevation, 200, 2500)
    
    # Calculate derived products
    # Slope (gradient magnitude)
    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx**2 + gy**2)
    slope_degrees = np.arctan(slope) * 180 / np.pi
    
    # Aspect (slope direction)
    aspect = np.arctan2(-gx, gy) * 180 / np.pi
    aspect = (aspect + 360) % 360  # Convert to 0-360 degrees
    
    # Hillshade (terrain visualization)
    def calculate_hillshade(elevation, azimuth=315, altitude=45):
        """Calculate hillshade for terrain visualization"""
        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)
        
        gy, gx = np.gradient(elevation)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        aspect_rad = np.arctan2(-gx, gy)
        
        hillshade = np.sin(altitude_rad) * np.sin(slope_rad) + \
                   np.cos(altitude_rad) * np.cos(slope_rad) * \
                   np.cos(azimuth_rad - aspect_rad)
        
        return np.clip(hillshade, 0, 1)
    
    hillshade = calculate_hillshade(elevation)
    
    # Visualize elevation analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Elevation
    im1 = axes[0,0].imshow(elevation, cmap='terrain', extent=[0, 50, 0, 50])
    axes[0,0].set_title('Elevation (m)')
    axes[0,0].set_xlabel('Distance (km)')
    axes[0,0].set_ylabel('Distance (km)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Slope
    im2 = axes[0,1].imshow(slope_degrees, cmap='YlOrRd', extent=[0, 50, 0, 50])
    axes[0,1].set_title('Slope (degrees)')
    axes[0,1].set_xlabel('Distance (km)')
    axes[0,1].set_ylabel('Distance (km)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Aspect
    im3 = axes[0,2].imshow(aspect, cmap='hsv', extent=[0, 50, 0, 50], vmin=0, vmax=360)
    axes[0,2].set_title('Aspect (degrees)')
    axes[0,2].set_xlabel('Distance (km)')
    axes[0,2].set_ylabel('Distance (km)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Hillshade
    im4 = axes[1,0].imshow(hillshade, cmap='gray', extent=[0, 50, 0, 50])
    axes[1,0].set_title('Hillshade')
    axes[1,0].set_xlabel('Distance (km)')
    axes[1,0].set_ylabel('Distance (km)')
    plt.colorbar(im4, ax=axes[1,0])
    
    # Elevation profile
    center_row = height // 2
    profile = elevation[center_row, :]
    axes[1,1].plot(x, profile, 'b-', linewidth=2)
    axes[1,1].set_title('Elevation Profile (Center Line)')
    axes[1,1].set_xlabel('Distance (km)')
    axes[1,1].set_ylabel('Elevation (m)')
    axes[1,1].grid(True, alpha=0.3)
    
    # Elevation histogram
    axes[1,2].hist(elevation.flatten(), bins=50, alpha=0.7, color='brown', edgecolor='black')
    axes[1,2].set_xlabel('Elevation (m)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Elevation Distribution')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Terrain statistics
    print("TERRAIN ANALYSIS RESULTS")
    print("=" * 30)
    
    print("Elevation Statistics:")
    print(f"  Minimum: {elevation.min():.1f} m")
    print(f"  Maximum: {elevation.max():.1f} m")
    print(f"  Mean: {elevation.mean():.1f} m")
    print(f"  Std Dev: {elevation.std():.1f} m")
    print(f"  Relief: {elevation.max() - elevation.min():.1f} m")
    
    print("\nSlope Statistics:")
    print(f"  Mean slope: {slope_degrees.mean():.1f}°")
    print(f"  Max slope: {slope_degrees.max():.1f}°")
    print(f"  Steep areas (>30°): {(slope_degrees > 30).sum()} pixels ({(slope_degrees > 30).mean()*100:.1f}%)")
    
    # Elevation zones
    zones = {
        'Valley': (elevation < 800),
        'Lowland': ((elevation >= 800) & (elevation < 1200)),
        'Highland': ((elevation >= 1200) & (elevation < 1600)),
        'Mountain': (elevation >= 1600)
    }
    
    print("\nElevation Zones:")
    total_pixels = height * width
    for zone_name, mask in zones.items():
        count = mask.sum()
        percentage = (count / total_pixels) * 100
        print(f"  {zone_name}: {count} pixels ({percentage:.1f}%)")
    
    return elevation, slope_degrees, aspect, hillshade

# Create elevation analysis
elev_data, slope_data, aspect_data, hillshade_data = create_elevation_analysis()
```

---

## 🔗 Data Sources and Getting Started

### Free Raster Data Sources
```python
def list_data_sources():
    """Comprehensive list of free raster data sources"""
    
    data_sources = {
        'Satellite Imagery': {
            'Landsat (USGS)': {
                'url': 'https://earthexplorer.usgs.gov/',
                'data_types': ['Multispectral imagery', 'Thermal', 'Panchromatic'],
                'resolution': '15-100m',
                'temporal': '1972-present',
                'coverage': 'Global'
            },
            'Sentinel (ESA)': {
                'url': 'https://scihub.copernicus.eu/',
                'data_types': ['Optical', 'Radar', 'Atmospheric'],
                'resolution': '10-60m',
                'temporal': '2014-present',
                'coverage': 'Global'
            },
            'MODIS (NASA)': {
                'url': 'https://modis.gsfc.nasa.gov/',
                'data_types': ['Land', 'Ocean', 'Atmosphere'],
                'resolution': '250m-1km',
                'temporal': '2000-present',
                'coverage': 'Global'
            }
        },
        'Elevation Data': {
            'SRTM (NASA)': {
                'url': 'https://www2.jpl.nasa.gov/srtm/',
                'data_types': ['Digital Elevation Model'],
                'resolution': '30m, 90m',
                'temporal': '2000',
                'coverage': '60°N-56°S'
            },
            'ASTER GDEM': {
                'url': 'https://asterweb.jpl.nasa.gov/gdem.asp',
                'data_types': ['Digital Elevation Model'],
                'resolution': '30m',
                'temporal': '2000-2013',
                'coverage': '83°N-83°S'
            }
        },
        'Climate Data': {
            'WorldClim': {
                'url': 'https://worldclim.org/',
                'data_types': ['Temperature', 'Precipitation', 'Bioclimatic'],
                'resolution': '30s-10m',
                'temporal': 'Current, Future scenarios',
                'coverage': 'Global'
            },
            'ECMWF ERA5': {
                'url': 'https://cds.climate.copernicus.eu/',
                'data_types': ['Reanalysis data', 'Weather variables'],
                'resolution': '0.25°',
                'temporal': '1979-present',
                'coverage': 'Global'
            }
        }
    }
    
    print("FREE RASTER DATA SOURCES")
    print("=" * 30)
    
    for category, sources in data_sources.items():
        print(f"\n{category.upper()}")
        print("-" * len(category))
        
        for source_name, details in sources.items():
            print(f"\n{source_name}:")
            print(f"  URL: {details['url']}")
            print(f"  Data Types: {', '.join(details['data_types'])}")
            print(f"  Resolution: {details['resolution']}")
            print(f"  Temporal: {details['temporal']}")
            print(f"  Coverage: {details['coverage']}")

# Display data sources
list_data_sources()
```

### Quick Start Code Template
```python
def raster_analysis_template():
    """Template for basic raster analysis workflow"""
    
    template_code = '''
# Basic Raster Analysis Template
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt

# 1. Load raster data
raster = rxr.open_rasterio('your_raster_file.tif')

# 2. Explore metadata
print(f"Shape: {raster.shape}")
print(f"CRS: {raster.rio.crs}")
print(f"Resolution: {raster.rio.resolution()}")
print(f"Bounds: {raster.rio.bounds()}")

# 3. Basic statistics
print(f"Min: {raster.min().values}")
print(f"Max: {raster.max().values}")
print(f"Mean: {raster.mean().values}")

# 4. Handle NoData
raster_clean = raster.where(raster != raster.rio.nodata)

# 5. Visualize
fig, ax = plt.subplots(figsize=(10, 8))
raster_clean.plot(ax=ax, cmap='viridis')
ax.set_title('Raster Data Visualization')
plt.show()

# 6. Save processed data
raster_clean.rio.to_raster('processed_raster.tif')
'''
    
    print("RASTER ANALYSIS TEMPLATE")
    print("=" * 25)
    print(template_code)

# Display template
raster_analysis_template()
```

---

## 🎯 Key Takeaways

1. **Raster Fundamentals**: Grid-based data representing continuous or discrete spatial phenomena
2. **Data Types**: Imagery (multi-spectral), grids (continuous surfaces), DEMs (elevation)
3. **Format Selection**: Choose based on use case - GeoTIFF for general use, NetCDF for climate, HDF for scientific data
4. **Structure Understanding**: Bands carry different information, resolution affects detail, NoData requires careful handling
5. **Real Applications**: Land cover mapping, vegetation monitoring, terrain analysis
6. **Data Access**: Abundant free sources available for practice and research

This foundation prepares you for advanced raster analysis, enabling confident work with satellite imagery, environmental data, and spatial modeling applications.