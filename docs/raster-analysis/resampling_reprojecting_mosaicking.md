# Resampling, Reprojecting, and Mosaicking

## Overview
Resampling, reprojecting, and mosaicking are essential transformations in raster processing that enable data integration, analysis standardization, and seamless dataset combination. These operations ensure that rasters from different sources, resolutions, and coordinate systems can work together effectively in spatial analysis workflows.

## Why These Operations Matter
- **Data Integration**: Combine datasets from different sensors and sources
- **Analysis Standardization**: Ensure consistent resolution and projection across datasets
- **Scale Matching**: Align data to appropriate resolution for analysis requirements
- **Seamless Coverage**: Create continuous datasets from multiple tiles or scenes
- **Performance Optimization**: Optimize resolution for computational efficiency

---

## 1️⃣ Resampling Methods: nearest, bilinear, cubic

### Understanding Resampling Methods
Resampling determines how pixel values are calculated when changing raster resolution or geometry.

```python
import rioxarray as rxr
import rasterio
from rasterio.enums import Resampling
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# Create sample high-resolution raster
def create_sample_raster():
    """Create high-resolution sample raster for resampling demonstration"""
    
    # High resolution grid (200x200)
    height, width = 200, 200
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Create detailed pattern with sharp edges and gradients
    data = (
        100 * np.sin(X * 2) * np.cos(Y * 2) +  # Smooth waves
        50 * ((X > 3) & (X < 7) & (Y > 3) & (Y < 7)) +  # Sharp rectangular feature
        25 * np.exp(-((X-2)**2 + (Y-8)**2) / 0.5) +  # Sharp peak
        10 * np.random.random((height, width))  # Noise
    )
    
    # Convert to xarray
    raster = xr.DataArray(
        data,
        coords={'y': y, 'x': x},
        dims=['y', 'x']
    )
    raster.rio.write_crs('EPSG:4326', inplace=True)
    
    return raster

# Demonstrate different resampling methods
def compare_resampling_methods():
    """Compare different resampling methods"""
    
    # Create high-resolution source
    high_res = create_sample_raster()
    
    # Target lower resolution (50x50 - 4x downsampling)
    target_height, target_width = 50, 50
    
    # Resampling methods to compare
    methods = {
        'Nearest Neighbor': Resampling.nearest,
        'Bilinear': Resampling.bilinear,
        'Cubic': Resampling.cubic,
        'Average': Resampling.average
    }
    
    resampled_results = {}
    
    # Apply each resampling method
    for method_name, resampling_enum in methods.items():
        resampled = high_res.rio.reproject(
            high_res.rio.crs,
            shape=(target_height, target_width),
            resampling=resampling_enum
        )
        resampled_results[method_name] = resampled
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original high resolution
    high_res.plot(ax=axes[0], cmap='viridis')
    axes[0].set_title(f'Original High Resolution\n{high_res.shape}')
    
    # Resampled versions
    for i, (method_name, resampled) in enumerate(resampled_results.items(), 1):
        resampled.plot(ax=axes[i], cmap='viridis')
        axes[i].set_title(f'{method_name}\n{resampled.shape}')
    
    # Hide unused subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Method characteristics
    method_info = {
        'Nearest Neighbor': {
            'description': 'Uses value of nearest pixel',
            'best_for': 'Categorical data, land cover',
            'preserves': 'Original values exactly',
            'artifacts': 'Blocky appearance'
        },
        'Bilinear': {
            'description': 'Linear interpolation between 4 nearest pixels',
            'best_for': 'Continuous data, DEMs',
            'preserves': 'Smooth transitions',
            'artifacts': 'Some blurring'
        },
        'Cubic': {
            'description': 'Cubic interpolation using 16 nearest pixels',
            'best_for': 'High-quality imagery',
            'preserves': 'Sharp edges and details',
            'artifacts': 'Possible overshooting'
        },
        'Average': {
            'description': 'Average of all pixels in target area',
            'best_for': 'Downsampling continuous data',
            'preserves': 'Statistical properties',
            'artifacts': 'Smoothing effect'
        }
    }
    
    print("RESAMPLING METHOD COMPARISON")
    print("=" * 35)
    
    for method, info in method_info.items():
        print(f"\n{method}:")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print(f"  Preserves: {info['preserves']}")
        print(f"  Artifacts: {info['artifacts']}")
    
    return high_res, resampled_results

# Compare resampling methods
original_raster, resampling_comparison = compare_resampling_methods()
```

### Choosing the Right Resampling Method
```python
def resampling_use_cases():
    """Demonstrate appropriate resampling method selection"""
    
    # Create different data types
    datasets = {}
    
    # 1. Land cover (categorical)
    land_cover = np.random.choice([1, 2, 3, 4, 5], size=(100, 100))
    datasets['Land Cover'] = xr.DataArray(
        land_cover,
        coords={'y': np.linspace(0, 10, 100), 'x': np.linspace(0, 10, 100)},
        dims=['y', 'x']
    )
    
    # 2. Temperature (continuous)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    temperature = 20 + 10 * np.sin(X) * np.cos(Y) + 2 * np.random.random((100, 100))
    datasets['Temperature'] = xr.DataArray(
        temperature,
        coords={'y': y, 'x': x},
        dims=['y', 'x']
    )
    
    # 3. Elevation (continuous with sharp features)
    elevation = 1000 + 500 * np.exp(-((X-5)**2 + (Y-5)**2) / 2) + 100 * np.random.random((100, 100))
    datasets['Elevation'] = xr.DataArray(
        elevation,
        coords={'y': y, 'x': x},
        dims=['y', 'x']
    )
    
    # Set CRS for all datasets
    for dataset in datasets.values():
        dataset.rio.write_crs('EPSG:4326', inplace=True)
    
    # Recommended resampling methods
    recommendations = {
        'Land Cover': Resampling.nearest,
        'Temperature': Resampling.bilinear,
        'Elevation': Resampling.cubic
    }
    
    # Apply recommended resampling
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, (data_type, dataset) in enumerate(datasets.items()):
        # Original
        dataset.plot(ax=axes[i, 0], cmap='viridis' if data_type != 'Land Cover' else 'tab10')
        axes[i, 0].set_title(f'{data_type} - Original')
        
        # Recommended method
        recommended = dataset.rio.reproject(
            dataset.rio.crs,
            shape=(25, 25),  # 4x downsampling
            resampling=recommendations[data_type]
        )
        recommended.plot(ax=axes[i, 1], cmap='viridis' if data_type != 'Land Cover' else 'tab10')
        axes[i, 1].set_title(f'{data_type} - Recommended\n({recommendations[data_type].name})')
        
        # Wrong method for comparison
        wrong_method = Resampling.cubic if data_type == 'Land Cover' else Resampling.nearest
        wrong_result = dataset.rio.reproject(
            dataset.rio.crs,
            shape=(25, 25),
            resampling=wrong_method
        )
        wrong_result.plot(ax=axes[i, 2], cmap='viridis' if data_type != 'Land Cover' else 'tab10')
        axes[i, 2].set_title(f'{data_type} - Wrong Method\n({wrong_method.name})')
    
    plt.tight_layout()
    plt.show()
    
    # Guidelines
    print("RESAMPLING METHOD SELECTION GUIDELINES")
    print("=" * 40)
    print("\nCategorical Data (Land Cover, Classifications):")
    print("  ✓ Use: Nearest Neighbor")
    print("  ✗ Avoid: Bilinear, Cubic (creates invalid intermediate values)")
    
    print("\nContinuous Data (Temperature, Precipitation):")
    print("  ✓ Use: Bilinear (good balance of quality and speed)")
    print("  ✓ Alternative: Cubic (higher quality, slower)")
    
    print("\nElevation Data (DEMs):")
    print("  ✓ Use: Cubic (preserves terrain features)")
    print("  ✓ Alternative: Bilinear (faster, acceptable quality)")
    
    print("\nDownsampling (reducing resolution):")
    print("  ✓ Use: Average (preserves statistical properties)")
    print("  ✓ Alternative: Bilinear")
    
    return datasets, recommendations

# Demonstrate use case selection
sample_datasets, method_recommendations = resampling_use_cases()
```

---

## 2️⃣ Changing Pixel Size with Rasterio

### Basic Resolution Changes
```python
def change_pixel_size():
    """Demonstrate changing pixel size with different approaches"""
    
    # Create sample raster with known resolution
    original = create_sample_raster()
    
    # Current resolution
    current_res = original.rio.resolution()
    print(f"Original resolution: {current_res}")
    
    # Method 1: Change by factor
    def resize_by_factor(raster, factor):
        """Resize raster by multiplication factor"""
        new_height = int(raster.sizes['y'] * factor)
        new_width = int(raster.sizes['x'] * factor)
        
        return raster.rio.reproject(
            raster.rio.crs,
            shape=(new_height, new_width),
            resampling=Resampling.bilinear
        )
    
    # Method 2: Change to specific resolution
    def resize_to_resolution(raster, target_resolution):
        """Resize raster to specific resolution"""
        return raster.rio.reproject(
            raster.rio.crs,
            resolution=target_resolution,
            resampling=Resampling.bilinear
        )
    
    # Method 3: Change to specific dimensions
    def resize_to_dimensions(raster, target_width, target_height):
        """Resize raster to specific dimensions"""
        return raster.rio.reproject(
            raster.rio.crs,
            shape=(target_height, target_width),
            resampling=Resampling.bilinear
        )
    
    # Apply different resizing methods
    resize_results = {}
    
    # Upsampling (2x finer resolution)
    resize_results['2x Upsampled'] = resize_by_factor(original, 2.0)
    
    # Downsampling (0.5x coarser resolution)
    resize_results['2x Downsampled'] = resize_by_factor(original, 0.5)
    
    # Specific resolution
    resize_results['0.1° Resolution'] = resize_to_resolution(original, 0.1)
    
    # Specific dimensions
    resize_results['Fixed 150x150'] = resize_to_dimensions(original, 150, 150)
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original
    original.plot(ax=axes[0], cmap='viridis')
    axes[0].set_title(f'Original\n{original.shape}\nRes: {current_res}')
    
    # Resized versions
    for i, (name, resized) in enumerate(resize_results.items(), 1):
        resized.plot(ax=axes[i], cmap='viridis')
        new_res = resized.rio.resolution()
        axes[i].set_title(f'{name}\n{resized.shape}\nRes: {new_res}')
    
    # Hide unused subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Resolution comparison table
    print("RESOLUTION CHANGE COMPARISON")
    print("=" * 35)
    print(f"{'Method':<20} {'Shape':<12} {'Resolution':<15} {'File Size':<10}")
    print("-" * 65)
    
    # Original
    orig_size = original.nbytes / (1024**2)
    print(f"{'Original':<20} {str(original.shape):<12} {str(current_res):<15} {orig_size:.1f} MB")
    
    # Resized versions
    for name, resized in resize_results.items():
        res = resized.rio.resolution()
        size = resized.nbytes / (1024**2)
        print(f"{name:<20} {str(resized.shape):<12} {str(res):<15} {size:.1f} MB")
    
    return resize_results

# Demonstrate pixel size changes
resize_examples = change_pixel_size()
```

### Advanced Resolution Matching
```python
def match_raster_resolutions():
    """Match multiple rasters to common resolution"""
    
    # Create rasters with different resolutions
    rasters = {}
    
    # High resolution (0.02° pixels)
    x_high = np.linspace(0, 5, 250)
    y_high = np.linspace(0, 5, 250)
    X_high, Y_high = np.meshgrid(x_high, y_high)
    high_res_data = np.sin(X_high * 2) * np.cos(Y_high * 2)
    
    rasters['High Resolution'] = xr.DataArray(
        high_res_data,
        coords={'y': y_high, 'x': x_high},
        dims=['y', 'x']
    )
    
    # Medium resolution (0.05° pixels)
    x_med = np.linspace(0, 5, 100)
    y_med = np.linspace(0, 5, 100)
    X_med, Y_med = np.meshgrid(x_med, y_med)
    med_res_data = np.exp(-((X_med-2.5)**2 + (Y_med-2.5)**2) / 2)
    
    rasters['Medium Resolution'] = xr.DataArray(
        med_res_data,
        coords={'y': y_med, 'x': x_med},
        dims=['y', 'x']
    )
    
    # Low resolution (0.1° pixels)
    x_low = np.linspace(0, 5, 50)
    y_low = np.linspace(0, 5, 50)
    X_low, Y_low = np.meshgrid(x_low, y_low)
    low_res_data = X_low + Y_low
    
    rasters['Low Resolution'] = xr.DataArray(
        low_res_data,
        coords={'y': y_low, 'x': x_low},
        dims=['y', 'x']
    )
    
    # Set CRS for all
    for raster in rasters.values():
        raster.rio.write_crs('EPSG:4326', inplace=True)
    
    # Method 1: Match to reference raster
    reference = rasters['Medium Resolution']
    matched_to_reference = {}
    
    for name, raster in rasters.items():
        if name != 'Medium Resolution':
            matched = raster.rio.reproject_match(
                reference,
                resampling=Resampling.bilinear
            )
            matched_to_reference[name] = matched
    
    # Method 2: Match to common resolution
    target_resolution = 0.05  # degrees
    matched_to_resolution = {}
    
    for name, raster in rasters.items():
        matched = raster.rio.reproject(
            raster.rio.crs,
            resolution=target_resolution,
            resampling=Resampling.bilinear
        )
        matched_to_resolution[name] = matched
    
    # Visualize original vs matched
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Original rasters
    for i, (name, raster) in enumerate(rasters.items()):
        raster.plot(ax=axes[0, i], cmap='viridis')
        axes[0, i].set_title(f'{name} - Original\n{raster.shape}')
    
    # Matched to reference
    axes[1, 0].text(0.5, 0.5, 'Reference\n(Medium Res)', ha='center', va='center', 
                    transform=axes[1, 0].transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    
    for i, (name, matched) in enumerate(matched_to_reference.items(), 1):
        matched.plot(ax=axes[1, i], cmap='viridis')
        axes[1, i].set_title(f'{name} - Matched to Ref\n{matched.shape}')
    
    # Matched to common resolution
    for i, (name, matched) in enumerate(matched_to_resolution.items()):
        matched.plot(ax=axes[2, i], cmap='viridis')
        axes[2, i].set_title(f'{name} - Common Res\n{matched.shape}')
    
    plt.tight_layout()
    plt.show()
    
    # Resolution summary
    print("RESOLUTION MATCHING SUMMARY")
    print("=" * 30)
    print("\nOriginal Resolutions:")
    for name, raster in rasters.items():
        res = raster.rio.resolution()
        print(f"  {name}: {res}")
    
    print(f"\nTarget Resolution: {target_resolution}°")
    print("\nMatched Resolutions:")
    for name, matched in matched_to_resolution.items():
        res = matched.rio.resolution()
        print(f"  {name}: {res}")
    
    return rasters, matched_to_reference, matched_to_resolution

# Demonstrate resolution matching
orig_rasters, ref_matched, res_matched = match_raster_resolutions()
```

---

## 3️⃣ Mosaicking Multiple Rasters with merge()

### Basic Mosaicking Operations
```python
def basic_mosaicking():
    """Demonstrate basic raster mosaicking with merge()"""
    
    from rasterio.merge import merge
    import rasterio
    from rasterio.transform import from_bounds
    
    # Create multiple overlapping raster tiles
    def create_raster_tile(bounds, data_pattern, filename):
        """Create a raster tile with specific bounds and pattern"""
        
        minx, miny, maxx, maxy = bounds
        width, height = 100, 100
        
        # Create coordinate grids
        x = np.linspace(minx, maxx, width)
        y = np.linspace(miny, maxy, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate data based on pattern
        if data_pattern == 'sine':
            data = 100 + 50 * np.sin(X) * np.cos(Y)
        elif data_pattern == 'exp':
            center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2
            data = 100 + 100 * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / 2)
        elif data_pattern == 'linear':
            data = 100 + 20 * X + 10 * Y
        else:
            data = 100 + 50 * np.random.random((height, width))
        
        # Create xarray
        raster = xr.DataArray(
            data,
            coords={'y': y, 'x': x},
            dims=['y', 'x']
        )
        raster.rio.write_crs('EPSG:4326', inplace=True)
        
        return raster
    
    # Create overlapping tiles
    tiles = {
        'Northwest': create_raster_tile((-5, 0, 0, 5), 'sine', 'tile_nw.tif'),
        'Northeast': create_raster_tile((0, 0, 5, 5), 'exp', 'tile_ne.tif'),
        'Southwest': create_raster_tile((-5, -5, 0, 0), 'linear', 'tile_sw.tif'),
        'Southeast': create_raster_tile((0, -5, 5, 0), 'random', 'tile_se.tif')
    }
    
    # Save tiles temporarily for merge operation
    tile_files = []
    for name, tile in tiles.items():
        filename = f'temp_{name.lower()}.tif'
        tile.rio.to_raster(filename)
        tile_files.append(filename)
    
    # Method 1: Simple merge (first raster takes precedence in overlaps)
    try:
        # Open files for merging
        src_files = [rasterio.open(f) for f in tile_files]
        
        # Merge with different methods
        mosaic_first, transform_first = merge(src_files, method='first')
        mosaic_last, transform_last = merge(src_files, method='last')
        mosaic_min, transform_min = merge(src_files, method='min')
        mosaic_max, transform_max = merge(src_files, method='max')
        
        # Close files
        for src in src_files:
            src.close()
        
        # Convert results to xarray for visualization
        def array_to_xarray(array, transform, crs='EPSG:4326'):
            height, width = array.shape[1], array.shape[2]
            x = np.linspace(transform.c, transform.c + width * transform.a, width)
            y = np.linspace(transform.f, transform.f + height * transform.e, height)
            
            return xr.DataArray(
                array[0],  # First band
                coords={'y': y, 'x': x},
                dims=['y', 'x']
            )
        
        mosaics = {
            'First': array_to_xarray(mosaic_first, transform_first),
            'Last': array_to_xarray(mosaic_last, transform_last),
            'Minimum': array_to_xarray(mosaic_min, transform_min),
            'Maximum': array_to_xarray(mosaic_max, transform_max)
        }
        
    except Exception as e:
        print(f"Merge operation failed: {e}")
        # Fallback: manual mosaicking with xarray
        mosaics = manual_mosaic_fallback(tiles)
    
    # Visualize tiles and mosaics
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Individual tiles
    for i, (name, tile) in enumerate(tiles.items()):
        tile.plot(ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Tile: {name}')
    
    # Mosaicked results
    for i, (method, mosaic) in enumerate(mosaics.items(), 4):
        if i < len(axes):
            mosaic.plot(ax=axes[i], cmap='viridis')
            axes[i].set_title(f'Mosaic: {method}')
    
    # Hide unused subplots
    for i in range(len(tiles) + len(mosaics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Clean up temporary files
    import os
    for filename in tile_files:
        try:
            os.remove(filename)
        except:
            pass
    
    return tiles, mosaics

def manual_mosaic_fallback(tiles):
    """Manual mosaicking fallback using xarray"""
    
    # Find common bounds
    all_bounds = []
    for tile in tiles.values():
        bounds = tile.rio.bounds()
        all_bounds.append(bounds)
    
    # Calculate overall extent
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    
    # Create target grid
    target_res = 0.05
    target_width = int((max_x - min_x) / target_res)
    target_height = int((max_y - min_y) / target_res)
    
    # Reproject all tiles to common grid
    reprojected_tiles = []
    for tile in tiles.values():
        reprojected = tile.rio.reproject(
            tile.rio.crs,
            transform=from_bounds(min_x, min_y, max_x, max_y, target_width, target_height),
            shape=(target_height, target_width),
            resampling=Resampling.bilinear
        )
        reprojected_tiles.append(reprojected)
    
    # Simple mosaicking methods
    stacked = xr.concat(reprojected_tiles, dim='tile')
    
    mosaics = {
        'First': reprojected_tiles[0],  # Just first tile
        'Mean': stacked.mean(dim='tile', skipna=True),
        'Maximum': stacked.max(dim='tile', skipna=True),
        'Minimum': stacked.min(dim='tile', skipna=True)
    }
    
    return mosaics

# Demonstrate basic mosaicking
tile_data, mosaic_results = basic_mosaicking()
```

### Advanced Mosaicking Techniques
```python
def advanced_mosaicking():
    """Demonstrate advanced mosaicking techniques"""
    
    # Create time series of raster tiles (e.g., monthly satellite images)
    def create_time_series_tiles():
        """Create time series of overlapping raster tiles"""
        
        import datetime
        
        tiles_by_time = {}
        base_date = datetime.date(2023, 1, 1)
        
        for month in range(1, 5):  # 4 months
            date = base_date.replace(month=month)
            
            # Create seasonal variation
            seasonal_factor = np.sin(2 * np.pi * month / 12)
            
            # Create overlapping tiles for this time period
            tiles = {}
            
            # Tile 1: Northwest
            x1 = np.linspace(-2, 2, 80)
            y1 = np.linspace(0, 4, 80)
            X1, Y1 = np.meshgrid(x1, y1)
            data1 = 100 + 50 * seasonal_factor + 20 * np.sin(X1) * np.cos(Y1) + 10 * np.random.random((80, 80))
            
            tiles['NW'] = xr.DataArray(
                data1, coords={'y': y1, 'x': x1}, dims=['y', 'x']
            )
            
            # Tile 2: Northeast  
            x2 = np.linspace(0, 4, 80)
            y2 = np.linspace(0, 4, 80)
            X2, Y2 = np.meshgrid(x2, y2)
            data2 = 120 + 30 * seasonal_factor + 25 * np.exp(-((X2-2)**2 + (Y2-2)**2) / 2) + 10 * np.random.random((80, 80))
            
            tiles['NE'] = xr.DataArray(
                data2, coords={'y': y2, 'x': x2}, dims=['y', 'x']
            )
            
            # Set CRS
            for tile in tiles.values():
                tile.rio.write_crs('EPSG:4326', inplace=True)
            
            tiles_by_time[date] = tiles
        
        return tiles_by_time
    
    # Create time series
    time_series_tiles = create_time_series_tiles()
    
    # Advanced mosaicking methods
    def weighted_mosaic(tiles, weights=None):
        """Create weighted mosaic"""
        
        if weights is None:
            weights = [1.0] * len(tiles)
        
        # Reproject all to common grid
        reference = list(tiles.values())[0]
        reprojected = []
        
        for tile in tiles.values():
            if tile.shape != reference.shape:
                reprojected_tile = tile.rio.reproject_match(reference)
            else:
                reprojected_tile = tile
            reprojected.append(reprojected_tile)
        
        # Weighted average
        weighted_sum = sum(w * tile for w, tile in zip(weights, reprojected))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum
    
    def feathered_mosaic(tiles, feather_distance=0.2):
        """Create feathered mosaic with smooth transitions"""
        
        # Simple implementation: distance-weighted blending
        reference = list(tiles.values())[0]
        result = reference.copy()
        
        # For demonstration, just average overlapping areas
        for tile in list(tiles.values())[1:]:
            if tile.shape == reference.shape:
                # Simple averaging in overlap areas
                overlap_mask = ~(np.isnan(result) | np.isnan(tile))
                result = xr.where(overlap_mask, (result + tile) / 2, result)
                result = xr.where(np.isnan(result) & ~np.isnan(tile), tile, result)
        
        return result
    
    # Apply advanced mosaicking to one time period
    sample_date = list(time_series_tiles.keys())[0]
    sample_tiles = time_series_tiles[sample_date]
    
    # Different mosaicking approaches
    mosaic_methods = {
        'Simple Average': lambda tiles: sum(tiles.values()) / len(tiles),
        'Weighted (NW=0.7, NE=0.3)': lambda tiles: weighted_mosaic(tiles, [0.7, 0.3]),
        'Feathered': lambda tiles: feathered_mosaic(tiles)
    }
    
    advanced_mosaics = {}
    for method_name, method_func in mosaic_methods.items():
        try:
            mosaic = method_func(sample_tiles)
            advanced_mosaics[method_name] = mosaic
        except Exception as e:
            print(f"Failed to create {method_name} mosaic: {e}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Individual tiles
    for i, (tile_name, tile) in enumerate(sample_tiles.items()):
        tile.plot(ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Tile: {tile_name}')
    
    # Advanced mosaics
    for i, (method_name, mosaic) in enumerate(advanced_mosaics.items(), 2):
        mosaic.plot(ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Mosaic: {method_name}')
    
    # Hide unused subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Temporal mosaicking
    print("TEMPORAL MOSAICKING EXAMPLE")
    print("=" * 30)
    
    # Create temporal composite (e.g., maximum NDVI over time)
    all_mosaics_by_time = {}
    
    for date, tiles in time_series_tiles.items():
        # Simple mosaic for each time period
        mosaic = sum(tiles.values()) / len(tiles)
        all_mosaics_by_time[date] = mosaic
        print(f"Created mosaic for {date}")
    
    # Stack temporal mosaics
    temporal_stack = xr.concat(
        list(all_mosaics_by_time.values()), 
        dim=xr.DataArray(list(all_mosaics_by_time.keys()), dims=['time'])
    )
    
    # Temporal composites
    temporal_max = temporal_stack.max(dim='time')
    temporal_mean = temporal_stack.mean(dim='time')
    temporal_std = temporal_stack.std(dim='time')
    
    # Visualize temporal composites
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    temporal_max.plot(ax=axes[0], cmap='viridis')
    axes[0].set_title('Temporal Maximum')
    
    temporal_mean.plot(ax=axes[1], cmap='viridis')
    axes[1].set_title('Temporal Mean')
    
    temporal_std.plot(ax=axes[2], cmap='viridis')
    axes[2].set_title('Temporal Std Dev')
    
    plt.tight_layout()
    plt.show()
    
    return time_series_tiles, advanced_mosaics, temporal_stack

# Demonstrate advanced mosaicking
time_tiles, advanced_mosaics, temporal_data = advanced_mosaicking()
```

---

## 4️⃣ Updating Metadata after Transformations

### Comprehensive Metadata Management
```python
def metadata_management():
    """Demonstrate comprehensive metadata management after transformations"""
    
    # Create sample raster with rich metadata
    def create_raster_with_metadata():
        """Create raster with comprehensive metadata"""
        
        # Create sample data
        x = np.linspace(-120, -110, 200)
        y = np.linspace(35, 45, 200)
        X, Y = np.meshgrid(x, y)
        
        # Simulate NDVI data
        ndvi_data = 0.3 + 0.4 * np.exp(-((X + 115)**2 + (Y - 40)**2) / 10) + 0.1 * np.random.random((200, 200))
        ndvi_data = np.clip(ndvi_data, -1, 1)  # Valid NDVI range
        
        # Create xarray with metadata
        raster = xr.DataArray(
            ndvi_data,
            coords={'y': y, 'x': x},
            dims=['y', 'x'],
            attrs={
                'long_name': 'Normalized Difference Vegetation Index',
                'standard_name': 'normalized_difference_vegetation_index',
                'units': 'dimensionless',
                'valid_range': [-1.0, 1.0],
                'scale_factor': 1.0,
                'add_offset': 0.0,
                'source': 'Landsat 8 OLI',
                'processing_level': 'L2',
                'creation_date': '2024-01-15T10:30:00Z',
                'spatial_resolution': '30m',
                'temporal_resolution': '16 days',
                'algorithm': 'NDVI = (NIR - Red) / (NIR + Red)',
                'quality_flag': 'good',
                'cloud_cover': 5.2,
                'sun_elevation': 45.6,
                'sensor_zenith': 2.1
            }
        )
        
        # Set CRS and NoData
        raster.rio.write_crs('EPSG:4326', inplace=True)
        raster.rio.write_nodata(-9999, inplace=True)
        
        return raster
    
    # Create original raster
    original_raster = create_raster_with_metadata()
    
    print("ORIGINAL METADATA:")
    print("=" * 20)
    for key, value in original_raster.attrs.items():
        print(f"  {key}: {value}")
    
    # Transformation 1: Resampling
    def update_metadata_after_resampling(raster, new_resolution, resampling_method):
        """Update metadata after resampling"""
        
        # Perform resampling
        resampled = raster.rio.reproject(
            raster.rio.crs,
            resolution=new_resolution,
            resampling=resampling_method
        )
        
        # Update metadata
        new_attrs = raster.attrs.copy()
        new_attrs.update({
            'spatial_resolution': f'{abs(new_resolution * 111000):.0f}m',  # Convert degrees to meters
            'resampling_method': resampling_method.name,
            'resampling_date': '2024-01-15T11:00:00Z',
            'processing_level': 'L3',
            'derived_from': 'Original L2 product',
            'grid_mapping': 'WGS84'
        })
        
        # Update processing history
        if 'processing_history' in new_attrs:
            new_attrs['processing_history'] += f'; Resampled to {new_resolution}° using {resampling_method.name}'
        else:
            new_attrs['processing_history'] = f'Resampled to {new_resolution}° using {resampling_method.name}'
        
        resampled.attrs = new_attrs
        
        return resampled
    
    # Transformation 2: Reprojection
    def update_metadata_after_reprojection(raster, target_crs):
        """Update metadata after reprojection"""
        
        # Perform reprojection
        reprojected = raster.rio.reproject(target_crs)
        
        # Update metadata
        new_attrs = raster.attrs.copy()
        new_attrs.update({
            'coordinate_system': str(target_crs),
            'reprojection_date': '2024-01-15T11:15:00Z',
            'reprojection_method': 'bilinear',
            'original_crs': str(raster.rio.crs)
        })
        
        # Update spatial resolution for projected CRS
        if target_crs.is_projected:
            res = reprojected.rio.resolution()
            new_attrs['spatial_resolution'] = f'{abs(res[0]):.0f}m x {abs(res[1]):.0f}m'
        
        # Update processing history
        if 'processing_history' in new_attrs:
            new_attrs['processing_history'] += f'; Reprojected to {target_crs}'
        else:
            new_attrs['processing_history'] = f'Reprojected to {target_crs}'
        
        reprojected.attrs = new_attrs
        
        return reprojected
    
    # Transformation 3: Clipping
    def update_metadata_after_clipping(raster, clip_bounds, clip_method):
        """Update metadata after clipping"""
        
        # Perform clipping (simplified)
        minx, miny, maxx, maxy = clip_bounds
        clipped = raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        
        # Update metadata
        new_attrs = raster.attrs.copy()
        new_attrs.update({
            'clipping_bounds': f'{minx}, {miny}, {maxx}, {maxy}',
            'clipping_method': clip_method,
            'clipping_date': '2024-01-15T11:30:00Z',
            'extent_modified': True,
            'original_extent': f'{raster.rio.bounds()}'
        })
        
        # Update processing history
        if 'processing_history' in new_attrs:
            new_attrs['processing_history'] += f'; Clipped to bounds {clip_bounds}'
        else:
            new_attrs['processing_history'] = f'Clipped to bounds {clip_bounds}'
        
        clipped.attrs = new_attrs
        
        return clipped
    
    # Apply transformations with metadata updates
    transformations = {}
    
    # Resampling
    resampled = update_metadata_after_resampling(
        original_raster, 
        new_resolution=0.01,  # 0.01 degrees
        resampling_method=Resampling.bilinear
    )
    transformations['Resampled'] = resampled
    
    # Reprojection
    reprojected = update_metadata_after_reprojection(
        original_raster,
        target_crs='EPSG:3857'  # Web Mercator
    )
    transformations['Reprojected'] = reprojected
    
    # Clipping
    clipped = update_metadata_after_clipping(
        original_raster,
        clip_bounds=(-118, 37, -112, 43),
        clip_method='bounding_box'
    )
    transformations['Clipped'] = clipped
    
    # Display updated metadata
    for transform_name, transformed_raster in transformations.items():
        print(f"\n{transform_name.upper()} METADATA:")
        print("=" * (len(transform_name) + 10))
        
        # Show key changes
        key_attrs = ['spatial_resolution', 'processing_level', 'processing_history', 
                    'coordinate_system', 'resampling_method', 'clipping_bounds']
        
        for key in key_attrs:
            if key in transformed_raster.attrs:
                print(f"  {key}: {transformed_raster.attrs[key]}")
    
    return original_raster, transformations

# Demonstrate metadata management
original_data, transformed_data = metadata_management()
```

### Export with Complete Metadata
```python
def export_with_complete_metadata():
    """Export rasters with complete metadata preservation"""
    
    # Create sample processed raster
    processed_raster = original_data.copy()
    
    # Add comprehensive processing metadata
    processing_metadata = {
        'title': 'Processed NDVI Dataset',
        'summary': 'NDVI derived from Landsat 8 imagery with quality filtering',
        'keywords': 'NDVI, vegetation, Landsat, remote sensing',
        'creator_name': 'Geospatial Analysis Team',
        'creator_email': 'gis@example.com',
        'institution': 'Research Institute',
        'project': 'Vegetation Monitoring Study',
        'license': 'CC BY 4.0',
        'references': 'Tucker, C.J. (1979). Red and photographic infrared linear combinations for monitoring vegetation',
        'comment': 'Quality controlled and atmospherically corrected',
        'acknowledgment': 'USGS for Landsat data provision',
        'conventions': 'CF-1.8',
        'metadata_link': 'https://example.com/metadata/ndvi_dataset',
        'doi': '10.5194/example-2024-1',
        'version': '1.0',
        'date_created': '2024-01-15T12:00:00Z',
        'date_modified': '2024-01-15T12:00:00Z',
        'geospatial_bounds': str(processed_raster.rio.bounds()),
        'geospatial_bounds_crs': str(processed_raster.rio.crs),
        'time_coverage_start': '2023-06-01T00:00:00Z',
        'time_coverage_end': '2023-08-31T23:59:59Z',
        'processing_software': 'Python rioxarray v0.15.0',
        'processing_environment': 'Linux Ubuntu 22.04'
    }
    
    # Update raster attributes
    processed_raster.attrs.update(processing_metadata)
    
    # Export with different metadata formats
    def export_geotiff_with_tags(raster, filename):
        """Export GeoTIFF with comprehensive tags"""
        
        # Prepare GDAL tags
        gdal_tags = {}
        
        # Map xarray attributes to GDAL tags
        attr_to_tag_mapping = {
            'title': 'TIFFTAG_IMAGEDESCRIPTION',
            'creator_name': 'TIFFTAG_ARTIST',
            'creation_date': 'TIFFTAG_DATETIME',
            'processing_software': 'TIFFTAG_SOFTWARE',
            'summary': 'DESCRIPTION',
            'keywords': 'KEYWORDS',
            'license': 'LICENSE',
            'version': 'VERSION'
        }
        
        for attr_key, tag_key in attr_to_tag_mapping.items():
            if attr_key in raster.attrs:
                gdal_tags[tag_key] = str(raster.attrs[attr_key])
        
        # Add custom tags
        gdal_tags.update({
            'AREA_OR_POINT': 'Area',
            'PROCESSING_LEVEL': raster.attrs.get('processing_level', 'L2'),
            'SENSOR': raster.attrs.get('source', 'Unknown'),
            'ALGORITHM': raster.attrs.get('algorithm', 'Not specified')
        })
        
        # Export with tags
        raster.rio.to_raster(filename, tags=gdal_tags)
        
        return filename
    
    def export_netcdf_with_metadata(raster, filename):
        """Export NetCDF with CF-compliant metadata"""
        
        # Prepare for NetCDF export
        raster_for_nc = raster.copy()
        
        # Add coordinate attributes
        raster_for_nc.x.attrs = {
            'standard_name': 'longitude',
            'long_name': 'longitude',
            'units': 'degrees_east',
            'axis': 'X'
        }
        
        raster_for_nc.y.attrs = {
            'standard_name': 'latitude', 
            'long_name': 'latitude',
            'units': 'degrees_north',
            'axis': 'Y'
        }
        
        # Add grid mapping
        raster_for_nc.attrs['grid_mapping'] = 'crs'
        
        # Export to NetCDF
        raster_for_nc.to_netcdf(filename)
        
        return filename
    
    # Export in different formats
    export_files = {}
    
    try:
        # GeoTIFF with tags
        geotiff_file = export_geotiff_with_tags(processed_raster, 'processed_ndvi.tif')
        export_files['GeoTIFF'] = geotiff_file
        
        # NetCDF with CF metadata
        netcdf_file = export_netcdf_with_metadata(processed_raster, 'processed_ndvi.nc')
        export_files['NetCDF'] = netcdf_file
        
        print("EXPORT SUMMARY")
        print("=" * 15)
        for format_name, filename in export_files.items():
            print(f"✓ {format_name}: {filename}")
        
        # Create metadata report
        metadata_report = f"""
DATASET METADATA REPORT
=======================

Dataset: {processed_raster.attrs.get('title', 'Untitled')}
Version: {processed_raster.attrs.get('version', 'Unknown')}
Created: {processed_raster.attrs.get('date_created', 'Unknown')}

SPATIAL INFORMATION:
- CRS: {processed_raster.rio.crs}
- Bounds: {processed_raster.rio.bounds()}
- Shape: {processed_raster.shape}
- Resolution: {processed_raster.rio.resolution()}

DATA INFORMATION:
- Variable: {processed_raster.attrs.get('long_name', 'Unknown')}
- Units: {processed_raster.attrs.get('units', 'Unknown')}
- Valid Range: {processed_raster.attrs.get('valid_range', 'Unknown')}
- NoData: {processed_raster.rio.nodata}

PROCESSING INFORMATION:
- Source: {processed_raster.attrs.get('source', 'Unknown')}
- Algorithm: {processed_raster.attrs.get('algorithm', 'Unknown')}
- Processing Level: {processed_raster.attrs.get('processing_level', 'Unknown')}
- Software: {processed_raster.attrs.get('processing_software', 'Unknown')}

QUALITY INFORMATION:
- Quality Flag: {processed_raster.attrs.get('quality_flag', 'Unknown')}
- Cloud Cover: {processed_raster.attrs.get('cloud_cover', 'Unknown')}%

CONTACT INFORMATION:
- Creator: {processed_raster.attrs.get('creator_name', 'Unknown')}
- Institution: {processed_raster.attrs.get('institution', 'Unknown')}
- Email: {processed_raster.attrs.get('creator_email', 'Unknown')}

LICENSE: {processed_raster.attrs.get('license', 'Unknown')}
"""
        
        # Save metadata report
        with open('metadata_report.txt', 'w') as f:
            f.write(metadata_report)
        
        print("\n✓ Metadata report: metadata_report.txt")
        
    except Exception as e:
        print(f"Export failed: {e}")
    
    return export_files, processed_raster.attrs

# Demonstrate complete metadata export
exported_files, final_metadata = export_with_complete_metadata()
```

---

## 🔧 Best Practices and Performance Tips

### Optimization Strategies
```python
def optimization_best_practices():
    """Best practices for efficient resampling, reprojection, and mosaicking"""
    
    best_practices = {
        'Resampling': {
            'Choose appropriate method': {
                'Categorical data': 'Use nearest neighbor',
                'Continuous data': 'Use bilinear or cubic',
                'Downsampling': 'Use average for statistical preservation'
            },
            'Performance tips': {
                'Chunk large datasets': 'Use dask chunks for memory management',
                'Batch operations': 'Combine multiple transformations',
                'Cache intermediate results': 'Save processed tiles for reuse'
            }
        },
        'Reprojection': {
            'CRS selection': {
                'Analysis CRS': 'Use equal-area projections for area calculations',
                'Display CRS': 'Use Web Mercator for web mapping',
                'Local analysis': 'Use appropriate UTM zones'
            },
            'Performance optimization': {
                'Minimize reprojections': 'Reproject once to analysis CRS',
                'Use appropriate resampling': 'Match method to data type',
                'Consider target resolution': 'Avoid unnecessary upsampling'
            }
        },
        'Mosaicking': {
            'Preparation': {
                'Align rasters first': 'Ensure consistent CRS and resolution',
                'Handle overlaps': 'Choose appropriate merge method',
                'Quality control': 'Check for seamless transitions'
            },
            'Large datasets': {
                'Process in chunks': 'Use spatial or temporal chunking',
                'Use pyramids': 'Build overviews for large mosaics',
                'Optimize storage': 'Use compression and tiling'
            }
        },
        'Metadata': {
            'Documentation': {
                'Processing history': 'Record all transformation steps',
                'Quality information': 'Include accuracy and limitations',
                'Provenance': 'Track data sources and lineage'
            },
            'Standards compliance': {
                'Use CF conventions': 'For NetCDF exports',
                'Include GDAL tags': 'For GeoTIFF exports',
                'Provide ISO metadata': 'For formal datasets'
            }
        }
    }
    
    print("OPTIMIZATION BEST PRACTICES")
    print("=" * 30)
    
    for category, practices in best_practices.items():
        print(f"\n{category.upper()}:")
        print("-" * len(category))
        
        for subcategory, tips in practices.items():
            print(f"\n  {subcategory}:")
            for tip_name, tip_desc in tips.items():
                print(f"    • {tip_name}: {tip_desc}")
    
    # Performance comparison example
    print("\n\nPERFORMANCE COMPARISON EXAMPLE")
    print("=" * 35)
    
    # Simulate performance metrics
    operations = {
        'Nearest Neighbor Resampling': {'time': 0.5, 'memory': 'Low', 'quality': 'Preserves values'},
        'Bilinear Resampling': {'time': 1.2, 'memory': 'Medium', 'quality': 'Smooth transitions'},
        'Cubic Resampling': {'time': 2.1, 'memory': 'High', 'quality': 'High quality'},
        'Simple Mosaic (First)': {'time': 0.8, 'memory': 'Low', 'quality': 'Sharp boundaries'},
        'Feathered Mosaic': {'time': 3.5, 'memory': 'High', 'quality': 'Smooth blending'}
    }
    
    print(f"{'Operation':<25} {'Time (s)':<10} {'Memory':<10} {'Quality':<20}")
    print("-" * 70)
    
    for operation, metrics in operations.items():
        print(f"{operation:<25} {metrics['time']:<10} {metrics['memory']:<10} {metrics['quality']:<20}")

# Display best practices
optimization_best_practices()
```

---

## 🎯 Key Takeaways

1. **Resampling Method Selection**: Choose based on data type - nearest for categorical, bilinear/cubic for continuous
2. **Resolution Management**: Match resolutions before analysis, consider computational efficiency
3. **Mosaicking Strategy**: Align rasters first, choose appropriate merge method for overlaps
4. **Metadata Preservation**: Maintain complete processing history and quality information
5. **Performance Optimization**: Use chunking for large datasets, minimize transformations
6. **Quality Control**: Validate results, check for artifacts and seamless transitions

These operations form the backbone of raster data preprocessing, enabling seamless integration of multi-source datasets for comprehensive spatial analysis.