# Raster Zonal Statistics

## Overview
Zonal statistics extract summary information from raster data within defined geographic zones (polygons). This technique is fundamental for environmental monitoring, land use analysis, and spatial data summarization, enabling quantitative analysis of continuous raster data across discrete geographic units.

## Why Zonal Statistics Matter
- **Data Summarization**: Convert continuous raster data into manageable statistics
- **Comparative Analysis**: Compare values across different geographic zones
- **Environmental Monitoring**: Track changes in vegetation, temperature, or precipitation by region
- **Land Use Assessment**: Analyze land cover patterns within administrative boundaries
- **Decision Support**: Provide quantitative data for planning and management decisions

---

## 1️⃣ Mean, Sum, Max, Min per Polygon

### Basic Zonal Statistics with rasterstats
```python
import geopandas as gpd
import rasterio
import numpy as np
from rasterstats import zonal_stats
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.plot import show
import xarray as xr
import rioxarray as rxr

# Create sample data for demonstration
def create_sample_data():
    """Create sample raster and polygon data"""
    
    # Create sample raster (NDVI-like data)
    height, width = 200, 200
    x = np.linspace(-10, 10, width)
    y = np.linspace(-10, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Simulate NDVI values (0 to 1)
    ndvi = (
        0.7 * np.exp(-((X-2)**2 + (Y-2)**2) / 8) +  # Forest patch
        0.5 * np.exp(-((X+3)**2 + (Y-1)**2) / 6) +  # Moderate vegetation
        0.3 * np.exp(-((X-1)**2 + (Y+3)**2) / 4) +  # Sparse vegetation
        0.1 + 0.1 * np.random.random((height, width))  # Background + noise
    )
    ndvi = np.clip(ndvi, 0, 1)  # Ensure valid NDVI range
    
    # Save as GeoTIFF
    from rasterio.transform import from_bounds
    transform = from_bounds(-10, -10, 10, 10, width, height)
    
    with rasterio.open(
        'sample_ndvi.tif', 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=ndvi.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(ndvi, 1)
    
    # Create sample polygons
    from shapely.geometry import Polygon
    
    polygons = [
        Polygon([(-8, -8), (-2, -8), (-2, -2), (-8, -2)]),  # Zone 1
        Polygon([(0, 0), (6, 0), (6, 6), (0, 6)]),          # Zone 2
        Polygon([(-4, 2), (2, 2), (2, 8), (-4, 8)]),        # Zone 3
        Polygon([(3, -7), (9, -7), (9, -1), (3, -1)])       # Zone 4
    ]
    
    zones_gdf = gpd.GeoDataFrame({
        'zone_id': ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D'],
        'zone_type': ['Agricultural', 'Forest', 'Mixed', 'Urban'],
        'geometry': polygons
    }, crs='EPSG:4326')
    
    return 'sample_ndvi.tif', zones_gdf

# Basic zonal statistics
def calculate_basic_zonal_stats():
    """Calculate basic statistics for each polygon zone"""
    
    # Create sample data
    raster_path, zones_gdf = create_sample_data()
    
    # Calculate zonal statistics
    stats = zonal_stats(
        zones_gdf,
        raster_path,
        stats=['count', 'min', 'max', 'mean', 'sum', 'std'],
        geojson_out=True
    )
    
    # Convert to GeoDataFrame
    stats_gdf = gpd.GeoDataFrame.from_features(stats, crs=zones_gdf.crs)
    
    # Display results
    print("BASIC ZONAL STATISTICS")
    print("=" * 25)
    
    for idx, row in stats_gdf.iterrows():
        zone_id = zones_gdf.iloc[idx]['zone_id']
        zone_type = zones_gdf.iloc[idx]['zone_type']
        
        print(f"\n{zone_id} ({zone_type}):")
        print(f"  Pixel Count: {row['count']:,}")
        print(f"  Mean NDVI: {row['mean']:.3f}")
        print(f"  Min NDVI: {row['min']:.3f}")
        print(f"  Max NDVI: {row['max']:.3f}")
        print(f"  Sum NDVI: {row['sum']:.1f}")
        print(f"  Std Dev: {row['std']:.3f}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show raster and zones
    with rasterio.open(raster_path) as src:
        show(src, ax=ax1, cmap='RdYlGn', title='NDVI Raster with Zones')
    zones_gdf.boundary.plot(ax=ax1, color='red', linewidth=2)
    zones_gdf.apply(lambda x: ax1.annotate(
        text=x['zone_id'], xy=x.geometry.centroid.coords[0],
        ha='center', fontsize=10, color='white', weight='bold'
    ), axis=1)
    
    # Bar plot of mean values
    zone_names = [zones_gdf.iloc[i]['zone_id'] for i in range(len(stats_gdf))]
    stats_gdf['zone_names'] = zone_names
    stats_gdf.plot(x='zone_names', y='mean', kind='bar', ax=ax2, color='green', alpha=0.7)
    ax2.set_title('Mean NDVI by Zone')
    ax2.set_ylabel('Mean NDVI')
    ax2.set_xlabel('Zone')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return stats_gdf, zones_gdf, raster_path

# Run basic analysis
zonal_results, zones, raster_file = calculate_basic_zonal_stats()
```

### Advanced Statistics and Percentiles
```python
def calculate_advanced_zonal_stats():
    """Calculate advanced statistics including percentiles and custom functions"""
    
    # Use existing sample data
    raster_path, zones_gdf = create_sample_data()
    
    # Advanced statistics with percentiles
    advanced_stats = zonal_stats(
        zones_gdf,
        raster_path,
        stats=['count', 'min', 'max', 'mean', 'median', 'sum', 'std'],
        percentiles=[10, 25, 75, 90],
        geojson_out=True
    )
    
    # Convert to DataFrame for easier handling
    stats_df = pd.DataFrame([feat['properties'] for feat in advanced_stats])
    stats_df['zone_id'] = zones_gdf['zone_id'].values
    stats_df['zone_type'] = zones_gdf['zone_type'].values
    
    # Custom statistics using add_stats parameter
    def coefficient_of_variation(x):
        """Calculate coefficient of variation"""
        return np.std(x) / np.mean(x) if np.mean(x) != 0 else 0
    
    def vegetation_vigor_index(x):
        """Custom vegetation vigor classification"""
        mean_ndvi = np.mean(x)
        if mean_ndvi > 0.7:
            return 'High'
        elif mean_ndvi > 0.4:
            return 'Moderate'
        else:
            return 'Low'
    
    # Calculate custom statistics
    custom_stats = zonal_stats(
        zones_gdf,
        raster_path,
        add_stats={'cv': coefficient_of_variation, 'vigor': vegetation_vigor_index}
    )
    
    # Add custom stats to main dataframe
    for i, custom in enumerate(custom_stats):
        stats_df.loc[i, 'cv'] = custom['cv']
        stats_df.loc[i, 'vigor'] = custom['vigor']
    
    # Display comprehensive results
    print("ADVANCED ZONAL STATISTICS")
    print("=" * 30)
    
    for idx, row in stats_df.iterrows():
        print(f"\n{row['zone_id']} ({row['zone_type']}):")
        print(f"  Basic Stats:")
        print(f"    Mean: {row['mean']:.3f} ± {row['std']:.3f}")
        print(f"    Range: {row['min']:.3f} - {row['max']:.3f}")
        print(f"    Median: {row['median']:.3f}")
        print(f"  Percentiles:")
        print(f"    10th: {row['percentile_10']:.3f}")
        print(f"    25th: {row['percentile_25']:.3f}")
        print(f"    75th: {row['percentile_75']:.3f}")
        print(f"    90th: {row['percentile_90']:.3f}")
        print(f"  Custom Metrics:")
        print(f"    Coefficient of Variation: {row['cv']:.3f}")
        print(f"    Vegetation Vigor: {row['vigor']}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Box plot equivalent using percentiles
    zones = stats_df['zone_id']
    
    # Mean vs Standard Deviation
    axes[0, 0].scatter(stats_df['mean'], stats_df['std'], 
                      c=range(len(stats_df)), cmap='viridis', s=100)
    for i, zone in enumerate(zones):
        axes[0, 0].annotate(zone, (stats_df.iloc[i]['mean'], stats_df.iloc[i]['std']))
    axes[0, 0].set_xlabel('Mean NDVI')
    axes[0, 0].set_ylabel('Standard Deviation')
    axes[0, 0].set_title('Mean vs Variability')
    
    # Percentile ranges
    x_pos = range(len(zones))
    axes[0, 1].errorbar(x_pos, stats_df['median'], 
                       yerr=[stats_df['median'] - stats_df['percentile_10'],
                             stats_df['percentile_90'] - stats_df['median']],
                       fmt='o', capsize=5, capthick=2)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(zones, rotation=45)
    axes[0, 1].set_ylabel('NDVI')
    axes[0, 1].set_title('Median with 10th-90th Percentile Range')
    
    # Coefficient of Variation
    axes[1, 0].bar(zones, stats_df['cv'], color='orange', alpha=0.7)
    axes[1, 0].set_ylabel('Coefficient of Variation')
    axes[1, 0].set_title('Variability Index by Zone')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Vegetation Vigor Distribution
    vigor_counts = stats_df['vigor'].value_counts()
    axes[1, 1].pie(vigor_counts.values, labels=vigor_counts.index, autopct='%1.0f%%')
    axes[1, 1].set_title('Vegetation Vigor Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return stats_df

# Calculate advanced statistics
advanced_results = calculate_advanced_zonal_stats()
```

---

## 2️⃣ Zonal Stats with rasterstats and xarray-spatial

### Using rasterstats Library
```python
def comprehensive_rasterstats_workflow():
    """Comprehensive workflow using rasterstats library"""
    
    # Create multi-band raster for demonstration
    def create_multiband_raster():
        """Create sample multi-band environmental data"""
        
        height, width = 150, 150
        x = np.linspace(0, 15, width)
        y = np.linspace(0, 15, height)
        X, Y = np.meshgrid(x, y)
        
        # Band 1: NDVI
        ndvi = 0.8 * np.exp(-((X-7.5)**2 + (Y-7.5)**2) / 20) + 0.2 * np.random.random((height, width))
        ndvi = np.clip(ndvi, 0, 1)
        
        # Band 2: Temperature (Celsius)
        temperature = 25 + 10 * np.sin(X/3) * np.cos(Y/3) + 3 * np.random.random((height, width))
        
        # Band 3: Precipitation (mm)
        precipitation = 100 + 50 * np.exp(-((X-10)**2 + (Y-5)**2) / 15) + 20 * np.random.random((height, width))
        
        # Stack bands
        data = np.stack([ndvi, temperature, precipitation])
        
        # Save as multi-band GeoTIFF
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 15, 15, width, height)
        
        with rasterio.open(
            'multiband_environmental.tif', 'w',
            driver='GTiff',
            height=height, width=width,
            count=3, dtype=data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(data)
        
        return 'multiband_environmental.tif'
    
    # Create sample polygons
    from shapely.geometry import Polygon
    
    polygons = [
        Polygon([(2, 2), (7, 2), (7, 7), (2, 7)]),      # Forest
        Polygon([(8, 8), (13, 8), (13, 13), (8, 13)]),  # Agricultural
        Polygon([(1, 9), (6, 9), (6, 14), (1, 14)]),    # Wetland
        Polygon([(9, 1), (14, 1), (14, 6), (9, 6)])     # Urban
    ]
    
    zones_gdf = gpd.GeoDataFrame({
        'zone_id': ['Forest', 'Agricultural', 'Wetland', 'Urban'],
        'area_ha': [2500, 2500, 2500, 2500],
        'geometry': polygons
    }, crs='EPSG:4326')
    
    # Create multi-band raster
    raster_path = create_multiband_raster()
    
    # Multi-band zonal statistics
    band_names = ['NDVI', 'Temperature', 'Precipitation']
    all_band_stats = {}
    
    for band_idx, band_name in enumerate(band_names, 1):
        stats = zonal_stats(
            zones_gdf,
            raster_path,
            band=band_idx,
            stats=['count', 'min', 'max', 'mean', 'sum', 'std'],
            prefix=f'{band_name}_'
        )
        all_band_stats[band_name] = stats
    
    # Combine all statistics
    combined_stats = zones_gdf.copy()
    
    for band_name, stats_list in all_band_stats.items():
        for i, stats in enumerate(stats_list):
            for stat_name, value in stats.items():
                combined_stats.loc[i, stat_name] = value
    
    # Display multi-band results
    print("MULTI-BAND ZONAL STATISTICS")
    print("=" * 30)
    
    for idx, row in combined_stats.iterrows():
        print(f"\n{row['zone_id']} Zone:")
        print(f"  NDVI: {row['NDVI_mean']:.3f} ± {row['NDVI_std']:.3f}")
        print(f"  Temperature: {row['Temperature_mean']:.1f}°C ± {row['Temperature_std']:.1f}")
        print(f"  Precipitation: {row['Precipitation_mean']:.1f}mm ± {row['Precipitation_std']:.1f}")
    
    # Advanced rasterstats features
    
    # 1. Categorical statistics (for land cover)
    categorical_stats = zonal_stats(
        zones_gdf,
        raster_path,
        band=1,  # Use NDVI band
        categorical=True,
        category_map={1: 'Water', 2: 'Vegetation', 3: 'Bare_Soil'}
    )
    
    # 2. Raster value extraction
    point_values = zonal_stats(
        zones_gdf.centroid,  # Use centroids as points
        raster_path,
        band=1,
        stats=['mean']  # Extract values at points
    )
    
    # 3. All touched pixels
    all_touched_stats = zonal_stats(
        zones_gdf,
        raster_path,
        band=1,
        stats=['mean'],
        all_touched=True  # Include pixels that touch polygon boundary
    )
    
    return combined_stats, raster_path, zones_gdf

# Run comprehensive rasterstats workflow
multiband_stats, env_raster, zones_data = comprehensive_rasterstats_workflow()
```

### Using xarray-spatial for Zonal Statistics
```python
def xarray_spatial_zonal_stats():
    """Demonstrate zonal statistics using xarray-spatial"""
    
    try:
        from xrspatial import zonal_stats as xs_zonal_stats
        from xrspatial.utils import ngjit
    except ImportError:
        print("xarray-spatial not available. Install with: pip install xarray-spatial")
        return None
    
    # Load raster data with xarray
    raster_da = rxr.open_rasterio('multiband_environmental.tif')
    
    # Create zone raster from polygons
    from rasterio.features import rasterize
    
    # Load zones
    _, zones_gdf = create_sample_data()
    
    # Create zone ID mapping
    zone_mapping = {zone_id: idx+1 for idx, zone_id in enumerate(zones_gdf['zone_id'])}
    zones_gdf['zone_numeric'] = zones_gdf['zone_id'].map(zone_mapping)
    
    # Rasterize zones to match raster grid
    with rasterio.open('multiband_environmental.tif') as src:
        transform = src.transform
        shape = src.shape
    
    zone_raster = rasterize(
        [(geom, zone_id) for geom, zone_id in zip(zones_gdf.geometry, zones_gdf['zone_numeric'])],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype='int32'
    )
    
    # Convert to xarray DataArray
    zone_da = xr.DataArray(
        zone_raster,
        coords=raster_da.coords[1:],  # Skip band dimension
        dims=['y', 'x']
    )
    
    # Calculate zonal statistics for each band
    results = {}
    
    for band_idx in range(raster_da.shape[0]):
        band_data = raster_da[band_idx]
        
        # Calculate statistics
        stats_result = xs_zonal_stats(
            zones=zone_da,
            values=band_data,
            zone_ids=list(zone_mapping.values())
        )
        
        results[f'Band_{band_idx+1}'] = stats_result
    
    # Display xarray-spatial results
    print("XARRAY-SPATIAL ZONAL STATISTICS")
    print("=" * 35)
    
    band_names = ['NDVI', 'Temperature', 'Precipitation']
    
    for zone_name, zone_id in zone_mapping.items():
        print(f"\n{zone_name}:")
        for band_idx, band_name in enumerate(band_names):
            band_key = f'Band_{band_idx+1}'
            if band_key in results and zone_id in results[band_key]:
                stats = results[band_key][zone_id]
                print(f"  {band_name}: Mean={stats.get('mean', 'N/A'):.3f}")
    
    return results, zone_da, raster_da

# Run xarray-spatial analysis
try:
    xarray_results, zone_raster, env_data = xarray_spatial_zonal_stats()
except Exception as e:
    print(f"xarray-spatial analysis skipped: {e}")
```

---

## 3️⃣ Working with LULC or NDVI Classes

### Land Use Land Cover (LULC) Analysis
```python
def lulc_zonal_analysis():
    """Analyze Land Use Land Cover data with zonal statistics"""
    
    # Create sample LULC raster
    def create_lulc_raster():
        """Create sample land cover classification"""
        
        height, width = 200, 200
        x = np.linspace(0, 20, width)
        y = np.linspace(0, 20, height)
        X, Y = np.meshgrid(x, y)
        
        # Create land cover classes
        lulc = np.zeros((height, width), dtype=np.uint8)
        
        # Water bodies (class 1)
        water_mask = ((X-5)**2 + (Y-15)**2) < 9
        lulc[water_mask] = 1
        
        # Forest (class 2)
        forest_mask = ((X-15)**2 + (Y-15)**2) < 16
        lulc[forest_mask] = 2
        
        # Agricultural (class 3)
        ag_mask = (X > 8) & (X < 18) & (Y > 2) & (Y < 8)
        lulc[ag_mask] = 3
        
        # Urban (class 4)
        urban_mask = (X > 2) & (X < 8) & (Y > 2) & (Y < 8)
        lulc[urban_mask] = 4
        
        # Grassland (class 5) - fill remaining areas
        lulc[lulc == 0] = 5
        
        # Add some noise for realism
        noise_mask = np.random.random((height, width)) < 0.05
        lulc[noise_mask] = np.random.choice([1, 2, 3, 4, 5], size=np.sum(noise_mask))
        
        # Save LULC raster
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 20, 20, width, height)
        
        with rasterio.open(
            'sample_lulc.tif', 'w',
            driver='GTiff',
            height=height, width=width,
            count=1, dtype=lulc.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(lulc, 1)
        
        return 'sample_lulc.tif', lulc
    
    # Create administrative zones
    from shapely.geometry import Polygon
    
    admin_zones = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),    # Southwest
        Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),  # Southeast  
        Polygon([(0, 10), (10, 10), (10, 20), (0, 20)]),  # Northwest
        Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]) # Northeast
    ]
    
    admin_gdf = gpd.GeoDataFrame({
        'admin_id': ['SW_District', 'SE_District', 'NW_District', 'NE_District'],
        'population': [50000, 75000, 30000, 45000],
        'geometry': admin_zones
    }, crs='EPSG:4326')
    
    # Create LULC data
    lulc_path, lulc_array = create_lulc_raster()
    
    # Define LULC classes
    lulc_classes = {
        1: 'Water',
        2: 'Forest', 
        3: 'Agricultural',
        4: 'Urban',
        5: 'Grassland'
    }
    
    # Calculate categorical zonal statistics
    lulc_stats = zonal_stats(
        admin_gdf,
        lulc_path,
        categorical=True,
        category_map=lulc_classes
    )
    
    # Process results into comprehensive DataFrame
    lulc_results = admin_gdf.copy()
    
    # Add pixel counts for each class
    for i, stats in enumerate(lulc_stats):
        total_pixels = sum(stats.values())
        lulc_results.loc[i, 'total_pixels'] = total_pixels
        
        for class_id, class_name in lulc_classes.items():
            pixel_count = stats.get(class_name, 0)
            percentage = (pixel_count / total_pixels * 100) if total_pixels > 0 else 0
            
            lulc_results.loc[i, f'{class_name}_pixels'] = pixel_count
            lulc_results.loc[i, f'{class_name}_percent'] = percentage
    
    # Calculate area statistics (assuming 100m pixel resolution)
    pixel_area_ha = 1  # 1 hectare per pixel for this example
    
    for class_name in lulc_classes.values():
        lulc_results[f'{class_name}_area_ha'] = lulc_results[f'{class_name}_pixels'] * pixel_area_ha
    
    # Display LULC analysis results
    print("LAND USE LAND COVER ANALYSIS")
    print("=" * 32)
    
    for idx, row in lulc_results.iterrows():
        print(f"\n{row['admin_id']} (Population: {row['population']:,}):")
        print(f"  Total Area: {row['total_pixels']:,} ha")
        
        for class_name in lulc_classes.values():
            area = row[f'{class_name}_area_ha']
            percent = row[f'{class_name}_percent']
            print(f"  {class_name}: {area:,.0f} ha ({percent:.1f}%)")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # LULC raster with zones
    with rasterio.open(lulc_path) as src:
        lulc_data = src.read(1)
    
    # Custom colormap for LULC
    colors = ['blue', 'darkgreen', 'yellow', 'red', 'lightgreen']
    from matplotlib.colors import ListedColormap
    lulc_cmap = ListedColormap(colors)
    
    im = axes[0, 0].imshow(lulc_data, cmap=lulc_cmap, vmin=1, vmax=5)
    admin_gdf.boundary.plot(ax=axes[0, 0], color='black', linewidth=2)
    axes[0, 0].set_title('Land Use Land Cover with Administrative Zones')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=class_name) 
                      for i, class_name in enumerate(lulc_classes.values())]
    axes[0, 0].legend(handles=legend_elements, loc='upper right')
    
    # Land cover composition by district
    districts = lulc_results['admin_id']
    class_names = list(lulc_classes.values())
    
    # Stacked bar chart
    bottom = np.zeros(len(districts))
    for i, class_name in enumerate(class_names):
        values = lulc_results[f'{class_name}_percent']
        axes[0, 1].bar(districts, values, bottom=bottom, label=class_name, color=colors[i])
        bottom += values
    
    axes[0, 1].set_title('Land Cover Composition by District (%)')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Urban vs Population correlation
    urban_percent = lulc_results['Urban_percent']
    population = lulc_results['population']
    
    axes[1, 0].scatter(population, urban_percent, s=100, alpha=0.7)
    for i, district in enumerate(districts):
        axes[1, 0].annotate(district, (population.iloc[i], urban_percent.iloc[i]))
    
    axes[1, 0].set_xlabel('Population')
    axes[1, 0].set_ylabel('Urban Area (%)')
    axes[1, 0].set_title('Urban Development vs Population')
    
    # Forest coverage comparison
    forest_area = lulc_results['Forest_area_ha']
    axes[1, 1].bar(districts, forest_area, color='darkgreen', alpha=0.7)
    axes[1, 1].set_title('Forest Coverage by District')
    axes[1, 1].set_ylabel('Forest Area (ha)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return lulc_results, lulc_path, admin_gdf

# Run LULC analysis
lulc_analysis_results, lulc_raster, admin_boundaries = lulc_zonal_analysis()
```

### NDVI Classification and Analysis
```python
def ndvi_classification_analysis():
    """Analyze NDVI data with vegetation health classifications"""
    
    # Create sample NDVI time series
    def create_ndvi_timeseries():
        """Create multi-temporal NDVI data"""
        
        height, width = 150, 150
        x = np.linspace(0, 15, width)
        y = np.linspace(0, 15, height)
        X, Y = np.meshgrid(x, y)
        
        # Base vegetation pattern
        base_ndvi = (
            0.8 * np.exp(-((X-7.5)**2 + (Y-7.5)**2) / 25) +  # Central forest
            0.4 * np.exp(-((X-3)**2 + (Y-12)**2) / 10) +     # Secondary patch
            0.3 * np.exp(-((X-12)**2 + (Y-3)**2) / 8) +      # Agricultural area
            0.1 + 0.1 * np.random.random((height, width))     # Background
        )
        
        # Simulate seasonal variation (3 time periods)
        seasons = ['Spring', 'Summer', 'Autumn']
        seasonal_multipliers = [0.7, 1.0, 0.6]  # Seasonal growth patterns
        
        ndvi_timeseries = {}
        
        for season, multiplier in zip(seasons, seasonal_multipliers):
            # Apply seasonal effect
            seasonal_ndvi = base_ndvi * multiplier
            
            # Add seasonal noise
            seasonal_ndvi += 0.05 * np.random.random((height, width))
            
            # Clip to valid NDVI range
            seasonal_ndvi = np.clip(seasonal_ndvi, -1, 1)
            
            ndvi_timeseries[season] = seasonal_ndvi
        
        return ndvi_timeseries, x, y
    
    # Create ecological zones
    from shapely.geometry import Polygon
    
    eco_zones = [
        Polygon([(2, 2), (8, 2), (8, 8), (2, 8)]),      # Core Forest
        Polygon([(9, 9), (14, 9), (14, 14), (9, 14)]),  # Agricultural
        Polygon([(1, 10), (6, 10), (6, 15), (1, 15)]),  # Wetland Buffer
        Polygon([(10, 1), (15, 1), (15, 6), (10, 6)])   # Grassland
    ]
    
    eco_gdf = gpd.GeoDataFrame({
        'zone_name': ['Core_Forest', 'Agricultural', 'Wetland_Buffer', 'Grassland'],
        'protection_status': ['Protected', 'Managed', 'Protected', 'Managed'],
        'geometry': eco_zones
    }, crs='EPSG:4326')
    
    # Generate NDVI time series
    ndvi_data, x_coords, y_coords = create_ndvi_timeseries()
    
    # NDVI classification thresholds
    ndvi_classes = {
        'Water/Bare': (-1.0, 0.1),
        'Sparse_Vegetation': (0.1, 0.3),
        'Moderate_Vegetation': (0.3, 0.5),
        'Dense_Vegetation': (0.5, 0.7),
        'Very_Dense_Vegetation': (0.7, 1.0)
    }
    
    # Analyze each season
    seasonal_results = {}
    
    for season, ndvi_array in ndvi_data.items():
        # Save seasonal NDVI as temporary raster
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 15, 15, len(x_coords), len(y_coords))
        
        temp_path = f'ndvi_{season.lower()}.tif'
        with rasterio.open(
            temp_path, 'w',
            driver='GTiff',
            height=ndvi_array.shape[0], width=ndvi_array.shape[1],
            count=1, dtype=ndvi_array.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(ndvi_array, 1)
        
        # Calculate zonal statistics
        season_stats = zonal_stats(
            eco_gdf,
            temp_path,
            stats=['count', 'min', 'max', 'mean', 'std'],
            percentiles=[25, 50, 75]
        )
        
        # Classify NDVI values
        classified_ndvi = np.zeros_like(ndvi_array, dtype=np.uint8)
        
        for class_idx, (class_name, (min_val, max_val)) in enumerate(ndvi_classes.items(), 1):
            mask = (ndvi_array >= min_val) & (ndvi_array < max_val)
            classified_ndvi[mask] = class_idx
        
        # Calculate class statistics
        class_stats = zonal_stats(
            eco_gdf,
            classified_ndvi,
            categorical=True,
            category_map={i: name for i, name in enumerate(ndvi_classes.keys(), 1)},
            transform=transform
        )
        
        # Store results
        seasonal_results[season] = {
            'continuous_stats': season_stats,
            'class_stats': class_stats,
            'classified_raster': classified_ndvi
        }
    
    # Create comprehensive results DataFrame
    results_df = eco_gdf.copy()
    
    # Add seasonal statistics
    for season in ndvi_data.keys():
        stats_list = seasonal_results[season]['continuous_stats']
        
        for i, stats in enumerate(stats_list):
            for stat_name, value in stats.items():
                results_df.loc[i, f'{season}_{stat_name}'] = value
    
    # Calculate vegetation health trends
    for idx, row in results_df.iterrows():
        spring_mean = row['Spring_mean']
        summer_mean = row['Summer_mean']
        autumn_mean = row['Autumn_mean']
        
        # Calculate seasonal change
        growing_season_change = summer_mean - spring_mean
        senescence_change = autumn_mean - summer_mean
        
        results_df.loc[idx, 'growing_season_change'] = growing_season_change
        results_df.loc[idx, 'senescence_change'] = senescence_change
        
        # Vegetation health classification
        if summer_mean > 0.6:
            health_status = 'Excellent'
        elif summer_mean > 0.4:
            health_status = 'Good'
        elif summer_mean > 0.2:
            health_status = 'Fair'
        else:
            health_status = 'Poor'
        
        results_df.loc[idx, 'vegetation_health'] = health_status
    
    # Display NDVI analysis results
    print("NDVI VEGETATION ANALYSIS")
    print("=" * 25)
    
    for idx, row in results_df.iterrows():
        print(f"\n{row['zone_name']} ({row['protection_status']}):")
        print(f"  Seasonal NDVI:")
        print(f"    Spring: {row['Spring_mean']:.3f} ± {row['Spring_std']:.3f}")
        print(f"    Summer: {row['Summer_mean']:.3f} ± {row['Summer_std']:.3f}")
        print(f"    Autumn: {row['Autumn_mean']:.3f} ± {row['Autumn_std']:.3f}")
        print(f"  Seasonal Changes:")
        print(f"    Growing Season: {row['growing_season_change']:+.3f}")
        print(f"    Senescence: {row['senescence_change']:+.3f}")
        print(f"  Overall Health: {row['vegetation_health']}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Seasonal NDVI maps
    seasons = list(ndvi_data.keys())
    for i, season in enumerate(seasons):
        ndvi_array = ndvi_data[season]
        im = axes[0, i].imshow(ndvi_array, cmap='RdYlGn', vmin=-0.2, vmax=1.0)
        eco_gdf.boundary.plot(ax=axes[0, i], color='black', linewidth=1)
        axes[0, i].set_title(f'{season} NDVI')
        
        if i == 2:  # Add colorbar to last subplot
            plt.colorbar(im, ax=axes[0, i])
    
    # Seasonal trends by zone
    zones = results_df['zone_name']
    spring_vals = results_df['Spring_mean']
    summer_vals = results_df['Summer_mean']
    autumn_vals = results_df['Autumn_mean']
    
    x_pos = np.arange(len(zones))
    width = 0.25
    
    axes[1, 0].bar(x_pos - width, spring_vals, width, label='Spring', alpha=0.8)
    axes[1, 0].bar(x_pos, summer_vals, width, label='Summer', alpha=0.8)
    axes[1, 0].bar(x_pos + width, autumn_vals, width, label='Autumn', alpha=0.8)
    
    axes[1, 0].set_xlabel('Ecological Zone')
    axes[1, 0].set_ylabel('Mean NDVI')
    axes[1, 0].set_title('Seasonal NDVI Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(zones, rotation=45)
    axes[1, 0].legend()
    
    # Vegetation health distribution
    health_counts = results_df['vegetation_health'].value_counts()
    axes[1, 1].pie(health_counts.values, labels=health_counts.index, autopct='%1.0f%%')
    axes[1, 1].set_title('Vegetation Health Distribution')
    
    # Growing season change
    growing_changes = results_df['growing_season_change']
    colors = ['red' if x < 0 else 'green' for x in growing_changes]
    axes[1, 2].bar(zones, growing_changes, color=colors, alpha=0.7)
    axes[1, 2].set_title('Growing Season NDVI Change')
    axes[1, 2].set_ylabel('NDVI Change (Summer - Spring)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return results_df, seasonal_results, eco_gdf

# Run NDVI classification analysis
ndvi_results, seasonal_data, ecological_zones = ndvi_classification_analysis()
```

---

## 4️⃣ Exporting Statistics as Tables or GeoDataFrames

### Export to Multiple Formats
```python
def export_zonal_statistics():
    """Export zonal statistics to various formats"""
    
    # Use results from previous analyses
    # Create comprehensive dataset combining all analyses
    
    # Simulate comprehensive zonal statistics dataset
    comprehensive_stats = pd.DataFrame({
        'zone_id': ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D'],
        'zone_type': ['Forest', 'Agricultural', 'Urban', 'Wetland'],
        'area_ha': [2500, 3200, 1800, 2100],
        'ndvi_mean': [0.75, 0.45, 0.25, 0.65],
        'ndvi_std': [0.12, 0.08, 0.15, 0.10],
        'ndvi_min': [0.45, 0.20, 0.05, 0.35],
        'ndvi_max': [0.95, 0.70, 0.55, 0.85],
        'temperature_mean': [22.5, 25.8, 28.2, 21.8],
        'precipitation_sum': [1250, 980, 750, 1450],
        'forest_percent': [85.2, 15.3, 2.1, 45.8],
        'urban_percent': [2.1, 8.5, 78.9, 1.2],
        'water_percent': [1.2, 0.8, 3.2, 25.6],
        'vegetation_health': ['Excellent', 'Good', 'Poor', 'Good']
    })
    
    # Create geometry for GeoDataFrame
    from shapely.geometry import Polygon
    
    geometries = [
        Polygon([(-2, -2), (2, -2), (2, 2), (-2, 2)]),
        Polygon([(3, 3), (7, 3), (7, 7), (3, 7)]),
        Polygon([(-1, 4), (3, 4), (3, 8), (-1, 8)]),
        Polygon([(5, -1), (9, -1), (9, 3), (5, 3)])
    ]
    
    # Create GeoDataFrame
    stats_gdf = gpd.GeoDataFrame(comprehensive_stats, geometry=geometries, crs='EPSG:4326')
    
    # 1. Export to CSV (tabular data only)
    print("Exporting to CSV...")
    stats_gdf.drop('geometry', axis=1).to_csv('zonal_statistics.csv', index=False)
    
    # 2. Export to Excel with multiple sheets
    print("Exporting to Excel...")
    with pd.ExcelWriter('zonal_statistics_comprehensive.xlsx', engine='openpyxl') as writer:
        # Main statistics
        stats_gdf.drop('geometry', axis=1).to_excel(writer, sheet_name='Zonal_Statistics', index=False)
        
        # Summary by zone type
        summary_by_type = stats_gdf.groupby('zone_type').agg({
            'area_ha': ['sum', 'mean'],
            'ndvi_mean': 'mean',
            'temperature_mean': 'mean',
            'precipitation_sum': 'sum',
            'forest_percent': 'mean',
            'urban_percent': 'mean'
        }).round(2)
        summary_by_type.to_excel(writer, sheet_name='Summary_by_Type')
        
        # Vegetation health summary
        health_summary = stats_gdf['vegetation_health'].value_counts()
        health_summary.to_excel(writer, sheet_name='Vegetation_Health')
    
    # 3. Export to GeoPackage (preserves geometry)
    print("Exporting to GeoPackage...")
    stats_gdf.to_file('zonal_statistics.gpkg', driver='GPKG')
    
    # 4. Export to Shapefile
    print("Exporting to Shapefile...")
    # Truncate column names for shapefile compatibility
    shapefile_gdf = stats_gdf.copy()
    column_mapping = {
        'zone_id': 'zone_id',
        'zone_type': 'zone_type',
        'area_ha': 'area_ha',
        'ndvi_mean': 'ndvi_mean',
        'ndvi_std': 'ndvi_std',
        'temperature_mean': 'temp_mean',
        'precipitation_sum': 'precip_sum',
        'forest_percent': 'forest_pct',
        'urban_percent': 'urban_pct',
        'vegetation_health': 'veg_health'
    }
    
    shapefile_gdf = shapefile_gdf[list(column_mapping.keys()) + ['geometry']]
    shapefile_gdf.columns = list(column_mapping.values()) + ['geometry']
    shapefile_gdf.to_file('zonal_statistics.shp')
    
    # 5. Export to GeoJSON
    print("Exporting to GeoJSON...")
    stats_gdf.to_file('zonal_statistics.geojson', driver='GeoJSON')
    
    # 6. Export to PostGIS (if connection available)
    def export_to_postgis():
        """Export to PostGIS database (example)"""
        try:
            from sqlalchemy import create_engine
            
            # Example connection (modify for your database)
            # engine = create_engine('postgresql://user:password@localhost:5432/database')
            # stats_gdf.to_postgis('zonal_statistics', engine, if_exists='replace')
            
            print("PostGIS export would require database connection")
        except ImportError:
            print("SQLAlchemy not available for PostGIS export")
    
    # 7. Create formatted report
    def create_formatted_report():
        """Create formatted HTML report"""
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Zonal Statistics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Zonal Statistics Analysis Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Zones Analyzed: {len(stats_gdf)}</p>
                <p>Total Area: {stats_gdf['area_ha'].sum():,.0f} hectares</p>
                <p>Average NDVI: {stats_gdf['ndvi_mean'].mean():.3f}</p>
                <p>Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>Detailed Statistics</h2>
            {stats_gdf.drop('geometry', axis=1).to_html(index=False, table_id='stats_table')}
            
            <h2>Zone Type Summary</h2>
            {stats_gdf.groupby('zone_type')['ndvi_mean'].agg(['count', 'mean', 'std']).round(3).to_html()}
            
        </body>
        </html>
        """
        
        with open('zonal_statistics_report.html', 'w') as f:
            f.write(html_report)
    
    create_formatted_report()
    
    # 8. Export metadata
    def create_metadata():
        """Create metadata file describing the analysis"""
        
        metadata = {
            'analysis_info': {
                'title': 'Zonal Statistics Analysis',
                'description': 'Comprehensive zonal statistics for environmental monitoring',
                'date_created': pd.Timestamp.now().isoformat(),
                'coordinate_system': 'EPSG:4326',
                'total_zones': len(stats_gdf)
            },
            'data_sources': {
                'raster_data': ['NDVI', 'Temperature', 'Precipitation'],
                'vector_data': 'Administrative boundaries',
                'analysis_software': 'Python with rasterstats, geopandas'
            },
            'statistics_calculated': {
                'continuous_variables': ['mean', 'std', 'min', 'max'],
                'categorical_variables': ['forest_percent', 'urban_percent', 'water_percent'],
                'derived_metrics': ['vegetation_health']
            },
            'export_formats': {
                'csv': 'Tabular data without geometry',
                'excel': 'Multi-sheet workbook with summaries',
                'gpkg': 'Spatial data with full attribute preservation',
                'shapefile': 'Compatible with most GIS software',
                'geojson': 'Web-compatible spatial format',
                'html': 'Formatted report for presentation'
            }
        }
        
        import json
        with open('zonal_statistics_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    create_metadata()
    
    # Display export summary
    print("\nEXPORT SUMMARY")
    print("=" * 15)
    print("Files created:")
    print("  ✓ zonal_statistics.csv - Tabular data")
    print("  ✓ zonal_statistics_comprehensive.xlsx - Multi-sheet Excel")
    print("  ✓ zonal_statistics.gpkg - GeoPackage")
    print("  ✓ zonal_statistics.shp (+ associated files) - Shapefile")
    print("  ✓ zonal_statistics.geojson - GeoJSON")
    print("  ✓ zonal_statistics_report.html - Formatted report")
    print("  ✓ zonal_statistics_metadata.json - Analysis metadata")
    
    # Show data preview
    print(f"\nData Preview ({len(stats_gdf)} zones):")
    print(stats_gdf.drop('geometry', axis=1).head())
    
    return stats_gdf

# Export all results
exported_data = export_zonal_statistics()
```

### Automated Reporting Workflow
```python
def automated_zonal_reporting():
    """Create automated reporting workflow for zonal statistics"""
    
    def generate_executive_summary(stats_gdf):
        """Generate executive summary from zonal statistics"""
        
        summary = {
            'total_area': stats_gdf['area_ha'].sum(),
            'zone_count': len(stats_gdf),
            'avg_ndvi': stats_gdf['ndvi_mean'].mean(),
            'forest_coverage': stats_gdf['forest_percent'].mean(),
            'urban_coverage': stats_gdf['urban_percent'].mean(),
            'health_distribution': stats_gdf['vegetation_health'].value_counts().to_dict()
        }
        
        return summary
    
    def create_dashboard_data(stats_gdf):
        """Prepare data for dashboard visualization"""
        
        dashboard_data = {
            'zone_metrics': stats_gdf[['zone_id', 'zone_type', 'ndvi_mean', 'area_ha']].to_dict('records'),
            'type_summary': stats_gdf.groupby('zone_type').agg({
                'area_ha': 'sum',
                'ndvi_mean': 'mean',
                'forest_percent': 'mean'
            }).round(2).to_dict(),
            'health_summary': stats_gdf['vegetation_health'].value_counts().to_dict()
        }
        
        return dashboard_data
    
    # Use existing comprehensive dataset
    stats_gdf = exported_data
    
    # Generate summary
    summary = generate_executive_summary(stats_gdf)
    
    # Create dashboard data
    dashboard_data = create_dashboard_data(stats_gdf)
    
    # Export for different use cases
    
    # 1. API-ready JSON
    api_data = {
        'metadata': {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_zones': len(stats_gdf),
            'coordinate_system': 'EPSG:4326'
        },
        'summary': summary,
        'zones': stats_gdf.drop('geometry', axis=1).to_dict('records')
    }
    
    with open('zonal_statistics_api.json', 'w') as f:
        json.dump(api_data, f, indent=2)
    
    # 2. Database-ready format
    db_ready = stats_gdf.copy()
    db_ready['geometry_wkt'] = db_ready['geometry'].apply(lambda x: x.wkt)
    db_ready = db_ready.drop('geometry', axis=1)
    db_ready.to_csv('zonal_statistics_database.csv', index=False)
    
    # 3. Visualization-ready format
    viz_data = {
        'zones': json.loads(stats_gdf.to_json()),
        'summary_charts': {
            'ndvi_by_type': stats_gdf.groupby('zone_type')['ndvi_mean'].mean().to_dict(),
            'area_by_type': stats_gdf.groupby('zone_type')['area_ha'].sum().to_dict(),
            'health_distribution': stats_gdf['vegetation_health'].value_counts().to_dict()
        }
    }
    
    with open('zonal_statistics_visualization.json', 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print("AUTOMATED REPORTING COMPLETE")
    print("=" * 30)
    print("Generated files:")
    print("  ✓ zonal_statistics_api.json - API-ready format")
    print("  ✓ zonal_statistics_database.csv - Database import format")
    print("  ✓ zonal_statistics_visualization.json - Visualization data")
    
    print(f"\nExecutive Summary:")
    print(f"  Total Area Analyzed: {summary['total_area']:,.0f} ha")
    print(f"  Number of Zones: {summary['zone_count']}")
    print(f"  Average NDVI: {summary['avg_ndvi']:.3f}")
    print(f"  Forest Coverage: {summary['forest_coverage']:.1f}%")
    print(f"  Urban Coverage: {summary['urban_coverage']:.1f}%")
    
    return api_data, dashboard_data

# Run automated reporting
api_output, dashboard_output = automated_zonal_reporting()
```

---

## Best Practices and Tips

### Performance Optimization
- **Use appropriate resampling**: Match raster resolution to analysis needs
- **Clip before analysis**: Reduce raster size to area of interest
- **Batch processing**: Process multiple zones simultaneously
- **Memory management**: Use chunked processing for large datasets

### Data Quality Considerations
- **Validate geometries**: Ensure polygons are valid and properly projected
- **Handle NoData values**: Account for missing or invalid raster pixels
- **Check spatial alignment**: Verify raster and vector data alignment
- **Quality control**: Review statistics for outliers or unexpected values

### Integration Opportunities
- **Time series analysis**: Track changes over multiple time periods
- **Multi-sensor fusion**: Combine different raster data sources
- **Automated monitoring**: Set up regular analysis workflows
- **Decision support**: Link statistics to management thresholds

This comprehensive guide provides the foundation for extracting meaningful insights from raster data using zonal statistics, enabling quantitative analysis across diverse environmental and land use applications.