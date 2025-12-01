# Raster Math and Band Operations

## Overview
Raster mathematics and band operations form the foundation of quantitative remote sensing analysis. These operations enable extraction of meaningful information from multispectral imagery, calculation of vegetation indices, and creation of derived products that reveal patterns invisible in individual bands.

## Why Raster Math Matters
- **Information Extraction**: Derive meaningful metrics from raw spectral data
- **Change Detection**: Quantify differences between time periods or conditions
- **Classification Support**: Create features for land cover classification
- **Quality Assessment**: Identify and handle data quality issues
- **Scientific Analysis**: Support research with quantitative measurements

---

## 1️⃣ Raster Algebra with NumPy (addition, ratio, masking)

### Basic Raster Arithmetic Operations
```python
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt

# Create sample multispectral data
def create_sample_multispectral():
    """Create sample multispectral raster data"""
    
    height, width = 100, 100
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    X, Y = np.meshgrid(x, y)
    
    # Simulate realistic spectral bands
    bands = {}
    
    # Blue band (lower reflectance)
    bands['blue'] = 0.1 + 0.05 * np.sin(X) + 0.02 * np.random.random((height, width))
    
    # Green band
    bands['green'] = 0.15 + 0.08 * np.sin(X * 1.5) + 0.03 * np.random.random((height, width))
    
    # Red band (vegetation absorption)
    vegetation_mask = ((X - 5)**2 + (Y - 5)**2) < 9
    bands['red'] = np.where(vegetation_mask, 
                           0.05 + 0.02 * np.random.random((height, width)),
                           0.25 + 0.1 * np.random.random((height, width)))
    
    # NIR band (vegetation reflection)
    bands['nir'] = np.where(vegetation_mask,
                           0.6 + 0.1 * np.random.random((height, width)),
                           0.3 + 0.05 * np.random.random((height, width)))
    
    # SWIR band
    bands['swir'] = 0.2 + 0.1 * np.exp(-((X-7)**2 + (Y-3)**2) / 5) + 0.03 * np.random.random((height, width))
    
    # Create xarray dataset
    data_vars = {}
    for band_name, band_data in bands.items():
        data_vars[band_name] = (['y', 'x'], band_data)
    
    ds = xr.Dataset(data_vars, coords={'y': y, 'x': x})
    ds.rio.write_crs('EPSG:4326', inplace=True)
    
    return ds

# Basic arithmetic operations
def basic_raster_arithmetic():
    """Demonstrate basic raster arithmetic operations"""
    
    # Load sample data
    ds = create_sample_multispectral()
    
    print("BASIC RASTER ARITHMETIC")
    print("=" * 23)
    
    # Addition: Combine bands
    visible_sum = ds.red + ds.green + ds.blue
    print(f"Visible sum range: {visible_sum.min().values:.3f} to {visible_sum.max().values:.3f}")
    
    # Subtraction: Band differences
    red_nir_diff = ds.nir - ds.red
    print(f"NIR-Red difference range: {red_nir_diff.min().values:.3f} to {red_nir_diff.max().values:.3f}")
    
    # Multiplication: Scaling
    scaled_nir = ds.nir * 10000  # Convert to integer reflectance
    print(f"Scaled NIR range: {scaled_nir.min().values:.0f} to {scaled_nir.max().values:.0f}")
    
    # Division: Band ratios
    nir_red_ratio = ds.nir / ds.red
    print(f"NIR/Red ratio range: {nir_red_ratio.min().values:.2f} to {nir_red_ratio.max().values:.2f}")
    
    # Power operations
    enhanced_contrast = ds.red ** 0.5  # Gamma correction
    print(f"Enhanced contrast range: {enhanced_contrast.min().values:.3f} to {enhanced_contrast.max().values:.3f}")
    
    # Logarithmic operations
    log_transform = np.log(ds.nir + 0.001)  # Add small value to avoid log(0)
    print(f"Log transform range: {log_transform.min().values:.3f} to {log_transform.max().values:.3f}")
    
    return ds, {
        'visible_sum': visible_sum,
        'red_nir_diff': red_nir_diff,
        'nir_red_ratio': nir_red_ratio,
        'enhanced_contrast': enhanced_contrast
    }

# Advanced mathematical operations
def advanced_raster_math():
    """Advanced mathematical operations on rasters"""
    
    ds = create_sample_multispectral()
    
    # Conditional operations
    def apply_conditional_math(data):
        """Apply conditional mathematical operations"""
        
        # Threshold-based operations
        high_nir = np.where(data.nir > 0.5, data.nir, 0)
        
        # Multiple conditions
        vegetation_pixels = np.where(
            (data.nir > data.red) & (data.nir > 0.3),
            1, 0
        )
        
        # Complex conditional expressions
        brightness = (data.red + data.green + data.blue) / 3
        bright_vegetation = np.where(
            (vegetation_pixels == 1) & (brightness > 0.15),
            data.nir / data.red,
            np.nan
        )
        
        return {
            'high_nir': high_nir,
            'vegetation_mask': vegetation_pixels,
            'bright_vegetation_ratio': bright_vegetation
        }
    
    # Statistical operations
    def calculate_band_statistics(data):
        """Calculate statistical measures across bands"""
        
        # Stack bands for multi-band operations
        band_stack = np.stack([data.blue, data.green, data.red, data.nir], axis=0)
        
        # Statistical measures
        band_mean = np.mean(band_stack, axis=0)
        band_std = np.std(band_stack, axis=0)
        band_max = np.max(band_stack, axis=0)
        band_min = np.min(band_stack, axis=0)
        
        # Coefficient of variation
        cv = band_std / (band_mean + 1e-8)  # Avoid division by zero
        
        return {
            'mean': band_mean,
            'std': band_std,
            'max': band_max,
            'min': band_min,
            'cv': cv
        }
    
    # Trigonometric operations
    def trigonometric_operations(data):
        """Apply trigonometric functions to raster data"""
        
        # Normalize data to 0-π range for trig functions
        normalized_nir = (data.nir - data.nir.min()) / (data.nir.max() - data.nir.min()) * np.pi
        
        # Trigonometric transformations
        sin_transform = np.sin(normalized_nir)
        cos_transform = np.cos(normalized_nir)
        
        # Phase relationships
        phase_diff = np.arctan2(data.nir - data.red, data.nir + data.red)
        
        return {
            'sin_nir': sin_transform,
            'cos_nir': cos_transform,
            'phase_difference': phase_diff
        }
    
    # Apply operations
    conditional_results = apply_conditional_math(ds)
    statistical_results = calculate_band_statistics(ds)
    trig_results = trigonometric_operations(ds)
    
    print("\nADVANCED RASTER MATH")
    print("=" * 20)
    print(f"Vegetation pixels: {conditional_results['vegetation_mask'].sum().values}")
    print(f"Mean spectral CV: {statistical_results['cv'].mean().values:.3f}")
    print(f"Phase difference range: {trig_results['phase_difference'].min().values:.3f} to {trig_results['phase_difference'].max().values:.3f}")
    
    return conditional_results, statistical_results, trig_results

# Masking operations
def raster_masking_operations():
    """Demonstrate various masking techniques"""
    
    ds = create_sample_multispectral()
    
    # Value-based masks
    def create_value_masks(data):
        """Create masks based on pixel values"""
        
        # Simple threshold mask
        bright_mask = data.nir > 0.4
        
        # Range mask
        moderate_red = (data.red >= 0.1) & (data.red <= 0.2)
        
        # Multi-band mask
        vegetation_mask = (data.nir > data.red) & (data.nir > 0.3)
        
        # Statistical mask (outliers)
        nir_mean = data.nir.mean()
        nir_std = data.nir.std()
        outlier_mask = np.abs(data.nir - nir_mean) > (2 * nir_std)
        
        return {
            'bright': bright_mask,
            'moderate_red': moderate_red,
            'vegetation': vegetation_mask,
            'outliers': outlier_mask
        }
    
    # Geometric masks
    def create_geometric_masks(data):
        """Create masks based on spatial patterns"""
        
        height, width = data.nir.shape
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Circular mask
        center_y, center_x = height // 2, width // 2
        radius = min(height, width) // 3
        circular_mask = ((y_coords - center_y)**2 + (x_coords - center_x)**2) <= radius**2
        
        # Rectangular mask
        rect_mask = (y_coords >= height//4) & (y_coords <= 3*height//4) & \
                   (x_coords >= width//4) & (x_coords <= 3*width//4)
        
        # Edge mask
        edge_width = 5
        edge_mask = (y_coords < edge_width) | (y_coords >= height - edge_width) | \
                   (x_coords < edge_width) | (x_coords >= width - edge_width)
        
        return {
            'circular': circular_mask,
            'rectangular': rect_mask,
            'edges': edge_mask
        }
    
    # Apply masks to data
    def apply_masks(data, masks):
        """Apply masks to raster data"""
        
        masked_results = {}
        
        for mask_name, mask in masks.items():
            # Apply mask (set masked values to NaN)
            masked_nir = np.where(mask, data.nir, np.nan)
            
            # Calculate statistics for masked data
            valid_pixels = np.sum(~np.isnan(masked_nir))
            mean_value = np.nanmean(masked_nir)
            
            masked_results[mask_name] = {
                'data': masked_nir,
                'valid_pixels': valid_pixels,
                'mean': mean_value
            }
        
        return masked_results
    
    # Create and apply masks
    value_masks = create_value_masks(ds)
    geometric_masks = create_geometric_masks(ds)
    
    value_results = apply_masks(ds, value_masks)
    geometric_results = apply_masks(ds, geometric_masks)
    
    print("\nMASKING OPERATIONS")
    print("=" * 18)
    
    print("Value-based masks:")
    for mask_name, result in value_results.items():
        print(f"  {mask_name}: {result['valid_pixels']} pixels, mean = {result['mean']:.3f}")
    
    print("Geometric masks:")
    for mask_name, result in geometric_results.items():
        print(f"  {mask_name}: {result['valid_pixels']} pixels, mean = {result['mean']:.3f}")
    
    return value_masks, geometric_masks, value_results

# Run basic operations
sample_data, arithmetic_results = basic_raster_arithmetic()
conditional_ops, statistical_ops, trig_ops = advanced_raster_math()
value_masks, geom_masks, mask_results = raster_masking_operations()
```

---

## 2️⃣ NDVI and Vegetation Indices from Multiband Images

### Vegetation Index Calculations
```python
def calculate_vegetation_indices():
    """Calculate various vegetation indices from multispectral data"""
    
    # Load sample multispectral data
    ds = create_sample_multispectral()
    
    # NDVI (Normalized Difference Vegetation Index)
    def calculate_ndvi(red, nir):
        """Calculate NDVI"""
        return (nir - red) / (nir + red)
    
    # EVI (Enhanced Vegetation Index)
    def calculate_evi(red, nir, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
        """Calculate Enhanced Vegetation Index"""
        return G * ((nir - red) / (nir + C1 * red - C2 * blue + L))
    
    # SAVI (Soil Adjusted Vegetation Index)
    def calculate_savi(red, nir, L=0.5):
        """Calculate Soil Adjusted Vegetation Index"""
        return ((nir - red) / (nir + red + L)) * (1 + L)
    
    # MSAVI (Modified Soil Adjusted Vegetation Index)
    def calculate_msavi(red, nir):
        """Calculate Modified Soil Adjusted Vegetation Index"""
        return (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
    
    # NDWI (Normalized Difference Water Index)
    def calculate_ndwi(green, nir):
        """Calculate Normalized Difference Water Index"""
        return (green - nir) / (green + nir)
    
    # NBR (Normalized Burn Ratio)
    def calculate_nbr(nir, swir):
        """Calculate Normalized Burn Ratio"""
        return (nir - swir) / (nir + swir)
    
    # Calculate all indices
    indices = {}
    
    # Basic vegetation indices
    indices['ndvi'] = calculate_ndvi(ds.red, ds.nir)
    indices['evi'] = calculate_evi(ds.red, ds.nir, ds.blue)
    indices['savi'] = calculate_savi(ds.red, ds.nir)
    indices['msavi'] = calculate_msavi(ds.red, ds.nir)
    
    # Water and burn indices
    indices['ndwi'] = calculate_ndwi(ds.green, ds.nir)
    indices['nbr'] = calculate_nbr(ds.nir, ds.swir)
    
    # Custom indices
    indices['green_ndvi'] = calculate_ndvi(ds.green, ds.nir)  # Green NDVI
    indices['red_edge_ndvi'] = (ds.nir - ds.red) / (ds.nir + ds.red + 0.16)  # Red-edge NDVI
    
    print("VEGETATION INDICES")
    print("=" * 18)
    
    for index_name, index_data in indices.items():
        valid_pixels = np.sum(~np.isnan(index_data))
        mean_val = np.nanmean(index_data)
        std_val = np.nanstd(index_data)
        
        print(f"{index_name.upper()}:")
        print(f"  Range: {np.nanmin(index_data):.3f} to {np.nanmax(index_data):.3f}")
        print(f"  Mean ± Std: {mean_val:.3f} ± {std_val:.3f}")
        print(f"  Valid pixels: {valid_pixels}")
    
    return indices

# Advanced vegetation analysis
def advanced_vegetation_analysis():
    """Advanced vegetation analysis techniques"""
    
    ds = create_sample_multispectral()
    indices = calculate_vegetation_indices()
    
    # Vegetation classification based on NDVI
    def classify_vegetation(ndvi):
        """Classify vegetation based on NDVI values"""
        
        classes = np.zeros_like(ndvi, dtype=int)
        
        # Classification thresholds
        classes = np.where(ndvi < 0, 0, classes)          # Water/No vegetation
        classes = np.where((ndvi >= 0) & (ndvi < 0.2), 1, classes)    # Bare soil/sparse
        classes = np.where((ndvi >= 0.2) & (ndvi < 0.4), 2, classes)  # Moderate vegetation
        classes = np.where((ndvi >= 0.4) & (ndvi < 0.6), 3, classes)  # Dense vegetation
        classes = np.where(ndvi >= 0.6, 4, classes)       # Very dense vegetation
        
        class_names = {
            0: 'Water/No vegetation',
            1: 'Bare soil/Sparse',
            2: 'Moderate vegetation',
            3: 'Dense vegetation',
            4: 'Very dense vegetation'
        }
        
        return classes, class_names
    
    # Vegetation health assessment
    def assess_vegetation_health(ndvi, evi, savi):
        """Assess vegetation health using multiple indices"""
        
        # Normalize indices to 0-1 range
        ndvi_norm = (ndvi - np.nanmin(ndvi)) / (np.nanmax(ndvi) - np.nanmin(ndvi))
        evi_norm = (evi - np.nanmin(evi)) / (np.nanmax(evi) - np.nanmin(evi))
        savi_norm = (savi - np.nanmin(savi)) / (np.nanmax(savi) - np.nanmin(savi))
        
        # Composite health index
        health_index = (ndvi_norm + evi_norm + savi_norm) / 3
        
        # Health classification
        health_classes = np.zeros_like(health_index, dtype=int)
        health_classes = np.where(health_index < 0.2, 0, health_classes)  # Poor
        health_classes = np.where((health_index >= 0.2) & (health_index < 0.4), 1, health_classes)  # Fair
        health_classes = np.where((health_index >= 0.4) & (health_index < 0.6), 2, health_classes)  # Good
        health_classes = np.where((health_index >= 0.6) & (health_index < 0.8), 3, health_classes)  # Very good
        health_classes = np.where(health_index >= 0.8, 4, health_classes)  # Excellent
        
        return health_index, health_classes
    
    # Phenology analysis
    def analyze_phenology(ndvi_timeseries):
        """Analyze vegetation phenology (simulated time series)"""
        
        # Simulate seasonal NDVI variation
        days = np.arange(365)
        seasonal_ndvi = 0.3 + 0.4 * np.sin(2 * np.pi * (days - 80) / 365)
        
        # Find key phenological dates
        max_ndvi_day = np.argmax(seasonal_ndvi)
        min_ndvi_day = np.argmin(seasonal_ndvi)
        
        # Growth phases
        spring_start = np.where(np.diff(seasonal_ndvi) > 0.001)[0][0] if len(np.where(np.diff(seasonal_ndvi) > 0.001)[0]) > 0 else 0
        autumn_start = np.where(np.diff(seasonal_ndvi) < -0.001)[0][0] if len(np.where(np.diff(seasonal_ndvi) < -0.001)[0]) > 0 else 200
        
        phenology_metrics = {
            'peak_day': max_ndvi_day,
            'minimum_day': min_ndvi_day,
            'spring_start': spring_start,
            'autumn_start': autumn_start,
            'growing_season_length': autumn_start - spring_start,
            'peak_ndvi': seasonal_ndvi[max_ndvi_day],
            'amplitude': np.max(seasonal_ndvi) - np.min(seasonal_ndvi)
        }
        
        return seasonal_ndvi, phenology_metrics
    
    # Apply analyses
    veg_classes, class_names = classify_vegetation(indices['ndvi'])
    health_index, health_classes = assess_vegetation_health(
        indices['ndvi'], indices['evi'], indices['savi']
    )
    seasonal_ndvi, phenology = analyze_phenology(indices['ndvi'])
    
    print("\nADVANCED VEGETATION ANALYSIS")
    print("=" * 29)
    
    # Vegetation class distribution
    unique_classes, class_counts = np.unique(veg_classes, return_counts=True)
    print("Vegetation class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / veg_classes.size) * 100
        print(f"  {class_names[cls]}: {percentage:.1f}%")
    
    # Health assessment summary
    unique_health, health_counts = np.unique(health_classes, return_counts=True)
    health_names = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
    print("\nVegetation health distribution:")
    for health, count in zip(unique_health, health_counts):
        percentage = (count / health_classes.size) * 100
        print(f"  {health_names[health]}: {percentage:.1f}%")
    
    # Phenology summary
    print(f"\nPhenology metrics:")
    print(f"  Peak NDVI day: {phenology['peak_day']} (NDVI = {phenology['peak_ndvi']:.3f})")
    print(f"  Growing season: {phenology['growing_season_length']} days")
    print(f"  Seasonal amplitude: {phenology['amplitude']:.3f}")
    
    return veg_classes, health_index, seasonal_ndvi, phenology

# Visualization of vegetation indices
def visualize_vegetation_indices():
    """Create comprehensive visualization of vegetation indices"""
    
    ds = create_sample_multispectral()
    indices = calculate_vegetation_indices()
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Original bands
    ds.red.plot(ax=axes[0], cmap='Reds', title='Red Band')
    ds.nir.plot(ax=axes[1], cmap='RdYlGn', title='NIR Band')
    ds.blue.plot(ax=axes[2], cmap='Blues', title='Blue Band')
    
    # Vegetation indices
    indices['ndvi'].plot(ax=axes[3], cmap='RdYlGn', vmin=-1, vmax=1, title='NDVI')
    indices['evi'].plot(ax=axes[4], cmap='RdYlGn', title='EVI')
    indices['savi'].plot(ax=axes[5], cmap='RdYlGn', title='SAVI')
    
    # Water and burn indices
    indices['ndwi'].plot(ax=axes[6], cmap='RdBu', vmin=-1, vmax=1, title='NDWI')
    indices['nbr'].plot(ax=axes[7], cmap='RdYlBu', vmin=-1, vmax=1, title='NBR')
    
    # Custom composite
    rgb_composite = np.stack([ds.red, ds.green, ds.blue], axis=-1)
    rgb_composite = np.clip(rgb_composite / rgb_composite.max(), 0, 1)
    axes[8].imshow(rgb_composite)
    axes[8].set_title('RGB Composite')
    axes[8].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Run vegetation analysis
vegetation_indices = calculate_vegetation_indices()
veg_analysis_results = advanced_vegetation_analysis()
vegetation_plot = visualize_vegetation_indices()
```

---

## 3️⃣ Handling NoData Values Properly

### NoData Management Strategies
```python
def handle_nodata_values():
    """Comprehensive NoData handling strategies"""
    
    # Create sample data with NoData values
    def create_data_with_nodata():
        """Create sample raster with various NoData scenarios"""
        
        height, width = 80, 80
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        
        # Create base data
        data = np.random.random((height, width)) * 0.8 + 0.1
        
        # Introduce different types of NoData
        # 1. Systematic gaps (sensor issues)
        data[10:15, :] = np.nan
        
        # 2. Cloud contamination (irregular patches)
        cloud_mask = ((np.arange(height)[:, None] - 40)**2 + (np.arange(width) - 30)**2) < 200
        data[cloud_mask] = np.nan
        
        # 3. Edge effects
        data[:5, :] = np.nan
        data[-5:, :] = np.nan
        data[:, :5] = np.nan
        data[:, -5:] = np.nan
        
        # 4. Random missing pixels
        random_mask = np.random.random((height, width)) < 0.05
        data[random_mask] = np.nan
        
        return xr.DataArray(data, coords={'y': y, 'x': x}, dims=['y', 'x'])
    
    # NoData detection and analysis
    def analyze_nodata_patterns(data_array):
        """Analyze NoData patterns in raster data"""
        
        # Basic NoData statistics
        total_pixels = data_array.size
        nodata_pixels = np.isnan(data_array).sum().values
        valid_pixels = total_pixels - nodata_pixels
        nodata_percentage = (nodata_pixels / total_pixels) * 100
        
        # Spatial distribution analysis
        nodata_mask = np.isnan(data_array)
        
        # Connected component analysis of NoData regions
        from scipy import ndimage
        labeled_gaps, num_gaps = ndimage.label(nodata_mask)
        
        # Gap size analysis
        gap_sizes = []
        for gap_id in range(1, num_gaps + 1):
            gap_size = np.sum(labeled_gaps == gap_id)
            gap_sizes.append(gap_size)
        
        analysis_results = {
            'total_pixels': total_pixels,
            'valid_pixels': valid_pixels,
            'nodata_pixels': nodata_pixels,
            'nodata_percentage': nodata_percentage,
            'num_gaps': num_gaps,
            'gap_sizes': gap_sizes,
            'largest_gap': max(gap_sizes) if gap_sizes else 0,
            'mean_gap_size': np.mean(gap_sizes) if gap_sizes else 0
        }
        
        return analysis_results, nodata_mask
    
    # NoData interpolation methods
    def interpolate_nodata(data_array, method='linear'):
        """Interpolate NoData values using various methods"""
        
        interpolated_results = {}
        
        # Method 1: Linear interpolation
        if method in ['linear', 'all']:
            linear_interp = data_array.interpolate_na(dim='x', method='linear')
            linear_interp = linear_interp.interpolate_na(dim='y', method='linear')
            interpolated_results['linear'] = linear_interp
        
        # Method 2: Nearest neighbor
        if method in ['nearest', 'all']:
            nearest_interp = data_array.interpolate_na(dim='x', method='nearest')
            nearest_interp = nearest_interp.interpolate_na(dim='y', method='nearest')
            interpolated_results['nearest'] = nearest_interp
        
        # Method 3: Cubic spline
        if method in ['cubic', 'all']:
            try:
                cubic_interp = data_array.interpolate_na(dim='x', method='cubic')
                cubic_interp = cubic_interp.interpolate_na(dim='y', method='cubic')
                interpolated_results['cubic'] = cubic_interp
            except:
                print("Cubic interpolation failed, using linear instead")
                interpolated_results['cubic'] = interpolated_results.get('linear', data_array)
        
        # Method 4: Mean filling
        if method in ['mean', 'all']:
            mean_value = data_array.mean(skipna=True)
            mean_filled = data_array.fillna(mean_value)
            interpolated_results['mean'] = mean_filled
        
        return interpolated_results
    
    # Advanced gap filling
    def advanced_gap_filling(data_array):
        """Advanced gap filling techniques"""
        
        # Method 1: Focal statistics (moving window)
        def focal_mean_fill(data, window_size=3):
            """Fill gaps using focal mean"""
            from scipy import ndimage
            
            # Create kernel
            kernel = np.ones((window_size, window_size)) / (window_size**2)
            
            # Apply focal mean where data is missing
            filled_data = data.copy()
            nodata_mask = np.isnan(data)
            
            # Iterative filling
            for iteration in range(5):  # Maximum 5 iterations
                # Calculate focal mean
                focal_mean = ndimage.convolve(
                    np.where(np.isnan(filled_data), 0, filled_data),
                    kernel, mode='constant', cval=0
                )
                
                # Count valid neighbors
                valid_count = ndimage.convolve(
                    (~np.isnan(filled_data)).astype(float),
                    kernel, mode='constant', cval=0
                )
                
                # Fill gaps where we have enough neighbors
                focal_mean = np.where(valid_count > 0, focal_mean / valid_count, np.nan)
                filled_data = np.where(nodata_mask & ~np.isnan(focal_mean), focal_mean, filled_data)
                
                # Update mask
                nodata_mask = np.isnan(filled_data)
                
                if not np.any(nodata_mask):
                    break
            
            return filled_data
        
        # Method 2: Distance-weighted interpolation
        def distance_weighted_fill(data):
            """Fill gaps using inverse distance weighting"""
            
            filled_data = data.copy()
            nodata_mask = np.isnan(data)
            
            if not np.any(nodata_mask):
                return filled_data
            
            # Get coordinates of valid and invalid pixels
            valid_coords = np.where(~np.isnan(data))
            invalid_coords = np.where(nodata_mask)
            
            valid_values = data[valid_coords]
            
            # For each invalid pixel, calculate weighted average
            for i, (row, col) in enumerate(zip(invalid_coords[0], invalid_coords[1])):
                # Calculate distances to all valid pixels
                distances = np.sqrt((valid_coords[0] - row)**2 + (valid_coords[1] - col)**2)
                
                # Avoid division by zero
                distances = np.maximum(distances, 1e-10)
                
                # Calculate weights (inverse distance)
                weights = 1.0 / distances
                
                # Limit to nearest neighbors for efficiency
                if len(weights) > 20:
                    nearest_indices = np.argsort(distances)[:20]
                    weights = weights[nearest_indices]
                    values = valid_values[nearest_indices]
                else:
                    values = valid_values
                
                # Calculate weighted average
                weighted_value = np.sum(weights * values) / np.sum(weights)
                filled_data[row, col] = weighted_value
            
            return filled_data
        
        # Apply advanced methods
        focal_filled = focal_mean_fill(data_array.values)
        idw_filled = distance_weighted_fill(data_array.values)
        
        # Convert back to xarray
        focal_result = data_array.copy()
        focal_result.values = focal_filled
        
        idw_result = data_array.copy()
        idw_result.values = idw_filled
        
        return {
            'focal_mean': focal_result,
            'inverse_distance': idw_result
        }
    
    # Quality assessment of gap filling
    def assess_fill_quality(original, filled, validation_mask=None):
        """Assess quality of gap filling"""
        
        if validation_mask is None:
            # Create validation mask by randomly removing some valid pixels
            valid_pixels = ~np.isnan(original)
            validation_indices = np.where(valid_pixels)
            n_validation = min(100, len(validation_indices[0]))
            
            if n_validation > 0:
                val_idx = np.random.choice(len(validation_indices[0]), n_validation, replace=False)
                validation_mask = np.zeros_like(original, dtype=bool)
                validation_mask[validation_indices[0][val_idx], validation_indices[1][val_idx]] = True
            else:
                return None
        
        # Extract validation data
        original_vals = original[validation_mask]
        filled_vals = filled[validation_mask]
        
        # Calculate metrics
        mae = np.mean(np.abs(original_vals - filled_vals))
        rmse = np.sqrt(np.mean((original_vals - filled_vals)**2))
        correlation = np.corrcoef(original_vals, filled_vals)[0, 1] if len(original_vals) > 1 else np.nan
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'n_validation': len(original_vals)
        }
    
    # Run NoData handling workflow
    print("NODATA HANDLING WORKFLOW")
    print("=" * 25)
    
    # Create test data
    test_data = create_data_with_nodata()
    
    # Analyze NoData patterns
    nodata_analysis, nodata_mask = analyze_nodata_patterns(test_data)
    
    print(f"NoData Analysis:")
    print(f"  Total pixels: {nodata_analysis['total_pixels']:,}")
    print(f"  Valid pixels: {nodata_analysis['valid_pixels']:,}")
    print(f"  NoData pixels: {nodata_analysis['nodata_pixels']:,} ({nodata_analysis['nodata_percentage']:.1f}%)")
    print(f"  Number of gaps: {nodata_analysis['num_gaps']}")
    print(f"  Largest gap: {nodata_analysis['largest_gap']} pixels")
    
    # Apply interpolation methods
    interpolated = interpolate_nodata(test_data, method='all')
    advanced_filled = advanced_gap_filling(test_data)
    
    # Assess quality
    print(f"\nInterpolation Quality Assessment:")
    for method_name, filled_data in {**interpolated, **advanced_filled}.items():
        quality = assess_fill_quality(test_data, filled_data)
        if quality:
            print(f"  {method_name}: RMSE = {quality['rmse']:.4f}, Correlation = {quality['correlation']:.3f}")
    
    return test_data, interpolated, advanced_filled, nodata_analysis

# Robust mathematical operations with NoData
def robust_math_with_nodata():
    """Perform robust mathematical operations handling NoData"""
    
    # Create sample data with NoData
    data_with_gaps = handle_nodata_values()[0]
    
    # Safe mathematical operations
    def safe_divide(numerator, denominator, fill_value=np.nan):
        """Safe division handling division by zero and NoData"""
        
        # Handle NoData in inputs
        valid_mask = ~(np.isnan(numerator) | np.isnan(denominator))
        
        # Handle division by zero
        zero_mask = (denominator == 0)
        
        # Perform division
        result = np.full_like(numerator, fill_value)
        safe_mask = valid_mask & ~zero_mask
        
        if np.any(safe_mask):
            result[safe_mask] = numerator[safe_mask] / denominator[safe_mask]
        
        return result
    
    def safe_log(data, base=np.e, fill_value=np.nan):
        """Safe logarithm handling negative values and NoData"""
        
        result = np.full_like(data, fill_value)
        valid_mask = ~np.isnan(data) & (data > 0)
        
        if np.any(valid_mask):
            if base == np.e:
                result[valid_mask] = np.log(data[valid_mask])
            else:
                result[valid_mask] = np.log(data[valid_mask]) / np.log(base)
        
        return result
    
    def safe_sqrt(data, fill_value=np.nan):
        """Safe square root handling negative values and NoData"""
        
        result = np.full_like(data, fill_value)
        valid_mask = ~np.isnan(data) & (data >= 0)
        
        if np.any(valid_mask):
            result[valid_mask] = np.sqrt(data[valid_mask])
        
        return result
    
    # Demonstrate safe operations
    print("\nROBUST MATH WITH NODATA")
    print("=" * 24)
    
    # Create test scenarios
    test_data = data_with_gaps.values
    
    # Safe division example
    numerator = test_data * 2
    denominator = test_data - 0.5  # Some values will be negative
    
    safe_ratio = safe_divide(numerator, denominator)
    valid_ratio_pixels = np.sum(~np.isnan(safe_ratio))
    
    print(f"Safe division: {valid_ratio_pixels} valid results")
    
    # Safe logarithm example
    safe_log_result = safe_log(test_data)
    valid_log_pixels = np.sum(~np.isnan(safe_log_result))
    
    print(f"Safe logarithm: {valid_log_pixels} valid results")
    
    # Safe square root example
    safe_sqrt_result = safe_sqrt(test_data - 0.3)  # Some negative values
    valid_sqrt_pixels = np.sum(~np.isnan(safe_sqrt_result))
    
    print(f"Safe square root: {valid_sqrt_pixels} valid results")
    
    return {
        'safe_ratio': safe_ratio,
        'safe_log': safe_log_result,
        'safe_sqrt': safe_sqrt_result
    }

# Run NoData handling examples
nodata_test_data, interpolation_results, advanced_fill_results, nodata_stats = handle_nodata_values()
robust_math_results = robust_math_with_nodata()
```

---

## 4️⃣ Combining Multiple Bands into One Array

### Multi-band Array Operations
```python
def combine_multiple_bands():
    """Demonstrate combining multiple bands into unified arrays"""
    
    # Create sample multi-band dataset
    ds = create_sample_multispectral()
    
    # Method 1: Stack bands into 3D array
    def stack_bands_3d(dataset, band_names=None):
        """Stack multiple bands into 3D array (bands, height, width)"""
        
        if band_names is None:
            band_names = ['blue', 'green', 'red', 'nir', 'swir']
        
        # Extract band data
        band_arrays = []
        for band_name in band_names:
            if band_name in dataset:
                band_arrays.append(dataset[band_name].values)
            else:
                print(f"Warning: Band {band_name} not found")
        
        # Stack into 3D array
        stacked_array = np.stack(band_arrays, axis=0)
        
        print(f"Stacked array shape: {stacked_array.shape}")
        print(f"Bands included: {band_names}")
        
        return stacked_array, band_names
    
    # Method 2: Create composite bands
    def create_composite_bands(dataset):
        """Create composite bands from multiple spectral bands"""
        
        composites = {}
        
        # RGB composite (natural color)
        rgb_composite = np.stack([
            dataset.red.values,
            dataset.green.values,
            dataset.blue.values
        ], axis=-1)
        composites['rgb_natural'] = rgb_composite
        
        # False color composite (NIR, Red, Green)
        false_color = np.stack([
            dataset.nir.values,
            dataset.red.values,
            dataset.green.values
        ], axis=-1)
        composites['false_color'] = false_color
        
        # SWIR composite (SWIR, NIR, Red)
        swir_composite = np.stack([
            dataset.swir.values,
            dataset.nir.values,
            dataset.red.values
        ], axis=-1)
        composites['swir_composite'] = swir_composite
        
        # Vegetation composite (NIR, Green, Blue)
        vegetation_composite = np.stack([
            dataset.nir.values,
            dataset.green.values,
            dataset.blue.values
        ], axis=-1)
        composites['vegetation'] = vegetation_composite
        
        return composites
    
    # Method 3: Principal Component Analysis
    def apply_pca_bands(stacked_array):
        """Apply PCA to reduce dimensionality of multi-band data"""
        
        from sklearn.decomposition import PCA
        
        # Reshape for PCA (pixels x bands)
        n_bands, height, width = stacked_array.shape
        reshaped_data = stacked_array.reshape(n_bands, -1).T
        
        # Remove NaN values for PCA
        valid_mask = ~np.isnan(reshaped_data).any(axis=1)
        valid_data = reshaped_data[valid_mask]
        
        if len(valid_data) == 0:
            print("No valid data for PCA")
            return None, None
        
        # Apply PCA
        pca = PCA(n_components=min(n_bands, 4))
        pca_result = pca.fit_transform(valid_data)
        
        # Reshape back to spatial dimensions
        pca_bands = np.full((len(reshaped_data), pca.n_components_), np.nan)
        pca_bands[valid_mask] = pca_result
        pca_bands = pca_bands.T.reshape(pca.n_components_, height, width)
        
        # PCA statistics
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        pca_info = {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'components': pca.components_,
            'n_components': pca.n_components_
        }
        
        print(f"PCA Results:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"  PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")
        
        return pca_bands, pca_info
    
    # Method 4: Band ratios and indices array
    def create_indices_array(dataset):
        """Create array of multiple spectral indices"""
        
        # Calculate various indices
        indices_dict = {}
        
        # Vegetation indices
        indices_dict['ndvi'] = (dataset.nir - dataset.red) / (dataset.nir + dataset.red)
        indices_dict['evi'] = 2.5 * ((dataset.nir - dataset.red) / (dataset.nir + 6 * dataset.red - 7.5 * dataset.blue + 1))
        indices_dict['savi'] = ((dataset.nir - dataset.red) / (dataset.nir + dataset.red + 0.5)) * 1.5
        
        # Water indices
        indices_dict['ndwi'] = (dataset.green - dataset.nir) / (dataset.green + dataset.nir)
        indices_dict['mndwi'] = (dataset.green - dataset.swir) / (dataset.green + dataset.swir)
        
        # Soil indices
        indices_dict['brightness'] = (dataset.red + dataset.green + dataset.blue) / 3
        indices_dict['greenness'] = dataset.green - dataset.red
        
        # Band ratios
        indices_dict['nir_red_ratio'] = dataset.nir / dataset.red
        indices_dict['swir_nir_ratio'] = dataset.swir / dataset.nir
        
        # Stack indices into array
        indices_list = []
        index_names = []
        
        for name, index_data in indices_dict.items():
            indices_list.append(index_data.values)
            index_names.append(name)
        
        indices_array = np.stack(indices_list, axis=0)
        
        print(f"Indices array shape: {indices_array.shape}")
        print(f"Indices included: {index_names}")
        
        return indices_array, index_names, indices_dict
    
    # Method 5: Texture analysis bands
    def calculate_texture_bands(data_array, window_size=3):
        """Calculate texture measures from single band"""
        
        from scipy import ndimage
        from skimage.feature import graycomatrix, graycoprops
        
        texture_bands = {}
        
        # Statistical texture measures
        # Mean
        mean_filter = np.ones((window_size, window_size)) / (window_size**2)
        texture_bands['mean'] = ndimage.convolve(data_array, mean_filter)
        
        # Standard deviation
        mean_img = texture_bands['mean']
        variance = ndimage.convolve((data_array - mean_img)**2, mean_filter)
        texture_bands['std'] = np.sqrt(variance)
        
        # Range
        max_filter = ndimage.maximum_filter(data_array, size=window_size)
        min_filter = ndimage.minimum_filter(data_array, size=window_size)
        texture_bands['range'] = max_filter - min_filter
        
        # Gradient magnitude
        grad_x = ndimage.sobel(data_array, axis=1)
        grad_y = ndimage.sobel(data_array, axis=0)
        texture_bands['gradient'] = np.sqrt(grad_x**2 + grad_y**2)
        
        # Stack texture bands
        texture_list = []
        texture_names = []
        
        for name, texture_data in texture_bands.items():
            texture_list.append(texture_data)
            texture_names.append(f'texture_{name}')
        
        texture_array = np.stack(texture_list, axis=0)
        
        return texture_array, texture_names
    
    # Execute all methods
    print("COMBINING MULTIPLE BANDS")
    print("=" * 24)
    
    # Stack bands
    stacked_bands, band_names = stack_bands_3d(ds)
    
    # Create composites
    composites = create_composite_bands(ds)
    print(f"Created {len(composites)} composite images")
    
    # Apply PCA
    pca_bands, pca_info = apply_pca_bands(stacked_bands)
    
    # Create indices array
    indices_array, index_names, indices_dict = create_indices_array(ds)
    
    # Calculate texture bands (using NIR band)
    texture_array, texture_names = calculate_texture_bands(ds.nir.values)
    
    # Combine everything into master array
    def create_master_array():
        """Combine all derived bands into single master array"""
        
        all_bands = []
        all_names = []
        
        # Original bands
        for i, name in enumerate(band_names):
            all_bands.append(stacked_bands[i])
            all_names.append(f'original_{name}')
        
        # PCA bands
        if pca_bands is not None:
            for i in range(pca_bands.shape[0]):
                all_bands.append(pca_bands[i])
                all_names.append(f'pca_{i+1}')
        
        # Indices
        for i, name in enumerate(index_names):
            all_bands.append(indices_array[i])
            all_names.append(name)
        
        # Texture bands
        for i, name in enumerate(texture_names):
            all_bands.append(texture_array[i])
            all_names.append(name)
        
        master_array = np.stack(all_bands, axis=0)
        
        print(f"Master array shape: {master_array.shape}")
        print(f"Total bands: {len(all_names)}")
        
        return master_array, all_names
    
    master_array, master_names = create_master_array()
    
    # Array statistics
    def analyze_band_array(array, names):
        """Analyze statistics of multi-band array"""
        
        stats = {}
        
        for i, name in enumerate(names):
            band_data = array[i]
            valid_data = band_data[~np.isnan(band_data)]
            
            if len(valid_data) > 0:
                stats[name] = {
                    'mean': np.mean(valid_data),
                    'std': np.std(valid_data),
                    'min': np.min(valid_data),
                    'max': np.max(valid_data),
                    'valid_pixels': len(valid_data)
                }
        
        return stats
    
    array_stats = analyze_band_array(master_array, master_names)
    
    print(f"\nBand Statistics Summary:")
    for name, stats in list(array_stats.items())[:5]:  # Show first 5
        print(f"  {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    return {
        'stacked_bands': stacked_bands,
        'composites': composites,
        'pca_bands': pca_bands,
        'indices_array': indices_array,
        'texture_array': texture_array,
        'master_array': master_array,
        'master_names': master_names,
        'array_stats': array_stats
    }

# Advanced band combination techniques
def advanced_band_combinations():
    """Advanced techniques for band combination and analysis"""
    
    ds = create_sample_multispectral()
    
    # Optimal band selection
    def select_optimal_bands(stacked_array, target_bands=3):
        """Select optimal bands using correlation analysis"""
        
        n_bands, height, width = stacked_array.shape
        reshaped_data = stacked_array.reshape(n_bands, -1)
        
        # Remove NaN values
        valid_mask = ~np.isnan(reshaped_data).any(axis=0)
        valid_data = reshaped_data[:, valid_mask]
        
        if valid_data.shape[1] == 0:
            return list(range(min(target_bands, n_bands)))
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(valid_data)
        
        # Select bands with lowest inter-correlation
        selected_bands = [0]  # Start with first band
        
        for _ in range(target_bands - 1):
            min_correlation = float('inf')
            best_band = None
            
            for candidate in range(n_bands):
                if candidate in selected_bands:
                    continue
                
                # Calculate mean correlation with selected bands
                mean_corr = np.mean([abs(correlation_matrix[candidate, selected]) 
                                   for selected in selected_bands])
                
                if mean_corr < min_correlation:
                    min_correlation = mean_corr
                    best_band = candidate
            
            if best_band is not None:
                selected_bands.append(best_band)
        
        return selected_bands
    
    # Weighted band combination
    def create_weighted_composite(bands_dict, weights=None):
        """Create weighted composite from multiple bands"""
        
        band_names = list(bands_dict.keys())
        
        if weights is None:
            weights = np.ones(len(band_names)) / len(band_names)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Create weighted composite
        composite = np.zeros_like(list(bands_dict.values())[0])
        
        for i, (name, band_data) in enumerate(bands_dict.items()):
            composite += weights[i] * band_data
        
        return composite
    
    # Apply advanced techniques
    stacked_bands, _ = stack_bands_3d(ds)
    optimal_bands = select_optimal_bands(stacked_bands, target_bands=3)
    
    # Create weighted composite
    band_dict = {
        'nir': ds.nir.values,
        'red': ds.red.values,
        'green': ds.green.values
    }
    
    # Vegetation-weighted composite
    veg_weights = [0.5, 0.3, 0.2]  # Emphasize NIR for vegetation
    veg_composite = create_weighted_composite(band_dict, veg_weights)
    
    print(f"\nADVANCED BAND COMBINATIONS")
    print("=" * 27)
    print(f"Optimal bands selected: {optimal_bands}")
    print(f"Vegetation composite range: {np.nanmin(veg_composite):.3f} to {np.nanmax(veg_composite):.3f}")
    
    return {
        'optimal_bands': optimal_bands,
        'vegetation_composite': veg_composite
    }

# Run band combination examples
band_combination_results = combine_multiple_bands()
advanced_combination_results = advanced_band_combinations()
```

---

## Best Practices and Tips

### Computational Efficiency
- **Vectorized Operations**: Use NumPy vectorized operations instead of loops
- **Memory Management**: Process large arrays in chunks when necessary
- **Data Types**: Choose appropriate data types to minimize memory usage
- **Lazy Evaluation**: Use dask for large datasets that don't fit in memory

### NoData Handling
- **Consistent NoData Values**: Use standardized NoData values across datasets
- **Propagation Rules**: Understand how NoData propagates through calculations
- **Quality Flags**: Maintain quality flags alongside data values
- **Validation**: Always validate interpolation results against known good data

### Index Calculation
- **Range Validation**: Ensure indices are within expected ranges
- **Atmospheric Correction**: Apply atmospheric correction before calculating indices
- **Temporal Consistency**: Use consistent methods across time series
- **Documentation**: Document index formulations and parameter choices

This comprehensive guide provides the foundation for performing sophisticated raster mathematics and band operations, enabling quantitative analysis of multispectral imagery for diverse remote sensing applications.