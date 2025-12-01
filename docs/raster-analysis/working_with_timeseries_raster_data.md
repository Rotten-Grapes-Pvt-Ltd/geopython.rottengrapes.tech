# Working with Time-Series Raster Data

## Overview
Time-series raster analysis enables monitoring of environmental changes, vegetation dynamics, and climate patterns over time. Using xarray and rioxarray, we can efficiently handle multi-dimensional datasets with temporal, spectral, and spatial dimensions, providing powerful tools for trend analysis and temporal pattern detection.

## Why Time-Series Raster Analysis Matters
- **Change Detection**: Monitor environmental changes over time
- **Trend Analysis**: Identify long-term patterns and cycles
- **Seasonal Monitoring**: Track seasonal vegetation and climate patterns
- **Anomaly Detection**: Identify unusual events or deviations
- **Forecasting**: Predict future conditions based on historical trends

---

## 1️⃣ Using xarray and rioxarray for 4D data (time, band, lat, lon)

### Creating and Managing 4D Datasets
```python
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create sample 4D time-series dataset
def create_4d_timeseries():
    """Create sample 4D raster dataset (time, band, lat, lon)"""
    
    # Define dimensions
    n_times = 24  # 2 years of monthly data
    n_bands = 4   # RGB + NIR
    n_lat = 50
    n_lon = 50
    
    # Create coordinate arrays
    times = pd.date_range('2022-01-01', periods=n_times, freq='MS')
    bands = ['red', 'green', 'blue', 'nir']
    lat = np.linspace(40.0, 41.0, n_lat)
    lon = np.linspace(-74.0, -73.0, n_lon)
    
    # Create meshgrids for spatial patterns
    LAT, LON = np.meshgrid(lat, lon, indexing='ij')
    
    # Generate realistic time-series data
    data = np.zeros((n_times, n_bands, n_lat, n_lon))
    
    for t_idx, time in enumerate(times):
        # Seasonal component (stronger in summer)
        seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (time.month - 3) / 12)
        
        # Spatial gradient (vegetation increases towards center)
        spatial_pattern = np.exp(-((LAT - 40.5)**2 + (LON + 73.5)**2) / 0.1)
        
        for b_idx, band in enumerate(bands):
            if band == 'nir':
                # NIR has higher values and stronger seasonal variation
                base_value = 0.6 + 0.3 * seasonal_factor
                band_data = base_value * spatial_pattern
            elif band == 'red':
                # Red has inverse relationship with vegetation
                base_value = 0.3 - 0.2 * seasonal_factor
                band_data = base_value * (1 - 0.5 * spatial_pattern)
            else:  # green, blue
                base_value = 0.4 + 0.2 * seasonal_factor
                band_data = base_value * spatial_pattern
            
            # Add noise
            band_data += 0.05 * np.random.random((n_lat, n_lon))
            
            # Clip to valid range
            data[t_idx, b_idx] = np.clip(band_data, 0, 1)
    
    # Create xarray Dataset
    ds = xr.Dataset({
        'reflectance': (['time', 'band', 'lat', 'lon'], data)
    }, coords={
        'time': times,
        'band': bands,
        'lat': lat,
        'lon': lon
    })
    
    # Add attributes
    ds.attrs['title'] = 'Sample Multi-temporal Satellite Data'
    ds.attrs['description'] = 'Simulated monthly reflectance data'
    ds['reflectance'].attrs['units'] = 'reflectance'
    ds['reflectance'].attrs['valid_range'] = [0, 1]
    
    # Set spatial reference
    ds.rio.write_crs('EPSG:4326', inplace=True)
    
    return ds

# Load and explore 4D dataset
def explore_4d_dataset():
    """Explore structure and properties of 4D dataset"""
    
    # Create sample dataset
    ds = create_4d_timeseries()
    
    print("4D DATASET STRUCTURE")
    print("=" * 20)
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords.keys())}")
    print(f"Data variables: {list(ds.data_vars.keys())}")
    print(f"Shape: {ds.reflectance.shape}")
    print(f"Size in memory: {ds.nbytes / 1e6:.1f} MB")
    
    # Basic dataset info
    print(f"\nTemporal Coverage:")
    print(f"  Start: {ds.time.min().values}")
    print(f"  End: {ds.time.max().values}")
    print(f"  Frequency: {pd.infer_freq(ds.time.values)}")
    
    print(f"\nSpatial Coverage:")
    print(f"  Lat range: {ds.lat.min().values:.3f} to {ds.lat.max().values:.3f}")
    print(f"  Lon range: {ds.lon.min().values:.3f} to {ds.lon.max().values:.3f}")
    print(f"  CRS: {ds.rio.crs}")
    
    print(f"\nSpectral Bands: {list(ds.band.values)}")
    
    # Demonstrate indexing and selection
    print(f"\nDATA ACCESS EXAMPLES:")
    print("=" * 20)
    
    # Select single time step
    single_time = ds.sel(time='2022-06-01')
    print(f"Single time step shape: {single_time.reflectance.shape}")
    
    # Select single band
    nir_band = ds.sel(band='nir')
    print(f"NIR band shape: {nir_band.reflectance.shape}")
    
    # Select spatial subset
    spatial_subset = ds.sel(lat=slice(40.2, 40.8), lon=slice(-73.8, -73.2))
    print(f"Spatial subset shape: {spatial_subset.reflectance.shape}")
    
    # Select time range
    summer_data = ds.sel(time=slice('2022-06-01', '2022-08-31'))
    print(f"Summer data shape: {summer_data.reflectance.shape}")
    
    return ds

# Create and explore dataset
sample_4d_dataset = explore_4d_dataset()
```

### Advanced 4D Data Operations
```python
def advanced_4d_operations():
    """Demonstrate advanced operations on 4D datasets"""
    
    # Use existing dataset
    ds = sample_4d_dataset
    
    # Calculate NDVI time series
    def calculate_ndvi_timeseries(dataset):
        """Calculate NDVI for entire time series"""
        
        red = dataset.sel(band='red').reflectance
        nir = dataset.sel(band='nir').reflectance
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red)
        
        # Create new dataset with NDVI
        ndvi_ds = xr.Dataset({
            'ndvi': (['time', 'lat', 'lon'], ndvi.values)
        }, coords={
            'time': dataset.time,
            'lat': dataset.lat,
            'lon': dataset.lon
        })
        
        ndvi_ds.rio.write_crs(dataset.rio.crs, inplace=True)
        ndvi_ds['ndvi'].attrs['long_name'] = 'Normalized Difference Vegetation Index'
        ndvi_ds['ndvi'].attrs['valid_range'] = [-1, 1]
        
        return ndvi_ds
    
    # Calculate NDVI
    ndvi_ts = calculate_ndvi_timeseries(ds)
    
    # Demonstrate chunking for large datasets
    chunked_ds = ds.chunk({'time': 6, 'lat': 25, 'lon': 25})
    print(f"Original chunks: {ds.chunks}")
    print(f"Chunked dataset: {chunked_ds.chunks}")
    
    # Memory-efficient operations with dask
    mean_reflectance = chunked_ds.reflectance.mean(dim=['lat', 'lon'])
    print(f"Mean reflectance computation (lazy): {type(mean_reflectance.data)}")
    
    # Compute result
    mean_result = mean_reflectance.compute()
    print(f"Mean reflectance shape: {mean_result.shape}")
    
    # Visualize 4D data structure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Time series at single pixel
    pixel_lat, pixel_lon = 40.5, -73.5
    pixel_ts = ndvi_ts.sel(lat=pixel_lat, lon=pixel_lon, method='nearest')
    
    axes[0, 0].plot(pixel_ts.time, pixel_ts.ndvi, 'o-')
    axes[0, 0].set_title(f'NDVI Time Series\nPixel: {pixel_lat:.2f}, {pixel_lon:.2f}')
    axes[0, 0].set_ylabel('NDVI')
    axes[0, 0].grid(True)
    
    # Spatial patterns for different times
    times_to_show = ['2022-03-01', '2022-07-01', '2022-11-01']
    seasons = ['Spring', 'Summer', 'Autumn']
    
    for i, (time_str, season) in enumerate(zip(times_to_show, seasons)):
        ndvi_map = ndvi_ts.sel(time=time_str)
        im = axes[0, i+1].imshow(ndvi_map.ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0, i+1].set_title(f'{season} NDVI\n{time_str}')
        if i == 2:
            plt.colorbar(im, ax=axes[0, i+1])
    
    # Band comparison over time
    band_means = ds.reflectance.mean(dim=['lat', 'lon'])
    
    for band in ds.band.values:
        band_data = band_means.sel(band=band)
        axes[1, 0].plot(band_data.time, band_data, 'o-', label=band)
    
    axes[1, 0].set_title('Mean Reflectance by Band')
    axes[1, 0].set_ylabel('Reflectance')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Seasonal cycle
    monthly_ndvi = ndvi_ts.groupby('time.month').mean()
    axes[1, 1].plot(monthly_ndvi.month, monthly_ndvi.ndvi.mean(dim=['lat', 'lon']), 'o-')
    axes[1, 1].set_title('Average Seasonal Cycle')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Mean NDVI')
    axes[1, 1].grid(True)
    
    # Spatial variability over time
    ndvi_std = ndvi_ts.ndvi.std(dim=['lat', 'lon'])
    axes[1, 2].plot(ndvi_ts.time, ndvi_std, 'o-', color='red')
    axes[1, 2].set_title('Spatial Variability Over Time')
    axes[1, 2].set_ylabel('NDVI Standard Deviation')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return ndvi_ts, chunked_ds

# Run advanced operations
ndvi_timeseries, chunked_dataset = advanced_4d_operations()
```

---

## 2️⃣ Loading NetCDF / HDF formats

### Working with NetCDF Files
```python
def work_with_netcdf():
    """Demonstrate loading and working with NetCDF files"""
    
    # Create sample NetCDF file
    def create_sample_netcdf():
        """Create sample NetCDF file with climate data"""
        
        # Create sample climate dataset
        n_time = 365  # Daily data for one year
        n_lat = 30
        n_lon = 40
        
        # Coordinates
        time = pd.date_range('2023-01-01', periods=n_time, freq='D')
        lat = np.linspace(35, 45, n_lat)
        lon = np.linspace(-120, -110, n_lon)
        
        # Create realistic climate patterns
        LAT, LON = np.meshgrid(lat, lon, indexing='ij')
        
        # Temperature with seasonal cycle and spatial gradient
        temp_data = np.zeros((n_time, n_lat, n_lon))
        precip_data = np.zeros((n_time, n_lat, n_lon))
        
        for t_idx, date in enumerate(time):
            # Seasonal temperature cycle
            day_of_year = date.dayofyear
            seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            # Spatial temperature gradient (cooler at higher latitudes)
            spatial_temp = seasonal_temp - 0.5 * (LAT - 40)
            
            # Add daily variation and noise
            temp_data[t_idx] = spatial_temp + 2 * np.random.random((n_lat, n_lon))
            
            # Precipitation (more in winter, spatial pattern)
            seasonal_precip = 2 + 3 * np.sin(2 * np.pi * (day_of_year - 350) / 365)
            spatial_precip = seasonal_precip * (1 + 0.5 * np.sin(LAT * np.pi / 180))
            
            # Random precipitation events
            precip_events = np.random.exponential(1, (n_lat, n_lon)) * seasonal_precip
            precip_data[t_idx] = np.where(np.random.random((n_lat, n_lon)) < 0.3, 
                                        precip_events, 0)
        
        # Create xarray Dataset
        climate_ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], temp_data),
            'precipitation': (['time', 'lat', 'lon'], precip_data)
        }, coords={
            'time': time,
            'lat': lat,
            'lon': lon
        })
        
        # Add metadata
        climate_ds.attrs.update({
            'title': 'Sample Climate Data',
            'institution': 'Sample Weather Service',
            'source': 'Simulated data for demonstration',
            'conventions': 'CF-1.6'
        })
        
        climate_ds['temperature'].attrs.update({
            'long_name': 'Daily Mean Temperature',
            'units': 'degrees_C',
            'standard_name': 'air_temperature'
        })
        
        climate_ds['precipitation'].attrs.update({
            'long_name': 'Daily Precipitation',
            'units': 'mm',
            'standard_name': 'precipitation_amount'
        })
        
        # Save to NetCDF
        climate_ds.to_netcdf('sample_climate.nc')
        
        return 'sample_climate.nc'
    
    # Create sample file
    netcdf_file = create_sample_netcdf()
    
    # Load NetCDF file
    print("LOADING NETCDF FILE")
    print("=" * 20)
    
    # Method 1: Basic xarray loading
    ds = xr.open_dataset(netcdf_file)
    print(f"Dataset loaded: {list(ds.data_vars.keys())}")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    
    # Method 2: Load with chunks for large files
    ds_chunked = xr.open_dataset(netcdf_file, chunks={'time': 30, 'lat': 15, 'lon': 20})
    print(f"Chunked loading: {ds_chunked.chunks}")
    
    # Method 3: Load specific variables only
    temp_only = xr.open_dataset(netcdf_file, data_vars=['temperature'])
    print(f"Temperature only: {list(temp_only.data_vars.keys())}")
    
    # Explore dataset structure
    print(f"\nDATASET INFORMATION:")
    print(f"File size: {ds.nbytes / 1e6:.1f} MB")
    print(f"Variables: {len(ds.data_vars)}")
    print(f"Global attributes: {len(ds.attrs)}")
    
    # Display metadata
    print(f"\nGLOBAL ATTRIBUTES:")
    for key, value in ds.attrs.items():
        print(f"  {key}: {value}")
    
    print(f"\nVARIABLE ATTRIBUTES (Temperature):")
    for key, value in ds.temperature.attrs.items():
        print(f"  {key}: {value}")
    
    return ds, netcdf_file

# Load and explore NetCDF
climate_dataset, climate_file = work_with_netcdf()
```

### Working with HDF Files
```python
def work_with_hdf():
    """Demonstrate loading and working with HDF files"""
    
    # Create sample HDF5 file (simulating satellite data)
    def create_sample_hdf():
        """Create sample HDF5 file with satellite-like data"""
        
        import h5py
        
        # Create sample satellite swath data
        n_along_track = 200
        n_across_track = 150
        n_bands = 6
        
        # Create realistic satellite data patterns
        along_track = np.arange(n_along_track)
        across_track = np.arange(n_across_track)
        
        # Simulate geolocation
        lat_data = 40 + 5 * np.sin(along_track / 50)[:, np.newaxis] * np.ones(n_across_track)
        lon_data = -120 + 10 * (across_track / n_across_track) + \
                   2 * np.sin(along_track / 30)[:, np.newaxis]
        
        # Simulate reflectance data
        reflectance_data = np.random.random((n_bands, n_along_track, n_across_track))
        
        # Add realistic patterns
        for band in range(n_bands):
            # Add spatial correlation
            for i in range(1, n_along_track-1):
                for j in range(1, n_across_track-1):
                    reflectance_data[band, i, j] = (
                        0.7 * reflectance_data[band, i, j] +
                        0.1 * (reflectance_data[band, i-1, j] + 
                               reflectance_data[band, i+1, j] +
                               reflectance_data[band, i, j-1] + 
                               reflectance_data[band, i, j+1]) / 4
                    )
        
        # Create HDF5 file
        with h5py.File('sample_satellite.h5', 'w') as f:
            # Create groups
            geo_group = f.create_group('Geolocation')
            data_group = f.create_group('Data')
            
            # Store geolocation
            geo_group.create_dataset('Latitude', data=lat_data)
            geo_group.create_dataset('Longitude', data=lon_data)
            
            # Store reflectance data
            refl_dataset = data_group.create_dataset('Reflectance', data=reflectance_data)
            
            # Add attributes
            f.attrs['title'] = 'Sample Satellite Data'
            f.attrs['instrument'] = 'Simulated Sensor'
            f.attrs['processing_level'] = 'L1B'
            
            geo_group['Latitude'].attrs['units'] = 'degrees_north'
            geo_group['Longitude'].attrs['units'] = 'degrees_east'
            
            refl_dataset.attrs['units'] = 'reflectance'
            refl_dataset.attrs['valid_range'] = [0.0, 1.0]
            refl_dataset.attrs['bands'] = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
        
        return 'sample_satellite.h5'
    
    # Create sample HDF file
    hdf_file = create_sample_hdf()
    
    # Load HDF file with xarray
    print("LOADING HDF5 FILE")
    print("=" * 17)
    
    try:
        # Method 1: Load with h5py first to explore structure
        import h5py
        
        with h5py.File(hdf_file, 'r') as f:
            print("HDF5 File Structure:")
            
            def print_structure(name, obj):
                print(f"  {name}: {type(obj).__name__}")
                if hasattr(obj, 'shape'):
                    print(f"    Shape: {obj.shape}")
                if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
                    print(f"    Attributes: {list(obj.attrs.keys())}")
            
            f.visititems(print_structure)
        
        # Method 2: Load specific datasets with xarray
        # Load reflectance data
        refl_da = xr.open_dataset(hdf_file, group='Data')['Reflectance']
        
        # Load geolocation
        lat_da = xr.open_dataset(hdf_file, group='Geolocation')['Latitude']
        lon_da = xr.open_dataset(hdf_file, group='Geolocation')['Longitude']
        
        print(f"\nReflectance data shape: {refl_da.shape}")
        print(f"Latitude data shape: {lat_da.shape}")
        print(f"Longitude data shape: {lon_da.shape}")
        
        # Create properly georeferenced dataset
        satellite_ds = xr.Dataset({
            'reflectance': (['band', 'along_track', 'across_track'], refl_da.values),
            'latitude': (['along_track', 'across_track'], lat_da.values),
            'longitude': (['along_track', 'across_track'], lon_da.values)
        }, coords={
            'band': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6'],
            'along_track': np.arange(refl_da.shape[1]),
            'across_track': np.arange(refl_da.shape[2])
        })
        
        print(f"\nSatellite dataset created:")
        print(f"  Variables: {list(satellite_ds.data_vars.keys())}")
        print(f"  Dimensions: {dict(satellite_ds.dims)}")
        
        return satellite_ds, hdf_file
        
    except ImportError:
        print("h5py not available. Install with: pip install h5py")
        return None, hdf_file

# Load and explore HDF
try:
    satellite_dataset, satellite_file = work_with_hdf()
except Exception as e:
    print(f"HDF example skipped: {e}")
    satellite_dataset = None
```

---

## 3️⃣ Calculating Temporal Trends (e.g., NDVI over time)

### Linear Trend Analysis
```python
def calculate_temporal_trends():
    """Calculate and analyze temporal trends in raster time series"""
    
    # Use NDVI time series from earlier
    ndvi_ds = ndvi_timeseries
    
    # Method 1: Linear trend using polyfit
    def calculate_linear_trend(data_array):
        """Calculate linear trend for each pixel"""
        
        # Convert time to numeric (days since start)
        time_numeric = (data_array.time - data_array.time[0]) / pd.Timedelta(days=1)
        
        # Calculate trend using polyfit along time dimension
        def trend_func(y):
            if np.isnan(y).all():
                return np.array([np.nan, np.nan])
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < 3:  # Need at least 3 points
                return np.array([np.nan, np.nan])
            return np.polyfit(time_numeric[valid_mask], y[valid_mask], 1)
        
        # Apply along time dimension
        trends = xr.apply_ufunc(
            trend_func,
            data_array,
            input_core_dims=[['time']],
            output_core_dims=[['coeff']],
            output_sizes={'coeff': 2},
            dask='allowed',
            output_dtypes=[float]
        )
        
        # Split into slope and intercept
        slope = trends.isel(coeff=0)
        intercept = trends.isel(coeff=1)
        
        # Convert slope to per-year units
        slope_per_year = slope * 365.25
        
        return slope_per_year, intercept
    
    # Calculate trends
    print("CALCULATING TEMPORAL TRENDS")
    print("=" * 27)
    
    ndvi_slope, ndvi_intercept = calculate_linear_trend(ndvi_ds.ndvi)
    
    # Calculate trend statistics
    print(f"NDVI Trend Statistics:")
    print(f"  Mean slope: {ndvi_slope.mean().values:.4f} NDVI/year")
    print(f"  Slope range: {ndvi_slope.min().values:.4f} to {ndvi_slope.max().values:.4f}")
    print(f"  Pixels with positive trend: {(ndvi_slope > 0).sum().values}")
    print(f"  Pixels with negative trend: {(ndvi_slope < 0).sum().values}")
    
    # Method 2: Seasonal trend decomposition
    def seasonal_trend_decomposition(data_array):
        """Decompose time series into trend, seasonal, and residual components"""
        
        # Calculate monthly climatology
        monthly_clim = data_array.groupby('time.month').mean()
        
        # Remove seasonal cycle
        deseasonalized = data_array.groupby('time.month') - monthly_clim
        
        # Calculate trend on deseasonalized data
        trend_slope, _ = calculate_linear_trend(deseasonalized)
        
        # Calculate seasonal amplitude
        seasonal_amplitude = monthly_clim.max(dim='month') - monthly_clim.min(dim='month')
        
        return trend_slope, seasonal_amplitude, monthly_clim
    
    # Perform seasonal decomposition
    seasonal_trend, seasonal_amp, monthly_climatology = seasonal_trend_decomposition(ndvi_ds.ndvi)
    
    print(f"\nSeasonal Analysis:")
    print(f"  Mean seasonal amplitude: {seasonal_amp.mean().values:.3f}")
    print(f"  Deseasonalized trend: {seasonal_trend.mean().values:.4f} NDVI/year")
    
    # Method 3: Significance testing
    def trend_significance(data_array, alpha=0.05):
        """Test statistical significance of trends using Mann-Kendall test"""
        
        from scipy import stats
        
        def mann_kendall_test(y):
            """Mann-Kendall trend test for single time series"""
            if np.isnan(y).all() or len(y) < 3:
                return np.array([np.nan, np.nan])
            
            valid_data = y[~np.isnan(y)]
            n = len(valid_data)
            
            # Calculate S statistic
            S = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    S += np.sign(valid_data[j] - valid_data[i])
            
            # Calculate variance
            var_S = n * (n-1) * (2*n+5) / 18
            
            # Calculate Z statistic
            if S > 0:
                Z = (S - 1) / np.sqrt(var_S)
            elif S < 0:
                Z = (S + 1) / np.sqrt(var_S)
            else:
                Z = 0
            
            # Calculate p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
            
            return np.array([Z, p_value])
        
        # Apply Mann-Kendall test
        mk_results = xr.apply_ufunc(
            mann_kendall_test,
            data_array,
            input_core_dims=[['time']],
            output_core_dims=[['stat']],
            output_sizes={'stat': 2},
            dask='allowed',
            output_dtypes=[float]
        )
        
        z_stat = mk_results.isel(stat=0)
        p_value = mk_results.isel(stat=1)
        
        # Determine significance
        significant = p_value < alpha
        
        return z_stat, p_value, significant
    
    # Test trend significance
    z_statistic, p_values, is_significant = trend_significance(ndvi_ds.ndvi)
    
    print(f"\nTrend Significance (α = 0.05):")
    print(f"  Significant pixels: {is_significant.sum().values} / {is_significant.size}")
    print(f"  Percentage significant: {(is_significant.sum() / is_significant.size * 100).values:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original time series at sample pixel
    sample_lat, sample_lon = ndvi_ds.lat[25], ndvi_ds.lon[25]
    pixel_ts = ndvi_ds.sel(lat=sample_lat, lon=sample_lon, method='nearest')
    
    axes[0, 0].plot(pixel_ts.time, pixel_ts.ndvi, 'o-', alpha=0.7)
    
    # Add trend line
    time_numeric = (pixel_ts.time - pixel_ts.time[0]) / pd.Timedelta(days=1)
    slope_val = ndvi_slope.sel(lat=sample_lat, lon=sample_lon, method='nearest').values
    intercept_val = ndvi_intercept.sel(lat=sample_lat, lon=sample_lon, method='nearest').values
    trend_line = slope_val * time_numeric + intercept_val
    
    axes[0, 0].plot(pixel_ts.time, trend_line, 'r-', linewidth=2, label=f'Trend: {slope_val*365:.3f}/year')
    axes[0, 0].set_title('NDVI Time Series with Trend')
    axes[0, 0].set_ylabel('NDVI')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Trend map
    im1 = axes[0, 1].imshow(ndvi_slope, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    axes[0, 1].set_title('NDVI Trend (per year)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Significance map
    im2 = axes[0, 2].imshow(is_significant, cmap='RdYlBu_r')
    axes[0, 2].set_title('Trend Significance (p < 0.05)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Seasonal climatology
    monthly_mean = monthly_climatology.mean(dim=['lat', 'lon'])
    axes[1, 0].plot(monthly_mean.month, monthly_mean, 'o-')
    axes[1, 0].set_title('Mean Seasonal Cycle')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Mean NDVI')
    axes[1, 0].grid(True)
    
    # Seasonal amplitude map
    im3 = axes[1, 1].imshow(seasonal_amp, cmap='viridis')
    axes[1, 1].set_title('Seasonal Amplitude')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Trend histogram
    valid_slopes = ndvi_slope.values[~np.isnan(ndvi_slope.values)]
    axes[1, 2].hist(valid_slopes, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(0, color='red', linestyle='--', label='No trend')
    axes[1, 2].set_title('Distribution of Trend Slopes')
    axes[1, 2].set_xlabel('NDVI Trend (per year)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return ndvi_slope, seasonal_trend, is_significant

# Calculate trends
trend_slope, seasonal_trend_slope, significance_mask = calculate_temporal_trends()
```

---

## 4️⃣ GroupBy time, date filtering, and rolling averages

### Temporal Grouping and Filtering
```python
def temporal_grouping_operations():
    """Demonstrate temporal grouping, filtering, and rolling operations"""
    
    # Use climate dataset from earlier
    ds = climate_dataset
    
    print("TEMPORAL GROUPING OPERATIONS")
    print("=" * 30)
    
    # 1. GroupBy operations
    print("1. GROUPBY OPERATIONS:")
    
    # Monthly means
    monthly_means = ds.groupby('time.month').mean()
    print(f"   Monthly climatology shape: {monthly_means.temperature.shape}")
    
    # Seasonal means
    seasonal_means = ds.groupby('time.season').mean()
    print(f"   Seasonal means: {list(seasonal_means.season.values)}")
    
    # Day of year climatology
    doy_climatology = ds.groupby('time.dayofyear').mean()
    print(f"   Day-of-year climatology: {doy_climatology.temperature.shape}")
    
    # Custom grouping by decade
    ds_with_decade = ds.assign_coords(decade=('time', ds.time.dt.year // 10 * 10))
    decade_means = ds_with_decade.groupby('decade').mean()
    print(f"   Decade grouping: {list(decade_means.decade.values)}")
    
    # 2. Date filtering
    print("\n2. DATE FILTERING:")
    
    # Filter by month
    summer_months = ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))
    print(f"   Summer months: {len(summer_months.time)} days")
    
    # Filter by season
    winter_data = ds.sel(time=ds.time.dt.season == 'DJF')
    print(f"   Winter data: {len(winter_data.time)} days")
    
    # Filter by date range
    spring_data = ds.sel(time=slice('2023-03-01', '2023-05-31'))
    print(f"   Spring period: {len(spring_data.time)} days")
    
    # Filter by day of week
    weekends = ds.sel(time=ds.time.dt.dayofweek.isin([5, 6]))
    print(f"   Weekend days: {len(weekends.time)} days")
    
    # Custom date filtering
    hot_days = ds.sel(time=ds.temperature.mean(dim=['lat', 'lon']) > 25)
    print(f"   Hot days (>25°C): {len(hot_days.time)} days")
    
    # 3. Rolling operations
    print("\n3. ROLLING OPERATIONS:")
    
    # Simple rolling mean
    temp_7day = ds.temperature.rolling(time=7, center=True).mean()
    print(f"   7-day rolling mean shape: {temp_7day.shape}")
    
    # Rolling with minimum periods
    temp_30day = ds.temperature.rolling(time=30, min_periods=20).mean()
    print(f"   30-day rolling mean (min 20): {temp_30day.shape}")
    
    # Rolling standard deviation
    temp_variability = ds.temperature.rolling(time=14).std()
    print(f"   14-day rolling std: {temp_variability.shape}")
    
    # Rolling sum for precipitation
    precip_monthly = ds.precipitation.rolling(time=30).sum()
    print(f"   30-day precipitation sum: {precip_monthly.shape}")
    
    return monthly_means, seasonal_means, temp_7day, precip_monthly

# Advanced temporal operations
def advanced_temporal_operations():
    """Advanced temporal analysis techniques"""
    
    ds = climate_dataset
    
    # 1. Anomaly calculation
    def calculate_anomalies(data_array, baseline_period=None):
        """Calculate anomalies relative to climatological mean"""
        
        if baseline_period:
            baseline_data = data_array.sel(time=slice(*baseline_period))
            climatology = baseline_data.groupby('time.dayofyear').mean()
        else:
            climatology = data_array.groupby('time.dayofyear').mean()
        
        # Calculate anomalies
        anomalies = data_array.groupby('time.dayofyear') - climatology
        
        return anomalies, climatology
    
    # Calculate temperature anomalies
    temp_anomalies, temp_climatology = calculate_anomalies(ds.temperature)
    
    print("ADVANCED TEMPORAL OPERATIONS")
    print("=" * 30)
    print(f"Temperature anomalies range: {temp_anomalies.min().values:.2f} to {temp_anomalies.max().values:.2f}°C")
    
    # 2. Extreme event detection
    def detect_extreme_events(data_array, threshold_percentile=95):
        """Detect extreme events based on percentile threshold"""
        
        # Calculate threshold
        threshold = data_array.quantile(threshold_percentile / 100, dim='time')
        
        # Identify extreme events
        extreme_events = data_array > threshold
        
        # Count events per pixel
        event_count = extreme_events.sum(dim='time')
        
        return extreme_events, event_count, threshold
    
    # Detect heat waves
    heat_waves, heat_wave_count, heat_threshold = detect_extreme_events(ds.temperature, 95)
    
    print(f"Heat wave detection (95th percentile):")
    print(f"  Mean threshold: {heat_threshold.mean().values:.1f}°C")
    print(f"  Total heat wave days: {heat_waves.sum().values}")
    
    # 3. Temporal correlation analysis
    def temporal_correlation(var1, var2, lag_days=0):
        """Calculate temporal correlation between variables"""
        
        if lag_days != 0:
            if lag_days > 0:
                var1_shifted = var1.shift(time=lag_days)
                var2_aligned = var2
            else:
                var1_shifted = var1
                var2_aligned = var2.shift(time=-lag_days)
        else:
            var1_shifted = var1
            var2_aligned = var2
        
        # Calculate correlation
        correlation = xr.corr(var1_shifted, var2_aligned, dim='time')
        
        return correlation
    
    # Calculate temperature-precipitation correlation
    temp_precip_corr = temporal_correlation(ds.temperature, ds.precipitation)
    
    print(f"Temperature-Precipitation correlation:")
    print(f"  Mean correlation: {temp_precip_corr.mean().values:.3f}")
    
    # 4. Seasonal cycle removal and trend analysis
    def remove_seasonal_cycle(data_array):
        """Remove seasonal cycle using harmonic analysis"""
        
        # Calculate day of year
        doy = data_array.time.dt.dayofyear
        
        # Fit harmonic functions (annual and semi-annual cycles)
        def fit_harmonics(y):
            if np.isnan(y).all():
                return np.full(len(y), np.nan)
            
            # Design matrix for harmonics
            n_harmonics = 2
            X = np.ones((len(y), 1 + 2 * n_harmonics))
            
            for h in range(1, n_harmonics + 1):
                X[:, 2*h-1] = np.cos(2 * np.pi * h * doy / 365.25)
                X[:, 2*h] = np.sin(2 * np.pi * h * doy / 365.25)
            
            # Fit model
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() < X.shape[1]:
                return np.full(len(y), np.nan)
            
            try:
                coeffs = np.linalg.lstsq(X[valid_mask], y[valid_mask], rcond=None)[0]
                seasonal_fit = X @ coeffs
                return y - seasonal_fit
            except:
                return np.full(len(y), np.nan)
        
        # Apply harmonic fitting
        deseasonalized = xr.apply_ufunc(
            fit_harmonics,
            data_array,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            dask='allowed',
            output_dtypes=[float]
        )
        
        return deseasonalized
    
    # Remove seasonal cycle from temperature
    temp_deseasonalized = remove_seasonal_cycle(ds.temperature)
    
    print(f"Seasonal cycle removal:")
    print(f"  Original std: {ds.temperature.std(dim='time').mean().values:.2f}°C")
    print(f"  Deseasonalized std: {temp_deseasonalized.std(dim='time').mean().values:.2f}°C")
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Monthly climatology
    monthly_temp = ds.temperature.groupby('time.month').mean()
    monthly_precip = ds.precipitation.groupby('time.month').mean()
    
    axes[0, 0].plot(monthly_temp.month, monthly_temp.mean(dim=['lat', 'lon']), 'o-', color='red')
    axes[0, 0].set_title('Monthly Temperature Climatology')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(monthly_precip.month, monthly_precip.mean(dim=['lat', 'lon']), 'o-', color='blue')
    axes[0, 1].set_title('Monthly Precipitation Climatology')
    axes[0, 1].set_ylabel('Precipitation (mm)')
    axes[0, 1].grid(True)
    
    # Time series with rolling averages
    temp_ts = ds.temperature.mean(dim=['lat', 'lon'])
    temp_7day = temp_ts.rolling(time=7, center=True).mean()
    temp_30day = temp_ts.rolling(time=30, center=True).mean()
    
    axes[1, 0].plot(temp_ts.time, temp_ts, alpha=0.3, color='gray', label='Daily')
    axes[1, 0].plot(temp_7day.time, temp_7day, color='blue', label='7-day mean')
    axes[1, 0].plot(temp_30day.time, temp_30day, color='red', label='30-day mean')
    axes[1, 0].set_title('Temperature with Rolling Averages')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Anomalies
    temp_anom_ts = temp_anomalies.mean(dim=['lat', 'lon'])
    axes[1, 1].plot(temp_anom_ts.time, temp_anom_ts, color='red', alpha=0.7)
    axes[1, 1].axhline(0, color='black', linestyle='--')
    axes[1, 1].set_title('Temperature Anomalies')
    axes[1, 1].set_ylabel('Temperature Anomaly (°C)')
    axes[1, 1].grid(True)
    
    # Heat wave frequency map
    im1 = axes[2, 0].imshow(heat_wave_count, cmap='Reds')
    axes[2, 0].set_title('Heat Wave Days Count')
    plt.colorbar(im1, ax=axes[2, 0])
    
    # Temperature-precipitation correlation map
    im2 = axes[2, 1].imshow(temp_precip_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2, 1].set_title('Temperature-Precipitation Correlation')
    plt.colorbar(im2, ax=axes[2, 1])
    
    plt.tight_layout()
    plt.show()
    
    return temp_anomalies, heat_waves, temp_deseasonalized

# Run temporal operations
monthly_climate, seasonal_climate, temp_smoothed, precip_cumulative = temporal_grouping_operations()
temp_anomalies, extreme_events, deseasonalized_temp = advanced_temporal_operations()
```

---

## Best Practices and Tips

### Performance Optimization
- **Use chunking**: Optimize chunk sizes for your analysis workflow
- **Lazy evaluation**: Leverage dask for memory-efficient processing
- **Selective loading**: Load only required variables and time periods
- **Compression**: Use compression when saving large time-series datasets

### Memory Management
- **Monitor memory usage**: Use `ds.nbytes` to check dataset size
- **Process in batches**: Break large time series into manageable chunks
- **Use appropriate data types**: Choose efficient dtypes for your data
- **Clean up intermediate results**: Delete unnecessary variables

### Analysis Considerations
- **Handle missing data**: Account for gaps in time series
- **Validate temporal consistency**: Check for irregular time steps
- **Consider temporal autocorrelation**: Account for serial correlation in statistics
- **Document metadata**: Preserve temporal and spatial metadata

This comprehensive guide provides the foundation for working with multi-dimensional time-series raster data, enabling sophisticated temporal analysis and monitoring applications across environmental and climate science domains.