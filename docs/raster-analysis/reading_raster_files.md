# Reading Raster Files in Python

## Overview
Reading raster files is the foundation of raster analysis in Python. This guide covers the essential libraries, methods, and techniques for loading and exploring raster data, from simple single-band images to complex multi-band satellite datasets.

## Why Master Raster Reading?
- **Data Access**: Load satellite imagery, DEMs, climate data, and aerial photos
- **Format Support**: Handle GeoTIFF, NetCDF, HDF, and 50+ other formats
- **Memory Management**: Efficiently work with large datasets
- **Metadata Extraction**: Access spatial reference, resolution, and band information
- **Integration Ready**: Prepare data for analysis workflows

---

## 1️⃣ Installing and Importing Libraries

### Installation
```bash
# Core raster libraries
pip install rasterio rioxarray

# Additional dependencies
pip install xarray netcdf4 h5netcdf

# For visualization
pip install matplotlib cartopy

# Complete geospatial stack
conda install -c conda-forge rasterio rioxarray xarray dask
```

### Essential Imports
```python
import rasterio
import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show
from rasterio.windows import Window
```

### Library Comparison
```python
# rasterio: Low-level, GDAL-based, memory efficient
# rioxarray: High-level, xarray integration, analysis-friendly
# xarray: N-dimensional arrays, metadata-rich, NetCDF native

print(f"Rasterio version: {rasterio.__version__}")
print(f"Rioxarray version: {rxr.__version__}")
print(f"Xarray version: {xr.__version__}")
```

---

## 2️⃣ Opening Single-band GeoTIFFs

### Basic File Opening with Rasterio
```python
# Open raster file
with rasterio.open('elevation.tif') as src:
    print(f"Driver: {src.driver}")
    print(f"Count: {src.count}")
    print(f"Width: {src.width}")
    print(f"Height: {src.height}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    
    # Read data
    elevation = src.read(1)  # Read first (and only) band
    print(f"Data shape: {elevation.shape}")
    print(f"Data type: {elevation.dtype}")
```

### Opening with Rioxarray
```python
# Open with rioxarray (recommended for analysis)
elevation_xr = rxr.open_rasterio('elevation.tif')
print(elevation_xr)

# Access data and coordinates
print(f"Data shape: {elevation_xr.shape}")
print(f"Coordinates: {list(elevation_xr.coords)}")
print(f"CRS: {elevation_xr.rio.crs}")
print(f"Resolution: {elevation_xr.rio.resolution()}")
```

### Handling Different File Formats
```python
# GeoTIFF
dem = rxr.open_rasterio('srtm_dem.tif')

# NetCDF
temperature = rxr.open_rasterio('temperature.nc')

# HDF5
modis = rxr.open_rasterio('MOD11A1.hdf', chunks=True)

# JPEG2000
sentinel = rxr.open_rasterio('sentinel_image.jp2')

# ASCII Grid
ascii_dem = rxr.open_rasterio('elevation.asc')
```

---

## 3️⃣ Opening Multi-band GeoTIFFs

### Landsat/Sentinel Multi-band Images
```python
# Open multi-band satellite image
landsat = rxr.open_rasterio('landsat8_scene.tif')
print(f"Bands: {landsat.shape[0]}")
print(f"Band names: {landsat.long_name if hasattr(landsat, 'long_name') else 'Not available'}")

# Access individual bands
red = landsat.sel(band=4)      # Red band
nir = landsat.sel(band=5)      # Near-infrared
blue = landsat.sel(band=2)     # Blue band

# Calculate NDVI
ndvi = (nir - red) / (nir + red)
```

### RGB Composite Creation
```python
# Create RGB composite
def create_rgb_composite(multi_band, red_idx=3, green_idx=2, blue_idx=1):
    """Create RGB composite from multi-band raster"""
    
    # Extract RGB bands (convert to 0-based indexing)
    red = multi_band.isel(band=red_idx-1)
    green = multi_band.isel(band=green_idx-1) 
    blue = multi_band.isel(band=blue_idx-1)
    
    # Stack bands
    rgb = xr.concat([red, green, blue], dim='band')
    rgb = rgb.assign_coords(band=['red', 'green', 'blue'])
    
    return rgb

# Usage
rgb_composite = create_rgb_composite(landsat, red_idx=4, green_idx=3, blue_idx=2)

# Plot RGB composite
fig, ax = plt.subplots(figsize=(10, 8))
rgb_composite.plot.imshow(ax=ax, robust=True)
ax.set_title('RGB Composite')
plt.show()
```

### Band Information and Statistics
```python
# Explore band information
for i, band in enumerate(landsat.band.values, 1):
    band_data = landsat.sel(band=band)
    print(f"Band {i}:")
    print(f"  Min: {band_data.min().values:.2f}")
    print(f"  Max: {band_data.max().values:.2f}")
    print(f"  Mean: {band_data.mean().values:.2f}")
    print(f"  Std: {band_data.std().values:.2f}")
    print(f"  NoData: {band_data.rio.nodata}")
```

---

## 4️⃣ Exploring Raster Metadata

### Comprehensive Metadata Extraction
```python
def explore_raster_metadata(filepath):
    """Extract comprehensive raster metadata"""
    
    with rasterio.open(filepath) as src:
        metadata = {
            'driver': src.driver,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'res': src.res,
            'units': src.crs.linear_units if src.crs else None,
            'compression': src.compression,
            'interleave': src.interleave,
            'tiled': src.is_tiled,
            'blockxsize': src.block_shapes[0][1] if src.block_shapes else None,
            'blockysize': src.block_shapes[0][0] if src.block_shapes else None,
        }
        
        # Additional metadata from tags
        metadata.update(src.tags())
        
        return metadata

# Usage
metadata = explore_raster_metadata('satellite_image.tif')
for key, value in metadata.items():
    print(f"{key}: {value}")
```

### CRS and Transform Details
```python
# Detailed CRS information
raster = rxr.open_rasterio('landsat_scene.tif')

print("=== CRS Information ===")
print(f"CRS: {raster.rio.crs}")
print(f"CRS String: {raster.rio.crs.to_string()}")
print(f"EPSG Code: {raster.rio.crs.to_epsg()}")
print(f"Units: {raster.rio.crs.linear_units}")
print(f"Is Geographic: {raster.rio.crs.is_geographic}")
print(f"Is Projected: {raster.rio.crs.is_projected}")

print("\n=== Transform Information ===")
transform = raster.rio.transform()
print(f"Transform: {transform}")
print(f"Pixel Width: {transform.a}")
print(f"Pixel Height: {-transform.e}")  # Negative because Y decreases
print(f"Upper-left X: {transform.c}")
print(f"Upper-left Y: {transform.f}")

print("\n=== Bounds Information ===")
bounds = raster.rio.bounds()
print(f"Bounds (left, bottom, right, top): {bounds}")
print(f"Width (degrees/meters): {bounds[2] - bounds[0]}")
print(f"Height (degrees/meters): {bounds[3] - bounds[1]}")
```

### Spatial Resolution Analysis
```python
def analyze_spatial_resolution(raster):
    """Analyze spatial resolution and coverage"""
    
    # Get resolution
    res_x, res_y = raster.rio.resolution()
    
    # Calculate coverage
    bounds = raster.rio.bounds()
    width_coverage = bounds[2] - bounds[0]
    height_coverage = bounds[3] - bounds[1]
    
    # Calculate total area (for projected CRS)
    if raster.rio.crs.is_projected:
        total_area = width_coverage * height_coverage
        pixel_area = abs(res_x * res_y)
        total_pixels = raster.size
        
        print(f"Pixel resolution: {abs(res_x):.2f} x {abs(res_y):.2f} {raster.rio.crs.linear_units}")
        print(f"Pixel area: {pixel_area:.2f} {raster.rio.crs.linear_units}²")
        print(f"Total coverage: {width_coverage:.2f} x {height_coverage:.2f} {raster.rio.crs.linear_units}")
        print(f"Total area: {total_area:.2f} {raster.rio.crs.linear_units}²")
        print(f"Total pixels: {total_pixels:,}")
    else:
        print(f"Pixel resolution: {abs(res_x):.6f}° x {abs(res_y):.6f}°")
        print(f"Coverage: {width_coverage:.6f}° x {height_coverage:.6f}°")

# Usage
analyze_spatial_resolution(raster)
```

---

## 5️⃣ Reading as NumPy Arrays

### Basic Array Reading
```python
# Method 1: Rasterio (returns numpy array)
with rasterio.open('elevation.tif') as src:
    elevation_np = src.read()  # All bands
    elevation_band1 = src.read(1)  # Single band
    
print(f"All bands shape: {elevation_np.shape}")  # (bands, height, width)
print(f"Single band shape: {elevation_band1.shape}")  # (height, width)

# Method 2: Rioxarray (returns xarray with numpy backend)
elevation_xr = rxr.open_rasterio('elevation.tif')
elevation_values = elevation_xr.values  # Extract numpy array
print(f"Xarray values shape: {elevation_values.shape}")
```

### Memory-Efficient Reading
```python
# Read specific windows for large files
def read_raster_window(filepath, window_bounds=None, band=1):
    """Read specific window from raster"""
    
    with rasterio.open(filepath) as src:
        if window_bounds:
            # Convert geographic bounds to pixel window
            window = rasterio.windows.from_bounds(*window_bounds, src.transform)
        else:
            # Read center quarter of image
            window = Window(
                col_off=src.width // 4,
                row_off=src.height // 4,
                width=src.width // 2,
                height=src.height // 2
            )
        
        # Read window
        data = src.read(band, window=window)
        
        # Get window transform
        window_transform = rasterio.windows.transform(window, src.transform)
        
        return data, window_transform, window

# Usage
subset_data, subset_transform, window_info = read_raster_window(
    'large_satellite_image.tif',
    window_bounds=(-120.5, 35.0, -119.5, 36.0)
)
print(f"Subset shape: {subset_data.shape}")
```

### Chunked Reading for Large Files
```python
# Use dask for lazy loading
raster_chunked = rxr.open_rasterio('very_large_file.tif', chunks={'x': 1024, 'y': 1024})
print(f"Chunks: {raster_chunked.chunks}")

# Compute statistics without loading full array
mean_value = raster_chunked.mean().compute()
std_value = raster_chunked.std().compute()
print(f"Mean: {mean_value.values:.2f}")
print(f"Std: {std_value.values:.2f}")
```

### Data Type Handling
```python
def analyze_data_types(raster_path):
    """Analyze and convert data types"""
    
    with rasterio.open(raster_path) as src:
        original_dtype = src.dtypes[0]
        data = src.read(1)
        
        print(f"Original dtype: {original_dtype}")
        print(f"Data range: {data.min()} to {data.max()}")
        print(f"Memory usage: {data.nbytes / 1024**2:.2f} MB")
        
        # Convert to different dtypes
        if original_dtype in ['uint16', 'int16']:
            # Convert to float32 for calculations
            data_float = data.astype(np.float32)
            print(f"Float32 memory: {data_float.nbytes / 1024**2:.2f} MB")
            
        elif original_dtype == 'float64':
            # Downcast to float32 to save memory
            data_float32 = data.astype(np.float32)
            print(f"Downcasted memory: {data_float32.nbytes / 1024**2:.2f} MB")
            
        return data

# Usage
data = analyze_data_types('landsat_scene.tif')
```

---

## 6️⃣ Extreme Cases and Error Handling

### Handling Corrupted or Missing Files
```python
def safe_raster_open(filepath):
    """Safely open raster with error handling"""
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try opening with rasterio
        with rasterio.open(filepath) as src:
            # Basic validation
            if src.count == 0:
                raise ValueError("No bands found in raster")
            
            if src.width == 0 or src.height == 0:
                raise ValueError("Invalid raster dimensions")
            
            # Check for valid CRS
            if src.crs is None:
                print("Warning: No CRS found in raster")
            
            # Read a small sample to test data integrity
            sample = src.read(1, window=Window(0, 0, min(10, src.width), min(10, src.height)))
            
            if np.all(np.isnan(sample)) or np.all(sample == src.nodata):
                print("Warning: Sample area contains only NoData values")
            
            return True, "File opened successfully"
            
    except rasterio.errors.RasterioIOError as e:
        return False, f"Rasterio error: {e}"
    except Exception as e:
        return False, f"General error: {e}"

# Usage
success, message = safe_raster_open('potentially_corrupted.tif')
print(f"Status: {success}, Message: {message}")
```

### Handling Large Files
```python
def handle_large_raster(filepath, max_memory_mb=500):
    """Handle large rasters with memory constraints"""
    
    with rasterio.open(filepath) as src:
        # Calculate file size
        file_size_mb = (src.width * src.height * src.count * 
                       np.dtype(src.dtypes[0]).itemsize) / 1024**2
        
        print(f"Estimated file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > max_memory_mb:
            print("File too large for memory, using chunked processing")
            
            # Calculate appropriate chunk size
            chunk_size = int(np.sqrt(max_memory_mb * 1024**2 / 
                           (src.count * np.dtype(src.dtypes[0]).itemsize)))
            
            # Process in chunks
            results = []
            for row in range(0, src.height, chunk_size):
                for col in range(0, src.width, chunk_size):
                    window = Window(
                        col, row,
                        min(chunk_size, src.width - col),
                        min(chunk_size, src.height - row)
                    )
                    
                    chunk_data = src.read(window=window)
                    # Process chunk (example: calculate mean)
                    chunk_mean = np.nanmean(chunk_data)
                    results.append(chunk_mean)
            
            overall_mean = np.mean(results)
            return overall_mean
        else:
            # Load entire file
            data = src.read()
            return np.nanmean(data)

# Usage
mean_value = handle_large_raster('huge_satellite_image.tif')
print(f"Overall mean: {mean_value:.2f}")
```

### NoData and Missing Value Handling
```python
def handle_nodata_values(raster_path):
    """Comprehensive NoData handling"""
    
    raster = rxr.open_rasterio(raster_path)
    
    print(f"Original NoData value: {raster.rio.nodata}")
    print(f"Data type: {raster.dtype}")
    
    # Check for various NoData representations
    nodata_checks = {
        'rio.nodata': raster.rio.nodata,
        'nan_count': np.isnan(raster).sum().values,
        'inf_count': np.isinf(raster).sum().values,
        'negative_values': (raster < 0).sum().values if raster.dtype.kind == 'u' else 0
    }
    
    for check, value in nodata_checks.items():
        print(f"{check}: {value}")
    
    # Set NoData values to NaN for calculations
    raster_clean = raster.where(raster != raster.rio.nodata)
    
    # Alternative: mask NoData values
    raster_masked = raster.where(~np.isnan(raster))
    
    return raster_clean, raster_masked

# Usage
clean_raster, masked_raster = handle_nodata_values('dem_with_nodata.tif')
```

---

## 7️⃣ Advanced Reading Techniques

### Reading with Overviews
```python
def read_with_overviews(filepath, overview_level=0):
    """Read raster using overviews for faster display"""
    
    with rasterio.open(filepath) as src:
        # Check available overviews
        overviews = src.overviews(1)  # For band 1
        print(f"Available overviews: {overviews}")
        
        if overview_level < len(overviews):
            # Read from specific overview
            overview_factor = overviews[overview_level]
            data = src.read(1, out_shape=(
                src.height // overview_factor,
                src.width // overview_factor
            ))
            print(f"Read overview {overview_level} with factor {overview_factor}")
        else:
            # Read full resolution
            data = src.read(1)
            print("Read full resolution")
        
        return data

# Usage
overview_data = read_with_overviews('large_image.tif', overview_level=2)
```

### Reading Specific Bands by Name
```python
def read_bands_by_name(filepath, band_names):
    """Read specific bands by name (for files with band descriptions)"""
    
    with rasterio.open(filepath) as src:
        # Get band descriptions
        band_descriptions = [src.descriptions[i] for i in range(src.count)]
        print(f"Available bands: {band_descriptions}")
        
        # Find band indices
        band_indices = []
        for name in band_names:
            try:
                idx = band_descriptions.index(name) + 1  # 1-based indexing
                band_indices.append(idx)
            except ValueError:
                print(f"Band '{name}' not found")
        
        # Read selected bands
        if band_indices:
            data = src.read(band_indices)
            return data, band_indices
        else:
            return None, []

# Usage for Sentinel-2 data
selected_bands, indices = read_bands_by_name(
    'sentinel2_scene.tif', 
    ['B04', 'B08', 'B11']  # Red, NIR, SWIR
)
```

### Reading Time Series Data
```python
def read_time_series_stack(file_pattern, time_dim='time'):
    """Read multiple raster files as time series"""
    
    import glob
    from datetime import datetime
    
    # Find all matching files
    files = sorted(glob.glob(file_pattern))
    print(f"Found {len(files)} files")
    
    # Extract dates from filenames (example pattern)
    dates = []
    for file in files:
        # Example: extract date from filename like 'data_20230101.tif'
        date_str = file.split('_')[-1].split('.')[0]
        date = datetime.strptime(date_str, '%Y%m%d')
        dates.append(date)
    
    # Open all files
    datasets = []
    for file, date in zip(files, dates):
        ds = rxr.open_rasterio(file)
        ds = ds.expand_dims(time=[date])
        datasets.append(ds)
    
    # Concatenate along time dimension
    time_series = xr.concat(datasets, dim='time')
    
    return time_series

# Usage
ndvi_time_series = read_time_series_stack('ndvi_*.tif')
print(f"Time series shape: {ndvi_time_series.shape}")
```

---

## 8️⃣ Performance Optimization

### Benchmarking Read Performance
```python
import time

def benchmark_reading_methods(filepath):
    """Compare performance of different reading methods"""
    
    methods = {}
    
    # Method 1: Rasterio
    start = time.time()
    with rasterio.open(filepath) as src:
        data1 = src.read()
    methods['rasterio'] = time.time() - start
    
    # Method 2: Rioxarray
    start = time.time()
    data2 = rxr.open_rasterio(filepath)
    methods['rioxarray'] = time.time() - start
    
    # Method 3: Rioxarray with chunks
    start = time.time()
    data3 = rxr.open_rasterio(filepath, chunks={'x': 512, 'y': 512})
    methods['rioxarray_chunked'] = time.time() - start
    
    # Method 4: Rasterio with window
    start = time.time()
    with rasterio.open(filepath) as src:
        window = Window(0, 0, src.width//2, src.height//2)
        data4 = src.read(window=window)
    methods['rasterio_window'] = time.time() - start
    
    print("Reading performance comparison:")
    for method, duration in methods.items():
        print(f"{method}: {duration:.3f} seconds")
    
    return methods

# Usage
performance = benchmark_reading_methods('large_raster.tif')
```

---

## 9️⃣ Integration Examples

### Reading for Machine Learning
```python
def prepare_raster_for_ml(raster_paths, target_size=(256, 256)):
    """Prepare raster data for machine learning"""
    
    datasets = []
    
    for path in raster_paths:
        # Read raster
        raster = rxr.open_rasterio(path)
        
        # Normalize to target size
        raster_resized = raster.rio.reproject(
            raster.rio.crs,
            shape=target_size,
            resampling=rasterio.enums.Resampling.bilinear
        )
        
        # Normalize values to 0-1
        raster_norm = (raster_resized - raster_resized.min()) / (raster_resized.max() - raster_resized.min())
        
        # Convert to numpy and reshape for ML
        data = raster_norm.values.reshape(-1, target_size[0], target_size[1])
        datasets.append(data)
    
    # Stack all datasets
    ml_ready_data = np.stack(datasets, axis=0)
    
    return ml_ready_data

# Usage
ml_data = prepare_raster_for_ml(['image1.tif', 'image2.tif', 'image3.tif'])
print(f"ML-ready data shape: {ml_data.shape}")  # (samples, bands, height, width)
```

---

## 🔗 Data Sources for Practice

### Free Raster Datasets
```python
# Download sample data programmatically
import requests
import zipfile
from pathlib import Path

def download_sample_rasters():
    """Download sample raster data for practice"""
    
    # Create data directory
    data_dir = Path('sample_raster_data')
    data_dir.mkdir(exist_ok=True)
    
    # Sample datasets (replace with actual URLs)
    datasets = {
        'landsat_sample.tif': 'https://example.com/landsat_sample.tif',
        'dem_sample.tif': 'https://example.com/dem_sample.tif',
        'temperature.nc': 'https://example.com/temperature.nc'
    }
    
    for filename, url in datasets.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            # response = requests.get(url)
            # filepath.write_bytes(response.content)
            print(f"Would download from: {url}")
    
    return data_dir

# Usage
# data_directory = download_sample_rasters()
```

### Real Data Sources
- **Landsat**: [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- **Sentinel**: [Copernicus Open Access Hub](https://scihub.copernicus.eu/)
- **MODIS**: [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/)
- **Climate Data**: [WorldClim](https://worldclim.org/), [ECMWF](https://www.ecmwf.int/)
- **Elevation**: [NASA SRTM](https://www2.jpl.nasa.gov/srtm/), [ASTER GDEM](https://asterweb.jpl.nasa.gov/gdem.asp)

---

## ⚠️ Common Pitfalls and Solutions

### Pitfall 1: Memory Issues with Large Files
```python
# ❌ Wrong: Loading entire large file
# large_raster = rxr.open_rasterio('100GB_file.tif').load()

# ✅ Correct: Use lazy loading and chunks
large_raster = rxr.open_rasterio('100GB_file.tif', chunks={'x': 1024, 'y': 1024})
subset = large_raster.isel(x=slice(0, 1000), y=slice(0, 1000)).load()
```

### Pitfall 2: Ignoring NoData Values
```python
# ❌ Wrong: Calculating statistics without handling NoData
# mean_value = raster.mean()

# ✅ Correct: Handle NoData properly
raster_clean = raster.where(raster != raster.rio.nodata)
mean_value = raster_clean.mean()
```

### Pitfall 3: Assuming Band Order
```python
# ❌ Wrong: Assuming RGB order
# red = raster.isel(band=0)
# green = raster.isel(band=1) 
# blue = raster.isel(band=2)

# ✅ Correct: Check band descriptions or metadata
with rasterio.open('satellite.tif') as src:
    band_descriptions = src.descriptions
    print(f"Band descriptions: {band_descriptions}")
    
# Or use band names if available
if hasattr(raster, 'long_name'):
    print(f"Band names: {raster.long_name}")
```

---

## 🎯 Key Takeaways

1. **Library Choice**: Use rasterio for low-level operations, rioxarray for analysis
2. **Memory Management**: Use chunks and windows for large files
3. **Metadata First**: Always explore CRS, transform, and NoData before analysis
4. **Error Handling**: Implement robust error checking for production code
5. **Performance**: Benchmark different methods for your specific use case
6. **Data Types**: Understand and optimize data types for memory efficiency

This foundation enables you to confidently read and explore any raster dataset, setting the stage for advanced raster analysis workflows.