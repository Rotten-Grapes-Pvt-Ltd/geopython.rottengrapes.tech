# Reading from Cloud and Web Sources

## Overview
Modern geospatial workflows increasingly rely on cloud-based data sources and web services for accessing large-scale raster datasets. Cloud-optimized formats, streaming capabilities, and open data platforms enable efficient access to global datasets without requiring local storage of massive files.

## Why Cloud-Based Raster Access Matters
- **Scalability**: Access petabytes of data without local storage constraints
- **Efficiency**: Stream only required data portions using spatial and temporal subsetting
- **Collaboration**: Share access to standardized datasets across teams and organizations
- **Cost-Effectiveness**: Pay-per-use model eliminates infrastructure maintenance
- **Real-Time Access**: Get latest data updates without manual downloads

---

## 1️⃣ Reading Cloud-Optimized GeoTIFFs (COG)

### Understanding COG Structure and Benefits
```python
import rasterio
import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from rasterio.session import AWSSession
import fsspec

# Basic COG reading
def read_basic_cog():
    """Read Cloud-Optimized GeoTIFF from web source"""
    
    # Example COG URL (USGS 3DEP elevation data)
    cog_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTMGL1/SRTMGL1_srtm.vrt"
    
    # Alternative: Landsat COG example
    landsat_cog = "https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/LC08_L1TP_139045_20170304_20170316_01_T1_B4.TIF"
    
    print("READING CLOUD-OPTIMIZED GEOTIFFS")
    print("=" * 35)
    
    try:
        # Method 1: Direct URL reading with rasterio
        with rasterio.open(landsat_cog) as src:
            print(f"COG Properties:")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bands: {src.count}")
            print(f"  CRS: {src.crs}")
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  Overviews: {src.overviews(1)}")
            
            # Check if it's a valid COG
            print(f"  Tiled: {src.profile.get('tiled', False)}")
            print(f"  Blocksize: {src.block_shapes[0]}")
            
            # Read a small subset (spatial window)
            window = rasterio.windows.Window(1000, 1000, 512, 512)
            subset = src.read(1, window=window)
            
            print(f"  Subset shape: {subset.shape}")
            print(f"  Subset data range: {subset.min()} to {subset.max()}")
        
        # Method 2: Using rioxarray for easier handling
        subset_bounds = (-122.5, 37.5, -122.0, 38.0)  # San Francisco area
        
        da = rxr.open_rasterio(
            landsat_cog,
            chunks=True,  # Enable dask for lazy loading
        ).rio.clip_box(*subset_bounds)
        
        print(f"\nrioxarray COG reading:")
        print(f"  Clipped shape: {da.shape}")
        print(f"  Chunks: {da.chunks}")
        print(f"  Memory usage: {da.nbytes / 1e6:.1f} MB")
        
        return da
        
    except Exception as e:
        print(f"COG reading failed: {e}")
        print("Creating synthetic COG example...")
        
        # Create synthetic data for demonstration
        height, width = 1024, 1024
        x = np.linspace(-122.5, -122.0, width)
        y = np.linspace(37.5, 38.0, height)
        
        # Simulate Landsat-like data
        data = np.random.randint(0, 65535, (height, width), dtype=np.uint16)
        
        synthetic_da = xr.DataArray(
            data,
            coords={'y': y, 'x': x},
            dims=['y', 'x']
        )
        synthetic_da.rio.write_crs('EPSG:4326', inplace=True)
        
        return synthetic_da

# Advanced COG operations
def advanced_cog_operations():
    """Demonstrate advanced COG reading techniques"""
    
    # COG with multiple overviews
    def read_cog_overviews(cog_url):
        """Read different overview levels from COG"""
        
        with rasterio.open(cog_url) as src:
            # Get overview information
            overviews = src.overviews(1)
            
            print(f"Available overviews: {overviews}")
            
            # Read at different resolutions
            overview_data = {}
            
            # Full resolution
            overview_data['full'] = src.read(1, out_shape=(512, 512))
            
            # Read specific overview levels
            for i, overview in enumerate(overviews[:2]):  # First 2 overviews
                overview_data[f'overview_{i}'] = src.read(
                    1, 
                    out_shape=(src.height // overview, src.width // overview)
                )
            
            return overview_data
    
    # Efficient spatial subsetting
    def efficient_spatial_subset(cog_url, bbox):
        """Efficiently read spatial subset from COG"""
        
        with rasterio.open(cog_url) as src:
            # Convert bbox to window
            window = rasterio.windows.from_bounds(*bbox, src.transform)
            
            # Read only the required window
            subset = src.read(1, window=window)
            
            # Get transform for the subset
            subset_transform = rasterio.windows.transform(window, src.transform)
            
            return subset, subset_transform
    
    # Band subsetting for multispectral data
    def read_specific_bands(cog_url, bands=[1, 2, 3]):
        """Read specific bands from multispectral COG"""
        
        with rasterio.open(cog_url) as src:
            # Read only specified bands
            band_data = src.read(bands)
            
            return band_data
    
    print("ADVANCED COG OPERATIONS")
    print("=" * 23)
    
    # Demonstrate with synthetic example
    print("COG optimization benefits:")
    print("  ✓ Tiled structure enables partial reading")
    print("  ✓ Overviews provide multi-resolution access")
    print("  ✓ Internal compression reduces transfer size")
    print("  ✓ HTTP range requests minimize bandwidth")
    
    return True

# Run COG examples
cog_data = read_basic_cog()
advanced_cog_operations()
```

### COG Validation and Optimization
```python
def validate_and_optimize_cog():
    """Validate COG structure and demonstrate optimization"""
    
    # COG validation using rio-cogeo
    def validate_cog_structure(file_path):
        """Validate if file is a proper COG"""
        
        try:
            from rio_cogeo.cogeo import cog_validate
            
            # Validate COG
            is_valid, errors, warnings = cog_validate(file_path)
            
            print(f"COG Validation Results:")
            print(f"  Valid COG: {is_valid}")
            
            if errors:
                print(f"  Errors: {len(errors)}")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"    - {error}")
            
            if warnings:
                print(f"  Warnings: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3 warnings
                    print(f"    - {warning}")
            
            return is_valid
            
        except ImportError:
            print("rio-cogeo not available. Install with: pip install rio-cogeo")
            return None
    
    # Create optimized COG
    def create_optimized_cog(input_file, output_file):
        """Convert regular GeoTIFF to optimized COG"""
        
        try:
            from rio_cogeo.cogeo import cog_translate
            from rio_cogeo.profiles import jpeg_profile, lzw_profile, deflate_profile
            
            # Choose profile based on data type
            profile = lzw_profile  # Good for most data types
            
            # Translate to COG
            cog_translate(
                input_file,
                output_file,
                profile,
                in_memory=False,
                quiet=True
            )
            
            print(f"COG created: {output_file}")
            
        except ImportError:
            print("rio-cogeo not available for COG creation")
    
    # COG info extraction
    def get_cog_info(cog_path):
        """Extract detailed COG information"""
        
        with rasterio.open(cog_path) as src:
            info = {
                'driver': src.driver,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'crs': str(src.crs),
                'transform': src.transform,
                'tiled': src.profile.get('tiled', False),
                'blockxsize': src.profile.get('blockxsize'),
                'blockysize': src.profile.get('blockysize'),
                'compress': src.profile.get('compress'),
                'interleave': src.profile.get('interleave'),
                'overviews': src.overviews(1) if src.count > 0 else []
            }
            
            return info
    
    print("COG VALIDATION AND OPTIMIZATION")
    print("=" * 32)
    
    # Best practices for COG
    cog_best_practices = {
        'Tiling': 'Use 512x512 or 1024x1024 pixel tiles',
        'Overviews': 'Include overviews with 2x downsampling',
        'Compression': 'Use LZW or DEFLATE for lossless, JPEG for lossy',
        'Data Type': 'Choose appropriate dtype (uint8, uint16, float32)',
        'Projection': 'Use Web Mercator (EPSG:3857) for web applications',
        'Metadata': 'Include proper spatial reference and statistics'
    }
    
    print("COG Best Practices:")
    for practice, description in cog_best_practices.items():
        print(f"  {practice}: {description}")
    
    return cog_best_practices

# Run validation examples
cog_practices = validate_and_optimize_cog()
```

---

## 2️⃣ Streaming rasters using rasterio and fsspec

### Basic Streaming with fsspec
```python
def stream_rasters_fsspec():
    """Demonstrate raster streaming using fsspec"""
    
    import fsspec
    from fsspec.implementations.http import HTTPFileSystem
    
    print("RASTER STREAMING WITH FSSPEC")
    print("=" * 30)
    
    # Setup HTTP filesystem
    fs = HTTPFileSystem()
    
    # Example streaming URLs
    streaming_urls = {
        'landsat': 'https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/LC08_L1TP_139045_20170304_20170316_01_T1_B4.TIF',
        'sentinel': 'https://sentinel-s2-l1c.s3.amazonaws.com/tiles/10/S/DG/2017/1/14/0/B02.jp2'
    }
    
    # Method 1: Direct streaming with rasterio
    def stream_with_rasterio(url):
        """Stream raster data using rasterio with fsspec"""
        
        try:
            # Open with fsspec-enabled rasterio
            with rasterio.open(url) as src:
                print(f"Streaming from: {url}")
                print(f"  Size: {src.width} x {src.height}")
                print(f"  Bands: {src.count}")
                
                # Read a small window to test streaming
                window = rasterio.windows.Window(0, 0, 256, 256)
                data = src.read(1, window=window)
                
                print(f"  Streamed window shape: {data.shape}")
                print(f"  Data range: {data.min()} - {data.max()}")
                
                return data
                
        except Exception as e:
            print(f"Streaming failed: {e}")
            return None
    
    # Method 2: Chunked streaming for large files
    def chunked_streaming(url, chunk_size=(512, 512)):
        """Stream raster in chunks to manage memory"""
        
        try:
            with rasterio.open(url) as src:
                height, width = src.height, src.width
                chunk_h, chunk_w = chunk_size
                
                print(f"Chunked streaming setup:")
                print(f"  Total size: {width} x {height}")
                print(f"  Chunk size: {chunk_w} x {chunk_h}")
                
                chunks_data = []
                
                # Stream in chunks
                for row in range(0, height, chunk_h):
                    for col in range(0, width, chunk_w):
                        # Calculate actual chunk size (handle edges)
                        actual_h = min(chunk_h, height - row)
                        actual_w = min(chunk_w, width - col)
                        
                        # Create window
                        window = rasterio.windows.Window(col, row, actual_w, actual_h)
                        
                        # Read chunk
                        chunk = src.read(1, window=window)
                        chunks_data.append({
                            'data': chunk,
                            'window': window,
                            'position': (row, col)
                        })
                        
                        # Limit to first few chunks for demo
                        if len(chunks_data) >= 4:
                            break
                    
                    if len(chunks_data) >= 4:
                        break
                
                print(f"  Streamed {len(chunks_data)} chunks")
                return chunks_data
                
        except Exception as e:
            print(f"Chunked streaming failed: {e}")
            return []
    
    # Method 3: Streaming with caching
    def stream_with_caching(url, cache_dir='./cache'):
        """Stream with local caching for repeated access"""
        
        import os
        
        # Setup cached filesystem
        cached_fs = fsspec.filesystem(
            'filecache',
            target_protocol='http',
            cache_storage=cache_dir
        )
        
        try:
            # Open cached file
            with cached_fs.open(url, 'rb') as f:
                with rasterio.open(f) as src:
                    print(f"Cached streaming from: {url}")
                    print(f"  Cache directory: {cache_dir}")
                    
                    # Check if file is cached
                    cache_info = cached_fs.cached_files[-1] if cached_fs.cached_files else None
                    if cache_info:
                        print(f"  Cache status: File cached locally")
                    
                    # Read subset
                    subset = src.read(1, window=rasterio.windows.Window(0, 0, 100, 100))
                    return subset
                    
        except Exception as e:
            print(f"Cached streaming failed: {e}")
            return None
    
    # Demonstrate streaming methods
    for name, url in streaming_urls.items():
        print(f"\n--- Streaming {name.upper()} ---")
        
        # Try basic streaming
        stream_data = stream_with_rasterio(url)
        
        if stream_data is not None:
            # Try chunked streaming
            chunk_data = chunked_streaming(url)
            
            # Try cached streaming
            cached_data = stream_with_caching(url)
            
            break  # Success with first URL
    
    return True

# Advanced streaming techniques
def advanced_streaming_techniques():
    """Advanced streaming patterns and optimizations"""
    
    print("\nADVANCED STREAMING TECHNIQUES")
    print("=" * 32)
    
    # Streaming with authentication
    def stream_with_auth(url, auth_method='bearer'):
        """Stream from authenticated sources"""
        
        # Example authentication patterns
        auth_examples = {
            'bearer': {
                'headers': {'Authorization': 'Bearer YOUR_TOKEN_HERE'}
            },
            'api_key': {
                'headers': {'X-API-Key': 'YOUR_API_KEY_HERE'}
            },
            'basic': {
                'auth': ('username', 'password')
            }
        }
        
        print(f"Authentication method: {auth_method}")
        print("Note: Replace with actual credentials")
        
        return auth_examples.get(auth_method, {})
    
    # Parallel streaming
    def parallel_streaming_setup():
        """Setup for parallel streaming of multiple files"""
        
        import concurrent.futures
        
        def stream_single_file(url):
            """Stream single file (for parallel execution)"""
            try:
                with rasterio.open(url) as src:
                    # Read small subset
                    data = src.read(1, window=rasterio.windows.Window(0, 0, 64, 64))
                    return {'url': url, 'data': data, 'success': True}
            except:
                return {'url': url, 'data': None, 'success': False}
        
        print("Parallel streaming setup:")
        print("  Use ThreadPoolExecutor for I/O-bound streaming")
        print("  Limit concurrent connections to avoid rate limiting")
        print("  Implement retry logic for failed requests")
        
        return stream_single_file
    
    # Streaming optimization tips
    streaming_optimizations = {
        'Connection Pooling': 'Reuse HTTP connections for multiple requests',
        'Range Requests': 'Use HTTP range headers for partial file access',
        'Compression': 'Enable gzip compression for metadata requests',
        'Caching': 'Cache frequently accessed tiles locally',
        'Retry Logic': 'Implement exponential backoff for failed requests',
        'Rate Limiting': 'Respect server rate limits and quotas'
    }
    
    print("\nStreaming Optimizations:")
    for optimization, description in streaming_optimizations.items():
        print(f"  {optimization}: {description}")
    
    # Error handling patterns
    error_patterns = {
        'Network Timeout': 'Set appropriate timeout values',
        'HTTP Errors': 'Handle 4xx/5xx status codes gracefully',
        'Partial Reads': 'Validate data integrity after streaming',
        'Memory Limits': 'Use chunked reading for large files',
        'Authentication': 'Refresh tokens before expiration'
    }
    
    print("\nError Handling Patterns:")
    for error_type, solution in error_patterns.items():
        print(f"  {error_type}: {solution}")
    
    return streaming_optimizations

# Run streaming examples
stream_rasters_fsspec()
advanced_streaming_techniques()
```

---

## 3️⃣ Accessing Open Datasets (AWS, STAC, GEE export)

### AWS Open Data Access
```python
def access_aws_open_data():
    """Access open datasets from AWS"""
    
    print("AWS OPEN DATA ACCESS")
    print("=" * 20)
    
    # AWS Open Data Registry examples
    aws_datasets = {
        'landsat': {
            'bucket': 'landsat-pds',
            'description': 'Landsat 8 imagery',
            'example_key': 'c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/'
        },
        'sentinel2': {
            'bucket': 'sentinel-s2-l1c',
            'description': 'Sentinel-2 Level-1C',
            'example_key': 'tiles/10/S/DG/2017/1/14/0/'
        },
        'naip': {
            'bucket': 'naip-visualization',
            'description': 'NAIP aerial imagery',
            'example_key': 'naip-visualization/al/2017/60cm/rgb/'
        },
        'terrain': {
            'bucket': 'terrain-tiles',
            'description': 'Mapzen terrain tiles',
            'example_key': 'terrarium/'
        }
    }
    
    # Method 1: Direct S3 access with rasterio
    def access_s3_direct(bucket, key):
        """Access S3 data directly with rasterio"""
        
        s3_url = f"s3://{bucket}/{key}"
        
        try:
            # Configure AWS session (anonymous for public data)
            aws_session = AWSSession(
                aws_access_key_id='',
                aws_secret_access_key='',
                aws_session_token='',
                region_name='us-west-2'
            )
            
            with rasterio.Env(session=aws_session):
                with rasterio.open(s3_url) as src:
                    print(f"S3 Dataset: {s3_url}")
                    print(f"  Dimensions: {src.width} x {src.height}")
                    print(f"  CRS: {src.crs}")
                    
                    return True
                    
        except Exception as e:
            print(f"S3 access failed: {e}")
            return False
    
    # Method 2: Using boto3 for S3 operations
    def list_s3_objects(bucket, prefix, max_keys=10):
        """List objects in S3 bucket"""
        
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            
            # Anonymous S3 client for public data
            s3_client = boto3.client(
                's3',
                config=Config(signature_version=UNSIGNED)
            )
            
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            if 'Contents' in response:
                print(f"Objects in s3://{bucket}/{prefix}:")
                for obj in response['Contents'][:5]:  # Show first 5
                    print(f"  {obj['Key']} ({obj['Size']} bytes)")
                
                return [obj['Key'] for obj in response['Contents']]
            else:
                print(f"No objects found in s3://{bucket}/{prefix}")
                return []
                
        except ImportError:
            print("boto3 not available. Install with: pip install boto3")
            return []
        except Exception as e:
            print(f"S3 listing failed: {e}")
            return []
    
    # Demonstrate AWS access
    for dataset_name, dataset_info in aws_datasets.items():
        print(f"\n--- {dataset_name.upper()} Dataset ---")
        print(f"Description: {dataset_info['description']}")
        print(f"Bucket: {dataset_info['bucket']}")
        
        # List objects
        objects = list_s3_objects(
            dataset_info['bucket'], 
            dataset_info['example_key']
        )
        
        if objects:
            # Try to access first raster file
            raster_files = [obj for obj in objects if obj.endswith(('.tif', '.TIF', '.jp2'))]
            if raster_files:
                success = access_s3_direct(dataset_info['bucket'], raster_files[0])
                if success:
                    break
    
    return aws_datasets

# STAC (SpatioTemporal Asset Catalog) access
def access_stac_data():
    """Access data through STAC catalogs"""
    
    print("\nSTAC DATA ACCESS")
    print("=" * 16)
    
    try:
        import pystac_client
        import planetary_computer
        
        # Method 1: Microsoft Planetary Computer STAC
        def access_planetary_computer():
            """Access Planetary Computer STAC catalog"""
            
            # Connect to Planetary Computer STAC API
            catalog = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace
            )
            
            print("Planetary Computer STAC:")
            
            # Search for Landsat data
            search = catalog.search(
                collections=["landsat-c2-l2"],
                bbox=[-122.5, 37.5, -122.0, 38.0],  # San Francisco
                datetime="2023-01-01/2023-12-31",
                limit=5
            )
            
            items = list(search.get_items())
            print(f"  Found {len(items)} Landsat items")
            
            if items:
                item = items[0]
                print(f"  Example item: {item.id}")
                print(f"  Assets: {list(item.assets.keys())}")
                
                # Access specific band
                if 'red' in item.assets:
                    red_asset = item.assets['red']
                    print(f"  Red band URL: {red_asset.href}")
                    
                    # Read with rioxarray
                    red_data = rxr.open_rasterio(red_asset.href, chunks=True)
                    print(f"  Red band shape: {red_data.shape}")
            
            return items
        
        # Method 2: Earth Search STAC
        def access_earth_search():
            """Access Earth Search STAC catalog"""
            
            catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
            
            print("\nEarth Search STAC:")
            
            # Search for Sentinel-2 data
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=[-122.5, 37.5, -122.0, 38.0],
                datetime="2023-06-01/2023-06-30",
                query={"eo:cloud_cover": {"lt": 10}},
                limit=3
            )
            
            items = list(search.get_items())
            print(f"  Found {len(items)} Sentinel-2 items")
            
            return items
        
        # Run STAC examples
        pc_items = access_planetary_computer()
        es_items = access_earth_search()
        
        return pc_items, es_items
        
    except ImportError:
        print("STAC libraries not available.")
        print("Install with: pip install pystac-client planetary-computer")
        return [], []

# Google Earth Engine export access
def access_gee_exports():
    """Access Google Earth Engine exported data"""
    
    print("\nGOOGLE EARTH ENGINE EXPORTS")
    print("=" * 30)
    
    # GEE export patterns
    gee_export_methods = {
        'Google Drive': {
            'description': 'Export to Google Drive, download manually',
            'pros': 'Easy setup, familiar interface',
            'cons': 'Manual download step required'
        },
        'Google Cloud Storage': {
            'description': 'Export to GCS bucket, access programmatically',
            'pros': 'Programmatic access, scalable',
            'cons': 'Requires GCS setup and billing'
        },
        'Asset Export': {
            'description': 'Export to GEE Asset, access via GEE API',
            'pros': 'Stays in GEE ecosystem',
            'cons': 'Limited to GEE platform'
        }
    }
    
    print("GEE Export Methods:")
    for method, info in gee_export_methods.items():
        print(f"\n{method}:")
        print(f"  Description: {info['description']}")
        print(f"  Pros: {info['pros']}")
        print(f"  Cons: {info['cons']}")
    
    # Example GEE export workflow (conceptual)
    gee_workflow = """
    # Google Earth Engine export workflow (run in GEE Code Editor or Python API)
    
    // 1. Process data in GEE
    var image = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_044034_20210101')
        .select(['SR_B4', 'SR_B3', 'SR_B2'])
        .multiply(0.0000275)
        .add(-0.2);
    
    // 2. Export to Google Drive
    Export.image.toDrive({
        image: image,
        description: 'landsat_export',
        scale: 30,
        region: geometry,
        fileFormat: 'GeoTIFF',
        formatOptions: {
            cloudOptimized: true
        }
    });
    
    // 3. Export to Google Cloud Storage
    Export.image.toCloudStorage({
        image: image,
        description: 'landsat_gcs_export',
        bucket: 'your-bucket-name',
        fileNamePrefix: 'landsat/',
        scale: 30,
        region: geometry,
        fileFormat: 'GeoTIFF',
        formatOptions: {
            cloudOptimized: true
        }
    });
    """
    
    print(f"\nExample GEE Export Code:")
    print(gee_workflow)
    
    # Accessing GEE exports programmatically
    def access_gcs_export(bucket_name, file_path):
        """Access GEE export from Google Cloud Storage"""
        
        gcs_url = f"gs://{bucket_name}/{file_path}"
        
        try:
            # Access GCS file with rasterio
            with rasterio.open(gcs_url) as src:
                print(f"GCS Export: {gcs_url}")
                print(f"  Dimensions: {src.width} x {src.height}")
                print(f"  Bands: {src.count}")
                
                return True
                
        except Exception as e:
            print(f"GCS access failed: {e}")
            return False
    
    print("\nAccessing GEE Exports:")
    print("  1. Set up authentication (gcloud auth or service account)")
    print("  2. Use gs:// URLs with rasterio for GCS exports")
    print("  3. Use Google Drive API for Drive exports")
    print("  4. Consider using Earth Engine Python API for direct access")
    
    return gee_export_methods

# Run open data access examples
aws_data = access_aws_open_data()
stac_items = access_stac_data()
gee_methods = access_gee_exports()
```

---

## 4️⃣ Basic Authentication for Private Resources

### Authentication Methods
```python
def authentication_methods():
    """Demonstrate various authentication methods for private resources"""
    
    print("AUTHENTICATION FOR PRIVATE RESOURCES")
    print("=" * 37)
    
    # Method 1: API Key authentication
    def api_key_auth():
        """API key authentication example"""
        
        import requests
        
        # Example API key patterns
        api_patterns = {
            'header': {
                'headers': {'X-API-Key': 'your_api_key_here'},
                'description': 'API key in request header'
            },
            'query': {
                'params': {'api_key': 'your_api_key_here'},
                'description': 'API key as query parameter'
            },
            'bearer': {
                'headers': {'Authorization': 'Bearer your_token_here'},
                'description': 'Bearer token authentication'
            }
        }
        
        print("API Key Authentication Patterns:")
        for pattern_name, pattern_info in api_patterns.items():
            print(f"\n{pattern_name.upper()}:")
            print(f"  Description: {pattern_info['description']}")
            if 'headers' in pattern_info:
                print(f"  Headers: {pattern_info['headers']}")
            if 'params' in pattern_info:
                print(f"  Parameters: {pattern_info['params']}")
        
        return api_patterns
    
    # Method 2: OAuth2 authentication
    def oauth2_auth():
        """OAuth2 authentication workflow"""
        
        oauth2_workflow = {
            'step1': 'Register application and get client credentials',
            'step2': 'Redirect user to authorization server',
            'step3': 'User grants permission and returns with auth code',
            'step4': 'Exchange auth code for access token',
            'step5': 'Use access token in API requests',
            'step6': 'Refresh token when expired'
        }
        
        print("\nOAuth2 Authentication Workflow:")
        for step, description in oauth2_workflow.items():
            print(f"  {step}: {description}")
        
        # Example OAuth2 implementation (conceptual)
        oauth2_example = """
        # OAuth2 example with requests-oauthlib
        from requests_oauthlib import OAuth2Session
        
        # Step 1: Initialize OAuth2 session
        client_id = 'your_client_id'
        client_secret = 'your_client_secret'
        redirect_uri = 'http://localhost:8080/callback'
        
        oauth = OAuth2Session(client_id, redirect_uri=redirect_uri)
        
        # Step 2: Get authorization URL
        authorization_url, state = oauth.authorization_url(
            'https://provider.com/oauth/authorize'
        )
        
        # Step 3: After user authorization, get token
        token = oauth.fetch_token(
            'https://provider.com/oauth/token',
            authorization_response=callback_url,
            client_secret=client_secret
        )
        
        # Step 4: Use token for API requests
        response = oauth.get('https://api.provider.com/data')
        """
        
        print(f"\nOAuth2 Implementation Example:")
        print(oauth2_example)
        
        return oauth2_workflow
    
    # Method 3: Certificate-based authentication
    def certificate_auth():
        """Certificate-based authentication"""
        
        cert_types = {
            'Client Certificates': {
                'description': 'Mutual TLS authentication',
                'use_case': 'High-security enterprise APIs',
                'implementation': 'requests.get(url, cert=("client.crt", "client.key"))'
            },
            'CA Certificates': {
                'description': 'Custom certificate authority',
                'use_case': 'Private/internal services',
                'implementation': 'requests.get(url, verify="ca-bundle.crt")'
            }
        }
        
        print("\nCertificate Authentication:")
        for cert_type, info in cert_types.items():
            print(f"\n{cert_type}:")
            print(f"  Description: {info['description']}")
            print(f"  Use case: {info['use_case']}")
            print(f"  Implementation: {info['implementation']}")
        
        return cert_types
    
    # Method 4: Cloud provider authentication
    def cloud_auth():
        """Cloud provider authentication methods"""
        
        cloud_methods = {
            'AWS': {
                'methods': ['IAM roles', 'Access keys', 'STS tokens'],
                'env_vars': ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'],
                'config_file': '~/.aws/credentials'
            },
            'Google Cloud': {
                'methods': ['Service accounts', 'Application default credentials'],
                'env_vars': ['GOOGLE_APPLICATION_CREDENTIALS'],
                'config_file': '~/.config/gcloud/application_default_credentials.json'
            },
            'Azure': {
                'methods': ['Service principals', 'Managed identities'],
                'env_vars': ['AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET'],
                'config_file': '~/.azure/credentials'
            }
        }
        
        print("\nCloud Provider Authentication:")
        for provider, info in cloud_methods.items():
            print(f"\n{provider}:")
            print(f"  Methods: {', '.join(info['methods'])}")
            print(f"  Environment variables: {', '.join(info['env_vars'])}")
            print(f"  Config file: {info['config_file']}")
        
        return cloud_methods
    
    # Run authentication examples
    api_auth = api_key_auth()
    oauth_auth = oauth2_auth()
    cert_auth = certificate_auth()
    cloud_auth_methods = cloud_auth()
    
    return api_auth, oauth_auth, cert_auth, cloud_auth_methods

# Practical authentication implementation
def implement_authenticated_access():
    """Practical implementation of authenticated raster access"""
    
    print("\nPRACTICAL AUTHENTICATION IMPLEMENTATION")
    print("=" * 40)
    
    # Secure credential management
    def secure_credential_management():
        """Best practices for credential management"""
        
        security_practices = {
            'Environment Variables': 'Store credentials in environment variables',
            'Config Files': 'Use secure config files with restricted permissions',
            'Key Management': 'Use cloud key management services (AWS KMS, etc.)',
            'Rotation': 'Implement regular credential rotation',
            'Least Privilege': 'Grant minimum required permissions',
            'Encryption': 'Encrypt credentials at rest and in transit'
        }
        
        print("Secure Credential Management:")
        for practice, description in security_practices.items():
            print(f"  {practice}: {description}")
        
        # Example secure implementation
        secure_example = """
        import os
        from rasterio.session import AWSSession
        
        # Method 1: Environment variables
        aws_session = AWSSession(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-west-2')
        )
        
        # Method 2: IAM roles (recommended for EC2/Lambda)
        aws_session = AWSSession()  # Uses instance profile
        
        # Method 3: Config file
        import configparser
        config = configparser.ConfigParser()
        config.read('~/.aws/credentials')
        
        aws_session = AWSSession(
            aws_access_key_id=config['default']['aws_access_key_id'],
            aws_secret_access_key=config['default']['aws_secret_access_key']
        )
        """
        
        print(f"\nSecure Implementation Example:")
        print(secure_example)
        
        return security_practices
    
    # Error handling for authentication
    def auth_error_handling():
        """Handle authentication errors gracefully"""
        
        error_scenarios = {
            'Invalid Credentials': 'Check credential format and validity',
            'Expired Tokens': 'Implement token refresh logic',
            'Permission Denied': 'Verify resource access permissions',
            'Rate Limiting': 'Implement exponential backoff',
            'Network Issues': 'Add retry logic with timeouts'
        }
        
        print("\nAuthentication Error Handling:")
        for error, solution in error_scenarios.items():
            print(f"  {error}: {solution}")
        
        # Example error handling
        error_example = """
        import time
        import random
        
        def authenticated_request(url, max_retries=3):
            for attempt in range(max_retries):
                try:
                    with rasterio.open(url) as src:
                        return src.read(1)
                        
                except rasterio.errors.RasterioIOError as e:
                    if '401' in str(e):  # Unauthorized
                        print("Authentication failed - check credentials")
                        break
                    elif '403' in str(e):  # Forbidden
                        print("Access denied - check permissions")
                        break
                    elif '429' in str(e):  # Rate limited
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limited - waiting {wait_time:.1f}s")
                        time.sleep(wait_time)
                    else:
                        print(f"Request failed: {e}")
                        
            return None
        """
        
        print(f"\nError Handling Example:")
        print(error_example)
        
        return error_scenarios
    
    # Run practical examples
    security_practices = secure_credential_management()
    error_handling = auth_error_handling()
    
    return security_practices, error_handling

# Run authentication examples
auth_methods = authentication_methods()
practical_auth = implement_authenticated_access()
```

---

## Best Practices and Tips

### Performance Optimization
- **Use COGs**: Leverage cloud-optimized formats for efficient streaming
- **Spatial Subsetting**: Request only required geographic areas
- **Temporal Filtering**: Limit time ranges to reduce data transfer
- **Caching**: Implement local caching for frequently accessed data

### Security Considerations
- **Credential Management**: Never hardcode credentials in source code
- **Access Control**: Use least-privilege principle for permissions
- **Token Rotation**: Implement regular credential rotation
- **Secure Transport**: Always use HTTPS for data transmission

### Cost Management
- **Data Transfer**: Monitor and optimize data transfer costs
- **Request Patterns**: Batch requests to minimize API calls
- **Storage Classes**: Use appropriate storage classes for different access patterns
- **Regional Access**: Access data from geographically close regions

This comprehensive guide enables efficient access to cloud-based raster resources while maintaining security and performance best practices for modern geospatial workflows.