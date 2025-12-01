# Automating Raster Workflows

## Overview
Automated raster workflows enable efficient processing of large datasets, reduce manual errors, and ensure consistent results across multiple processing tasks. By implementing robust automation scripts, you can handle routine data downloads, processing operations, and maintenance tasks with minimal manual intervention.

## Why Automate Raster Workflows
- **Efficiency**: Process large datasets automatically without manual intervention
- **Consistency**: Ensure standardized processing across all datasets
- **Reliability**: Reduce human errors and implement robust error handling
- **Scalability**: Handle increasing data volumes without proportional effort increase
- **Monitoring**: Track processing status and maintain audit trails

---

## 1️⃣ Automating Raster Downloads (e.g., from Sentinel/S3)

### Automated Sentinel Data Downloads
```python
import os
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

# Setup logging
def setup_logging(log_file='raster_automation.log'):
    """Setup logging configuration"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Sentinel data download automation
def automate_sentinel_downloads():
    """Automate Sentinel-2 data downloads"""
    
    class SentinelDownloader:
        def __init__(self, download_dir='./sentinel_data'):
            self.download_dir = Path(download_dir)
            self.download_dir.mkdir(exist_ok=True)
            self.session = requests.Session()
            
        def search_products(self, bbox, date_range, cloud_cover=20):
            """Search for Sentinel products"""
            
            # Example using Copernicus Open Access Hub API
            search_params = {
                'bbox': bbox,
                'start_date': date_range[0],
                'end_date': date_range[1],
                'cloud_cover': f'[0 TO {cloud_cover}]',
                'product_type': 'S2MSI1C'
            }
            
            logger.info(f"Searching Sentinel products: {search_params}")
            
            # Simulate search results
            products = [
                {
                    'id': 'S2A_MSIL1C_20231201T103321_N0509_R108_T32TQM_20231201T123456',
                    'title': 'S2A_MSIL1C_20231201T103321_N0509_R108_T32TQM_20231201T123456.SAFE',
                    'size': '1.2 GB',
                    'cloud_cover': 15.2,
                    'date': '2023-12-01',
                    'download_url': 'https://example.com/sentinel/product1.zip'
                },
                {
                    'id': 'S2B_MSIL1C_20231203T103319_N0509_R108_T32TQM_20231203T134567',
                    'title': 'S2B_MSIL1C_20231203T103319_N0509_R108_T32TQM_20231203T134567.SAFE',
                    'size': '1.1 GB',
                    'cloud_cover': 8.5,
                    'date': '2023-12-03',
                    'download_url': 'https://example.com/sentinel/product2.zip'
                }
            ]
            
            logger.info(f"Found {len(products)} products")
            return products
        
        def download_product(self, product, max_retries=3):
            """Download single product with retry logic"""
            
            product_path = self.download_dir / f"{product['id']}.zip"
            
            if product_path.exists():
                logger.info(f"Product already exists: {product['id']}")
                return product_path
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Downloading {product['id']} (attempt {attempt + 1})")
                    
                    # Simulate download (replace with actual download logic)
                    response = self.session.get(
                        product['download_url'],
                        stream=True,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        with open(product_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        logger.info(f"Downloaded: {product['id']}")
                        return product_path
                    else:
                        logger.warning(f"Download failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Download error (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            
            logger.error(f"Failed to download after {max_retries} attempts: {product['id']}")
            return None
        
        def batch_download(self, bbox, date_range, cloud_cover=20):
            """Batch download products"""
            
            products = self.search_products(bbox, date_range, cloud_cover)
            downloaded_files = []
            
            for product in products:
                file_path = self.download_product(product)
                if file_path:
                    downloaded_files.append(file_path)
            
            logger.info(f"Batch download complete: {len(downloaded_files)} files")
            return downloaded_files
    
    # Usage example
    downloader = SentinelDownloader()
    
    # Define search parameters
    bbox = [2.0, 41.0, 3.0, 42.0]  # Longitude, Latitude bounds
    date_range = ['2023-12-01', '2023-12-31']
    
    # Download products
    downloaded_files = downloader.batch_download(bbox, date_range, cloud_cover=15)
    
    return downloaded_files

# AWS S3 automated downloads
def automate_s3_downloads():
    """Automate downloads from AWS S3 buckets"""
    
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    
    class S3Downloader:
        def __init__(self, bucket_name, download_dir='./s3_data'):
            self.bucket_name = bucket_name
            self.download_dir = Path(download_dir)
            self.download_dir.mkdir(exist_ok=True)
            
            # Anonymous S3 client for public buckets
            self.s3_client = boto3.client(
                's3',
                config=Config(signature_version=UNSIGNED)
            )
        
        def list_objects(self, prefix, max_keys=1000):
            """List objects in S3 bucket"""
            
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=max_keys
                )
                
                objects = response.get('Contents', [])
                logger.info(f"Found {len(objects)} objects with prefix: {prefix}")
                
                return objects
                
            except Exception as e:
                logger.error(f"Failed to list S3 objects: {e}")
                return []
        
        def download_file(self, s3_key, local_path=None):
            """Download single file from S3"""
            
            if local_path is None:
                local_path = self.download_dir / Path(s3_key).name
            
            if local_path.exists():
                logger.info(f"File already exists: {local_path}")
                return local_path
            
            try:
                logger.info(f"Downloading: {s3_key}")
                self.s3_client.download_file(
                    self.bucket_name,
                    s3_key,
                    str(local_path)
                )
                
                logger.info(f"Downloaded: {local_path}")
                return local_path
                
            except Exception as e:
                logger.error(f"Download failed: {e}")
                return None
        
        def batch_download_by_pattern(self, prefix, file_pattern='*.tif'):
            """Download files matching pattern"""
            
            import fnmatch
            
            objects = self.list_objects(prefix)
            downloaded_files = []
            
            for obj in objects:
                if fnmatch.fnmatch(obj['Key'], f"*{file_pattern.replace('*', '')}"):
                    file_path = self.download_file(obj['Key'])
                    if file_path:
                        downloaded_files.append(file_path)
            
            logger.info(f"Pattern download complete: {len(downloaded_files)} files")
            return downloaded_files
    
    # Usage example
    s3_downloader = S3Downloader('landsat-pds')
    
    # Download Landsat files
    landsat_files = s3_downloader.batch_download_by_pattern(
        'c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/',
        '*.TIF'
    )
    
    return landsat_files

# Scheduled downloads
def setup_scheduled_downloads():
    """Setup scheduled download tasks"""
    
    import schedule
    
    def daily_download_task():
        """Daily download task"""
        
        logger.info("Starting daily download task")
        
        # Calculate date range (last 7 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        date_range = [
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        ]
        
        # Run download
        downloader = SentinelDownloader()
        bbox = [2.0, 41.0, 3.0, 42.0]
        
        try:
            files = downloader.batch_download(bbox, date_range)
            logger.info(f"Daily task completed: {len(files)} files downloaded")
        except Exception as e:
            logger.error(f"Daily task failed: {e}")
    
    # Schedule tasks
    schedule.every().day.at("02:00").do(daily_download_task)
    schedule.every().monday.at("01:00").do(lambda: logger.info("Weekly maintenance"))
    
    logger.info("Scheduled tasks configured")
    
    # Run scheduler (in production, use proper task scheduler)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)
    
    return schedule

# Run download automation examples
sentinel_files = automate_sentinel_downloads()
s3_files = automate_s3_downloads()
scheduler = setup_scheduled_downloads()
```

---

## 2️⃣ Batch Clipping, Resampling, and Reprojecting

### Batch Processing Framework
```python
import rasterio
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Batch processing class
class BatchRasterProcessor:
    def __init__(self, output_dir='./processed', n_workers=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_workers = n_workers or mp.cpu_count()
        
    def clip_raster(self, input_file, clip_geometry, output_file=None):
        """Clip raster to geometry"""
        
        if output_file is None:
            output_file = self.output_dir / f"clipped_{Path(input_file).name}"
        
        try:
            logger.info(f"Clipping: {input_file}")
            
            # Load and clip raster
            with rasterio.open(input_file) as src:
                clipped, transform = rasterio.mask.mask(
                    src, [clip_geometry], crop=True
                )
                
                # Update metadata
                profile = src.profile.copy()
                profile.update({
                    'height': clipped.shape[1],
                    'width': clipped.shape[2],
                    'transform': transform
                })
                
                # Write clipped raster
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(clipped)
            
            logger.info(f"Clipped: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Clipping failed for {input_file}: {e}")
            return None
    
    def resample_raster(self, input_file, target_resolution, output_file=None, resampling_method=Resampling.bilinear):
        """Resample raster to target resolution"""
        
        if output_file is None:
            output_file = self.output_dir / f"resampled_{Path(input_file).name}"
        
        try:
            logger.info(f"Resampling: {input_file}")
            
            with rasterio.open(input_file) as src:
                # Calculate new dimensions
                transform = src.transform
                new_width = int(src.width * (transform.a / target_resolution))
                new_height = int(src.height * (abs(transform.e) / target_resolution))
                
                # Create new transform
                new_transform = rasterio.transform.from_bounds(
                    *src.bounds, new_width, new_height
                )
                
                # Resample data
                resampled = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=resampling_method
                )
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': new_transform
                })
                
                # Write resampled raster
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(resampled)
            
            logger.info(f"Resampled: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Resampling failed for {input_file}: {e}")
            return None
    
    def reproject_raster(self, input_file, target_crs, output_file=None, resampling_method=Resampling.bilinear):
        """Reproject raster to target CRS"""
        
        if output_file is None:
            output_file = self.output_dir / f"reprojected_{Path(input_file).name}"
        
        try:
            logger.info(f"Reprojecting: {input_file}")
            
            with rasterio.open(input_file) as src:
                # Calculate transform and dimensions for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                # Reproject
                with rasterio.open(output_file, 'w', **profile) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=resampling_method
                        )
            
            logger.info(f"Reprojected: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Reprojection failed for {input_file}: {e}")
            return None
    
    def batch_process(self, input_files, operations, parallel=True):
        """Batch process multiple files"""
        
        def process_single_file(file_path):
            """Process single file through operation chain"""
            
            current_file = file_path
            results = {'input': file_path, 'operations': []}
            
            for operation in operations:
                op_type = operation['type']
                op_params = operation.get('params', {})
                
                if op_type == 'clip':
                    current_file = self.clip_raster(current_file, **op_params)
                elif op_type == 'resample':
                    current_file = self.resample_raster(current_file, **op_params)
                elif op_type == 'reproject':
                    current_file = self.reproject_raster(current_file, **op_params)
                
                results['operations'].append({
                    'type': op_type,
                    'output': current_file,
                    'success': current_file is not None
                })
                
                if current_file is None:
                    break
            
            results['final_output'] = current_file
            return results
        
        logger.info(f"Starting batch processing: {len(input_files)} files")
        
        if parallel and len(input_files) > 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                results = list(executor.map(process_single_file, input_files))
        else:
            results = [process_single_file(f) for f in input_files]
        
        # Summary
        successful = sum(1 for r in results if r['final_output'] is not None)
        logger.info(f"Batch processing complete: {successful}/{len(input_files)} successful")
        
        return results

# Batch processing examples
def run_batch_processing():
    """Run batch processing examples"""
    
    processor = BatchRasterProcessor()
    
    # Example input files (replace with actual files)
    input_files = [
        './data/raster1.tif',
        './data/raster2.tif',
        './data/raster3.tif'
    ]
    
    # Define processing operations
    operations = [
        {
            'type': 'reproject',
            'params': {'target_crs': 'EPSG:3857'}
        },
        {
            'type': 'resample',
            'params': {'target_resolution': 30}
        }
    ]
    
    # Run batch processing
    results = processor.batch_process(input_files, operations, parallel=True)
    
    # Process results
    for result in results:
        if result['final_output']:
            logger.info(f"Successfully processed: {result['input']} -> {result['final_output']}")
        else:
            logger.error(f"Failed to process: {result['input']}")
    
    return results

# Advanced batch operations
def advanced_batch_operations():
    """Advanced batch processing with custom operations"""
    
    class AdvancedBatchProcessor(BatchRasterProcessor):
        def calculate_ndvi(self, input_file, red_band=4, nir_band=5, output_file=None):
            """Calculate NDVI from multispectral raster"""
            
            if output_file is None:
                output_file = self.output_dir / f"ndvi_{Path(input_file).stem}.tif"
            
            try:
                logger.info(f"Calculating NDVI: {input_file}")
                
                with rasterio.open(input_file) as src:
                    red = src.read(red_band).astype(float)
                    nir = src.read(nir_band).astype(float)
                    
                    # Calculate NDVI
                    ndvi = (nir - red) / (nir + red)
                    ndvi = np.where((nir + red) == 0, 0, ndvi)
                    
                    # Update profile
                    profile = src.profile.copy()
                    profile.update({
                        'count': 1,
                        'dtype': 'float32'
                    })
                    
                    # Write NDVI
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(ndvi.astype('float32'), 1)
                
                logger.info(f"NDVI calculated: {output_file}")
                return output_file
                
            except Exception as e:
                logger.error(f"NDVI calculation failed for {input_file}: {e}")
                return None
        
        def apply_cloud_mask(self, input_file, cloud_mask_file, output_file=None):
            """Apply cloud mask to raster"""
            
            if output_file is None:
                output_file = self.output_dir / f"masked_{Path(input_file).name}"
            
            try:
                logger.info(f"Applying cloud mask: {input_file}")
                
                with rasterio.open(input_file) as src, rasterio.open(cloud_mask_file) as mask_src:
                    data = src.read()
                    mask = mask_src.read(1)
                    
                    # Apply mask (assuming 0 = clear, 1 = cloud)
                    masked_data = np.where(mask == 0, data, src.nodata or 0)
                    
                    # Write masked raster
                    profile = src.profile.copy()
                    with rasterio.open(output_file, 'w', **profile) as dst:
                        dst.write(masked_data)
                
                logger.info(f"Cloud mask applied: {output_file}")
                return output_file
                
            except Exception as e:
                logger.error(f"Cloud masking failed for {input_file}: {e}")
                return None
    
    # Usage example
    advanced_processor = AdvancedBatchProcessor()
    
    # Custom operations chain
    custom_operations = [
        {
            'type': 'reproject',
            'params': {'target_crs': 'EPSG:32633'}
        },
        {
            'type': 'custom',
            'function': 'calculate_ndvi',
            'params': {'red_band': 4, 'nir_band': 8}
        }
    ]
    
    logger.info("Advanced batch processing configured")
    return advanced_processor

# Run batch processing
batch_results = run_batch_processing()
advanced_processor = advanced_batch_operations()
```

---

## 3️⃣ Writing Modular Functions and Logging Outputs

### Modular Function Design
```python
from functools import wraps
import json
from typing import List, Dict, Optional, Union
import traceback

# Configuration management
class ProcessingConfig:
    def __init__(self, config_file='processing_config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        
        default_config = {
            'processing': {
                'default_crs': 'EPSG:4326',
                'default_resolution': 30,
                'resampling_method': 'bilinear',
                'compression': 'lzw',
                'tiled': True,
                'blocksize': 512
            },
            'paths': {
                'input_dir': './input',
                'output_dir': './output',
                'temp_dir': './temp',
                'log_dir': './logs'
            },
            'download': {
                'max_retries': 3,
                'timeout': 300,
                'chunk_size': 8192
            }
        }
        
        try:
            with open(self.config_file, 'r') as f:
                user_config = json.load(f)
            
            # Merge with defaults
            config = default_config.copy()
            config.update(user_config)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            return config
            
        except FileNotFoundError:
            logger.info("Using default configuration")
            return default_config
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

# Decorator for logging and error handling
def log_execution(func):
    """Decorator to log function execution"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"Completed {func_name} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func_name} after {execution_time:.2f}s: {e}")
            logger.debug(traceback.format_exc())
            raise
    
    return wrapper

# Modular processing functions
class RasterProcessingModules:
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    @log_execution
    def validate_input(self, file_path: Union[str, Path]) -> bool:
        """Validate input raster file"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Input file does not exist: {file_path}")
            return False
        
        try:
            with rasterio.open(file_path) as src:
                if src.count == 0:
                    logger.error(f"No bands found in: {file_path}")
                    return False
                
                if src.crs is None:
                    logger.warning(f"No CRS defined in: {file_path}")
                
                logger.info(f"Valid input: {file_path} ({src.width}x{src.height}, {src.count} bands)")
                return True
                
        except Exception as e:
            logger.error(f"Invalid raster file {file_path}: {e}")
            return False
    
    @log_execution
    def standardize_raster(self, input_file: Path, output_file: Path = None) -> Optional[Path]:
        """Standardize raster to common format and projection"""
        
        if output_file is None:
            output_file = Path(self.config.get('paths.output_dir')) / f"std_{Path(input_file).name}"
        
        target_crs = self.config.get('processing.default_crs')
        compression = self.config.get('processing.compression')
        tiled = self.config.get('processing.tiled')
        blocksize = self.config.get('processing.blocksize')
        
        try:
            with rasterio.open(input_file) as src:
                # Check if reprojection needed
                if str(src.crs) != target_crs:
                    logger.info(f"Reprojecting from {src.crs} to {target_crs}")
                    
                    # Calculate target transform and dimensions
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds
                    )
                else:
                    transform = src.transform
                    width, height = src.width, src.height
                
                # Update profile
                profile = src.profile.copy()
                profile.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'compress': compression,
                    'tiled': tiled,
                    'blockxsize': blocksize,
                    'blockysize': blocksize
                })
                
                # Write standardized raster
                with rasterio.open(output_file, 'w', **profile) as dst:
                    if str(src.crs) != target_crs:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=target_crs,
                                resampling=Resampling.bilinear
                            )
                    else:
                        dst.write(src.read())
            
            logger.info(f"Standardized: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Standardization failed: {e}")
            return None
    
    @log_execution
    def create_overview(self, input_file: Path, overview_levels: List[int] = None) -> bool:
        """Create overview pyramids for raster"""
        
        if overview_levels is None:
            overview_levels = [2, 4, 8, 16]
        
        try:
            with rasterio.open(input_file, 'r+') as src:
                src.build_overviews(overview_levels, Resampling.average)
                src.update_tags(ns='rio_overview', resampling='average')
            
            logger.info(f"Overviews created for {input_file}: {overview_levels}")
            return True
            
        except Exception as e:
            logger.error(f"Overview creation failed for {input_file}: {e}")
            return False
    
    @log_execution
    def calculate_statistics(self, input_file: Path) -> Optional[Dict]:
        """Calculate raster statistics"""
        
        try:
            with rasterio.open(input_file) as src:
                stats = {}
                
                for i in range(1, src.count + 1):
                    band_data = src.read(i, masked=True)
                    
                    band_stats = {
                        'min': float(band_data.min()),
                        'max': float(band_data.max()),
                        'mean': float(band_data.mean()),
                        'std': float(band_data.std()),
                        'count': int(band_data.count()),
                        'nodata_count': int(band_data.mask.sum())
                    }
                    
                    stats[f'band_{i}'] = band_stats
                
                # Overall statistics
                stats['metadata'] = {
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'crs': str(src.crs),
                    'bounds': src.bounds
                }
            
            logger.info(f"Statistics calculated for {input_file}")
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation failed for {input_file}: {e}")
            return None

# Processing pipeline
class ProcessingPipeline:
    def __init__(self, config_file='processing_config.json'):
        self.config = ProcessingConfig(config_file)
        self.modules = RasterProcessingModules(self.config)
        self.results = []
    
    def add_step(self, step_name: str, function, **kwargs):
        """Add processing step to pipeline"""
        
        step = {
            'name': step_name,
            'function': function,
            'params': kwargs,
            'status': 'pending'
        }
        
        self.results.append(step)
        return self
    
    def execute(self, input_files: List[Path]) -> Dict:
        """Execute processing pipeline"""
        
        logger.info(f"Executing pipeline with {len(input_files)} files")
        
        pipeline_results = {
            'input_files': input_files,
            'steps': [],
            'successful_files': [],
            'failed_files': []
        }
        
        for file_path in input_files:
            file_results = {'file': file_path, 'steps': []}
            current_file = file_path
            
            # Execute each step
            for step in self.results:
                step_result = {
                    'name': step['name'],
                    'input': current_file,
                    'output': None,
                    'success': False,
                    'error': None
                }
                
                try:
                    logger.info(f"Executing {step['name']} on {current_file}")
                    
                    # Execute step function
                    result = step['function'](current_file, **step['params'])
                    
                    if result:
                        step_result['output'] = result
                        step_result['success'] = True
                        current_file = result
                    else:
                        step_result['error'] = "Function returned None"
                        break
                        
                except Exception as e:
                    step_result['error'] = str(e)
                    logger.error(f"Step {step['name']} failed: {e}")
                    break
                
                file_results['steps'].append(step_result)
            
            # Determine overall success
            if all(step['success'] for step in file_results['steps']):
                pipeline_results['successful_files'].append(file_path)
                file_results['final_output'] = current_file
            else:
                pipeline_results['failed_files'].append(file_path)
                file_results['final_output'] = None
            
            pipeline_results['steps'].append(file_results)
        
        # Summary
        success_count = len(pipeline_results['successful_files'])
        total_count = len(input_files)
        
        logger.info(f"Pipeline complete: {success_count}/{total_count} files successful")
        
        return pipeline_results

# Usage example
def create_processing_pipeline():
    """Create and configure processing pipeline"""
    
    # Initialize pipeline
    pipeline = ProcessingPipeline()
    
    # Add processing steps
    pipeline.add_step(
        'validate',
        pipeline.modules.validate_input
    ).add_step(
        'standardize',
        pipeline.modules.standardize_raster
    ).add_step(
        'create_overviews',
        pipeline.modules.create_overview,
        overview_levels=[2, 4, 8]
    ).add_step(
        'calculate_stats',
        pipeline.modules.calculate_statistics
    )
    
    return pipeline

# Run modular processing
config = ProcessingConfig()
pipeline = create_processing_pipeline()
```

---

## 4️⃣ Cleanup and Backup Scripts

### Automated Cleanup Scripts
```python
import shutil
from datetime import datetime, timedelta
import glob

class CleanupManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.temp_dir = Path(self.config.get('paths.temp_dir'))
        self.log_dir = Path(self.config.get('paths.log_dir'))
        self.output_dir = Path(self.config.get('paths.output_dir'))
    
    @log_execution
    def cleanup_temp_files(self, max_age_days: int = 7) -> int:
        """Clean up temporary files older than specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        if not self.temp_dir.exists():
            logger.info("Temp directory does not exist")
            return 0
        
        for file_path in self.temp_dir.rglob('*'):
            if file_path.is_file():
                file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_age < cutoff_date:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Deleted temp file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        return cleaned_count
    
    @log_execution
    def cleanup_logs(self, max_age_days: int = 30, keep_count: int = 10) -> int:
        """Clean up old log files"""
        
        if not self.log_dir.exists():
            logger.info("Log directory does not exist")
            return 0
        
        # Get all log files
        log_files = list(self.log_dir.glob('*.log'))
        log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        cleaned_count = 0
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        # Keep recent files, delete old ones
        for i, log_file in enumerate(log_files):
            file_age = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if i >= keep_count and file_age < cutoff_date:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Deleted log file: {log_file}")
                except Exception as e:
                    logger.error(f"Failed to delete {log_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} log files")
        return cleaned_count
    
    @log_execution
    def disk_usage_check(self, warning_threshold: float = 0.8) -> Dict:
        """Check disk usage and warn if approaching limits"""
        
        usage_info = {}
        
        for dir_name, dir_path in [
            ('temp', self.temp_dir),
            ('output', self.output_dir),
            ('logs', self.log_dir)
        ]:
            if dir_path.exists():
                total, used, free = shutil.disk_usage(dir_path)
                usage_percent = used / total
                
                usage_info[dir_name] = {
                    'total_gb': total / (1024**3),
                    'used_gb': used / (1024**3),
                    'free_gb': free / (1024**3),
                    'usage_percent': usage_percent
                }
                
                if usage_percent > warning_threshold:
                    logger.warning(f"High disk usage in {dir_name}: {usage_percent:.1%}")
                else:
                    logger.info(f"Disk usage {dir_name}: {usage_percent:.1%}")
        
        return usage_info

# Backup management
class BackupManager:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.backup_dir = Path(self.config.get('paths.backup_dir', './backups'))
        self.backup_dir.mkdir(exist_ok=True)
    
    @log_execution
    def create_backup(self, source_dir: Path, backup_name: str = None) -> Optional[Path]:
        """Create compressed backup of directory"""
        
        if backup_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{source_dir.name}_{timestamp}"
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            logger.info(f"Creating backup: {source_dir} -> {backup_path}")
            
            import tarfile
            
            with tarfile.open(backup_path, 'w:gz') as tar:
                tar.add(source_dir, arcname=source_dir.name)
            
            backup_size = backup_path.stat().st_size / (1024**2)  # MB
            logger.info(f"Backup created: {backup_path} ({backup_size:.1f} MB)")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    @log_execution
    def rotate_backups(self, pattern: str, keep_count: int = 5) -> int:
        """Rotate backups, keeping only recent ones"""
        
        backup_files = list(self.backup_dir.glob(pattern))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        removed_count = 0
        
        for backup_file in backup_files[keep_count:]:
            try:
                backup_file.unlink()
                removed_count += 1
                logger.debug(f"Removed old backup: {backup_file}")
            except Exception as e:
                logger.error(f"Failed to remove backup {backup_file}: {e}")
        
        logger.info(f"Backup rotation complete: removed {removed_count} old backups")
        return removed_count
    
    @log_execution
    def verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity"""
        
        try:
            import tarfile
            
            with tarfile.open(backup_path, 'r:gz') as tar:
                # Try to list contents
                members = tar.getmembers()
                logger.info(f"Backup verification successful: {len(members)} files in {backup_path}")
                return True
                
        except Exception as e:
            logger.error(f"Backup verification failed for {backup_path}: {e}")
            return False

# Automated maintenance
class MaintenanceScheduler:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.cleanup_manager = CleanupManager(config)
        self.backup_manager = BackupManager(config)
    
    @log_execution
    def daily_maintenance(self):
        """Daily maintenance tasks"""
        
        logger.info("Starting daily maintenance")
        
        # Cleanup temp files
        self.cleanup_manager.cleanup_temp_files(max_age_days=1)
        
        # Check disk usage
        usage_info = self.cleanup_manager.disk_usage_check()
        
        # Log summary
        logger.info("Daily maintenance completed")
        return usage_info
    
    @log_execution
    def weekly_maintenance(self):
        """Weekly maintenance tasks"""
        
        logger.info("Starting weekly maintenance")
        
        # Cleanup older temp files
        self.cleanup_manager.cleanup_temp_files(max_age_days=7)
        
        # Cleanup old logs
        self.cleanup_manager.cleanup_logs(max_age_days=30)
        
        # Create backup of output directory
        output_dir = Path(self.config.get('paths.output_dir'))
        if output_dir.exists():
            backup_path = self.backup_manager.create_backup(output_dir)
            
            if backup_path:
                # Verify backup
                self.backup_manager.verify_backup(backup_path)
                
                # Rotate old backups
                self.backup_manager.rotate_backups('backup_output_*.tar.gz', keep_count=4)
        
        logger.info("Weekly maintenance completed")
    
    def setup_maintenance_schedule(self):
        """Setup automated maintenance schedule"""
        
        import schedule
        
        # Schedule maintenance tasks
        schedule.every().day.at("01:00").do(self.daily_maintenance)
        schedule.every().sunday.at("02:00").do(self.weekly_maintenance)
        
        logger.info("Maintenance schedule configured")
        return schedule

# Usage example
def run_maintenance_example():
    """Run maintenance operations example"""
    
    config = ProcessingConfig()
    maintenance = MaintenanceScheduler(config)
    
    # Run immediate maintenance
    maintenance.daily_maintenance()
    
    # Setup scheduled maintenance
    schedule = maintenance.setup_maintenance_schedule()
    
    logger.info("Maintenance system initialized")
    return maintenance

# Run cleanup and backup examples
maintenance_system = run_maintenance_example()
```

---

## Best Practices and Tips

### Workflow Design
- **Modular Architecture**: Design reusable, testable components
- **Configuration Management**: Use external config files for flexibility
- **Error Handling**: Implement comprehensive error handling and recovery
- **Progress Tracking**: Provide clear progress indicators and logging

### Performance Optimization
- **Parallel Processing**: Use multiprocessing for CPU-bound tasks
- **Memory Management**: Monitor and optimize memory usage
- **Disk I/O**: Minimize disk operations and use efficient formats
- **Caching**: Cache intermediate results when appropriate

### Monitoring and Maintenance
- **Comprehensive Logging**: Log all operations with appropriate detail levels
- **Health Checks**: Implement system health monitoring
- **Automated Cleanup**: Regular cleanup of temporary and old files
- **Backup Strategy**: Implement reliable backup and recovery procedures

This comprehensive automation framework enables robust, scalable raster processing workflows with proper error handling, logging, and maintenance capabilities.