
import os
import io
import asyncio
import logging
import hashlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# ==========================================
#  EXTERNAL DEPENDENCIES
# ==========================================
# pip install boto3 aioboto3 pillow torch transformers pyodbc tqdm aiofiles

import torch
from PIL import Image
from tqdm.asyncio import tqdm as async_tqdm
from transformers import (
    AutoModel,
    AutoProcessor,                      
    Blip2Processor,
    Blip2ForConditionalGeneration,
    InstructBlipProcessor,  # Add this
    InstructBlipForConditionalGeneration,
)

# AWS
import boto3
import aioboto3

# SQL Server
import pyodbc


def s3_to_public_url(s3_path: str, bucket: str) -> str:
    """Convert s3://bucket/key to https://bucket.s3.amazonaws.com/key"""
    if not s3_path or not s3_path.startswith("s3://"):
        return s3_path
    
    # Remove s3:// prefix
    path = s3_path[5:]
    
    # Split bucket and key
    first_slash = path.index('/')
    bucket = path[:first_slash]
    key = path[first_slash + 1:]
    
    return f"https://{bucket}.s3.amazonaws.com/{key}"




# ==========================================
#  CONFIGURATION
# ==========================================

@dataclass
class Config:
    """
    Centralized configuration for the pipeline.
    Modify these values according to your environment.
    """
    
    # ----- AWS Configuration -----
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    
    # S3 Buckets
    S3_PRIVATE_BUCKET: str = "imageshopprivate"  # Original images (private)
    S3_PUBLIC_BUCKET: str = "imageshoppublic"    # Compressed images (public)
    S3_PRIVATE_PREFIX: str = "original-images/"                   # Folder prefix in private bucket
    S3_PUBLIC_PREFIX: str = "thumbnails/"                # Folder prefix in public bucket
    
    # ----- SQL Server Configuration -----
    SQL_SERVER: str = ""
    SQL_DATABASE: str = ""
    SQL_USERNAME: str = ""
    SQL_PASSWORD: str = ""
    SQL_DRIVER: str = ""
    
    # ----- Processing Configuration -----
    BATCH_SIZE_FETCH: int = 50          # Images to fetch from S3 per batch
    BATCH_SIZE_GPU: int = 8             # Images per GPU batch (adjust based on VRAM)
    MAX_IMAGES_TO_PROCESS: int = None    # Limit for testing (set to None for all)
    MAX_CONCURRENT_DOWNLOADS: int = 20  # Parallel S3 downloads
    MAX_CONCURRENT_UPLOADS: int = 20    # Parallel S3 uploads
    THREAD_POOL_SIZE: int = 8           # CPU compression threads
    
    # ----- Compression Configuration -----
    TARGET_SIZE_KB: int = 200           # Target compressed size (100-250KB)
    MAX_DIMENSION: int = 800            # Max width/height for thumbnail
    MIN_QUALITY: int = 30               # Minimum JPEG quality
    MAX_QUALITY: int = 85               # Maximum JPEG quality
    
    # ----- Checkpoint Configuration -----
    CHECKPOINT_FILE: str = "processing_checkpoint.json"
    CHECKPOINT_INTERVAL: int = 100      # Save checkpoint every N images
    
    # ----- Logging -----
    LOG_FILE: str = "image_categorization.log"
    LOG_LEVEL: int = logging.INFO


# ==========================================
#  DATA MODELS
# ==========================================

@dataclass
class ImageMetadata:
    """
    Represents processed image metadata.
    Maps directly to SQL Server table columns.
    """
    # Primary identification
    image_id: str = ""                      # SHA256 hash of original image
    original_filename: str = ""
    
    # AWS paths
    s3_original_path: str = ""              # s3://private-bucket/images/...
    s3_compressed_path: str = ""            # s3://public-bucket/thumbnails/...
    
    # Image properties
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    width: int = 0
    height: int = 0
    format: str = ""
    
    # AI-generated metadata
    main_category: str = ""
    sub_category: str = ""
    caption: str = ""
    tags: str = ""                          # Comma-separated AI-generated tags
    
    # Confidence scores
    category_confidence: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Processing status
    status: str = "pending"                 # pending, processing, completed, failed
    error_message: str = ""


# ==========================================
#  LOGGING SETUP
# ==========================================

def setup_logging(config: Config) -> logging.Logger:
    """
    Configure structured logging for production use.
    Logs to both file and console.
    """
    logger = logging.getLogger("ImageCategorization")
    logger.setLevel(config.LOG_LEVEL)
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ==========================================
#  SQL SERVER DATABASE MANAGER
# ==========================================

class SQLServerManager:
    """
    Manages SQL Server connections and operations.
    Uses connection pooling for efficiency.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._connection_string = self._build_connection_string()
        self._connection_pool: List[pyodbc.Connection] = []
        self._pool_size = 5
    
    def _build_connection_string(self) -> str:
        """Build ODBC connection string for SQL Server."""
        return (
            f"DRIVER={self.config.SQL_DRIVER};"
            f"SERVER={self.config.SQL_SERVER},1433;"
            f"DATABASE={self.config.SQL_DATABASE};"
            f"UID={self.config.SQL_USERNAME};"
            f"PWD={self.config.SQL_PASSWORD};"
            "Encrypt=yes;"
            "TrustServerCertificate=no;"
            "Connection Timeout=30;"
        )
    
    def get_connection(self) -> pyodbc.Connection:
        """Get connection from pool or create new one."""
        if self._connection_pool:
            return self._connection_pool.pop()
        return pyodbc.connect(self._connection_string)
    
    def return_connection(self, conn: pyodbc.Connection):
        """Return connection to pool."""
        if len(self._connection_pool) < self._pool_size:
            self._connection_pool.append(conn)
        else:
            conn.close()
    
    def initialize_database(self):
        """
        Create the images metadata table if it doesn't exist.
        Includes proper indexes for query optimization.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create table with all required columns
        create_table_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='image_metadata' AND xtype='U')
        CREATE TABLE image_metadata (
            -- Primary Key
            id BIGINT IDENTITY(1,1) PRIMARY KEY,
            image_id NVARCHAR(64) NOT NULL UNIQUE,  -- SHA256 hash
            
            -- File Information
            original_filename NVARCHAR(500) NOT NULL,
            s3_original_path NVARCHAR(1000) NOT NULL,
            s3_compressed_path NVARCHAR(1000) NOT NULL,
            
            -- Image Properties
            original_size_bytes BIGINT NOT NULL,
            compressed_size_bytes INT NOT NULL,
            width INT NOT NULL,
            height INT NOT NULL,
            format NVARCHAR(10) NOT NULL,
            
            -- AI-Generated Metadata
            main_category NVARCHAR(100) NOT NULL,
            sub_category NVARCHAR(100) NOT NULL,
            caption NVARCHAR(MAX) NOT NULL,
            tags NVARCHAR(MAX) NOT NULL,  -- Comma-separated
            category_confidence FLOAT NOT NULL,
            
            -- Timestamps
            created_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
            processed_at DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
            
            -- Status
            status NVARCHAR(20) NOT NULL DEFAULT 'pending',
            error_message NVARCHAR(MAX) NULL,
            
            -- Indexes for common queries
            INDEX idx_main_category (main_category),
            INDEX idx_sub_category (sub_category),
            INDEX idx_status (status),
            INDEX idx_created_at (created_at),
            INDEX idx_processed_at (processed_at)
        );
        
        -- Full-text index for tags and caption search (if not exists)
        IF NOT EXISTS (SELECT * FROM sys.fulltext_indexes WHERE object_id = OBJECT_ID('image_metadata'))
        BEGIN
            -- Note: Full-text catalog must be created first by DBA
            -- CREATE FULLTEXT CATALOG ImageCatalog AS DEFAULT;
            -- CREATE FULLTEXT INDEX ON image_metadata(tags, caption) KEY INDEX PK__image_metadata;
            PRINT 'Full-text index should be created by DBA for tags and caption search';
        END
        """
        
        try:
            cursor.execute(create_table_sql)
            conn.commit()
            self.logger.info(" Database table initialized successfully")
        except Exception as e:
            self.logger.error(f" Failed to initialize database: {e}")
            raise
        finally:
            self.return_connection(conn)
    
    def insert_batch(self, metadata_list: List[ImageMetadata]) -> int:
        """
        Insert batch of image metadata records.
        Uses parameterized queries for security.
        Returns number of successfully inserted records.
        """
        if not metadata_list:
            return 0
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        insert_sql = """
        INSERT INTO image_metadata (
            image_id, original_filename, s3_original_path, s3_compressed_path,
            original_size_bytes, compressed_size_bytes, width, height, format,
            main_category, sub_category, caption, tags, category_confidence,
            created_at, processed_at, status, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        inserted = 0
        for meta in metadata_list:
            try:
                cursor.execute(insert_sql, (
                    meta.image_id,
                    meta.original_filename,
                    meta.s3_original_path,
                    meta.s3_compressed_path,
                    meta.original_size_bytes,
                    meta.compressed_size_bytes,
                    meta.width,
                    meta.height,
                    meta.format,
                    meta.main_category,
                    meta.sub_category,
                    meta.caption,
                    meta.tags,
                    meta.category_confidence,
                    meta.created_at,
                    meta.processed_at,
                    meta.status,
                    meta.error_message
                ))
                inserted += 1
            except pyodbc.IntegrityError:
                # Duplicate image_id - skip
                self.logger.warning(f" Duplicate image skipped: {meta.image_id[:16]}...")
            except Exception as e:
                self.logger.error(f" Insert failed for {meta.original_filename}: {e}")
        
        conn.commit()
        self.return_connection(conn)
        return inserted
    
    def get_processed_ids(self) -> set:
        """Get set of already processed image IDs for deduplication."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT image_id FROM image_metadata WHERE status = 'completed'")
        ids = {row[0] for row in cursor.fetchall()}
        
        self.return_connection(conn)
        return ids
    
    def close_all(self):
        """Close all pooled connections."""
        for conn in self._connection_pool:
            conn.close()
        self._connection_pool.clear()


# ==========================================
#  AWS S3 MANAGER (ASYNC)
# ==========================================

class S3Manager:
    """
    Async AWS S3 operations manager.
    Handles both download from private bucket and upload to public bucket.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._session = None
    
    @asynccontextmanager
    async def get_session(self):
        """Create aioboto3 session with credentials."""
        session = aioboto3.Session(
            aws_access_key_id=self.config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.config.AWS_SECRET_ACCESS_KEY,
            region_name=self.config.AWS_REGION
        )
        yield session
    
    # async def list_images(
    #     self, 
    #     max_images: Optional[int] = None,
    #     continuation_token: Optional[str] = None
    # ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    #     """
    #     List images from private S3 bucket.
        
    #     Returns:
    #         Tuple of (list of image objects, next continuation token)
    #     """
    #     async with self.get_session() as session:
    #         async with session.client('s3') as s3:
    #             images = []
    #             params = {
    #                 'Bucket': self.config.S3_PRIVATE_BUCKET,
    #                 'Prefix': self.config.S3_PRIVATE_PREFIX,
    #                 'MaxKeys': min(1000, max_images) if max_images else 1000
    #             }
                   
    #             if continuation_token:
    #                 params['ContinuationToken'] = continuation_token
                
    #             response = await s3.list_objects_v2(**params)
                
    #             for obj in response.get('Contents', []):
    #                 key = obj['Key']
    #                 # Filter only image files
    #                 if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
    #                     images.append({
    #                         'key': key,
    #                         'size': obj['Size'],
    #                         'last_modified': obj['LastModified']
    #                     })
                        
    #                     if max_images and len(images) >= max_images:
    #                         break
                
    #             next_token = response.get('NextContinuationToken')
                
    #             self.logger.info(f" Listed {len(images)} images from S3")
    #             return images, next_token





    async def list_images(
        self, 
        max_images: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List ALL images from private S3 bucket using pagination.
        No more 1000 image limit!
        """
        async with self.get_session() as session:
            async with session.client('s3') as s3:
                images = []
                continuation_token = None
                
                # Keep fetching until no more pages OR we hit max_images
                while True:
                    params = {
                        'Bucket': self.config.S3_PRIVATE_BUCKET,
                        'Prefix': self.config.S3_PRIVATE_PREFIX,
                        'MaxKeys': 1000  # S3 API max per request
                    }
                    
                    # Add continuation token for pagination
                    if continuation_token:
                        params['ContinuationToken'] = continuation_token
                    
                    response = await s3.list_objects_v2(**params)
                    
                    # Process images from this page
                    for obj in response.get('Contents', []):
                        key = obj['Key']
                        # Filter only image files
                        if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                            images.append({
                                'key': key,
                                'size': obj['Size'],
                                'last_modified': obj['LastModified']
                            })
                            
                            # Stop if we hit the max_images limit
                            if max_images and len(images) >= max_images:
                                self.logger.info(f"✓ Reached max_images limit: {max_images}")
                                return images
                    
                    # Check if there are more pages
                    if response.get('IsTruncated'):
                        continuation_token = response.get('NextContinuationToken')
                        self.logger.info(f"📄 Fetched {len(images)} images so far, continuing...")
                    else:
                        # No more pages
                        break
                
                self.logger.info(f"✓ Total images found: {len(images)}")
                return images




















    
    async def download_image(self, key: str) -> Tuple[bytes, str]:
        """
        Download single image from private bucket.
        
        Returns:
            Tuple of (image bytes, original key)
        """
        async with self.get_session() as session:
            async with session.client('s3') as s3:
                try:
                    response = await s3.get_object(
                        Bucket=self.config.S3_PRIVATE_BUCKET,
                        Key=key
                    )
                    data = await response['Body'].read()
                    return data, key
                except Exception as e:
                    self.logger.error(f" Download failed for {key}: {e}")
                    raise
    
    async def download_batch(
        self, 
        keys: List[str],
        semaphore: asyncio.Semaphore
    ) -> List[Tuple[bytes, str]]:
        """
        Download multiple images concurrently with rate limiting.
        """
        async def download_with_semaphore(key: str):
            async with semaphore:
                return await self.download_image(key)
        
        tasks = [download_with_semaphore(key) for key in keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed downloads
        successful = []
        for result in results:
            if isinstance(result, Exception):
                continue
            successful.append(result)
        
        return successful
    
    async def upload_compressed(
        self, 
        image_bytes: bytes, 
        original_key: str
    ) -> str:
        """
        Upload compressed image to public bucket.
        
        Returns:
            S3 path of uploaded compressed image
        """
        # Generate compressed image key
        filename = os.path.basename(original_key)
        name, _ = os.path.splitext(filename)
        compressed_key = f"{self.config.S3_PUBLIC_PREFIX}{name}_thumb.jpg"
        
        async with self.get_session() as session:
            async with session.client('s3') as s3:
                await s3.put_object(
                    Bucket=self.config.S3_PUBLIC_BUCKET,
                    Key=compressed_key,
                    Body=image_bytes,
                    ContentType='image/jpeg',
                    CacheControl='max-age=31536000',  # 1 year cache
                )
        
        return f"https://{self.config.S3_PUBLIC_BUCKET}.s3.amazonaws.com/{compressed_key}"
    
    async def upload_batch(
        self,
        images: List[Tuple[bytes, str]],  # (compressed_bytes, original_key)
        semaphore: asyncio.Semaphore
    ) -> List[Tuple[str, str]]:
        """
        Upload batch of compressed images.
        
        Returns:
            List of (original_key, compressed_s3_path)
        """
        async def upload_with_semaphore(img_bytes: bytes, key: str):
            async with semaphore:
                compressed_path = await self.upload_compressed(img_bytes, key)
                return key, compressed_path
        
        tasks = [upload_with_semaphore(img, key) for img, key in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        for result in results:
            if isinstance(result, Exception):
                continue
            successful.append(result)
        
        return successful


# ==========================================
#  IMAGE COMPRESSOR
# ==========================================

class ImageCompressor:
    """
    CPU-bound image compression.
    Runs in thread pool for parallelism.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_SIZE)
    
    def compress_single(
        self, 
        image_bytes: bytes, 
        original_key: str
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compress single image to target size.
        
        Returns:
            Tuple of (compressed_bytes, metadata_dict)
        """
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            original_format = img.format or "JPEG"
            
            # Convert to RGB if necessary (for PNG with transparency, etc.)
            if img.mode in ('RGBA', 'P', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            original_width, original_height = img.size
            
            # Resize if larger than max dimension
            if max(img.size) > self.config.MAX_DIMENSION:
                img.thumbnail(
                    (self.config.MAX_DIMENSION, self.config.MAX_DIMENSION),
                    Image.Resampling.LANCZOS
                )
            
            # Binary search for optimal quality
            target_bytes = self.config.TARGET_SIZE_KB * 1024
            quality = self.config.MAX_QUALITY
            min_quality = self.config.MIN_QUALITY
            
            best_bytes = None
            best_quality = quality
            
            while min_quality <= quality:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                size = buffer.tell()
                
                if size <= target_bytes:
                    best_bytes = buffer.getvalue()
                    best_quality = quality
                    break
                
                quality -= 5
            
            # If still too large, use minimum quality
            if best_bytes is None:
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=min_quality, optimize=True)
                best_bytes = buffer.getvalue()
                best_quality = min_quality
            
            # Generate image hash for deduplication
            image_hash = hashlib.sha256(image_bytes).hexdigest()
            
            metadata = {
                'image_id': image_hash,
                'original_filename': os.path.basename(original_key),
                'original_size_bytes': len(image_bytes),
                'compressed_size_bytes': len(best_bytes),
                'width': img.size[0],
                'height': img.size[1],
                'format': original_format,
                's3_original_path': s3_to_public_url(
        f"s3://{self.config.S3_PRIVATE_BUCKET}/{original_key}",
        self.config.S3_PRIVATE_BUCKET),
            }
            
            return best_bytes, metadata
            
        except Exception as e:
            self.logger.error(f" Compression failed for {original_key}: {e}")
            raise
    
    async def compress_batch(
        self, 
        images: List[Tuple[bytes, str]]
    ) -> List[Tuple[bytes, Dict[str, Any]]]:
        """
        Compress batch of images using thread pool.
        
        Returns:
            List of (compressed_bytes, metadata_dict)
        """
        loop = asyncio.get_event_loop()
        
        tasks = [
            loop.run_in_executor(
                self.executor,
                self.compress_single,
                img_bytes,
                key
            )
            for img_bytes, key in images
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        for result in results:
            if isinstance(result, Exception):
                continue
            successful.append(result)
        
        return successful
    
    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=True)


# ==========================================
# 🧠 AI CATEGORIZATION ENGINE
# ==========================================

class AICategorizationEngine:
    """
    AI-powered image categorization using:
    - SigLIP for zero-shot category classification
    - BLIP-2 for caption and tag generation
    
    IMPORTANT: All GPU operations are sequential (not thread-safe).
    Batching is used for efficiency.
    """
    
    
    




    # Use exact taxonomy from main.py
    TAXONOMY = {
    "Backgrounds": [
        "Background","White Background","Black Background","Christmas Background",
        "Cool Background","Cute Background","Pink Background","Aesthetic Backgrounds",
        "Blue Background","Fall Background","Red Background","Green Background",
        "Halloween Background","Purple Background","Beach Background","Space Background",
        "Anime Background","Flower Background","Galaxy Background","Gold Background",
        "Yellow Background","Grey Background","Heart Background","Light Blue Background",
        "Rainbow Background","Winter Background","Marble Background","Office Background",
        "Spring Background","Birthday Background","Stockphoto-Graf"
    ],

    "Nature": [
        "Nature Background","Sunset","Flowers","Sunrise","Autumn","Water",
        "Beach","Ocean","Desert","Star","Moon","Spring","Exercise",
        "Summer","Forest","Camping","Rain","Hiking","Clouds","Lake",
        "Winter","Mountains","Underwater","Wind","Recycling","Moss",
        "Dusk","Nature Wallpaper","People In Nature","Gardening","Wildlife",
        "Fall Wallpaper","Red Moon","Fall Leaves","Landscapes","Ocean Background",
        "Travel Mania"
    ],

    "Business": [
        "Stock Market","Recession","Money","Business Casual","Office",
        "Cryptocurrency","Teleworking","Business","Digital Marketing",
        "Customer Service","Work From Home","Marketing","Bankruptcy",
        "Globalization","Economics","Fintech","Biotechnology","Business Plan",
        "Business Card","Small Business","Infographics","Conference Room",
        "Business Card Design","Happy Work Anniversary","Conference Call",
        "Business Man","Business Woman","Hybrid Work","Business Travel",
        "Teamwork"
    ],

    "People": [
        "People","Team Work","Children","Families","Women","Group Photo",
        "Friends","Baby","Party","City Streets","Sporting Event","Concerts",
        "Doctor","Boy","Men","Business People","Faces","Kids","Wedding",
        "Happy People","Crowd Of People","People Icons","People Walking",
        "People Eating","Mom And Son","Dad And Son","Siblings","Senior Citizens",
        "Authentic People","Diverse People","Drawings Of People",
        "Silhouettes Of People","Large Group Of People"
    ],

    "Medical": [
        "Hospital","Nurse","Positive Pregnancy Test","Heart","Doctor",
        "Eye","Eyeball","Bacteria","Medical","Science","Mri","Brain",
        "Medicine","Blood","X Ray","Dental","Skin Disease","Cell",
        "Magnetic Resonance Imaging","Dental Implant","Medical Background",
        "Coughing","Cancer","Heart Anatomy"
    ],

    "Food": [
        "Food","Food Truck","Healthy Food","Diet","Mexican Food",
        "Food Icons","Food Clipart","Food Vectors","Seafood","Food Bank",
        "Food Pantry","Fast Food","Italian Food","Food Delivery","Food Waste",
        "Food Background","Food Truck Mockup","Food Safety","Chinese Food",
        "Canned Food","Dog Food","Junk Food","Indian Food","Food Manufacturing"
    ],

    "Technology": [
        "Technology Background","Laptop","Cryptocurrency","Computer","Podcast",
        "Innovation","Artificial Intelligence","Cybersecurity","Fintech",
        "Information Technology","Biotechnology","Robotics","Smartphone",
        "Blockchain Technology","Generative Ai","Nanotechnology","Science Fiction",
        "Old Computer","Medical Technology","Computer Clipart","Computer Drawing",
        "Cloud Technology","Healthcare Technology","Financial Technology",
        "Construction Technology","Tech Company","Computer Wallpaper",
        "Technical Difficulties Screen","Computer Cartoon","Technology Clipart",
        "Tierney"
    ],

    "Travel": [
        "Passport","Beach","Cruise","Luggage","Airport","Travel Insurance",
        "Spring Break","Travel Agent","Family Travel","Beach Background",
        "Luxury Travel","Beach Sunset","Flower Pictures","Summer Vacation",
        "Palm Tree","World Travel","Business Travel","Adventure Travel",
        "Travel Background","Holiday Travel","Space Travel","Roadtrip",
        "Vacation Mode","Out Of Office"
    ],

    "Winter": [
        "Winter Background","Winter Wallpaper","Winter Banner","Winter Border","Winter Holiday",
        "Winter Scene","Winter Wonderland","Snowy Trees","Winter Landscapes","Winter Solstice",
        "Winter Home","Winter Storm","Winter Road","Winter Trees","Winter Sky",
        "Winter Hat","Family Winter","Winter Coat","Winter Wedding","Winter Sports",
        "Winter Flowers","Winter Break","Winter Vacation","Winter Texture","Winter Icon",
        "Winter Logo","Winter Lights","Dog Winter","Winter Village","Winter Jacket Mockup"
    ],

    "Clipart": [
        "Heart Clipart","Flower Clipart","Book Clipart","Star Clipart","Sun Clipart",
        "Basketball Clipart","Dog Clipart","Fall Clipart","Football Clipart","Apple Clipart",
        "Butterfly Clipart","Cat Clipart","Car Clipart","Fire Clipart","Fish Clipart",
        "Money Clipart","School Clipart","Snowflake Clipart","Tree Clipart","Happy Birthday Clipart",
        "Baseball Clipart","Bee Clipart","Dinosaur Clipart","House Clipart","Airplane Clipart",
        "Camera Clipart","Cow Clipart","Crown Clipart","Pizza Clipart","Christmas Clipart",
        "Christmas Tree Clipart","Halloween Clipart","Pumpkin Clipart","Turkey Clipart",
        "Summer Clipart","Arrow Clipart"
    ],

    "Healthcare": [
        "Ambulance","Doctors Office Background","Dermatologist","Mental Health","Nutrition",
        "Pediatrician","Emergency Room","Surgery","Wellness","Nurse",
        "Preventative Medicine","Medical Insurance","Sports Medicine","Telemedicine","Doctors Office",
        "Hospital Room Background","First Aid","Operation","Positive Covid Test Picture","Medical Equipment",
        "Dentist Office","Alternative Medicine","Medical Wallpaper","Hospital Room","Medical Logo",
        "Medical Technology","Healthy Tonsils","Medical Images","Hospital Sign","Mental Wellness"
    ],

    "Family": [
        "Home","Welcome To The Family","Family Vacation","Family Dinner","Pregnancy",
        "Baby","Parenting","Wedding","Marriage","Children",
        "Happy Family","Family Travel","Family Quotes","Engagement","Family Therapy",
        "Siblings","Family Fun","Family Planning","Grandparents","Family Clipart",
        "Family And Friends","Family Drawing","Family Cartoon","Family Outside"
    ],

    "Wallpaper": [
        "Cool Wallpapers","Cute Wallpapers","Aesthetic Wallpaper","Black Wallpaper","Pink Wallpaper",
        "Anime Wallpaper","4K Wallpaper","Desktop Wallpaper","Blue Wallpaper","Background Wallpaper",
        "Purple Wallpaper","Fall Wallpaper","Red Wallpaper","Space Wallpaper","Stitch Wallpaper",
        "Green Wallpaper","Black And White Wallpaper","Flower Wallpaper","Galaxy Wallpaper","Heart Wallpaper",
        "Beach Wallpaper","Beautiful Wallpaper","Butterfly Wallpaper","3D Wallpaper","Nature Wallpaper",
        "Football Wallpaper","Laptop Wallpaper","Sunset Wallpaper","Winter Wallpaper","Wolf Wallpaper",
        "Christmas Wallpaper","Halloween Wallpaper","Computer Wallpaper","Cat Wallpaper","Dog Wallpaper"
    ],

    "Drawings": [
        "Butterfly Drawing","Cool Drawing","Flower Drawing","Rose Drawing","Skull Drawing",
        "Dragon Drawing","Eye Drawing","Heart Drawing","Mushroom Drawing","Tree Drawing",
        "Christmas Drawing","Frog Drawing","Horse Drawing","Bunny Drawing","Car Drawing",
        "Cow Drawing","Fish Drawing","Hand Drawing","Snake Drawing","Sunflower Drawing",
        "Wolf Drawing","Ai Drawing","Anime Drawing","Bee Drawing","Bird Drawing",
        "Elephant Drawing","Fire Drawing","Fox Drawing","Girl Drawing","Lion Drawing",
        "Moon Drawing","Pumpkin Drawing","Shark Drawing","Skeleton Drawing",
        "Soccer Ball Drawing","Football Drawing"
    ],

    "Mockups": [
        "Hoodie Mockup","Wall Art Mockup","Menu Mockup","Tshirt Mockup","Book Mockup",
        "Poster Mockup","Beanie Mockup","Business Card Mockup","Website Mockup","Magazine Mockup",
        "Billboard Mockup","Tote Bag Mockup","Hat Mockup","Laptop Mockup","Logo Mockup",
        "Phone Mockup","Sticker Mockup","Box Mockup","Sweatpants Mockup","Clothing Mockup",
        "Varsity Jacket Mockup","Shorts Mockup","Brochure Mockup","Trucker Hat Mockup",
        "Computer Mockup","App Mockup","Banner Mockup","Flyer Mockup","Postcard Mockup",
        "Product Mockup"
    ],

    "Valentines Day": [
        "Happy Valentines Day","Valentines Day Background","Valentines Day Banner","Valentines Day Border",
        "Valentines Day Vector","Valentines Day Clipart","Valentines Day Dinner",
        "Valentines Day Celebration","Valentines Day Party","Valentines Day Flowers",
        "Valentines Day Card","Valentines Day Heart","Valentines Day Chocolate",
        "Valentines Day Text","Valentines Day Couple","Valentines Day Food",
        "Valentines Day Presents","Valentines Day Gifts","Valentines Day Flyer",
        "Valentines Day Poster","Valentines Day Icon","Valentines Day Cookies",
        "Valentines Day Sale","Valentines Day Date","Dog Valentine","Cat Valentine",
        "Will You Be My Valentine","Valentine Heart","Valentines SVG","Valentines PNG"
    ]
}




















    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Models will be loaded lazily
        self._siglip_model = None
        self._siglip_processor = None
        self._blip_model = None
        self._blip_processor = None
        
        self.logger.info(f" AI Engine initialized on: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    
    




    def load_models(self):
        """
        Load AI models - using same models as main.py for consistency.
        """
        self.logger.info("🔹 Loading SigLIP model for classification...")
        self._siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip-so400m-patch14-384"
        )
        self._siglip_model = AutoModel.from_pretrained(
            "google/siglip-so400m-patch14-384"
        ).to(self.device)
        self._siglip_model.eval()
        
        self.logger.info("🔹 Loading InstructBLIP model for captioning...")
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        
        self._blip_processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl"
        )
        self._blip_model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        self.logger.info("✅ All AI models loaded successfully")










    # def _classify_category(
    #     self, 
    #     images: List[Image.Image]
    # ) -> List[Tuple[str, str, float]]:
    #     """
    #     Classify images into main and sub categories.
        
    #     Returns:
    #         List of (main_category, sub_category, confidence)
    #     """
    #     results = []
        
    #     for img in images:
    #         # Step 1: Classify main category
    #         inputs = self._siglip_processor(
    #             text=self.MAIN_CATEGORIES,
    #             images=img,
    #             return_tensors="pt",
    #             padding=True
    #         ).to(self.device)
            
    #         with torch.no_grad():
    #             outputs = self._siglip_model(**inputs)
    #             probs = torch.sigmoid(outputs.logits_per_image).squeeze()
            
    #         main_idx = probs.argmax().item()
    #         main_category = self.MAIN_CATEGORIES[main_idx]
    #         main_confidence = probs[main_idx].item()
            
    #         # Step 2: Classify sub category
    #         sub_cats = self.SUB_CATEGORIES[main_category]
    #         inputs_sub = self._siglip_processor(
    #             text=sub_cats,
    #             images=img,
    #             return_tensors="pt",
    #             padding=True
    #         ).to(self.device)
            
    #         with torch.no_grad():
    #             outputs_sub = self._siglip_model(**inputs_sub)
    #             probs_sub = torch.sigmoid(outputs_sub.logits_per_image).squeeze()
            
    #         sub_idx = probs_sub.argmax().item()
    #         sub_category = sub_cats[sub_idx]
            
    #         results.append((main_category, sub_category, main_confidence))
        
    #     return results


    










    # LOCATION: In AICategorizationEngine class, replace _classify_category method

    def _classify_category(
    self, 
    images: List[Image.Image]
    ) -> List[Tuple[str, str, float]]:
        """
        Classify images into main and sub categories using main.py approach.
        
        Returns:
            List of (main_category, sub_category, confidence)
        """
        results = []
        
        # Get main category keys
        main_keys = list(self.TAXONOMY.keys())
        
        for img in images:
            # Step 1: Classify main category
            inputs = self._siglip_processor(
                text=main_keys,
                images=img,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._siglip_model(**inputs)
                probs = torch.sigmoid(outputs.logits_per_image).squeeze()
            
            main_idx = probs.argmax().item()
            main_category = main_keys[main_idx]
            main_confidence = probs[main_idx].item()
            
            # Step 2: Classify sub category from the matched main category
            sub_keys = self.TAXONOMY[main_category]
            inputs_sub = self._siglip_processor(
                text=sub_keys,
                images=img,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs_sub = self._siglip_model(**inputs_sub)
                probs_sub = torch.sigmoid(outputs_sub.logits_per_image).squeeze()
            
            sub_idx = probs_sub.argmax().item()
            sub_category = sub_keys[sub_idx]
            
            results.append((main_category, sub_category, main_confidence))
        
        return results









    # def _generate_captions_and_tags(
    #     self, 
    #     images: List[Image.Image]
    # ) -> List[Tuple[str, str]]:
    #     """
    #     Generate captions and extract meaningful tags.
        
    #     Returns:
    #         List of (caption, tags_string)
    #     """
    #     results = []
        
    #     for img in images:
    #         # Generate detailed caption
    #         caption_prompt = "Describe this image in detail, including the main subject, setting, colors, mood, and any notable elements."
            
    #         inputs = self._blip_processor(
    #             images=img,
    #             text=caption_prompt,
    #             return_tensors="pt"
    #         ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
    #         with torch.no_grad():
    #             generated_ids = self._blip_model.generate(
    #                 **inputs,
    #                 max_new_tokens=100,
    #                 min_length=20,
    #                 num_beams=5,
    #                 temperature=0.7
    #             )
            
    #         caption = self._blip_processor.batch_decode(
    #             generated_ids, 
    #             skip_special_tokens=True
    #         )[0].strip()
            
    #         # Generate tags
    #         tags_prompt = "List 5-10 relevant keywords or tags for this image, separated by commas:"
            
    #         inputs_tags = self._blip_processor(
    #             images=img,
    #             text=tags_prompt,
    #             return_tensors="pt"
    #         ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
    #         with torch.no_grad():
    #             tags_ids = self._blip_model.generate(
    #                 **inputs_tags,
    #                 max_new_tokens=50,
    #                 num_beams=3
    #             )
            
    #         tags_raw = self._blip_processor.batch_decode(
    #             tags_ids,
    #             skip_special_tokens=True
    #         )[0].strip()
            
    #         # Clean up tags
    #         tags = self._clean_tags(tags_raw, caption)
            
    #         results.append((caption, tags))
        
    #     return results




    # def _generate_captions_and_tags(
    #     self, 
    #     images: List[Image.Image]
    # ) -> List[Tuple[str, str]]:
    #     """
    #     Generate captions and extract meaningful tags.
        
    #     Returns:
    #         List of (caption, tags_string)
    #     """
    #     results = []
        
    #     for img in images:
    #         # Generate detailed caption - NO TEXT PROMPT
    #         inputs = self._blip_processor(
    #             images=img,
    #             return_tensors="pt"
    #         ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
    #         with torch.no_grad():
    #             generated_ids = self._blip_model.generate(
    #                 **inputs,
    #                 max_new_tokens=100,
    #                 min_length=20,
    #                 num_beams=5,
    #                 # temperature removed - not valid for BLIP-2
    #             )
            
    #         caption = self._blip_processor.batch_decode(
    #             generated_ids, 
    #             skip_special_tokens=True
    #         )[0].strip()
            
    #         # Generate tags from caption (don't generate separately)
    #         # Extract keywords from the generated caption
    #         tags = self._clean_tags("", caption)  # Pass empty string for tags_raw
            
    #         results.append((caption, tags))
        
    #     return results



    
    


    def _generate_captions_and_tags(
        self, 
        images: List[Image.Image]
    ) -> List[Tuple[str, str]]:
        """
        Generate captions and extract meaningful tags using main.py approach.
        
        Returns:
            List of (caption, tags_string)
        """
        results = []
        
        for img in images:
            # Generate detailed caption with specific prompt
            prompt = "Describe the image in detail including objects, style, lighting and mood."
            
            inputs = self._blip_processor(
                images=img,
                text=prompt,
                return_tensors="pt"
            ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
            
            with torch.no_grad():
                generated_ids = self._blip_model.generate(
                    **inputs,
                    max_new_tokens=90,
                    do_sample=False
                )
            
            caption = self._blip_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Extract keywords from caption (main.py approach)
            keywords = ", ".join(sorted(set([
                w.strip(".,").lower()
                for w in caption.split()
                if len(w) > 4
            ])))
            
            results.append((caption, keywords))
        
        return results














    # LOCATION: In AICategorizationEngine class, add this NEW method after _generate_captions_and_tags

    def _clean_tags_enhanced(self, tag_words: set, caption: str) -> str:
        """
        Enhanced tag cleaning with better filtering and keyword extraction.
        
        Args:
            tag_words: Set of words from AI-generated tag responses
            caption: Generated caption for additional keyword extraction
            
        Returns:
            Comma-separated string of cleaned tags
        """
        # Extended stop words list
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
            'those', 'image', 'picture', 'photo', 'photograph', 'shows',
            'showing', 'depicts', 'depicting', 'features', 'featuring',
            'visible', 'question', 'answer', 'description', 'detail', 'detailed'
        }
        
        # Combine AI tags with caption keywords
        all_tags = set(tag_words)
        
        # Extract meaningful words from caption
        for word in caption.split():
            word = word.strip('.,!?;:()[]"\'').lower()
            # Only add nouns/adjectives (length > 3, alphabetic, not stop word)
            if len(word) > 3 and word.isalpha() and word not in stop_words:
                all_tags.add(word)
        
        # Sort alphabetically and limit to top 12 most relevant tags
        tags_list = sorted(list(all_tags))[:12]
        
        return ', '.join(tags_list)




    def _clean_tags(self, tags_raw: str, caption: str) -> str:
        """
        Clean and deduplicate tags.
        Combines AI-generated tags with key terms from caption.
        """
        # Extract words from tags response
        tag_words = set()
        
        for tag in tags_raw.replace(',', ' ').split():
            tag = tag.strip().lower()
            if len(tag) > 2 and tag.isalpha():
                tag_words.add(tag)
        
        # Extract key nouns/adjectives from caption (simple heuristic)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
            'those', 'image', 'picture', 'photo', 'photograph', 'shows',
            'showing', 'depicts', 'depicting', 'features', 'featuring'
        }
        
        for word in caption.split():
            word = word.strip('.,!?;:()[]"\'').lower()
            if len(word) > 3 and word.isalpha() and word not in stop_words:
                tag_words.add(word)
        
        # Limit to top 10 tags
        tags_list = sorted(list(tag_words))[:10]
        return ', '.join(tags_list)
    
    def process_batch(
        self,
        compressed_images: List[Tuple[bytes, Dict[str, Any]]]
    ) -> List[ImageMetadata]:
        """
        Process batch of compressed images through AI pipeline.
        
        Args:
            compressed_images: List of (compressed_bytes, metadata_dict)
            
        Returns:
            List of complete ImageMetadata objects
        """
        if not compressed_images:
            return []
        
        # Convert bytes to PIL Images
        pil_images = []
        for img_bytes, _ in compressed_images:
            try:
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
            except Exception as e:
                self.logger.warning(f" Failed to open image: {e}")
                pil_images.append(None)
        
        # Filter out failed images
        valid_indices = [i for i, img in enumerate(pil_images) if img is not None]
        valid_images = [pil_images[i] for i in valid_indices]
        
        if not valid_images:
            return []
        
        # Classify categories
        categories = self._classify_category(valid_images)
        
        # Generate captions and tags
        captions_tags = self._generate_captions_and_tags(valid_images)
        
        # Build complete metadata
        results = []
        for idx, valid_idx in enumerate(valid_indices):
            _, meta_dict = compressed_images[valid_idx]
            
            main_cat, sub_cat, confidence = categories[idx]
            caption, tags = captions_tags[idx]
            
            metadata = ImageMetadata(
                image_id=meta_dict['image_id'],
                original_filename=meta_dict['original_filename'],
                s3_original_path=meta_dict['s3_original_path'],
                s3_compressed_path=meta_dict.get('s3_compressed_path', ''),
                original_size_bytes=meta_dict['original_size_bytes'],
                compressed_size_bytes=meta_dict['compressed_size_bytes'],
                width=meta_dict['width'],
                height=meta_dict['height'],
                format=meta_dict['format'],
                main_category=main_cat,
                sub_category=sub_cat,
                caption=caption,
                tags=tags,
                category_confidence=confidence,
                status='completed',
                processed_at=datetime.utcnow()
            )
            results.append(metadata)
        
        return results
    
    def unload_models(self):
        """Free GPU memory."""
        del self._siglip_model
        del self._siglip_processor
        del self._blip_model
        del self._blip_processor
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info(" AI models unloaded from memory")


# ==========================================
#  CHECKPOINT MANAGER
# ==========================================

class CheckpointManager:
    """
    Manages processing checkpoints for crash recovery.
    """
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.checkpoint_file = config.CHECKPOINT_FILE
    
    def save(self, state: Dict[str, Any]):
        """Save checkpoint state to file."""
        state['timestamp'] = datetime.utcnow().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        self.logger.debug(f" Checkpoint saved: {state.get('processed_count', 0)} images")
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state if exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    state = json.load(f)
                self.logger.info(f" Checkpoint loaded: {state.get('processed_count', 0)} images processed previously")
                return state
            except Exception as e:
                self.logger.warning(f" Failed to load checkpoint: {e}")
        return None
    
    def clear(self):
        """Remove checkpoint file after successful completion."""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            self.logger.info("Checkpoint file cleared")


# ==========================================
#  MAIN PIPELINE ORCHESTRATOR
# ==========================================

class ImageCategorizationPipeline:
    """
    Main pipeline orchestrating the entire flow:
    1. Fetch images from AWS S3
    2. Compress images
    3. Categorize with AI
    4. Upload compressed to public bucket
    5. Store metadata in SQL Server
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        
        # Initialize components
        self.s3_manager = S3Manager(config, self.logger)
        self.compressor = ImageCompressor(config, self.logger)
        self.ai_engine = AICategorizationEngine(config, self.logger)
        self.db_manager = SQLServerManager(config, self.logger)
        self.checkpoint = CheckpointManager(config, self.logger)
        
        # Processing state
        self.processed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
    
    async def run(self):
        """
        Execute the complete pipeline.
        """
        self.logger.info("=" * 60)
        self.logger.info(" STARTING IMAGE CATEGORIZATION PIPELINE")
        self.logger.info("=" * 60)
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Initialize database
            self.logger.info("\n Step 1: Initializing database...")
            self.db_manager.initialize_database()
            
            # Get already processed IDs for deduplication
            processed_ids = self.db_manager.get_processed_ids()
            self.logger.info(f"   Found {len(processed_ids)} already processed images")
            
            # Step 2: Load AI models
            self.logger.info("\ Step 2: Loading AI models...")
            self.ai_engine.load_models()
            
            # Step 3: List images from S3
            self.logger.info("\n Step 3: Listing images from S3...")
            images_to_process = await self.s3_manager.list_images(
                max_images=self.config.MAX_IMAGES_TO_PROCESS
            )
            
            if not images_to_process:
                self.logger.warning(" No images found to process!")
                return
            
            self.logger.info(f"   Found {len(images_to_process)} images to process")
            
            # Step 4: Process in batches
            self.logger.info("\n Step 4: Processing images in batches...")
            
            # Semaphores for rate limiting
            download_semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_DOWNLOADS)
            upload_semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_UPLOADS)
            
            # Process in batches
            batch_size = self.config.BATCH_SIZE_FETCH
            total_batches = (len(images_to_process) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(images_to_process))
                batch = images_to_process[batch_start:batch_end]
                
                self.logger.info(f"\n Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} images)")
                
                # 4a: Download batch
                keys = [img['key'] for img in batch]
                downloaded = await self.s3_manager.download_batch(keys, download_semaphore)
                self.logger.info(f"   ✓ Downloaded: {len(downloaded)}/{len(batch)}")
                
                # 4b: Filter already processed
                filtered = []
                for img_bytes, key in downloaded:
                    img_hash = hashlib.sha256(img_bytes).hexdigest()
                    if img_hash not in processed_ids:
                        filtered.append((img_bytes, key))
                    else:
                        self.skipped_count += 1
                
                if not filtered:
                    self.logger.info("    All images in batch already processed, skipping...")
                    continue
                
                self.logger.info(f"   ✓ After dedup: {len(filtered)} new images")
                
                # 4c: Compress batch
                compressed = await self.compressor.compress_batch(filtered)
                self.logger.info(f"   ✓ Compressed: {len(compressed)}/{len(filtered)}")
                
                # 4d: Upload compressed images
                upload_data = [(comp_bytes, meta['original_filename']) 
                               for comp_bytes, meta in compressed]
                # Actually we need original key not filename for path construction
                upload_data = []
                for i, (comp_bytes, meta) in enumerate(compressed):
                    original_key = filtered[i][1]  # Get original S3 key
                    upload_data.append((comp_bytes, original_key))
                
                uploaded = await self.s3_manager.upload_batch(upload_data, upload_semaphore)
                self.logger.info(f"   ✓ Uploaded: {len(uploaded)}/{len(compressed)}")
                
                # Update metadata with compressed paths
                path_map = {key: path for key, path in uploaded}
                for comp_bytes, meta in compressed:
                    original_key = f"s3://{self.config.S3_PRIVATE_BUCKET}/" + \
                                   meta['s3_original_path'].split('/')[-1]
                    # Find matching key
                    for key, path in path_map.items():
                        if meta['original_filename'] in key:
                            meta['s3_compressed_path'] = path
                            break
                
                # 4e: AI categorization (GPU - sequential for thread safety)
                # Process in smaller GPU batches
                gpu_batch_size = self.config.BATCH_SIZE_GPU
                all_metadata = []
                
                for gpu_start in range(0, len(compressed), gpu_batch_size):
                    gpu_end = min(gpu_start + gpu_batch_size, len(compressed))
                    gpu_batch = compressed[gpu_start:gpu_end]
                    
                    batch_metadata = self.ai_engine.process_batch(gpu_batch)
                    all_metadata.extend(batch_metadata)
                
                self.logger.info(f"   ✓ Categorized: {len(all_metadata)}/{len(compressed)}")
                
                # 4f: Update compressed paths from upload results
                for meta in all_metadata:
                    for key, path in uploaded:
                        if meta.original_filename in key:
                            meta.s3_compressed_path = path
                            break
                
                # 4g: Save to database
                inserted = self.db_manager.insert_batch(all_metadata)
                self.processed_count += inserted
                self.failed_count += len(all_metadata) - inserted
                
                self.logger.info(f"   ✓ Saved to DB: {inserted}/{len(all_metadata)}")
                
                # 4h: Save checkpoint
                if self.processed_count % self.config.CHECKPOINT_INTERVAL == 0:
                    self.checkpoint.save({
                        'processed_count': self.processed_count,
                        'last_batch': batch_idx + 1,
                        'total_batches': total_batches
                    })
                
                # Add processed IDs to set for future batches
                for meta in all_metadata:
                    processed_ids.add(meta.image_id)
            
            # Step 5: Cleanup
            self.logger.info("\n Step 5: Cleanup...")
            self.ai_engine.unload_models()
            self.compressor.shutdown()
            self.db_manager.close_all()
            self.checkpoint.clear()
            
            # Final report
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info(" PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"   Total processed: {self.processed_count}")
            self.logger.info(f"   Skipped (dups):  {self.skipped_count}")
            self.logger.info(f"   Failed:          {self.failed_count}")
            self.logger.info(f"   Duration:        {duration:.1f} seconds")
            self.logger.info(f"   Speed:           {self.processed_count / max(duration, 1):.1f} images/sec")
            
        except Exception as e:
            self.logger.error(f" Pipeline failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Save checkpoint for recovery
            self.checkpoint.save({
                'processed_count': self.processed_count,
                'error': str(e),
                'status': 'failed'
            })
            raise


# ==========================================
#  ENTRY POINT
# ==========================================

async def main():
    """
    Main entry point.
    Configure settings and run pipeline.
    """
    # Create configuration
    config = Config(
        # AWS Settings - UPDATE THESE
        AWS_REGION="us-east-1",
        S3_PRIVATE_BUCKET="imageshopprivate",
        S3_PUBLIC_BUCKET="imageshoppublic",
        
        # SQL Server Settings - UPDATE THESE
        SQL_SERVER="sql1003.site4now.net",
        SQL_DATABASE="db_ab683f_aiimages",
        
        # Processing Settings
        MAX_IMAGES_TO_PROCESS=None,  # Set to None for all images
        BATCH_SIZE_FETCH=50,        # Images per S3 batch
        BATCH_SIZE_GPU=8,           # Images per GPU batch
        TARGET_SIZE_KB=200,         # Compressed image size
    )
    
    # Run pipeline
    pipeline = ImageCategorizationPipeline(config)
    await pipeline.run()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       OPTIMIZED IMAGE CATEGORIZATION PIPELINE                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Features:                                                   ║
    ║  • AWS S3 Integration (Private → Public bucket)              ║
    ║  • AI-powered categorization (SigLIP + BLIP-2)               ║
    ║  • Intelligent compression (100-250KB)                       ║
    ║  • Microsoft SQL Server storage                              ║
    ║  • Checkpoint recovery for crash safety                      ║
    ║  • Scalable async/batch processing                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
