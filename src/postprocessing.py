# Constants
DEFAULT_KERNEL_SIZE = 3

# Standard Libraries
import os
import logging
from typing import Any, Tuple, Union

# Scientific Computing and Image Processing
import numpy as np
import cv2

# Geospatial Libraries
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

# Setup logging
logger = logging.getLogger(__name__)

# =====================================================================
# 1. MORPHOLOGICAL SMOOTHING
# =====================================================================

def apply_morphological_smoothing(input_path: str, output_path: str, kernel_size: int = DEFAULT_KERNEL_SIZE) -> None:
    """
    Cleans classification noise by removing 'lonely' pixels and filling small holes.
    Uses a hierarchical approach: smooths total vegetation footprint first,
    then overlays smoothed canopy blobs.

    Inputs:
        input_path: Path to the input raster file.
        output_path: Path to save the smoothed raster file.
        kernel_size: Size of the structuring element for morphological operations.
    """
    try:
        # 1. Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Load the classification raster
        with rasterio.open(input_path) as src:
            classification_raster = src.read(1)
            profile = src.profile

        # 3. Define the 'Structuring Element'
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 4. Smooth Total Vegetation Footprint (Classes 1 & 2)
        # This acts as a 'safe zone' to prevent holes in parks.
        veg_mask = np.where((classification_raster == 1) | (classification_raster == 2), 255, 0).astype(np.uint8)
        smoothed_footprint = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel)

        # 5. Smooth Canopy (Class 1) independently
        canopy_mask = np.where(classification_raster == 1, 255, 0).astype(np.uint8)
        smoothed_canopy = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, kernel)

        # 6. RECONSTRUCTION
        # Initialize with 'Neither' (Class 3)
        smoothed_map = np.full_like(classification_raster, 3)

        # Fill in the Green background (Class 2) wherever the vegetation footprint exists
        smoothed_map[smoothed_footprint == 255] = 2

        # Overlay the cohesive Canopy (Class 1) on top
        smoothed_map[smoothed_canopy == 255] = 1

        # Restore NoData (Class 0)
        smoothed_map[classification_raster == 0] = 0

        # 7. Write the smoothed raster
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(smoothed_map.astype(np.uint8), 1)

        print(f"Smooth raster saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error during morphological smoothing: {e}")

# =====================================================================
# 2. VECTORIZATION HELPERS
# =====================================================================

def extract_to_gdf(image: np.ndarray, target_classes: Union[int, Tuple[int, ...]], transform: Any, crs: Any, mask_all: np.ndarray) -> Union[gpd.GeoDataFrame, None]:
    """
    Internal helper to convert pixel clusters into a GeoDataFrame.

    Inputs:
        image: The raster image as a numpy array.
        target_classes: Single class ID or tuple of class IDs to extract.
        transform: Affine transform of the raster.
        crs: Coordinate Reference System of the raster.
        mask_all: Mask indicating valid data areas.
    """
    try:
        # 1. Determine the class mask
        # Use np.isin for tuples (Meld) or direct comparison for single classes
        if isinstance(target_classes, tuple):
            class_mask = np.isin(image, target_classes)
        else:
            class_mask = (image == target_classes)

        # 2. Ensure we don't vectorize NoData areas
        final_mask = class_mask & (mask_all > 0)

        # 3. Extract polygons
        results = (
            {'properties': {'class_id': str(target_classes)}, 'geometry': shape(geom)}
            for geom, val in shapes(image, mask=final_mask, transform=transform)
        )

        features = list(results)
        if not features:
            return None

        # 4. Return as GeoDataFrame
        return gpd.GeoDataFrame.from_features(features, crs=crs)
    except Exception as e:
        logger.error(f"Error during vectorization: {e}")
        return None

# =====================================================================
# 3. VECTORIZATION MAIN FUNCTION
# =====================================================================

def vectorise_raster(input_path: str, output_canopy: str, output_green: str) -> None:
    """
    Converts raster classification into separate vector folders.

    Inputs:
        input_path: Path to the smoothed .tif file.
        output_canopy: Folder path for the Canopy (Class 1) shapefile.
        output_green: Folder path for the Total Green Cover (Classes 1 & 2) shapefile.
    """
    try:
        # 1. Safety: Ensure output directories exist
        for folder in [output_canopy, output_green]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # 2. Load the raster data
        with rasterio.open(input_path) as src:
            image = src.read(1)
            mask_all = src.dataset_mask()
            transform = src.transform
            crs = src.crs

        # 3. Vectorise Canopy Only (Class 1)
        print("Vectorising Canopy Cover...")
        canopy_gdf = extract_to_gdf(image, 1, transform, crs, mask_all)
        if canopy_gdf is not None:
            out_file = os.path.join(output_canopy, "canopy_cover.shp")
            canopy_gdf.to_file(out_file)
            print(f"Successfully saved Canopy to {out_file}")

        # 4. Vectorise Total Green Cover (Meld of Classes 1 and 2)
        # This aligns with the GLA 'Total Vegetation' narrative
        print("Vectorising Total Green Cover (Meld)...")
        green_gdf = extract_to_gdf(image, (1, 2), transform, crs, mask_all)
        if green_gdf is not None:
            out_file = os.path.join(output_green, "green_cover.shp")
            green_gdf.to_file(out_file)
            print(f"Successfully saved Green Cover to {out_file}")
    except Exception as e:
        logger.error(f"Error during raster vectorization: {e}")
