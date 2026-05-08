# Constants
DEFAULT_WFS_VERSION = "2.0.0"
DEFAULT_OUTPUT_FORMAT = "json"
METADATA_WFS_URL = "https://environment.data.gov.uk/spatialdata/survey-index-files/wfs"
DSM_ID = "dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:LIDAR_Composite_1m_First_Return_DSM_2022_extents"
DTM_ID = (
    "dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:LIDAR_Composite_1m_DTM_2022_extents"
)
NATIONAL_LIDAR_ID = "dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:National_LIDAR_Programme_Index_Catalogue"
VERTICAL_PHOTO_ID = (
    "dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:Vertical_photography_index_catalogue"
)

# Standard Libraries
import json
import logging
import os
import pickle

import numpy as np

# Geospatial and Image Processing Libraries
import rasterio

# Web Services
from owslib.wfs import WebFeatureService

# Scikit-Image and SciPy

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================================
# 1. LIDAR DATA SERVICES
# =====================================================================


def print_defra_lidar_indices() -> None:
    """
    Prints the indices of DEFRA LIDAR datasets available via WFS.
    """
    try:
        # 1. Connect to WFS service
        wfs = WebFeatureService(url=METADATA_WFS_URL, version=DEFAULT_WFS_VERSION)

        # 2. Iterate and print contents
        for i, index in enumerate(list(wfs.contents)):
            print(f"{i + 1}.	 {index}")
    except Exception as e:
        logging.error(f"Error fetching DEFRA LIDAR indices: {e}")


def view_lidar_composite_dates(bbox: tuple, type: str) -> None:
    """
    Fetches and prints metadata for LIDAR composite datasets based on bounding box and type.

    Args:
        bbox (tuple): Bounding box coordinates (minx, miny, maxx, maxy).
        type (str): Dataset type, either 'DSM' or 'DTM'.

    Raises:
        ValueError: If the type is not 'DSM' or 'DTM'.
    """
    try:
        # 1. Select the correct dataset ID based on type
        if type == "DSM":
            ID = DSM_ID
        elif type == "DTM":
            ID = DTM_ID
        else:
            raise ValueError("Error: The 'type' variable can only be DSM or DTM.")

        # 2. Connect to WFS service
        wfs = WebFeatureService(url=METADATA_WFS_URL, version=DEFAULT_WFS_VERSION)

        # 3. Get features within bounding box
        response = wfs.getfeature(
            typename=ID, bbox=bbox, outputFormat=DEFAULT_OUTPUT_FORMAT
        )
        data = json.loads(response.read())

        # 4. Print metadata for each feature
        for i, feature in enumerate(data["features"]):
            props = feature["properties"]
            print(
                f"{i + 1}.	Filename: {props.get('filename')} | Start date: {props.get('sd_flown')} | "
                f"End date: {props.get('ed_flown')} | Resolution: {props.get('resolution')}m"
            )
    except Exception as e:
        logging.error(f"Error fetching LIDAR composite dates: {e}")


def view_national_lidar_programme_dates(bbox: tuple) -> None:
    """
    Fetches and prints metadata for the National LIDAR Programme.
    """
    try:
        # 1. Connect to WFS service
        wfs = WebFeatureService(url=METADATA_WFS_URL, version=DEFAULT_WFS_VERSION)

        # 2. Get features within bounding box
        response = wfs.getfeature(
            typename=NATIONAL_LIDAR_ID, bbox=bbox, outputFormat=DEFAULT_OUTPUT_FORMAT
        )
        data = json.loads(response.read())

        # 3. Print metadata for each feature
        for i, feature in enumerate(data["features"]):
            props = feature["properties"]
            print(
                f"{i + 1}.\tTile: {props.get('tilename')} | Date: {props.get('surveys')} | Resolution: {props.get('resolution')}m"
            )
    except Exception as e:
        logging.error(f"Error fetching National LIDAR Programme dates: {e}")


def view_vertical_photography_dates(bbox: tuple) -> None:
    """
    Fetches and prints metadata for vertical photography.
    """
    try:
        # 1. Connect to WFS service
        wfs = WebFeatureService(url=METADATA_WFS_URL, version=DEFAULT_WFS_VERSION)

        # 2. Get features within bounding box
        response = wfs.getfeature(
            typename=VERTICAL_PHOTO_ID, bbox=bbox, outputFormat=DEFAULT_OUTPUT_FORMAT
        )
        data = json.loads(response.read())

        # 3. Print metadata for each feature
        for i, feature in enumerate(data["features"]):
            props = feature["properties"]
            print(
                f"{i + 1}.\tImage type: {props.get('type')} | Date: {props.get('year')} | Res: {props.get('resolution')}m"
            )
    except Exception as e:
        logging.error(f"Error fetching vertical photography dates: {e}")


# =====================================================================
# 2. RASTER PROCESSING UTILITIES
# =====================================================================


def rasterize_feature_stack(
    feature_stack_path: str, metadata_path: str, output_path: str, buffer_px: int
) -> None:
    """
    Loads a .npy feature stack and its .pkl metadata to save as a multi-band TIF.

    Args:
        feature_stack_path (str): Path to the .npy file (e.g., fstack.npy).
        metadata_path (str): Path to the .pkl file containing original_metadata.
        output_path (str): Final destination for the .tif file.
        buffer_px (int): Buffer size in pixels for cropping.
    """
    try:
        # 1. Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 2. Load the data from the provided paths
        feature_stack = np.load(feature_stack_path)

        with open(metadata_path, "rb") as f:
            original_metadata = pickle.load(f)

        # 3. Prepare the new metadata
        meta = original_metadata.copy()

        # Update transform to account for the 30px crop (shifting the origin)
        # This is critical for spatial alignment in GIS software
        old_transform = meta["transform"]
        new_transform = old_transform  # Placeholder for actual transformation logic

        meta.update(
            {
                "count": feature_stack.shape[0],  # total bands (features + labels)
                "height": feature_stack.shape[1],  # target height (e.g., 210)
                "width": feature_stack.shape[2],  # target width (e.g., 210)
                "dtype": "float32",
                "transform": new_transform,
            }
        )

        # 4. Write the multi-band TIF
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(feature_stack.astype(np.float32))

        print(f"Rasterized stack ({feature_stack.shape[0]} bands) saved: {output_path}")
    except Exception as e:
        logging.error(f"Error rasterizing feature stack: {e}")
