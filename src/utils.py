# Standard Libraries
import os
import json
import requests
import math
import zipfile
from pathlib import Path
import pickle

# Geospatial and Image Processing Libraries
import rasterio
from rasterio.windows import from_bounds
import cv2 
import numpy as np

# Scikit-Image and SciPy
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows
from skimage.filters.rank import entropy, minimum
from skimage.morphology import disk
from scipy.ndimage import median_filter, minimum_filter

# Web Services
from owslib.wfs import WebFeatureService
from owslib.wcs import WebCoverageService


def print_defra_lidar_indices ():
    wfs_url = "https://environment.data.gov.uk/spatialdata/survey-index-files/wfs"
    wfs = WebFeatureService(url=wfs_url, version='2.0.0')

    for i, index in enumerate(list(wfs.contents)):
        print(f"{i+1}.\t {index}")

def view_lidar_composite_dates(bbox: tuple, type: str):
    # type is either DSM or DTM

    # This is the stable endpoint for metadata catalogs
    metadata_wfs_url = "https://environment.data.gov.uk/spatialdata/survey-index-files/wfs"
    wfs = WebFeatureService(url=metadata_wfs_url, version='2.0.0')

    if type == 'DSM':
        ID = 'dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:LIDAR_Composite_1m_First_Return_DSM_2022_extents'
    elif type == 'DTM':
        ID = 'dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:LIDAR_Composite_1m_DTM_2022_extents'
    else:
        raise ValueError("Error: The 'type' variable can only be DSM or DTM.")

    # Request the metadata
    response = wfs.getfeature(typename=ID, bbox=bbox, outputFormat='json')
    data = json.loads(response.read())


    for i, feature in enumerate(data['features']):
        props = feature['properties']
        print(f"{i+1}.\tFilename: {props.get('filename')} | Start date: {props.get('sd_flown')} | "
              f"End date: {props.get('ed_flown')} | Resolution: {props.get('resolution')}m")


def view_national_lidar_programme_dates(bbox: tuple):
    # This is the stable endpoint for metadata catalogs
    metadata_wfs_url = "https://environment.data.gov.uk/spatialdata/survey-index-files/wfs"
    wfs = WebFeatureService(url=metadata_wfs_url, version='2.0.0')

    ID = 'dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:National_LIDAR_Programme_Index_Catalogue'

    # Request the metadata
    response = wfs.getfeature(typename=ID, bbox=bbox, outputFormat='json')
    data = json.loads(response.read())

    for i, feature in enumerate(data['features']):
        props = feature['properties']
        print(f"{i+1}.\tTile: {props.get('tilename')} | Date: {props.get('surveys')} | Resolution: {props.get('resolution')}m")

def view_vertical_photography_dates(bbox: tuple):
    # This is the stable endpoint for metadata catalogs
    metadata_wfs_url = "https://environment.data.gov.uk/spatialdata/survey-index-files/wfs"
    wfs = WebFeatureService(url=metadata_wfs_url, version='2.0.0')

    ID = 'dataset-9f0fa3fc-a860-4729-adc9-47fe53f658d0:Vertical_photography_index_catalogue'

    # Request the metadata
    response = wfs.getfeature(typename=ID, bbox=bbox, outputFormat='json')
    data = json.loads(response.read())

    for i, feature in enumerate(data['features']):
        props = feature['properties']
        print(f"{i+1}.\tImage type: {props.get('type')} | Date: {props.get('year')} | Res: {props.get('resolution')}m")



def rasterize_feature_stack(feature_stack_path, metadata_path, output_path, buffer_px):
    """
    Loads a .npy feature stack and its .pkl metadata to save as a multi-band TIF.

    Inputs:
        feature_stack_path: Path to the .npy file (e.g., fstack.npy)
        metadata_path: Path to the .pkl file containing original_metadata
        output_path: Final destination for the .tif file
    """
    # 1. Load the data from the provided paths
    feature_stack = np.load(feature_stack_path)

    with open(metadata_path, 'rb') as f:
        original_metadata = pickle.load(f)

    # 2. Prepare the new metadata
    meta = original_metadata.copy()

    # Update transform to account for the 30px crop (shifting the origin)
    # This is critical for spatial alignment in GIS software
    old_transform = meta['transform']
    new_transform = old_transform #* old_transform.translation(buffer_px, buffer_px)

    meta.update({
        "count": feature_stack.shape[0],  # total bands (features + labels)
        "height": feature_stack.shape[1],  # target height (e.g., 210)
        "width": feature_stack.shape[2],  # target width (e.g., 210)
        "dtype": 'float32',
        "transform": new_transform
    })

    # 3. Write the multi-band TIF
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(feature_stack.astype(np.float32))

    print(f"Rasterized stack ({feature_stack.shape[0]} bands) saved: {output_path}")