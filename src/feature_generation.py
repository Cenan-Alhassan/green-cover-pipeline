# The file contains the functions used in 2.2 Feature Engineering of 01_data_acquisition_and_preparation.ipynb

# Constants
DEFAULT_BUFFER_PX = 30
GLCM_LEVELS = 32

# Standard Libraries
import os
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Any, Dict, Union

# Scientific Computing and Image Processing
import numpy as np
import cv2

# Geospatial Libraries
import rasterio
from rasterio.windows import from_bounds

# Scikit-Image and SciPy
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows
from skimage.filters.rank import entropy, minimum
from skimage.morphology import disk
from scipy.ndimage import median_filter, minimum_filter

# Setup logging
logger = logging.getLogger(__name__)

# =====================================================================
# 1. RASTER CLIPPING UTILITIES
# =====================================================================

# The function must take a tile and a list of raster paths
# loops through all core features. Returns a list of metadata and rasters

# TO DO: create checks for file existence
def clip_rasters_by_tiles(core_features: List[str], tiles: List[Path], outputs: List[Path], buffer_px: int = DEFAULT_BUFFER_PX) -> None:
    """
    Clips multiple core feature rasters to the extent of a labelled tile + a buffer.
    Returns lists containing the clipped rasters and their metadata. #WRONG

    Inputs:
        core_features: List of paths to the original large rasters.
        tiles: List of paths to the small, labelled tiles for spatial reference.
        outputs: List of folder paths to save the clipped results.
        buffer_px: Expansion size in pixels (default 30).
    """
    try:
        # 1. Perform the clipping and saving functionalities tile by tile
        for tile_path, output_path in zip(tiles, outputs):
            print(f"Clipping core features to {tile_path.stem}.")
            # 2. Get the spatial anchor from your manual labels
            with rasterio.open(tile_path) as tile:
                tile_bounds = tile.bounds
                res = tile.res[0]  # Standard 1m resolution

            # 3. Define the buffered area (30m expansion for Feature #26's 27px disk)
            buffered_bounds = (
                tile_bounds.left - (buffer_px * res),
                tile_bounds.bottom - (buffer_px * res),
                tile_bounds.right + (buffer_px * res),
                tile_bounds.top + (buffer_px * res)
            )

            clipped_rasters_list = []
            metadata_list = []

            # 4. Process each core raster individually
            for path in core_features:
                with rasterio.open(path) as src:
                    # Align the window to the core feature's grid
                    window = from_bounds(*buffered_bounds, transform=src.transform)

                    # Read data (boundless=True handles tiles near Westminster edge)
                    data = src.read(window=window, boundless=True, fill_value=0)

                    # Update metadata for the new small file
                    metadata = src.meta.copy()
                    metadata.update({
                        "driver": "GTiff",
                        "height": window.height,
                        "width": window.width,
                        "transform": src.window_transform(window)
                    })

                    clipped_rasters_list.append(data)
                    metadata_list.append(metadata)

            # 5. Save the clipped rasters to the relevant folder
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for path, raster, metadata in zip(core_features, clipped_rasters_list, metadata_list):
                file_name = os.path.basename(path)
                output = os.path.join(output_path, f"{file_name}")

                with rasterio.open(output, "w", **metadata) as dest:
                    dest.write(raster)

            print(f"Core features saved to {os.path.dirname(output)}.\n")
    except Exception as e:
        logger.error(f"Error clipping rasters: {e}")
        raise

# =====================================================================
# 2. TEXTURE CALCULATION HELPERS
# =====================================================================

def calculate_glcm_map(image: np.ndarray, prop: str, window_size: int, dist: int) -> np.ndarray:
    """
    Generates a texture map by applying GLCM to a sliding window.

    Inputs:
        image: The input 2D image array.
        prop: The GLCM property to extract (e.g., 'dissimilarity').
        window_size: Size of the square sliding window.
        dist: Distance between pixel pairs for GLCM.
    """
    try:
        # 1. Quantize the image to fewer levels to speed up calculation (0-31)
        img_min, img_max = image.min(), image.max()
        img_quantized = (((image - img_min) / (img_max - img_min + 1e-6)) * (GLCM_LEVELS - 1)).astype(np.uint8)

        # 2. Create sliding windows (patches)
        # The image here is still the 'buffered' version (e.g., 270x270)
        patches = view_as_windows(img_quantized, (window_size, window_size))

        # 3. Initialize the output map for the area within the buffer
        h, w = patches.shape[0], patches.shape[1]
        texture_map = np.zeros((h, w), dtype=np.float32)

        # 4. Calculate GLCM for each patch
        # This is the 'heavy' loop. We use angles=[0] as per report specs.
        for i in range(h):
            for j in range(w):
                patch = patches[i, j]
                glcm = graycomatrix(patch, distances=[dist], angles=[0], levels=GLCM_LEVELS, symmetric=True, normed=True)
                texture_map[i, j] = graycoprops(glcm, prop)[0, 0]

        # 5. Calculate padding to match the original input shape (e.g., 270x270)
        # This handles the 'off-by-one' error for even window sizes
        pad_total_h = image.shape[0] - texture_map.shape[0]
        pad_total_w = image.shape[1] - texture_map.shape[1]

        pad_h_before = pad_total_h // 2
        pad_h_after = pad_total_h - pad_h_before

        pad_w_before = pad_total_w // 2
        pad_w_after = pad_total_w - pad_w_before

        return np.pad(texture_map, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), mode='edge')
    except Exception as e:
        logger.error(f"Error calculating GLCM map: {e}")
        return np.zeros_like(image, dtype=np.float32)

# =====================================================================
# 3. FEATURE STACK GENERATION
# =====================================================================

#def generate_features(clipped_buffered_rasters, tile_path, buffer_px=30):
def generate_features(tiled_core_features: str, core_features: List[str], tiles: List[Path], outputs: List[Path], buffer_px: int = DEFAULT_BUFFER_PX) -> None:
    """
    Calculates features from Appendix 2 (excludes features 4 & 5, includes feature 44)
    Inputs:
        tiled_core_features: Directory containing the clipped core feature folders.
        core_features: List of core feature names.
        tiles: List of paths to the labelled tiles.
        outputs: List of paths for the final .npy stacks.
        buffer_px: The buffer size used during clipping (default 30).
    Returns:
        feature_stack: A 3D NumPy array (42, 210, 210)
    """
    try:
        for tile_path, output_path in zip(tiles, outputs):
            print(f"Generating feature stack for {tile_path.stem}.")
            # 1. Unpack rasters

            # for each tile, raw_tiled_rasters is a list of paths, each: feature stacks + current tile + loop through features
            raw_tiled_rasters = [Path(tiled_core_features, tile_path.stem, feature) for feature in core_features]

            tiled_rasters = []
            for raw_raster_path in raw_tiled_rasters:
                with rasterio.open(raw_raster_path) as src:
                    tiled_rasters.append(src.read(1))

            dtm, dsm, blue, green, red, nir = tiled_rasters

            # 2. GENERATE FEATURES

            # (Features 1-6) Raw Bands (Blue, Green, Red, NIR). Features 4 and 5 are not included.
            f1, f2, f3, f6 = blue, green, red, nir

            # (Features 7, 8, 9) Convert RGB to HSV for Hue, Saturation, Value
            # We must stack BGR for OpenCV
            bgr_stack = np.stack([blue, green, red], axis=-1).astype(np.float32)
            hsv = cv2.cvtColor(bgr_stack, cv2.COLOR_BGR2HSV)
            f7, f8, f9 = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # (Features 10, 11, 12) CIR Hue (HSV using Blue, Green, NIR)
            # The report uses NIR in place of Red for a CIR-based HSV
            cir_bgr_stack = np.stack([blue, green, nir], axis=-1).astype(np.float32)
            cir_hsv = cv2.cvtColor(cir_bgr_stack, cv2.COLOR_BGR2HSV)
            f10, f11, f12 = cir_hsv[:, :, 0], cir_hsv[:, :, 1], cir_hsv[:, :, 2]

            # (Feature 13) Flipped Hue to handle circular color values
            f13 = f7.copy()
            f13[f7 < 90] += 90
            f13[f7 >= 90] -= 90

            # (Feature 14) Greyscale conversion of the RGB stack
            f14 = cv2.cvtColor(bgr_stack, cv2.COLOR_BGR2GRAY)

            # (Feature 15) NDVI: (NIR - Red) / (NIR + Red)
            # 1e-6 is added to the denominator to prevent division by zero
            f15 = (nir - red) / (nir + red + 1e-6)

            # (Feature 16) NDVI numerator by value: (NIR - Red) / HSV_Value
            f16 = (nir - red) / (f9 + 1e-6)

            # (Features 17, 18, 19) Smooth NDVI with medianBlur and kernel size 3, 5, and 7, respectively
            # Note: medianBlur requires the input to be in uint8

            # Convert NDVI to 0-255 range
            f15_8u = ((f15 + 1.0) * 127.5).astype(np.uint8)

            # Perform median blur on the 8-bit image
            f17_8u = cv2.medianBlur(f15_8u, 5)
            f18_8u = cv2.medianBlur(f15_8u, 3)
            f19_8u = cv2.medianBlur(f15_8u, 7)

            # Convert back to -1.0 to 1.0 range
            f17 = (f17_8u.astype(np.float32) / 127.5) - 1.0
            f18 = (f18_8u.astype(np.float32) / 127.5) - 1.0
            f19 = (f19_8u.astype(np.float32) / 127.5) - 1.0

            # (Feature 20) DSM Mindiff cutoff: smallest difference between a pixel and its 4 neighbors, capped at 2m
            # We use a 3x3 kernel to find the minimum difference to neighbors
            diffs = []
            for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                shifted = np.roll(dsm, shift, axis=(0, 1))
                diffs.append(np.abs(dsm - shifted))
            f20 = np.min(np.stack(diffs), axis=0)
            f20[f20 > 2.0] = 2.0

            # (Feature 21) DSM Maxdiff: largest height difference between a pixel and its 4 neighbors
            diffs_max = []
            for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                shifted = np.roll(dsm, shift, axis=(0, 1))
                diffs_max.append(np.abs(dsm - shifted))
            f21 = np.max(np.stack(diffs_max), axis=0)

            # (Feature 22) Height above ground: difference between DSM and DTM
            f22 = dsm - dtm

            # (Feature 23) Function of max height change in disk(2) footprint of the height above ground
            # Uses a circular neighborhood to find local height variability
            f23 = cv2.dilate(f22.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) - \
                cv2.erode(f22.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            # (Feature 24) Entropy of NDVI using disk(5) footprint
            # Note: entropy requires skimage.filters.rank, which typically uses uint8
            f24 = entropy(f15_8u, disk(5))

            # (Feature 25) Entropy of NDVI using disk(20) footprint
            f25 = entropy(f15_8u, disk(20))

            # (Feature 26) Entropy of NDVI using disk(27) footprint
            # This is the feature that defined our 30px buffer
            f26 = entropy(f15_8u, disk(27))

            # (Feature 27) Entropy of Hue using disk(10) footprint
            # Hue is already 0-179, suitable for entropy
            f27 = entropy(f7.astype(np.uint8), disk(10))

            # (Feature 28) NDVI neighbourhood minimum in disk(5)
            f28 = minimum(f15_8u, disk(5))

            # (Feature 29) NDVI neighbourhood minimum in disk(10)
            f29 = minimum(f15_8u, disk(10))

            # (Feature 30) NDVI neighbourhood minimum in disk(20)
            f30 = minimum(f15_8u, disk(20))

            # (Feature 31) Smooth NDVI (f17) neighbourhood minimum with disk(10)
            f31 = minimum(f17.astype(np.uint8), disk(10))

            # (Feature 32) Saturation (f8) neighbourhood median with disk(5) (using SciPy)
            f32 = median_filter(f8, size=5)

            # (Feature 33) Hue (f7) neighbourhood median with disk(5)
            f33 = median_filter(f7, size=5)

            # (Feature 34) GLCM Dissimilarity for NIR, 25x25 patch, dist 2
            f34 = calculate_glcm_map(f6, 'dissimilarity', 25, 2)

            # (Feature 35) GLCM Dissimilarity for NIR, 10x10 patch, dist 1
            f35 = calculate_glcm_map(f6, 'dissimilarity', 10, 1)

            # (Feature 36) GLCM Correlation for NIR, 25x25 patch, dist 2
            f36 = calculate_glcm_map(f6, 'correlation', 25, 2)

            # (Feature 37) GLCM Correlation for NDVI, 25x25 patch, dist 2
            f37 = calculate_glcm_map(f15, 'correlation', 25, 2)

            # (Feature 38) GLCM Dissimilarity for NDVI, 25x25 patch, dist 2
            f38 = calculate_glcm_map(f15, 'dissimilarity', 25, 2)

            # (Feature 39) Min of GLCM Dissimilarity for NDVI, 10x10 patch, dist 2
            # We apply a minimum filter to the texture map generated above
            f39 = minimum_filter(f38, size=10)

            # (Feature 40) Median of GLCM Dissimilarity for NDVI, 10x10 patch, dist 2
            f40 = median_filter(f38, size=10)

            # (Features 41, 42, 43) Blurred Blue, Green, and Red using 9x9 patch [cite: 607-609]
            f41 = cv2.blur(blue, (9, 9))
            f42 = cv2.blur(green, (9, 9))
            f43 = cv2.blur(red, (9, 9))

            # (Feature 44) Binary flag for pixels where height above ground (f22) is > 4m
            # 1 for True, 0 for False
            f44 = (f22 > 4.0).astype(np.float32)

            # 3. CROP BACK TO ORIGINAL AREA
            # We now crop all features from the buffered size (e.g., 270x270)
            # back to the label size (210x210) by removing the 30px buffer on all sides.

            features = [f1, f2, f3, f6, f7, f8, f9, f10,
                        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
                        f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,
                        f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,
                        f41, f42, f43, f44]


            # We first stack all the features together
            feature_stack = np.stack(features)

            # We then slice the array: [features, row_start:row_end, col_start:col_end]
            # which removes the buffer pixels
            feature_stack = feature_stack[:, buffer_px:-buffer_px, buffer_px:-buffer_px]

            # We finally append the labels to the feature stack
            with rasterio.open(tile_path) as src:
                labels = src.read(1)
                tile_meta = src.profile.copy()  # Extract the exact spatial metadata

            # feature_stack shape: (44, 210, 210), label shape: (210, 210), must add z axis to label
            labels = labels[np.newaxis, :, :]
            final_stack = np.append(feature_stack, labels, axis=0)

            # 4. Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 5. Save the feature stack and metadata
            np.save(output_path, final_stack)

            tile_meta.update({
                "count": final_stack.shape[0],
                "dtype": 'float32'
            })

            meta_output_path = output_path.with_suffix('.pkl')
            with open(meta_output_path, 'wb') as f:
                pickle.dump(tile_meta, f)

            print(f"Feature stack saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        raise
