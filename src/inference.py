"""
inference.py
Handles the windowed classification of Sentinel-2 and LiDAR data across Westminster.
Includes multi-date probability merging and spatial activation for faint vegetation.
"""

# Constants
DEFAULT_SIGMA = 1.0
DEFAULT_THRESHOLD = 0.15
DEFAULT_STEEPNESS = 5.0
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_BUFFER = 30
GLCM_LEVELS = 32

# Standard Libraries
import datetime
import os
import logging
from typing import Dict, List, Optional, Any, Tuple

# Scientific Computing and Image Processing
import numpy as np
import cv2
import joblib

# Geospatial Libraries
import rasterio
from rasterio.windows import Window

# Image Processing (Scikit-Image & SciPy)
from skimage.filters.rank import entropy, minimum
from skimage.morphology import disk
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows
from scipy.ndimage import median_filter, minimum_filter, gaussian_filter

# Setup logging
logger = logging.getLogger(__name__)

# =====================================================================
# 1. MATHEMATICAL & POST-PROCESSING HELPERS
# =====================================================================

def spatial_activation_math(prob_array: np.ndarray, sigma: float = DEFAULT_SIGMA, threshold: float = DEFAULT_THRESHOLD,
                            steepness: float = DEFAULT_STEEPNESS) -> np.ndarray:
    """
    Enhances faint vegetation signals using spatial consensus.
    If a faint pixel is part of a cohesive "blob", its probability is amplified
    via a Sigmoid Activation Function.

    Inputs:
        prob_array: 2D numpy array of class probabilities (0.0 to 1.0).
        sigma: Spread of the Gaussian filter used to measure spatial consensus.
        threshold: The probability score required to trigger amplification.
        steepness: The multiplier (k) of the sigmoid function.
    """
    try:
        # 1. Spatial Consensus (Measure "strength in numbers")
        smoothed = gaussian_filter(prob_array, sigma=sigma)

        # 2. The Sigmoid Amplifier (f(x) = 1 / (1 + e^-k(x - x0)))
        # We calculate the boosted values for the entire array
        boosted = 1.0 / (1.0 + np.exp(-steepness * (smoothed - threshold)))

        # 3. Apply Threshold Gatekeeper
        # "Any vegetation with a probability below this threshold is not amplified."
        activated_prob = np.where(smoothed >= threshold, boosted, prob_array)

        return activated_prob
    except Exception as e:
        logger.error(f"Error in spatial_activation_math: {e}")
        return prob_array

def calculate_glcm_map_optimized(image: np.ndarray, prop: str, window_size: int, dist: int) -> np.ndarray:
    """
    Optimized GLCM texture calculation using quantization and striding.

    Inputs:
        image: The input image array.
        prop: GLCM property to calculate (e.g., 'dissimilarity').
        window_size: Size of the sliding window.
        dist: Distance for GLCM.
    """
    try:
        # 1. Quantize ONCE for the whole chunk (0-31)
        img_min, img_max = image.min(), image.max()
        img_quantized = (((image - img_min) / (max(img_max - img_min, 1e-6))) * (GLCM_LEVELS - 1)).astype(np.uint8)

        # 2. Extract patches using a sliding window
        patches = view_as_windows(img_quantized, (window_size, window_size))

        # 3. OPTIMIZATION: Sub-sampling (Stride = 2 for 4x speedup)
        stride = 2
        patches_sub = patches[::stride, ::stride]
        h_sub, w_sub = patches_sub.shape[0], patches_sub.shape[1]

        texture_map_sub = np.zeros((h_sub, w_sub), dtype=np.float32)

        # 4. Process the sub-sampled patches
        for i in range(h_sub):
            for j in range(w_sub):
                patch = patches_sub[i, j]
                glcm = graycomatrix(patch, distances=[dist], angles=[0], levels=GLCM_LEVELS, symmetric=True, normed=True)
                texture_map_sub[i, j] = graycoprops(glcm, prop)[0, 0]

        # 5. Resize back to original spatial dimensions
        texture_map = cv2.resize(texture_map_sub, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        return texture_map
    except Exception as e:
        logger.error(f"Error in calculate_glcm_map_optimized: {e}")
        return np.zeros_like(image, dtype=np.float32)

# =====================================================================
# 2. FEATURE GENERATION HELPERS
# =====================================================================

def generate_static_height_features(dsm_path: str, dtm_path: str, read_window: Window) -> Dict[str, np.ndarray]:
    """
    Computes static features based on LiDAR data (DSM/DTM).
    These features do not change across temporal dates.

    Inputs:
        dsm_path: Path to the Digital Surface Model.
        dtm_path: Path to the Digital Terrain Model.
        read_window: Rasterio Window object for reading.
    """
    try:
        # 1. Load the DSM and DTM data for the specific window
        with rasterio.open(dsm_path) as dsm_f, rasterio.open(dtm_path) as dtm_f:
            # Boundless read ensures 0-padding if the window extends outside the boundary
            dsm = dsm_f.read(1, window=read_window, boundless=True).astype(np.float32)
            dtm = dtm_f.read(1, window=read_window, boundless=True).astype(np.float32)

        # 2. Feature 22: Height above ground
        f22 = dsm - dtm

        # 3. Feature 20: DSM Mindiff cutoff
        diffs = []
        for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = np.roll(dsm, shift, axis=(0, 1))
            diffs.append(np.abs(dsm - shifted))
        f20 = np.min(np.stack(diffs), axis=0)
        f20[f20 > 2.0] = 2.0

        # 4. Feature 21: DSM Maxdiff
        diffs_max = []
        for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            shifted = np.roll(dsm, shift, axis=(0, 1))
            diffs_max.append(np.abs(dsm - shifted))
        f21 = np.max(np.stack(diffs_max), axis=0)

        # 5. Feature 23: Local height variability (Morphological Gradient)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        f23 = cv2.dilate(f22, kernel) - cv2.erode(f22, kernel)

        # 6. Feature 44: Binary flag for height > 4m (Canopy Shield)
        f44 = (f22 > 4.0).astype(np.float32)

        return {'f20': f20, 'f21': f21, 'f22': f22, 'f23': f23, 'f44': f44}
    except Exception as e:
        logger.error(f"Error generating static height features: {e}")
        raise

def generate_temporal_spectral_features(date_files: Dict[str, str], read_window: Window) -> Dict[str, np.ndarray]:
    """
    Computes spectral and texture features for a specific satellite observation date.

    Inputs:
        date_files: Dictionary containing paths to spectral bands (blue, green, red, nir).
        read_window: Rasterio Window object for reading.
    """
    try:
        # 1. Load spectral bands
        with rasterio.open(date_files['blue']) as b, \
                rasterio.open(date_files['green']) as g, \
                rasterio.open(date_files['red']) as r, \
                rasterio.open(date_files['nir']) as n:
            blue = b.read(1, window=read_window, boundless=True).astype(np.float32)
            green = g.read(1, window=read_window, boundless=True).astype(np.float32)
            red = r.read(1, window=read_window, boundless=True).astype(np.float32)
            nir = n.read(1, window=read_window, boundless=True).astype(np.float32)

        # 2. Raw Bands (f1, f2, f3, f6)
        f1, f2, f3, f6 = blue, green, red, nir

        # 3. RGB to HSV (f7, f8, f9)
        bgr_stack = np.stack([blue, green, red], axis=-1)
        hsv = cv2.cvtColor(bgr_stack, cv2.COLOR_BGR2HSV)
        f7, f8, f9 = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 4. CIR Hue (B, G, NIR) (f10, f11, f12)
        cir_bgr_stack = np.stack([blue, green, nir], axis=-1)
        cir_hsv = cv2.cvtColor(cir_bgr_stack, cv2.COLOR_BGR2HSV)
        f10, f11, f12 = cir_hsv[:, :, 0], cir_hsv[:, :, 1], cir_hsv[:, :, 2]

        # 5. Flipped Hue (f13)
        f13 = f7.copy()
        f13[f7 < 90] += 90
        f13[f7 >= 90] -= 90

        # 6. Greyscale & NDVI (f14, f15)
        f14 = cv2.cvtColor(bgr_stack, cv2.COLOR_BGR2GRAY)
        f15 = (nir - red) / (nir + red + 1e-6)
        f15_8u = ((f15 + 1.0) * 127.5).astype(np.uint8)

        # 7. NDVI numerator by value (f16)
        f16 = (nir - red) / (f9 + 1e-6)

        # 8. Median Blurs (f17, f18, f19)
        f17 = (cv2.medianBlur(f15_8u, 5).astype(np.float32) / 127.5) - 1.0
        f18 = (cv2.medianBlur(f15_8u, 3).astype(np.float32) / 127.5) - 1.0
        f19 = (cv2.medianBlur(f15_8u, 7).astype(np.float32) / 127.5) - 1.0

        # 9. Entropy and Neighborhood filters (f24-f31)
        f24 = entropy(f15_8u, disk(5))
        f25 = entropy(f15_8u, disk(20))
        f26 = entropy(f15_8u, disk(27))
        f27 = entropy(f7.astype(np.uint8), disk(10))
        f28 = minimum(f15_8u, disk(5))
        f29 = minimum(f15_8u, disk(10))
        f30 = minimum(f15_8u, disk(20))
        f31 = minimum(f17.astype(np.uint8), disk(10))

        # 10. SciPy Median Filters (f32, f33)
        f32 = median_filter(f8, size=5)
        f33 = median_filter(f7, size=5)

        # 11. GLCM Texture Maps (f34-f40)
        f34 = calculate_glcm_map_optimized(f6, 'dissimilarity', 25, 2)
        f35 = calculate_glcm_map_optimized(f6, 'dissimilarity', 10, 1)
        f36 = calculate_glcm_map_optimized(f6, 'correlation', 25, 2)
        f37 = calculate_glcm_map_optimized(f15, 'correlation', 25, 2)
        f38 = calculate_glcm_map_optimized(f15, 'dissimilarity', 25, 2)
        f39 = minimum_filter(f38, size=10)
        f40 = median_filter(f38, size=10)

        # 12. Blurred RGB (f41, f42, f43)
        f41 = cv2.blur(blue, (9, 9))
        f42 = cv2.blur(green, (9, 9))
        f43 = cv2.blur(red, (9, 9))

        return {
            'f1': f1, 'f2': f2, 'f3': f3, 'f6': f6, 'f7': f7, 'f8': f8, 'f9': f9,
            'f10': f10, 'f11': f11, 'f12': f12, 'f13': f13, 'f14': f14, 'f15': f15,
            'f16': f16, 'f17': f17, 'f18': f18, 'f19': f19, 'f24': f24, 'f25': f25,
            'f26': f26, 'f27': f27, 'f28': f28, 'f29': f29, 'f30': f30, 'f31': f31,
            'f32': f32, 'f33': f33, 'f34': f34, 'f35': f35, 'f36': f36, 'f37': f37,
            'f38': f38, 'f39': f39, 'f40': f40, 'f41': f41, 'f42': f42, 'f43': f43
        }
    except Exception as e:
        logger.error(f"Error generating temporal spectral features: {e}")
        raise

# =====================================================================
# 3. THE UNIFIED CLASSIFICATION ENGINE
# =====================================================================

def soft_classify(
        feature_matrices: List[np.ndarray],
        model: Any,
        height: int,
        width: int,
        mean_prob: bool = False,
        spatial_activation_params: Optional[Dict[str, Dict[str, float]]] = None
) -> np.ndarray:
    """
    Performs multi-date classification using either Mean or Max Probability Merging.
    Optionally applies spatial activation to boost faint vegetation signals.

    Inputs:
        feature_matrices: List of feature matrices (one per date).
        model: Trained ML model.
        height: Height of the chunk.
        width: Width of the chunk.
        mean_prob: Whether to return mean probabilities.
        spatial_activation_params: Parameters for spatial activation.
    """
    try:
        # 1. Prediction & Temporal Composite
        all_date_probs = []

        for X in feature_matrices:
            probs = model.predict_proba(X)
            all_date_probs.append(probs.reshape(height, width, -1))

        # 2. THE ROUTING LOGIC (Mean vs Max)
        if mean_prob:
            # Calculate MEAN probability across dates
            temporal_stack = np.stack(all_date_probs, axis=0)
            composite_probs = np.mean(temporal_stack, axis=0)

            # Capture max_prob_composite to preserve NoData accurately
            max_prob_composite = np.maximum.reduce(all_date_probs)
        else:
            # Calculate MAXIMUM probability across dates
            composite_probs = np.maximum.reduce(all_date_probs)
            max_prob_composite = composite_probs

        # 3. Extract base probabilities
        prob_canopy = composite_probs[:, :, 1]
        prob_green = composite_probs[:, :, 2]
        prob_neither = composite_probs[:, :, 3]

        # 4. Optional Spatial Activation (The Sigmoid Amplifier)
        if spatial_activation_params is not None:

            if 'Canopy' in spatial_activation_params:
                params = spatial_activation_params['Canopy']
                prob_canopy = spatial_activation_math(
                    prob_canopy,
                    sigma=params.get('sigma', DEFAULT_SIGMA),
                    threshold=params.get('threshold', DEFAULT_THRESHOLD),
                    steepness=params.get('steepness', DEFAULT_STEEPNESS)
                )

            if 'Green' in spatial_activation_params:
                params = spatial_activation_params['Green']
                prob_green = spatial_activation_math(
                    prob_green,
                    sigma=params.get('sigma', DEFAULT_SIGMA),
                    threshold=params.get('threshold', 0.25),
                    steepness=params.get('steepness', DEFAULT_STEEPNESS)
                )

            # Re-normalize to ensure probabilities sum to exactly 1.0
            enh_sum = prob_canopy + prob_green + prob_neither
            enh_sum[enh_sum == 0] = 1e-6

            prob_canopy = prob_canopy / enh_sum
            prob_green = prob_green / enh_sum
            prob_neither = prob_neither / enh_sum

        # 5. Re-stack the final processed probabilities
        final_prob_stack = np.stack([prob_canopy, prob_green, prob_neither], axis=2)

        # 6. The Output Router
        if mean_prob:
            # Returns the (Bands, Height, Width) format required by Rasterio
            return final_prob_stack.transpose(2, 0, 1).astype(np.float32)

        else:
            # HARD LABELS
            winning_indices = np.argmax(final_prob_stack, axis=2)
            hard_classification = (winning_indices + 1).astype(np.uint8)

            # Explicitly preserve NoData (Class 0) using the max_prob_composite
            orig_indices = np.argmax(max_prob_composite, axis=2)
            orig_class = (orig_indices).astype(np.uint8)
            hard_classification = np.where(orig_class == 0, 0, hard_classification)

            return hard_classification
    except Exception as e:
        logger.error(f"Error in soft_classify: {e}")
        raise

# =====================================================================
# 4. THE UNIVERSAL ORCHESTRATOR
# =====================================================================

def process_area_in_chunks(
        dsm_path: str,
        dtm_path: str,
        dates_paths: List[Dict[str, str]],
        output_path: str,
        model_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        buffer: int = DEFAULT_BUFFER,
        stopping_chunk: Optional[int] = None,
        mean_prob: bool = False,
        spatial_activation: Optional[Dict[str, Dict[str, float]]] = None
) -> None:
    """
    Orchestrates the windowed classification of an entire geographic area.
    Handles dynamic temporal merging, buffer management, and output routing.

    Inputs:
        dsm_path: Path to the Digital Surface Model.
        dtm_path: Path to the Digital Terrain Model.
        dates_paths: List of dictionaries containing paths to spectral bands for each date.
        output_path: Destination path for the final GeoTIFF.
        model_path: Path to the trained .joblib RandomForest model.
        chunk_size: Size of the processing window.
        buffer: Overlap pixels to prevent edge artifacts in texture filters.
        stopping_chunk: If set, stops processing after this many chunks.
        mean_prob: If True, outputs a multi-band probability raster.
        spatial_activation: Dictionary of parameters for the Sigmoid amplifier.
    """
    try:
        # 1. Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Suppress numpy division warnings for empty blocks
        np.seterr(all="ignore")

        print(f"[{datetime.datetime.now()}] Starting Westminster Pipeline...")

        # 3. Load the model ONCE before the loop begins
        print("Loading Random Forest model into memory...")
        model = joblib.load(model_path)

        with rasterio.open(dsm_path) as dsm_src:
            profile = dsm_src.profile
            width = dsm_src.width
            height = dsm_src.height

            # 4. DYNAMIC PROFILE SETUP
            if mean_prob:
                # 3 bands (Canopy, Green, Neither), Float32
                profile.update(count=3, dtype='float32', blockxsize=chunk_size, blockysize=chunk_size, tiled=True, nodata=0)
                print("Mode: Soft Classification (Probability Maps)")
            else:
                # 1 band, Uint8
                profile.update(count=1, dtype='uint8', blockxsize=chunk_size, blockysize=chunk_size, tiled=True)
                print("Mode: Hard Classification")

            print(f"Creating output raster at {output_path}...")

            with rasterio.open(output_path, 'w', **profile) as dst:

                chunk_num = 0

                # --- START CHUNKING LOOP ---
                for y in range(0, height, chunk_size):
                    for x in range(0, width, chunk_size):
                        chunk_num += 1

                        # 5. EARLY STOPPING CHECK
                        if stopping_chunk is not None and chunk_num > stopping_chunk:
                            print(f"[{datetime.datetime.now()}] Stopping chunk limit ({stopping_chunk}) reached. "
                                f"Terminating early.")
                            return

                        # 6. DEFINE WINDOWS
                        read_window = Window(x - buffer, y - buffer, chunk_size + (2 * buffer), chunk_size + (2 * buffer))
                        write_window = Window(x, y, min(chunk_size, width - x), min(chunk_size, height - y))

                        print(f"Processing Chunk {chunk_num} | Window at X:{x}, Y:{y}")

                        # 7. GENERATE FEATURES
                        static_height_dict = generate_static_height_features(dsm_path, dtm_path, read_window)

                        feature_matrices_by_date = []
                        for i, date_files in enumerate(dates_paths):
                            spectral_dict = generate_temporal_spectral_features(date_files, read_window)

                            # Merge dictionaries and sort numerically to guarantee model feature order
                            full_chunk_dict = {**static_height_dict, **spectral_dict}
                            ordered_keys = sorted(full_chunk_dict.keys(), key=lambda k: int(k[1:]))

                            # Stack into a (pixels, 42) matrix
                            X_date = np.column_stack([full_chunk_dict[k].flatten() for k in ordered_keys]).astype(
                                np.float32)
                            feature_matrices_by_date.append(X_date)

                        # 8. UNIFIED CLASSIFICATION ENGINE
                        chunk_output = soft_classify(
                            feature_matrices=feature_matrices_by_date,
                            model=model,
                            height=read_window.height,
                            width=read_window.width,
                            mean_prob=mean_prob,
                            spatial_activation_params=spatial_activation
                        )

                        # 9. CROP TO CLEAN AREA (Remove Buffer)
                        if mean_prob:
                            # Shape: (Bands, Height, Width)
                            clean_output = chunk_output[:, buffer: buffer + write_window.height,
                                        buffer: buffer + write_window.width]
                            dst.write(clean_output, window=write_window)
                        else:
                            # Shape: (Height, Width)
                            clean_output = chunk_output[buffer: buffer + write_window.height,
                                        buffer: buffer + write_window.width]
                            dst.write(clean_output.astype(np.uint8), 1, window=write_window)

                print(f"[{datetime.datetime.now()}] Westminster processing finalized successfully.")
    except Exception as e:
        logger.error(f"Error in process_area_in_chunks: {e}")
        raise
