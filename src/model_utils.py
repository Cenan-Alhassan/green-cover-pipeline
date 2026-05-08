import datetime
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

BALANCED_RANDOM_STATE = 42

# Setup logging
logger = logging.getLogger(__name__)

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# =====================================================================
# 1. DATA PREPARATION FUNCTIONS
# =====================================================================


def prepare_spatial_split(
    feature_stack: np.ndarray,
    train_ratio: float = 0.7,
    test_gap_px: int = 5,
    cv_gap_px: int = 3,
    n_folds: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slices a feature stack into training and testing sets with spatial gaps
    to ensure data independence across any tile resolution.

    Inputs:
        feature_stack: NumPy array (Bands, Height, Width)
        train_ratio: Float, percentage of rows for training (default 0.7)
        test_gap_px: Int, rows to discard between Train and Test sets
        cv_gap_px: Int, rows to discard between CV folds
        n_folds: Int, number of cross-validation folds
    """
    try:
        # 1. Get dimensions (Height is the row count used for splitting)
        n_rows = feature_stack.shape[1]

        # 2. Calculate the Train/Test Split
        # Determine the end of the training block based on ratio
        train_end_idx = int(n_rows * train_ratio)
        test_start_idx = train_end_idx + test_gap_px

        # 3. Slice into raw sets [Bands, Rows, Cols]
        # We transpose to [Rows * Cols, Bands] for the ML model
        train_block = feature_stack[:, :train_end_idx, :]
        test_block = feature_stack[:, test_start_idx:, :]

        # Reshape to (Pixels, Features)
        X_train = train_block[:-1, :, :].reshape(feature_stack.shape[0] - 1, -1).T
        y_train = train_block[-1, :, :].ravel()

        X_test = test_block[:-1, :, :].reshape(feature_stack.shape[0] - 1, -1).T
        y_test = test_block[-1, :, :].ravel()

        # 4. Dynamic CV Fold Logic
        # We divide the training rows (minus the gaps) by the number of folds
        total_cv_gap_rows = (n_folds - 1) * cv_gap_px
        available_train_rows = train_end_idx - total_cv_gap_rows
        rows_per_fold = available_train_rows // n_folds

        folds = np.full(y_train.shape, -1)  # -1 indicates discarded "gap" pixels

        current_row = 0
        pixels_per_row = feature_stack.shape[2]

        for fold_id in range(n_folds):
            start_pixel = current_row * pixels_per_row
            end_pixel = (current_row + rows_per_fold) * pixels_per_row

            # Assign pixels to the current fold
            folds[start_pixel:end_pixel] = fold_id

            # Advance row counter by the fold size + the gap
            current_row += rows_per_fold + cv_gap_px

        return X_train, y_train, X_test, y_test, folds
    except Exception as e:
        logger.error(f"Error in prepare_spatial_split: {e}")
        raise


def get_ml_model_input(
    stack_paths: list[Path],
    train_ratio: float = 0.7,
    test_gap_px: int = 5,
    cv_gap_px: int = 3,
    n_folds: int = 3,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Iterates through processed feature stack paths, splits each tile,
    and concatenates them into a master training and testing dataset.

    Inputs:
        stack_paths: List of Paths to the .npy feature stacks.
        All other parameters are passed directly to prepare_spatial_split.
    """
    try:
        # 1. Initialize lists to collect data from each tile
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        cv_folds_list = []
        train_tile_ids_list, test_tile_ids_list = [], []

        print(f"Processing {len(stack_paths)} tiles...")

        # 2. Iterate through each tile path
        for i, path in enumerate(stack_paths):
            # Load the feature stack (assumed already cropped of
            # buffer in generate_features)
            stack_np = np.load(path)

            # Call the generalized splitter
            X_tr, y_tr, X_te, y_te, folds = prepare_spatial_split(
                stack_np,
                train_ratio=train_ratio,
                test_gap_px=test_gap_px,
                cv_gap_px=cv_gap_px,
                n_folds=n_folds,
            )

            # Create unique tile ID masks for traceability
            train_ids = np.full((X_tr.shape[0],), i)
            test_ids = np.full((X_te.shape[0],), i)

            # Collect results
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            X_test_list.append(X_te)
            y_test_list.append(y_te)
            cv_folds_list.append(folds)
            train_tile_ids_list.append(train_ids)
            test_tile_ids_list.append(test_ids)

        # 3. Perform final concatenation into master arrays
        X_train_final = np.vstack(X_train_list)
        y_train_final = np.concatenate(y_train_list)
        X_test_final = np.vstack(X_test_list)
        y_test_final = np.concatenate(y_test_list)
        cv_folds_final = np.concatenate(cv_folds_list)
        train_ids_final = np.concatenate(train_tile_ids_list)
        test_ids_final = np.concatenate(test_tile_ids_list)

        # 4. Print final dimensions
        print("\n--- Final Dataset Dimensions ---")
        print(f"X_train: {X_train_final.shape}")
        print(f"y_train: {y_train_final.shape}")
        print(f"X_test:  {X_test_final.shape}")
        print(f"y_test:  {y_test_final.shape}")
        print(f"CV Folds: {cv_folds_final.shape}")
        print("--------------------------------\n")

        return (
            X_train_final,
            y_train_final,
            X_test_final,
            y_test_final,
            cv_folds_final,
            train_ids_final,
            test_ids_final,
        )
    except Exception as e:
        logger.error(f"Error in get_ml_model_input: {e}")
        raise


# =====================================================================
# 2. MODEL EVALUATION AND SELECTION FUNCTIONS
# =====================================================================


def get_top_simplicity_candidates(
    cv_results: dict[str, Any], quantile: float = 0.99, top_n: int = 10
) -> pd.DataFrame:
    """
    Filters for the top performance tier and sorts by simplicity (GLA method).

    Inputs:
        cv_results: Dictionary containing cross-validation results.
        quantile: Quantile threshold for top performance.
        top_n: Number of top candidates to return.
    """
    try:
        results = pd.DataFrame(cv_results)

        # 1. Filter for models in the top percentile of accuracy
        threshold = results["mean_test_score"].quantile(quantile)
        top_models = results[results["mean_test_score"] >= threshold].copy()

        # 2. Sort by simplicity (lower depth, then fewer features)
        # We sort ascending so the 'simplest' appear first
        simplest_best_sorted = top_models.sort_values(
            by=["param_max_depth", "param_max_features"], ascending=True
        )

        return simplest_best_sorted.head(top_n)
    except Exception as e:
        logger.error(f"Error in get_top_simplicity_candidates: {e}")
        raise


def save_top_candidates_to_disk(
    top_candidates_df: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    tile_names: list[str],
    stack_paths: list[Path],
    meta_paths: list[Path],
    output_root: str,
) -> None:
    """
    Trains and audits top candidates using consolidated helpers.

    Inputs:
        top_candidates_df: DataFrame of top model candidates.
        X_train, y_train: Training features and labels.
        train_ids: IDs for training tiles.
        X_test, y_test: Testing features and labels.
        test_ids: IDs for testing tiles.
        tile_names: List of tile names.
        stack_paths: List of paths to feature stacks.
        meta_paths: List of paths to metadata files.
        output_root: Root directory for saving model data.
    """
    try:
        feat_names = get_feature_names()

        for i in range(len(top_candidates_df)):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            print(f"processing model {i + 1}...")

            params = top_candidates_df.iloc[i]["params"]
            rf = RandomForestClassifier(
                **params,
                class_weight="balanced",
                random_state=BALANCED_RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)

            # 1. Use the 'Engine' to evaluate the model
            metrics = evaluate_model(
                rf, X_train, y_train, train_ids, X_test, y_test, test_ids, tile_names
            )

            # 2. Setup Folder for results
            folder = os.path.join(output_root, f"model_{i + 1}_{timestamp}")
            os.makedirs(folder, exist_ok=True)

            # 3. Use the 'Reporter' to write metrics
            write_metrics_report(
                metrics,
                os.path.join(folder, "metrics.txt"),
                params,
                model_index=i,
                feature_names=feat_names,
            )

            # 4. Save diagnostic rasters
            save_diagnostic_rasters(rf, folder, stack_paths, meta_paths)

            # 5. Save the Random Forest model to disk
            joblib.dump(rf, Path(folder, "rf_model.joblib"))

            print(f"Diagnostic data saved to {folder}")
    except Exception as e:
        logger.error(f"Error in save_top_candidates_to_disk: {e}")
        raise


def get_feature_names() -> list[str]:
    """
    Returns the list of feature labels used in the GLA methodology.
    Excludes features 4 and 5; includes feature 44.
    """
    try:
        # Features 1, 2, 3, 6-43, then 44
        names = [f"f{i + 1 if i < 3 else i + 3}" for i in range(41)]
        names.append("f44")
        return names
    except Exception as e:
        logger.error(f"Error in get_feature_names: {e}")
        return []


def evaluate_per_tile(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    tile_names: list[str],
) -> dict[str, dict[str, float]]:
    """
    Calculates both Training and Testing balanced accuracy for every tile.

    Inputs:
        model: Trained ML model.
        X_train, y_train: Training data.
        train_ids: IDs for training tiles.
        X_test, y_test: Testing data.
        test_ids: IDs for testing tiles.
        tile_names: Names of the tiles.
    """
    try:
        # 1. Get unique tile IDs from both sets
        unique_indices = np.unique(np.concatenate([train_ids, test_ids]))
        results = {}

        # 2. Iterate through each tile and evaluate
        for idx in unique_indices:
            name = tile_names[idx] if idx < len(tile_names) else f"Tile {idx}"

            # 3. Slice and predict for the Training portion of this tile
            tr_mask = train_ids == idx
            tr_acc = balanced_accuracy_score(
                y_train[tr_mask], model.predict(X_train[tr_mask])
            )

            # 4. Slice and predict for the Testing portion of this tile
            te_mask = test_ids == idx
            te_acc = balanced_accuracy_score(
                y_test[te_mask], model.predict(X_test[te_mask])
            )

            results[name] = {"train": tr_acc, "test": te_acc, "gap": tr_acc - te_acc}
        return results
    except Exception as e:
        logger.error(f"Error in evaluate_per_tile: {e}")
        return {}


def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: np.ndarray,
    tile_names: list[str],
) -> dict[str, Any]:
    """
    The 'Engine': Calculates all metrics and returns a master dictionary.

    Inputs:
        model: Trained ML model.
        X_train, y_train: Training data.
        train_ids: Training tile IDs.
        X_test, y_test: Testing data.
        test_ids: Testing tile IDs.
        tile_names: Tile names.
    """
    try:
        train_preds = model.predict(X_train)
        y_pred = model.predict(X_test)

        return {
            "train_acc": balanced_accuracy_score(y_train, train_preds),
            "test_acc": balanced_accuracy_score(y_test, y_pred),
            "tile_results": evaluate_per_tile(
                model, X_train, y_train, train_ids, X_test, y_test, test_ids, tile_names
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "class_report": classification_report(
                y_test, y_pred, zero_division=0
            ),  # String format for text reports
            "importances": model.feature_importances_,
        }
    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        raise


def write_metrics_report(
    metrics_dict: dict[str, Any],
    output_path: str,
    params_dict: dict[str, Any],
    model_index: int | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """
    The 'Reporter': Handles your specific .txt formatting requirements.

    Inputs:
        metrics_dict: Dictionary of model metrics.
        output_path: Path to save the report.
        params_dict: Dictionary of model parameters.
        model_index: Optional index of the model.
        feature_names: Optional list of feature names.
    """
    try:
        # 1. Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 2. Get sorted indices of feature importances
        indices = np.argsort(metrics_dict["importances"])[::-1]

        # 3. Write report content to file
        with open(output_path, "w") as f:
            f.write("=" * 30 + "\n")
            title = (
                f" MODEL {model_index + 1} PERFORMANCE REPORT"
                if model_index is not None
                else " MODEL PERFORMANCE REPORT"
            )
            f.write(f"{title}\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Parameters: {params_dict}\n\n")

            f.write("--- Global Accuracies ---\n")
            f.write(f"Training Balanced Acc: {metrics_dict['train_acc']:.4f}\n")
            f.write(f"Testing Balanced Acc:  {metrics_dict['test_acc']:.4f}\n")
            f.write(
                "Generalization Gap:    "
                f"{metrics_dict['train_acc'] - metrics_dict['test_acc']:.4f}\n\n"
            )

            f.write("--- Per-Tile Accuracy ---\n")
            for name, data in metrics_dict["tile_results"].items():
                f.write(f"--- {name} ---\n")
                f.write(f"Training Accuracy: {data['train']:.4f}\n")
                f.write(f"Testing Accuracy: {data['test']:.4f}\n")
                f.write(f"Generalization Gap: {data['gap']:.4f}\n")
            f.write("\n")

            f.write("--- Confusion Matrix ---\n")
            f.write(np.array2string(metrics_dict["confusion_matrix"]) + "\n\n")
            f.write("--- Detailed Classification Report ---\n")
            f.write(metrics_dict["class_report"] + "\n")

            if not feature_names:
                feature_names = get_feature_names()

            f.write("--- Top 10 Feature Importances ---\n")
            for rank in range(min(10, len(indices))):
                idx = indices[rank]
                f.write(
                    f"{rank + 1}. {feature_names[idx]}: "
                    f"{metrics_dict['importances'][idx]:.4f}\n"
                )
    except Exception as e:
        logger.error(f"Error in write_metrics_report: {e}")
        raise


def save_diagnostic_rasters(
    model: Any, output_folder: str, stack_paths: list[Path], meta_paths: list[Path]
) -> None:
    """
    Generates GeoTIFF error maps.
    If correct: 0.
    If incorrect: Shows the predicted class value (1, 2, or 3).

    Inputs:
        model: Trained ML model.
        output_folder: Directory to save diagnostic rasters.
        stack_paths: Paths to feature stacks.
        meta_paths: Paths to metadata files.
    """
    try:
        # 1. Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 2. Iterate through each tile stack and metadata
        for stack_path, meta_path in zip(stack_paths, meta_paths, strict=True):
            # Load data
            stack = np.load(stack_path)
            # All channels except the last one are features
            X_full = stack[:-1, :, :].reshape(stack.shape[0] - 1, -1).T
            # The last channel is the ground truth label
            y_true = stack[-1, :, :]

            # 3. Predict for the entire tile
            y_pred = model.predict(X_full).reshape(stack.shape[1], stack.shape[2])

            # 4. Apply the 'Identity Error' logic
            # 0 = Correct. Any other value = the class the model MISTAKENLY predicted.
            error_map = np.where(y_pred != y_true, y_pred, 0).astype(np.uint8)

            # 5. Load metadata and update for single-band error map
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)

            meta.update({"count": 1, "dtype": "uint8"})

            # 6. Save the error map as a GeoTIFF
            out_name = f"{stack_path.stem}_classification_errors.tif"
            with rasterio.open(
                os.path.join(output_folder, out_name), "w", **meta
            ) as dst:
                dst.write(error_map, 1)
    except Exception as e:
        logger.error(f"Error in save_diagnostic_rasters: {e}")
        raise


def plot_feature_importance(model: Any, title: str = "Feature Importance") -> None:
    """
    Generates a bar chart showing the top contributors to the classification.

    Inputs:
        model: Trained ML model.
        title: Title for the plot.
    """
    try:
        importances = model.feature_importances_
        feature_names = get_feature_names()
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(
            range(len(importances)),
            importances[indices],
            align="center",
            color="skyblue",
        )
        plt.xticks(
            range(len(importances)), [feature_names[i] for i in indices], rotation=90
        )
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error in plot_feature_importance: {e}")
