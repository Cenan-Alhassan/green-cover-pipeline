# Constants
RANDOM_STATE = 42
N_JOBS = -1

# Standard Libraries
import logging
import os
import sys
from pathlib import Path

import joblib

# Scientific and Machine Learning Libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

# Project-specific imports
# This allows importing from the parent directory where 'params.py' is located
sys.path.append(os.path.abspath(".."))
import params

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================================
# HYPERPARAMETER SEARCH
# =====================================================================


def execute_search() -> None:
    """
    Executes a hyperparameter search for a Random Forest Classifier using spatial cross-validation.

    Loads training data, defines the search space, and saves the results to disk.

    Inputs:
        None
    """
    try:
        # 1. Load the necessary data
        ml_numpy_input: Path = params.ml_numpy_input_folder()
        X_train: np.ndarray = np.load(Path(ml_numpy_input, "X_train.npy"))
        y_train: np.ndarray = np.load(Path(ml_numpy_input, "y_train.npy"))
        folds: np.ndarray = np.load(Path(ml_numpy_input, "folds.npy"))

        # 2. Initialize the Spatial Cross-Validation
        # 'folds' is the array returned by our data_prep script
        ps: PredefinedSplit = PredefinedSplit(folds)

        # 3. Define the Search Space (aligned with GLA standards)
        param_dist: dict = {
            "n_estimators": list(range(30, 101)),
            "max_depth": list(range(6, 19)),
            "min_samples_leaf": list(range(50, 1001)),
            "max_samples": np.linspace(0.3, 1.0, 71).tolist(),
            "max_features": list(range(3, 21)),
        }

        # 4. Setup and Run Search
        search: RandomizedSearchCV = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
            param_distributions=param_dist,
            n_iter=1200,
            cv=ps,
            scoring="balanced_accuracy",
            verbose=3,
            n_jobs=1,
            random_state=RANDOM_STATE,
            refit=False,  # We manually audit and refit the 'Simplest Best' later
        )

        logging.info("Starting hyperparameter search...")
        search.fit(X_train, y_train)

        # 5. Save Search Results to Disk
        output_path: Path = params.hyperparameter_search_file(is_output=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        joblib.dump(search, output_path)
        logging.info(
            f"Hyperparameter Search Results Successfully Saved to {output_path}"
        )

    except Exception as e:
        logging.error(f"Error during hyperparameter search: {e}")


if __name__ == "__main__":
    execute_search()
