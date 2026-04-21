import joblib
import sys
import os
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

sys.path.append(os.path.abspath('..'))
import params

def execute_search():

    # Load the necessary data
    ml_numpy_input = params.ml_numpy_input_folder()
    X_train = np.load(Path(ml_numpy_input, 'X_train.npy'))
    y_train = np.load(Path(ml_numpy_input, 'y_train.npy'))
    folds = np.load(Path(ml_numpy_input, 'folds.npy'))

    # 1. Initialize the Spatial Cross-Validation
    # 'folds' is the array returned by our data_prep script
    ps = PredefinedSplit(folds)

    # 2. Define the Search Space (aligned with GLA standards)
    param_dist = {
        'n_estimators': list(range(30, 101)),
        'max_depth': list(range(6, 19)),
        'min_samples_leaf': list(range(50, 1001)),
        'max_samples': np.linspace(0.3, 1.0, 71).tolist(),
        'max_features': list(range(3, 21))
    }

    # 3. Setup and Run Search
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=1200,
        cv=ps,
        scoring='balanced_accuracy',
        verbose=3,
        n_jobs=1,
        random_state=42,
        refit=False  # We manually audit and refit the 'Simplest Best' later
    )

    search.fit(X_train, y_train)

    # 4. Save Search Results to Disk
    joblib.dump(search, params.hyperparamter_search_file(is_output=True))
    print(f"Hyperparameter Search Results Successfully Saved to {params.hyperparamter_search_file(is_output=True)}")

if __name__ == '__main__':
    execute_search()