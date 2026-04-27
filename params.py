from pathlib import Path

USING_CUSTOM_DATA = False

def data_folder(force_custom_data=False):
    if force_custom_data or USING_CUSTOM_DATA:
        return Path('..', 'data', 'user_data')
    return Path('..', 'data', 'westminster_data')

# 01_data_acquisition_and_preparation
def full_core_features_folder():
    return Path(data_folder(), "01_data_acquisition_and_preparation", "full_core_features")

def labelled_tiles_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "01_data_acquisition_and_preparation", "labelled_tiles")

def tiled_core_features_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "01_data_acquisition_and_preparation", "tiled_core_features")

def feature_stacks_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "01_data_acquisition_and_preparation", "feature_stacks")


# 02_model_training_and_postprocessing
def hyperparameter_search_file(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "search_results.joblib")

def ml_numpy_input_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "ml_numpy_inputs")

def simplest_best_models_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "simplest_best_models")

def diagnostic_rasters_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "diagnostic_rasters")

def production_model_file(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "production_model.joblib")

def model_inference_file(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "green_cover_map", "1_model_inference.tif")

def smoothed_cover_file(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "green_cover_map", "2_smoothed_cover.tif")

def vectorised_green_cover_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "green_cover_map", "3_vectorised_green_cover")
def vectorised_canopy_cover_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "green_cover_map", "3_vectorised_canopy_cover")

def enhanced_green_cover_folder(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "02_model_training_and_postprocessing", "green_cover_map", "4_enhanced_green_cover")


# 03_accuracy_assessment_and_result_analysis
def validation_points_file():
    return Path(data_folder(), "03_accuracy_assessment_and_result_analysis", "validation_points.gpkg")

def sampled_validation_points_file(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "03_accuracy_assessment_and_result_analysis", "sampled_validation_points.gpkg")

def confusion_matrix_table(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "03_accuracy_assessment_and_result_analysis", "confusion_matrix_table.csv")

def accuracy_metrics_table(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "03_accuracy_assessment_and_result_analysis", "accuracy_metrics_table.csv")

def area_boundary_vector_file():
    return Path(data_folder(), "03_accuracy_assessment_and_result_analysis", "area_boundary.gpkg")

def error_plot_file(is_output=False):
    return Path(data_folder(force_custom_data=is_output),
                "03_accuracy_assessment_and_result_analysis", "error_plot.png")