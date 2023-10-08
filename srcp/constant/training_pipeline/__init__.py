import os

SAVED_MODEL_DIR =os.path.join("saved_models")
# Common constant variable for training pipeline
TARGET_COLUMN = "Labels"
PIPELINE_NAME: str = "chestxray_classification_model"
ARTIFACT_NAME: str = "artifact"
DATA_DIR: str = os.path.join("data")
IMG_DATA_DIR: str = os.path.join("data", 'data_dir')
FILE_NAME: str = "padchest_pneumonia_normal_en.csv"


TRAIN_FILE_NAME: str = "train_data.csv"
TEST_FILE_NAME: str = "test_data.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

EPOCHS = 1
BATCH_SIZE = 32

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessor.pkl"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.80
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
