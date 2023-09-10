import os

SAVED_MODEL_DIR: str = os.path.join('saved_model')



TARGET_COLUMN: str = 'class'
PIPELINE_NAME: str = 'chestxray_classifier'
ARTIFACT_NAME: str = 'artifacts'

FILE_NAME: str = 'abc.csv'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'

PREPROCESSOR_OBJECT_FILE_NAME: str = 'preprocessing.pkl'
MODEL_NAME: str = 'model.pkl'
SCHEMA_FILE_NAME = os.path.join('config', 'schema.yaml')


#EDA Preprocessing
DROP_COLUMNS: str = 'drop_columns'

'''
DATA INGESTION CONSTANT
'''
DATA_INGESTION_DIR_NAME: str = 'chest_x_ray'
DATA_INGESTION_FEATURE_STORE_SIR: str = 'feature_store'
DATA_INGESTION_INGESTED_DIR: str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20



