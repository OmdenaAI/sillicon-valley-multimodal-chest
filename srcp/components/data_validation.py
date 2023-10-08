from distutils import dir_util
from srcp.data_access.data_manager import ChestXrayData
from srcp.constant.training_pipeline import SCHEMA_FILE_PATH
from srcp.constant.training_pipeline import TARGET_COLUMN
from srcp.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from srcp.entity.config_entity import DataValidationConfig
from srcp.exception import CustomException
from srcp.logger import logging
from srcp.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys


class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                        data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise  CustomException(e,sys)
        
    def validate_inputs(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Check model inputs for na values and filter."""
        try:
            validated_data = input_data.copy()
            validated_data = validated_data[self._schema_config['DataInputSchema'].keys()]
            validated_data.dropna(inplace=True)
            return validated_data
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["DataInputSchema"].keys())
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            error_message = ""
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            #Reading data from train and test file location
            train_dataframe = ChestXrayData(train_file_path).read_data()
            test_dataframe = ChestXrayData(test_file_path).read_data()

            #Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"{error_message}Test dataframe does not contain all columns.\n"
        
            if len(error_message)>0:
                raise Exception(error_message)
            

            train_dataframe = self.validate_inputs(train_dataframe)
            test_dataframe = self.validate_inputs(test_dataframe)

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting Valid train and test file path.")

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)