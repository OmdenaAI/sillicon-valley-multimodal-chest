
from srcp.exception import CustomException
from srcp.logger import logging
from srcp.entity.config_entity import DataIngestionConfig
from srcp.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split

import os,sys
import pandas as pd
from srcp.data_access.data_manager import ChestXrayData
from srcp.utils.main_utils import read_yaml_file
from srcp.constant.training_pipeline import SCHEMA_FILE_PATH
from srcp.constant import training_pipeline 


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Export mongo db collection record as data frame into feature
        """
        try:
            logging.info("Exporting data from mongodb to feature store")
            data_dir: str = os.path.join(
            training_pipeline.DATA_DIR, training_pipeline.FILE_NAME
            )
            dataframe = ChestXrayData(data_path=data_dir).read_data()
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path            

            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except  Exception as e:
            raise  CustomException(e,sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Feature store dataset will be split into train and test file
        """

        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.export_data_into_feature_store()
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)