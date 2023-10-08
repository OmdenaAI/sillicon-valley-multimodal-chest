import sys

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


from srcp.constant.training_pipeline import TARGET_COLUMN
from srcp.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from srcp.entity.config_entity import DataTransformationConfig
from srcp.exception import CustomException
from srcp.logger import logging
from srcp.utils.main_utils import save_numpy_array_data, save_object
from srcp.data_access.data_manager import ChestXrayData
from srcp.constant.training_pipeline import SCHEMA_FILE_PATH
from srcp.utils.main_utils import read_yaml_file, write_yaml_file

from feature_engine.encoding import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer

from srcp.ml.preprocess import preprocessing as pp


class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """

        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    def get_data_transformer_object(self)->Pipeline:
        try:
            preprocessor = CountVectorizer()
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            
            train_df = ChestXrayData(self.data_validation_artifact.valid_train_file_path).read_data()
            test_df = ChestXrayData(self.data_validation_artifact.valid_test_file_path).read_data()

            # preprocessing object
            preprocessor = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

            #training dataframe
            input_feature_train_df = train_df['Report_en_ChatGPT']
            input_feature_train_df = pp.NlpPreprocessor(input_feature_train_df)
            target_feature_train_df = train_df[TARGET_COLUMN]

            #testing dataframe
            input_feature_test_df = test_df['Report_en_ChatGPT']

            input_feature_test_df = pp.NlpPreprocessor(input_feature_test_df)
            target_feature_test_df = test_df[TARGET_COLUMN]

            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df.values)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=transformed_input_train_feature.toarray(), )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=transformed_input_test_feature.toarray(),)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor,)
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
