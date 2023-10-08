
from srcp.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from srcp.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from srcp.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from srcp.entity.artifact_entity import ModelTrainerArtifact
from srcp.components.data_ingestion import DataIngestion
from srcp.components.data_validation import DataValidation
from srcp.components.data_transformation import DataTransformation
from srcp.components.model_trainer import ModelTrainer
import os 
import sys
from srcp.logger import logging
from srcp.exception import CustomException


class TrainingPipeline:
    is_pipeline_running = False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config)
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact, data_validation_artifact: DataValidationArtifact ):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact, data_validation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact:DataIngestionArtifact=self.start_data_ingestion()
            data_validation_artifact: DataValidationArtifact=self.start_data_validation(data_ingestion_artifact
                                                                                        =data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact
                                                                          =data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact, data_validation_artifact)
        except Exception as e:
            raise CustomException(e, sys)