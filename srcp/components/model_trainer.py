

from srcp.exception import CustomException
from srcp.logger import logging
from srcp.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, DataValidationArtifact
from srcp.entity.config_entity import ModelTrainerConfig
from srcp.constant.training_pipeline import SCHEMA_FILE_PATH
from srcp.utils.main_utils import read_yaml_file, write_yaml_file
from srcp.utils.main_utils import load_numpy_array_data
import os
import sys

import numpy as np
# to build the model
from sklearn.linear_model import Lasso
from srcp.data_access.data_manager import ChestXrayData
from srcp.constant.training_pipeline import TARGET_COLUMN

from srcp.ml.metric.classification_metric import get_classification_score
from srcp.ml.model.estimator import NLPModel
from srcp.utils.main_utils import save_object,load_object
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from srcp.ml.model.loss import get_weighted_loss, compute_class_freqs    
from srcp.ml.preprocess import preprocessing as pp
from srcp.constant import training_pipeline 
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
        data_transformation_artifact:DataTransformationArtifact,
        data_validation_artifact: DataValidationArtifact,
        ):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            self.data_validation_artifact=data_validation_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)

    def train_model(self,x_train, y_train):
        try:
            lin_model = LinearSVC()
            lin_model.fit(x_train, y_train)
            return lin_model
        except Exception as e:
            raise e
        
    def train_model_final(self, text_predictions, image_predictions, Y):
        # create the base pre-trained model
        stacked_predictions = np.column_stack(text_predictions, image_predictions)

        fusion_classifier = RandomForestClassifier()
        fusion_classifier.fit(stacked_predictions, Y)
        return fusion_classifier
    
    def train_model_cv(self):
        # create the base pre-trained model
        base_model = DenseNet121(include_top=False) # weights='./drive/MyDrive/Colab/MMML/densenet.hdf5'
        x = base_model.output
        # add a global spatial average pooling layer
        x = GlobalAveragePooling2D()(x)
        # and a logistic layer
        predictions = Dense(2, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        return model
            
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_df = ChestXrayData(self.data_validation_artifact.valid_train_file_path).read_data()
            test_df = ChestXrayData(self.data_validation_artifact.valid_test_file_path).read_data()

            train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].map(lambda x: 1 if x=='pneumonia' else 0)
            test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].map(lambda x: 1 if x=='pneumonia' else 0)

            #loading training array and testing array
            x_train = load_numpy_array_data(train_file_path)
            x_test = load_numpy_array_data(test_file_path)
            y_train = train_df[TARGET_COLUMN]
            y_test = test_df[TARGET_COLUMN]

            model_nlp = self.train_model(x_train, y_train)
            y_train_pred = model_nlp.predict(x_train)
            classification_train_metric =  get_classification_score(y_true=y_train, y_pred=y_train_pred)
            print(classification_train_metric)
            if classification_train_metric.model_f1_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")
        
            y_test_pred = model_nlp.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            #Overfitting and Underfitting
            diff = abs(classification_train_metric.model_f1_score-classification_test_metric.model_f1_score)
            
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")
            
            # =========================
            train_set_path = training_pipeline.IMG_DATA_DIR
            test_set_path = training_pipeline.IMG_DATA_DIR
            epoch = training_pipeline.EPOCHS
            batch_size = training_pipeline.BATCH_SIZE   
            labels = y_train.unique() #['normal', 'pneumonia']

            train_generator, test_generator = pp.data_generator(train_set=train_df, 
                                                               train_set_path=train_set_path, 
                                                               test_set=test_df, 
                                                               test_set_path=test_set_path, 
                                                               batch_size = batch_size)
            
            # Train the model
            model_cv = self.train_model_cv(y_train)
            pos_weights, neg_weights = compute_class_freqs(y_train)

            # Compile the model
            model_cv.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights), metrics=['accuracy'])
            model_cv.fit(train_generator=train_generator, epochs=epoch, validation_data=test_generator)

            # ========================================================
            text_predictions = y_train_pred
            image_predictions = model_cv.predict_generator(test_generator, steps = len(test_generator))

            fusion_classifier = self.train_model_final(text_predictions, image_predictions, y_train)

            fusion_y_train_pred = fusion_classifier.predict(x_train)
            fusion_y_test_pred = fusion_classifier.predict(x_test)

            classification_final_metric_train =  get_classification_score(y_true=y_train, y_pred=fusion_y_train_pred)
            classification_final_metric_test =  get_classification_score(y_true=y_test, y_pred=fusion_y_test_pred)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            final_model = NLPModel(model=fusion_classifier)
            save_object(self.model_trainer_config.trained_model_file_path, obj=final_model)

            #model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
            train_metric_artifact=classification_final_metric_train,
            test_metric_artifact=classification_final_metric_test)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e,sys)