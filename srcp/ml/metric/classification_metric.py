from srcp.entity.artifact_entity import ClassificationMetricArtifact
from srcp.exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score   
import os,sys

def get_classification_score(y_true, y_pred)->ClassificationMetricArtifact:
    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        model_f1_score = f1_score(y_true, y_pred)

        classification_metric =  ClassificationMetricArtifact(accuracy=accuracy, 
                                                        precision=precision,
                                                        recall=recall,
                                                        model_f1_score=model_f1_score)
        return classification_metric
    except Exception as e:
        raise CustomException(e,sys)