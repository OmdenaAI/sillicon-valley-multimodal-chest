from srcp.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os

#Write a code to train model and check the accuracy.
class NLPModel:
    def __init__(self, model):
        try:
            self.model = model
        except Exception as e:
            raise e
    
    def predict(self, x):
        try:
            y_hat = self.model.predict(x)
            return y_hat
        except Exception as e:
            raise e
    
