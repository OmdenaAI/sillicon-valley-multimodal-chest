import sys
import pandas as pd
from source.exception import CustomeException

class DataAcess:
    '''
    reading dataset
    '''
    def __init__(self, path):
        self.path = path
    
    def read_data(self):
        '''
        reading the dataset from a given path
        '''
        try:
            data = pd.read_csv(self.path)
        except Exception as e:
            raise CustomeException(e, sys)
        return data
