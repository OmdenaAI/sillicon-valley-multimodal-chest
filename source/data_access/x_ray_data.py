import pandas as pd

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
        except:
            raise Exception()
        return data
