from typing import List
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# remove stop words

tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
detokenizer = TreebankWordDetokenizer()
Vectorizer = CountVectorizer()

# nltk.download('stopwords')
stop=stopwords.words('english')
stop.remove('no')



class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Temporal elapsed time transformer."""
    def __init__(self, variables: List[str], reference_variable: str):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # so that we do not over-write the original dataframe
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""
    def __init__(self, variables: List[str], mappings: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)
        return X
    
def pre_process(text):
    sentences = re.findall(r'"(.*?)"', text)
    if len(sentences)==0:
        return(text)
    elif len(sentences) >= 2:
        second_sentence = sentences[1]
        return(second_sentence)
    else:
        return(sentences[0])
    

def NlpPreprocessor(X: pd.Series) -> pd.Series:
    X = X.copy()
    X = X.str.lower()
    X = X.apply(lambda x : pre_process(x))

    # we remove stop words from training part, but leave validation and test untouchable
    X = X.map(lambda x: tokenizer.tokenize(str(x)))
    X = X.map(lambda x: [ps.stem(word) for word in x if not word in stop])
    X = X.map(lambda x: detokenizer.detokenize(x))
    return X


# Create TensorFlow generators for training and validation
batch_size = 32

def data_generator(train_set, train_set_path, test_set, test_set_path, batch_size = batch_size):
    # this block was constructed with the tutorial developed by Saurabh Bhardwaj

    #image_directory = "images/pneumonia_normal_images"
    image_height = 224
    image_width = 224
    

    # Create an ImageDataGenerator for data augmentation and preprocessing
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        samplewise_center=True,  # Define your preprocess_image function here
        rescale=1.0/255.0,  # Normalize pixel values
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_set,
        directory=train_set_path,
        x_col='ImageID',
        y_col='Labels',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_set,
        directory=test_set_path,
        x_col='ImageID',
        y_col='Labels',
        target_size=(image_height, image_width),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )

    return train_generator, test_generator


