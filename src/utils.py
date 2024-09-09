import os
import sys

import numpy as np 
import pandas as pd
import dill # type: ignore
import pickle

from sklearn.base import BaseEstimator, TransformerMixin # type: ignore
from sklearn.metrics import r2_score # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        DIR_PATH = os.path.dirname(file_path)
        
        os.makedirs(DIR_PATH, exist_ok= True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
# Custom transformer for creating interaction terms and additional features
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # TODO: engineer other features with analysis
        # # Create interaction terms
        # X['longitude_latitude_interaction'] = X['longitude'] * X['latitude']

        # # Binning housing median age
        # bins = [0, 20, 40, np.inf]
        # labels = ['new', 'moderate', 'old']
        # X['housingMedianAge_binned'] = pd.cut(X['housingMedianAge'], bins=bins, labels=labels)

        # # Interaction terms
        # X['age_income_interaction'] = X['housingMedianAge'] * X['medianIncome']
        # X['age_ocean_interaction'] = X['housingMedianAge'].astype(str) + "_" + X['oceanProximity']

        # Rooms per Household and Bedrooms per Room
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["bedrooms_per_rooms"] = X["total_bedrooms"] / X["total_rooms"]
        X["bedrooms_per_households"] = X["total_bedrooms"] / X["households"]
        # X['rooms_population_interaction'] = X['totalRooms'] * X['population']

        # Population per Household
        X["population_per_household"] = X["population"] / X["households"]

        # # Interaction with Ocean Proximity
        # X['ocean_longitude_interaction'] = X['longitude'].astype(str) + "_" + X['oceanProximity']
        # X['ocean_latitude_interaction'] = X['latitude'].astype(str) + "_" + X['oceanProximity']

        return X

# Function for log transformation of the column
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()  # Create a copy to avoid altering the original data
        for colname in self.columns:
            if (X[colname] == 1.0).all():
                X[colname] = np.log(X[colname] + 1)
            else:
                X[colname] = np.log(X[colname])
        return X

def evaluate_models(X_train, y_train,X_test,y_test,models, param):
    '''
    
    '''
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3, n_jobs = -1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            
            logging.info(f"Best {model} model evaluated with best parameters {gs.best_params_} getting training score of {gs.best_score_}")

            # test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = train_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)