import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            # define paths to data preprocessing objects and models
            MODEL_PATH = 'artifacts/model.pkl'
            IMPUTER_PATH = 'artifacts/imputer.pkl'
            FEAT_ENGINEERING_PATH = 'artifacts/featengineering.pkl'
            LOG_TRANSFORMER_PATH = 'artifacts/logtransformer.pkl'
            PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'
            
            # load objects
            model = load_object(file_path = MODEL_PATH)
            imputer = load_object(file_path= IMPUTER_PATH)
            feat_engineer = load_object(file_path= FEAT_ENGINEERING_PATH)
            transformer = load_object(file_path= LOG_TRANSFORMER_PATH)
            preprocessor = load_object(file_path= PREPROCESSOR_PATH)
            
            # perform imputation
            imputed_data = pd.DataFrame(
                imputer.fit_transform(features), columns=features.columns
            )
            # Manually convert data types back to original
            numerical_features = [
                'longitude',
                'latitude',
                'housing_median_age',
                'total_bedrooms',
                'total_rooms',
                'population',
                'households',
                'median_income'
            ]
            for column in numerical_features:
                imputed_data[column] = imputed_data[column].astype(float)
            
            engineered_data = feat_engineer.transform(imputed_data)
            engineered_transformed_data = transformer.transform(engineered_data)
            data_scaled = preprocessor.transform(engineered_transformed_data)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class HousingData:
    def __init__(self,
        longitude: float,
        latitude : float,
        housing_median_age : float,
        total_rooms : float,
        total_bedrooms : float,
        population : float,
        households : float,
        median_income : float,
        ocean_proximity : str
    ):
        self.longitude = longitude,
        self.latitude  = latitude,
        self.housing_median_age = housing_median_age,
        self.total_rooms  = total_rooms,
        self.total_bedrooms  = total_bedrooms,
        self.population  = population,
        self.households  = households,
        self.median_income  = median_income,
        self.ocean_proximity = ocean_proximity
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'longitude': self.longitude,
                'latitude': self.latitude,
                'housing_median_age': self.housing_median_age,
                'total_rooms': self.total_rooms,
                'total_bedrooms': self.total_bedrooms,
                'population': self.population,
                'households': self.households,
                'median_income': self.median_income,
                'ocean_proximity': self.ocean_proximity,
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
    