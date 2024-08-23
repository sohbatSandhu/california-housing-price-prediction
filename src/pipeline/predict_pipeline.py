import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            MODEL_PATH = 'artifacts/model.pkl'
            PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = MODEL_PATH)
            preprocessor = load_object(file_path= PREPROCESSOR_PATH)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomerData:
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
    