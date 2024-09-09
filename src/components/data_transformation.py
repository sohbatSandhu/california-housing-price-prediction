import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.pipeline import Pipeline, make_pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler # type: ignore

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, FeatureEngineering, LogTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', "preprocessor.pkl")
    imputer_obj_file_path= os.path.join('artifacts', "imputer.pkl")
    featengineering_obj_file_path= os.path.join('artifacts', "featengineering.pkl")
    logtransformer_obj_file_path= os.path.join('artifacts', "logtransformer.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            categorical_features, numerical_features, robust_feats, std_feats, _, _, _, _ = self.get_separated_features()

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            # imputation preprocessor
            imputer_processor = make_column_transformer(
                (SimpleImputer(strategy="median", fill_value="missing_values"), numerical_features),
                (SimpleImputer(strategy="most_frequent", fill_value="missing_values"), categorical_features),
            )

            logging.info("Built Imputer")

            # Make tranformers for each feature
            categorical_transformer = make_pipeline(
                OneHotEncoder(handle_unknown="ignore", sparse_output=False), StandardScaler()
            )

            std_transformer = make_pipeline(StandardScaler())

            robust_transformer = make_pipeline(RobustScaler())
            
            # Make Preprocessor
            preprocessor = make_column_transformer(
                (std_transformer, std_feats),
                (robust_transformer, robust_feats),
                (categorical_transformer, categorical_features),
            )
            
            logging.info("Built Preprocesser")
            
            return imputer_processor, preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, TRAIN_PATH, TEST_PATH):
        try:
            train_df= pd.read_csv(TRAIN_PATH)
            test_df= pd.read_csv(TEST_PATH)
            
            logging.info("Reading train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            imputer_processor, preprocessing_obj= self.get_data_transformer_object()
            
            categorical_features, numerical_features, _, _, _, engineered_features, log_features, target = self.get_separated_features()
            
            X_train= train_df.drop(columns=[target], axis=1)
            y_train= train_df[target]

            X_test= test_df.drop(columns=[target], axis=1)
            y_test= test_df[target]
            
            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            # perform imputation
            imputed_X_train = pd.DataFrame(
                imputer_processor.fit_transform(X_train), columns=X_train.columns
            )
            imputed_X_test = pd.DataFrame(
                imputer_processor.transform(X_test), columns=X_train.columns
            )
            
            logging.info("Imputation Performed")

            # Manually convert data types back to original
            for column in numerical_features:
                imputed_X_train[column] = imputed_X_train[column].astype(float)
                imputed_X_test[column] = imputed_X_test[column].astype(float)

            # apply feature endineering and data transformation to train and test data again
            feat_engineer = FeatureEngineering()
            transformer = LogTransformer(columns=log_features)
            
            logging.info("Perform feature engineering")

            # perform feature engineering
            engineered_X_train = feat_engineer.fit_transform(imputed_X_train)
            engineered_X_test = feat_engineer.transform(imputed_X_test)

            logging.info(f"Engineered Features: {engineered_features}")
            logging.info("Perform Log transformation on Highly skewed features")
            
            # perform log transformation
            transformed_engineered_X_train = transformer.fit_transform(engineered_X_train)
            transformed_engineered_X_test = transformer.transform(engineered_X_test)
            
            logging.info(f"Log transformed Features: {log_features}")

            X_train_arr=preprocessing_obj.fit_transform(transformed_engineered_X_train)
            
            X_test_arr=preprocessing_obj.transform(transformed_engineered_X_test)
            
            train_arr = np.c_[
                X_train_arr, np.array(y_train)
            ]
            
            test_arr = np.c_[
                X_test_arr, np.array(y_test)
            ]
            
            save_object(
                file_path=self.data_transformation_config.imputer_obj_file_path,
                obj=imputer_processor
            )
            logging.info(f"Saved imputer object.")

            save_object(
                file_path=self.data_transformation_config.featengineering_obj_file_path,
                obj=feat_engineer
            )
            logging.info(f"Saved feature engineering object.")
            
            save_object(
                file_path=self.data_transformation_config.logtransformer_obj_file_path,
                obj=transformer
            )
            logging.info(f"Saved Log transformer object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.imputer_obj_file_path,
                self.data_transformation_config.featengineering_obj_file_path,
                self.data_transformation_config.logtransformer_obj_file_path,
            )    

        except Exception as e:
            raise CustomException(e, sys)
    
    def get_separated_features(self):
            categorical_features = ['ocean_proximity']
            
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
            
            num_feats = [
                "longitude",
                "latitude",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "median_income",
                "rooms_per_household",
                "bedrooms_per_rooms",
                "bedrooms_per_households",
                "population_per_household",
            ]

            std_feats = ["latitude", "longitude", "housing_median_age"]

            robust_feats = [
                "total_bedrooms",
                "total_rooms",
                "population",
                "households",
                "median_income",
                "rooms_per_household",
                "bedrooms_per_rooms",
                "bedrooms_per_households",
                "population_per_household",
            ]
            
            engineered_features = [
                "rooms_per_household",
                "bedrooms_per_rooms",
                "bedrooms_per_households",
                "population_per_household",
            ]
            
            log_features = [
                "total_bedrooms",
                "total_rooms",
                "population",
                "households",
                "rooms_per_household",
                "bedrooms_per_rooms",
                "bedrooms_per_households",
                "population_per_household",
            ]
            
            target = "median_house_value"
            
            return categorical_features, numerical_features, robust_feats, std_feats, num_feats, engineered_features, log_features, target
