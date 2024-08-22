import os
import sys
from src.logger import logging
from src.exception import CustomException

import subprocess
from pymongo import MongoClient

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    TRAIN_DATA_PATH: str = os.path.join('artifacts', "train.csv")
    TEST_DATA_PATH: str = os.path.join('artifacts', "test.csv")
    RAW_DATA_PATH: str = os.path.join('artifacts', "data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")
        try:
            # Download dataset from kaggle if required
            data = KaggleCaliforniaHousingDataset()
            data.download_kaggle_dataset()
            
            # load data in a dataframe
            df = pd.read_csv('./notebooks/data/housing.csv')
            
            # establish connection with database
            database = DatabaseConnection()
            
            # create database if it does not exist
            if database.database_exists():
                logging.info("Database exists")
            else:
                database.create_database()
                logging.info("Database did not exist. Created database successfully.")
            
            # create and store data in database if needed
            if database.data_exists_in_database():
                logging.info("Data already stored in database.")
            else:
                database.create_data_in_database('housing')
                database.store_data_in_mongodb(df, 'housing')
                logging.info(f"Data did not exist. Created stored successfully.")
            
            # Make artifact directory for train data if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.TRAIN_DATA_PATH), exist_ok = True)
            
            # store housing data in artifacts folder
            df.to_csv(self.ingestion_config.RAW_DATA_PATH, index = False, header= True)
            
            # split data
            logging.info("Train test split initated")
            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)
            
            # store full 
            train_set.to_csv(self.ingestion_config.TRAIN_DATA_PATH, index= False, header= True)
            test_set.to_csv(self.ingestion_config.TEST_DATA_PATH, index= False, header= True)
            
            logging.info("Ingestion of the data is completed")
            
            return(
                self.ingestion_config.TRAIN_DATA_PATH,
                self.ingestion_config.TEST_DATA_PATH
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def storing_process_database(self):
        # TODO : implement the complete storing process of the database into the database
        pass 

@dataclass
class DatabaseConnectionConfig:
    mongo_uri = os.getenv("MONGODB_USER_NAME")
    username = os.getenv("MONGODB_USER_NAME")
    password = os.getenv("MONGODB_PASSWORD")
    db_name = os.getenv("DATABASE_NAME")

class DatabaseConnection:
    def __init__(self):
        self.database_config = DatabaseConnectionConfig()
        self.client = MongoClient(self.mongo_uri) # connects client to database
        logging.info("Client connection successful")
        
    
    # check if database exists in project cluster
    def database_exists(self) -> bool:
        return self.database_config.db_name in self.client.list_database_names()

    # create database in project cluster and stores a database variable
    def create_database(self):
        try:
            self.db = self.client[self.db_name]
            logging.info("Created empty database for storing California Housing Data")
        except Exception as e:
            raise CustomException(e, sys)
    
    # check if dataframe is in database
    def data_exists_in_database(self, collection_name) -> bool:
        return collection_name in self.db.collection_names()
    
    # create collection for future storage of dataframe in database
    def create_data_in_database(self, collection_name : str):
        try:
            self.db.create_collection(collection_name)
        except Exception as e:
            raise CustomException(e, sys)
    
    # store data in dataframe datatype in database collection
    def store_data_in_mongodb(self, df, collection_name : str):
        try:
            collection = self.db[collection_name]
            if collection.count_documents == df.shape[0]: # check if the correct data is stored in the collection
                logging.info("Data already stored in database")
            else:
                collection.delete_many({}) # empties collection for storing new data
                data_dict = df.to_dict(orient="records") # convert dataframe to document format
                collection.insert_many(data_dict)
                logging.info(f"Stored {collection_name} data in database")
        except Exception as e:
            raise CustomException(e, sys)

@dataclass
class KaggleConfig:
    DATASET = os.getenv("KAGGLE_DATASET")
    DATA_PATH = './notebooks/data'

class KaggleCaliforniaHousingDataset:
    def __init__(self):
        self.kaggle_config = KaggleConfig()
    
    # download Kaggle dataset
    def download_kaggle_dataset(self):
        try:
            if not os.path.exists(os.path.join(self.kaggle_config.DATA_PATH, "housing.csv")):
                subprocess.run(['kaggle', 'datasets', 'download', '-d', self.kaggle_config.dataset, '-p', self.kaggle_config.dataset, '--unzip'], check=True)
                logging.info("Downloaded Data from Kaggle successfully")
            else:
                logging.info("Data already downloaded")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while downloading the dataset: {e}")
            
if __name__ == "__main__":
    ingestion = DataIngestion()
    train_df, test_df = ingestion.initiate_data_ingestion()
    