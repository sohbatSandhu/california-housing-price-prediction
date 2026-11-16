"""Load California housing data from MongoDB or a cached/downloaded CSV."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


REQUIRED_COLUMNS = {
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity", "median_house_value",
}


def validate_dataset(frame):
    if frame.empty:
        raise ValueError("Housing dataset must not be empty.")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError("Housing dataset is missing required columns: " + ", ".join(sorted(missing)))
    return frame


@dataclass
class DataIngestionConfig:
    TRAIN_DATA_PATH: str = os.path.join("artifacts", "train.csv")
    TEST_DATA_PATH: str = os.path.join("artifacts", "test.csv")
    RAW_DATA_PATH: str = os.path.join("artifacts", "data.csv")


@dataclass
class DatabaseConnectionConfig:
    mongo_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", ""))
    db_name: str = field(default_factory=lambda: os.getenv("DATABASE_NAME", ""))
    collection_name: str = field(
        default_factory=lambda: os.getenv("MONGODB_COLLECTION", "housing")
    )


@dataclass
class KaggleConfig:
    DATASET: str = field(default_factory=lambda: os.getenv("KAGGLE_DATASET", ""))
    DATA_PATH: str = "./notebooks/data"


class DatabaseConnection:
    def __init__(self, config=None):
        self.database_config = config or DatabaseConnectionConfig()
        if not self.database_config.mongo_uri or not self.database_config.db_name:
            raise ValueError("Set both MONGODB_URI and DATABASE_NAME to use MongoDB.")
        self.client = MongoClient(
            self.database_config.mongo_uri, serverSelectionTimeoutMS=5000
        )
        self.db = self.client[self.database_config.db_name]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()

    def database_exists(self):
        return self.database_config.db_name in self.client.list_database_names()

    def create_database(self):
        # MongoDB creates databases on the first successful write.
        return self.db

    def data_exists_in_database(self, collection_name):
        return self.db[collection_name].count_documents({}) > 0

    def create_data_in_database(self, collection_name):
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)

    def read_data_from_mongodb(self, collection_name):
        frame = pd.DataFrame(list(self.db[collection_name].find({}, {"_id": 0})))
        return validate_dataset(frame)

    def store_data_in_mongodb(self, frame, collection_name):
        validate_dataset(frame)
        collection = self.db[collection_name]
        existing_count = collection.count_documents({})
        if existing_count:
            # This is a persistent dataset cache, not a destructive refresh.
            logging.info("Reusing the existing housing collection (%s records).", existing_count)
            return existing_count
        records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        result = collection.insert_many(records)
        logging.info("Stored %s housing records in MongoDB.", len(result.inserted_ids))
        return len(result.inserted_ids)


class KaggleCaliforniaHousingDataset:
    def __init__(self, config=None):
        self.kaggle_config = config or KaggleConfig()

    def download_kaggle_dataset(self):
        target = Path(self.kaggle_config.DATA_PATH) / "housing.csv"
        if target.is_file():
            logging.info("Using the cached housing CSV.")
            return target
        if not self.kaggle_config.DATASET:
            raise ValueError("Set KAGGLE_DATASET when the local housing.csv is missing.")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", self.kaggle_config.DATASET,
                "-p", self.kaggle_config.DATA_PATH, "--unzip",
            ],
            check=True,
        )
        if not target.is_file():
            raise FileNotFoundError("Kaggle download did not produce housing.csv.")
        return target


class DataIngestion:
    def __init__(self, config=None, database_config=None, kaggle_config=None):
        self.ingestion_config = config or DataIngestionConfig()
        self.database_config = database_config or DatabaseConnectionConfig()
        self.kaggle_config = kaggle_config or KaggleConfig()

    def _load_csv(self):
        path = KaggleCaliforniaHousingDataset(self.kaggle_config).download_kaggle_dataset()
        return validate_dataset(pd.read_csv(path))

    def _load_dataset(self):
        if not self.database_config.mongo_uri:
            logging.info("MongoDB is not configured; using the local CSV workflow.")
            return self._load_csv()
        with DatabaseConnection(self.database_config) as database:
            collection = self.database_config.collection_name
            if database.data_exists_in_database(collection):
                logging.info("Loading the stored housing dataset from MongoDB.")
                return database.read_data_from_mongodb(collection)
            frame = self._load_csv()
            database.store_data_in_mongodb(frame, collection)
            return frame

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated.")
        try:
            frame = self._load_dataset()
            for path in (
                self.ingestion_config.RAW_DATA_PATH,
                self.ingestion_config.TRAIN_DATA_PATH,
                self.ingestion_config.TEST_DATA_PATH,
            ):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(self.ingestion_config.RAW_DATA_PATH, index=False)
            train_set, test_set = train_test_split(
                frame, test_size=0.2, random_state=42
            )
            train_set.to_csv(self.ingestion_config.TRAIN_DATA_PATH, index=False)
            test_set.to_csv(self.ingestion_config.TEST_DATA_PATH, index=False)
            logging.info("Saved raw, training, and test CSV artifacts.")
            return (
                self.ingestion_config.TRAIN_DATA_PATH,
                self.ingestion_config.TEST_DATA_PATH,
            )
        except Exception as error:
            raise CustomException(error, sys) from error


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    load_dotenv(dotenv_path=".env", override=False)
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    transformation = DataTransformation()
    train_array, test_array, _, _, _, _ = transformation.initiate_data_transformation(
        train_path, test_path
    )
    print(ModelTrainer().initiate_model_trainer(train_array, test_array))
