import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.components.data_ingestion import (
    DataIngestion, DataIngestionConfig, DatabaseConnection,
    DatabaseConnectionConfig, KaggleCaliforniaHousingDataset, KaggleConfig,
)
from src.exception import CustomException


def housing_frame():
    return pd.DataFrame({
        "longitude": [-122.0] * 10,
        "latitude": [37.0] * 10,
        "housing_median_age": list(range(10, 20)),
        "total_rooms": [100.0] * 10,
        "total_bedrooms": [20.0] * 10,
        "population": [50.0] * 10,
        "households": [10.0] * 10,
        "median_income": [3.5] * 10,
        "ocean_proximity": ["NEAR BAY"] * 10,
        "median_house_value": list(range(100000, 100010)),
    })


def mongo_client(documents=None):
    """Stateful stand-in for storage; no network or real credentials are used."""
    records = [dict(row) for row in (documents or [])]
    client = MagicMock()
    database = client.__getitem__.return_value
    collection = database.__getitem__.return_value
    collection.count_documents.side_effect = lambda query: len(records)
    collection.find.side_effect = lambda query, projection: [
        {key: value for key, value in row.items() if key != "_id"}
        for row in records
    ]

    def insert(rows):
        identifiers = []
        for row in rows:
            saved = dict(row)
            saved["_id"] = len(records) + 1
            identifiers.append(saved["_id"])
            records.append(saved)
        return SimpleNamespace(inserted_ids=identifiers)

    collection.insert_many.side_effect = insert
    return client, database, collection, records


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = DataIngestionConfig(
            TRAIN_DATA_PATH=str(self.root / "artifacts/train.csv"),
            TEST_DATA_PATH=str(self.root / "artifacts/test.csv"),
            RAW_DATA_PATH=str(self.root / "artifacts/data.csv"),
        )
        self.kaggle = KaggleConfig(DATASET="", DATA_PATH=str(self.root / "cache"))
        self.database_config = DatabaseConnectionConfig(
            mongo_uri="mongodb://example.invalid", db_name="test_housing"
        )

    def save_csv(self, frame=None):
        path = Path(self.kaggle.DATA_PATH) / "housing.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        (housing_frame() if frame is None else frame).to_csv(path, index=False)
        return path

    def ingestion(self, mongo=False):
        return DataIngestion(
            config=self.paths,
            database_config=self.database_config if mongo else DatabaseConnectionConfig(
                mongo_uri="", db_name=""
            ),
            kaggle_config=self.kaggle,
        )

    def test_local_csv_preserves_artifacts_and_deterministic_80_20_split(self):
        self.save_csv()
        with patch("src.components.data_ingestion.MongoClient") as connect:
            result = self.ingestion().initiate_data_ingestion()
            connect.assert_not_called()
        self.assertEqual(result, (self.paths.TRAIN_DATA_PATH, self.paths.TEST_DATA_PATH))
        self.assertEqual(len(pd.read_csv(self.paths.RAW_DATA_PATH)), 10)
        self.assertEqual(len(pd.read_csv(result[0])), 8)
        self.assertEqual(len(pd.read_csv(result[1])), 2)
        first = Path(result[0]).read_bytes()
        self.ingestion().initiate_data_ingestion()
        self.assertEqual(first, Path(result[0]).read_bytes())

    def test_stored_mongo_dataset_is_used_without_csv_or_kaggle(self):
        rows = housing_frame().to_dict("records")
        rows[0]["_id"] = "database-only-field"
        client, _, collection, _ = mongo_client(rows)
        with patch("src.components.data_ingestion.MongoClient", return_value=client), patch.object(
            DataIngestion, "_load_csv", side_effect=AssertionError("CSV must not be read")
        ):
            self.ingestion(mongo=True).initiate_data_ingestion()
        raw = pd.read_csv(self.paths.RAW_DATA_PATH)
        self.assertNotIn("_id", raw.columns)
        pd.testing.assert_frame_equal(raw, housing_frame())
        collection.find.assert_called_once_with({}, {"_id": 0})
        collection.insert_many.assert_not_called()
        client.close.assert_called_once()

    def test_empty_mongo_collection_is_populated_then_reused_without_duplicates(self):
        self.save_csv()
        client, _, collection, records = mongo_client()
        with patch("src.components.data_ingestion.MongoClient", return_value=client):
            self.ingestion(mongo=True).initiate_data_ingestion()
            with patch.object(DataIngestion, "_load_csv", side_effect=AssertionError("cache miss")):
                self.ingestion(mongo=True).initiate_data_ingestion()
        self.assertEqual(len(records), 10)
        collection.insert_many.assert_called_once()
        collection.delete_many.assert_not_called()
        self.assertEqual(client.close.call_count, 2)

    def test_invalid_stored_dataset_fails_without_fallback_and_closes_client(self):
        client, _, _, _ = mongo_client([{"unexpected": "field"}])
        with patch("src.components.data_ingestion.MongoClient", return_value=client), patch.object(
            DataIngestion, "_load_csv"
        ) as load_csv:
            with self.assertRaisesRegex(CustomException, "missing required columns"):
                self.ingestion(mongo=True).initiate_data_ingestion()
            load_csv.assert_not_called()
        client.close.assert_called_once()
        self.assertFalse(Path(self.paths.RAW_DATA_PATH).exists())

    def test_empty_csv_is_rejected_before_artifact_writes(self):
        self.save_csv(housing_frame().iloc[:0])
        with self.assertRaisesRegex(CustomException, "must not be empty"):
            self.ingestion().initiate_data_ingestion()
        self.assertFalse(Path(self.paths.RAW_DATA_PATH).exists())

    def test_csv_missing_required_columns_is_rejected(self):
        self.save_csv(housing_frame().drop(columns="median_house_value"))
        with self.assertRaisesRegex(CustomException, "median_house_value"):
            self.ingestion().initiate_data_ingestion()

    def test_mongo_connection_failure_is_not_silently_replaced_with_local_data(self):
        client, _, collection, _ = mongo_client()
        collection.count_documents.side_effect = RuntimeError("connection unavailable")
        with patch("src.components.data_ingestion.MongoClient", return_value=client), patch.object(
            DataIngestion, "_load_csv"
        ) as load_csv:
            with self.assertRaisesRegex(CustomException, "connection unavailable"):
                self.ingestion(mongo=True).initiate_data_ingestion()
            load_csv.assert_not_called()
        client.close.assert_called_once()


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.config = DatabaseConnectionConfig(
            mongo_uri="mongodb://example.invalid", db_name="test_housing"
        )

    def test_configuration_reads_current_environment_at_instantiation(self):
        with patch.dict(os.environ, {
            "MONGODB_URI": "mongodb://example.invalid",
            "DATABASE_NAME": "test_housing",
            "MONGODB_COLLECTION": "fixture",
            "KAGGLE_DATASET": "owner/dataset",
        }, clear=True):
            self.assertEqual(DatabaseConnectionConfig(), DatabaseConnectionConfig(
                mongo_uri="mongodb://example.invalid", db_name="test_housing",
                collection_name="fixture"
            ))
            self.assertEqual(KaggleConfig().DATASET, "owner/dataset")

    def test_missing_uri_or_database_name_does_not_connect(self):
        with patch("src.components.data_ingestion.MongoClient") as connect:
            for config in [
                DatabaseConnectionConfig(mongo_uri="", db_name="test_housing"),
                DatabaseConnectionConfig(mongo_uri="mongodb://example.invalid", db_name=""),
            ]:
                with self.assertRaisesRegex(ValueError, "MONGODB_URI and DATABASE_NAME"):
                    DatabaseConnection(config)
            connect.assert_not_called()

    def test_existing_collection_is_preserved_when_store_is_called_again(self):
        original = housing_frame().to_dict("records")
        client, _, collection, records = mongo_client(original)
        with patch("src.components.data_ingestion.MongoClient", return_value=client):
            with DatabaseConnection(self.config) as database:
                self.assertEqual(database.store_data_in_mongodb(housing_frame(), "housing"), 10)
        self.assertEqual(records, original)
        collection.count_documents.assert_called_once_with({})
        collection.insert_many.assert_not_called()
        collection.delete_many.assert_not_called()

    def test_missing_values_become_bson_null_without_mutating_input(self):
        frame = housing_frame()
        frame.loc[0, "total_bedrooms"] = float("nan")
        client, _, _, records = mongo_client()
        with patch("src.components.data_ingestion.MongoClient", return_value=client):
            with DatabaseConnection(self.config) as database:
                database.store_data_in_mongodb(frame, "housing")
        self.assertIsNone(records[0]["total_bedrooms"])
        self.assertTrue(pd.isna(frame.loc[0, "total_bedrooms"]))
        self.assertNotIn("_id", frame.columns)

    def test_collection_creation_uses_current_pymongo_api_and_is_idempotent(self):
        client, db, _, _ = mongo_client()
        db.list_collection_names.return_value = ["housing"]
        with patch("src.components.data_ingestion.MongoClient", return_value=client):
            with DatabaseConnection(self.config) as database:
                database.create_data_in_database("housing")
                db.create_collection.assert_not_called()
                database.create_data_in_database("other")
        db.create_collection.assert_called_once_with("other")


class KaggleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = KaggleConfig(DATASET="owner/dataset", DATA_PATH=str(self.path))

    def test_existing_csv_never_invokes_download(self):
        (self.path / "housing.csv").write_text("cached")
        with patch("src.components.data_ingestion.subprocess.run") as run:
            self.assertEqual(
                KaggleCaliforniaHousingDataset(self.config).download_kaggle_dataset(),
                self.path / "housing.csv"
            )
            run.assert_not_called()

    def test_download_uses_dataset_and_output_directory_and_requires_csv(self):
        def download(command, check):
            self.assertTrue(check)
            (self.path / "housing.csv").write_text("downloaded")

        with patch("src.components.data_ingestion.subprocess.run", side_effect=download) as run:
            KaggleCaliforniaHousingDataset(self.config).download_kaggle_dataset()
        run.assert_called_once_with([
            "kaggle", "datasets", "download", "-d", "owner/dataset",
            "-p", str(self.path), "--unzip",
        ], check=True)

    def test_download_error_is_propagated(self):
        with patch("src.components.data_ingestion.subprocess.run", side_effect=
                   subprocess.CalledProcessError(1, "kaggle")):
            with self.assertRaises(subprocess.CalledProcessError):
                KaggleCaliforniaHousingDataset(self.config).download_kaggle_dataset()

    def test_successful_command_without_housing_csv_is_rejected(self):
        with patch("src.components.data_ingestion.subprocess.run"):
            with self.assertRaisesRegex(FileNotFoundError, "did not produce housing.csv"):
                KaggleCaliforniaHousingDataset(self.config).download_kaggle_dataset()

    def test_missing_dataset_configuration_never_invokes_download(self):
        self.config.DATASET = ""
        with patch("src.components.data_ingestion.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "KAGGLE_DATASET"):
                KaggleCaliforniaHousingDataset(self.config).download_kaggle_dataset()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
