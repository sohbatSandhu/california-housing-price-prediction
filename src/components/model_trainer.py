import os, sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor  # type: ignore
from sklearn.linear_model import LinearRegression, Ridge  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from xgboost import XGBRegressor  # type: ignore

from sklearn.metrics import r2_score  # type: ignore

from src.exception import CustomException  # type: ignore
from src.logger import logging  # type: ignore

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainingConfig:
    TRAINED_MODEL_FILE_PATH = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            logging.info("Initialize models and hyper-parameters")

            models, params = self.init_models_and_params()
            
            logging.info("Evaluating Models on pre-processed training data")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("No best model found")
                raise CustomException("No best model found")
            logging.info("Best Model found on training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.TRAINED_MODEL_FILE_PATH,
                obj=best_model,
            )
            
            logging.info("Saved Best Model")

            predicted = best_model.predict(X_test)

            score = r2_score(y_test, predicted)
            logging.info("")
            
            return score

        except Exception as e:
            raise CustomException(e, sys)

    def init_models_and_params(self):
        models = {
            "Linear Regressor": LinearRegression(),
            "Ridge Regressor": Ridge(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Bagging Regressor": BaggingRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "XGBRegressor": XGBRegressor(),
        }

        params = {
            "Linear Regressor": {},
            "Ridge Regressor": {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "Decision Tree Regressor": {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
            },
            "Random Forest Regressor": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10],
                # "min_samples_leaf": [1, 2, 4],
                # "max_features": ["sqrt", "log2"],
            },
            "Bagging Regressor": {
                "n_estimators": [10, 50, 100],
                "max_samples": [0.5, 0.8, 1.0],
                "max_features": [0.5, 0.8, 1.0],
                # "bootstrap": [True, False],
                # "bootstrap_features": [True, False],
            },
            "Gradient Boosting Regressor": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 1.0],
                "max_depth": [3, 5, 7],
                # "subsample": [0.5, 0.8, 1.0],
            },
            "AdaBoost Regressor": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1],
                "loss": ["linear", "square", "exponential"],
            },
            "XGBRegressor": {
                "n_estimators": [100, 200, 500],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [5, 7, 9],
                # "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.3],
                # "subsample": [0.8, 0.9, 1.0],
                # "colsample_bytree": [0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [1, 1.5, 2],
            }
        }
        
        return models, params
