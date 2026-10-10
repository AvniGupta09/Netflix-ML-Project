import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.components.data_transformation import DataTransformation
from src.NetflixMLProject.utils import evaluate_models, save_object


class ModelTrainer:
    def __init__(self, save_path="artifacts/models/best_model.pkl"):
        self.save_path = save_path

    def default_models(self):
        return {
            "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        }

    def default_params(self):
        return {
            "DecisionTree": {"max_depth": [5, 10, None]},
            "RandomForest": {"n_estimators": [100, 200], "max_depth": [10, None]},
            "GradientBoosting": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
            "LogisticRegression": {"C": [0.1, 1, 10]},
        }

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test,
                                models: dict = None, params: dict = None,
                                metric: str = "f1_macro"):
        try:
            logging.info("Starting model training and comparison")
            models = models or self.default_models()
            params = params if params is not None else self.default_params()

            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            for name, res in report.items():
                logging.info(f"{name} -> accuracy={res['accuracy']:.3f}, f1_macro={res['f1_macro']:.3f}")

            best_name = max(report, key=lambda n: report[n][metric])
            best_model = report[best_name]["model"]
            best_score = report[best_name][metric]

            logging.info(f"Best model: {best_name} ({metric}={best_score:.3f})")

            if best_score < 0.5:
                logging.warning("Best model score is below 0.5 -- consider revisiting features")

            save_object(self.save_path, best_model)
            return best_name, best_model, report
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Example standalone usage (normally called from a pipeline instead)
    

    dt = DataTransformation()
    df = dt.initiate_data_transformation()
    X, y, _ = dt.get_classification_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    trainer = ModelTrainer()
    name, model, report = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    print(f"Best model: {name}")