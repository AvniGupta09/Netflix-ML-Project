import sys
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.components.data_transformation import DataTransformation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.NetflixMLProject.utils import save_object, evaluate_models
from src.NetflixMLProject.components.model_monitoring import ModelMonitor


class RatingClassifier:
    def __init__(self):
        self.dt = DataTransformation()

    def run(self):
        try:
            df = self.dt.initiate_data_transformation()
            X, y, encoders = self.dt.get_classification_features(df)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
                "LogisticRegression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
            }
            params = {
                "DecisionTree": {"max_depth": [5, 10, None]},
                "RandomForest": {"n_estimators": [100, 200], "max_depth": [10, None]},
                "LogisticRegression": {"clf__C": [0.1, 1, 10]},
            }

            report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            ModelMonitor().log_report("task3_rating_classification", report, notes="final run")

            # NOTE: report accuracy AND f1_macro -- accuracy alone is misleading
            # given the class imbalance found in EDA.
            for name, res in report.items():
                logging.info(f"{name} -> accuracy={res['accuracy']:.3f}, f1_macro={res['f1_macro']:.3f}")
                print(f"{name}: accuracy={res['accuracy']:.3f}  f1_macro={res['f1_macro']:.3f}")

            best_name = max(report, key=lambda n: report[n]["f1_macro"])
            best_model = report[best_name]["model"]
            print(f"\nBest model: {best_name}")

            y_pred = best_model.predict(X_test)
            print(classification_report(y_test, y_pred, zero_division=0))
            print(confusion_matrix(y_test, y_pred))

            save_object("artifacts/models/best_model_task3.pkl", best_model)
            return report
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    clf = RatingClassifier()
    clf.run()