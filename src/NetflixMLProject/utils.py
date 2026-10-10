import os
import sys
import joblib
from sklearn.metrics import accuracy_score, f1_score

from src.NetflixMLProject.exception import CustomException


def save_object(file_path, obj):
    """Pickle any object (model, encoder, scaler, similarity matrix) to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a previously pickled object."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict = None):
    """
    Trains each model in `models` (optionally grid-searched using `params`),
    returns a dict of {model_name: {"accuracy":.., "f1_macro":.., "model": fitted_model}}.

    models: {"RandomForest": RandomForestClassifier(), ...}
    params: {"RandomForest": {"n_estimators": [100, 200]}, ...}  (optional)
    """
    try:
        from sklearn.model_selection import GridSearchCV

        report = {}
        for name, model in models.items():
            if params and name in params and params[name]:
                gs = GridSearchCV(model, params[name], cv=3, scoring="f1_macro", n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            report[name] = {"accuracy": acc, "f1_macro": f1, "model": model}

        return report
    except Exception as e:
        raise CustomException(e, sys)