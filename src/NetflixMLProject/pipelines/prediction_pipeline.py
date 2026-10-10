import sys
import pandas as pd

from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.utils import load_object


class PredictPipeline:
    """
    Inference-only path -- no retraining. Loads saved model + encoders and
    transforms/predicts on new input. This is what app.py should import.
    """

    def __init__(self,
                 model_path="artifacts/models/best_model_task3.pkl",
                 encoders_path="artifacts/models/classification_encoders.pkl"):
        self.model_path = model_path
        self.encoders_path = encoders_path
        self._model = None
        self._encoders = None

    def _load(self):
        if self._model is None:
            self._model = load_object(self.model_path)
        if self._encoders is None:
            self._encoders = load_object(self.encoders_path)

    def predict(self, input_dict: dict):
        """
        input_dict example:
        {"type": "Movie", "duration_num": 90, "release_year": 2020, "country": "United States"}
        """
        try:
            self._load()
            df = pd.DataFrame([input_dict])

            for col in ["type", "country"]:
                le = self._encoders[col]
                # unseen category fallback
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))

            df = df[["type", "duration_num", "release_year", "country"]]

            pred_encoded = self._model.predict(df)
            target_le = self._encoders["rating_target"]
            pred_label = target_le.inverse_transform(pred_encoded)

            logging.info(f"Prediction for input {input_dict}: {pred_label[0]}")
            return pred_label[0]
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = PredictPipeline()
    sample = {"type": "Movie", "duration_num": 90, "release_year": 2020, "country": "United States"}
    print(pipeline.predict(sample))