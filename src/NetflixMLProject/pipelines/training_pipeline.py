import sys
from sklearn.model_selection import train_test_split


from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.components.data_ingestion import DataIngestion
from src.NetflixMLProject.components.data_transformation import DataTransformation
from src.NetflixMLProject.components.model_trainer import ModelTrainer

class TrainPipeline:
    """
    End-to-end training run: ingestion -> transformation -> training.
    Run with: python -m src.pipeline.train_pipeline
    Used for Task 3 (and adaptable to Task 6 by swapping the feature-builder call).
    """

    def run(self):
        try:
            logging.info("=== TRAIN PIPELINE STARTED ===")

            # 1. Ingestion
            ingestion = DataIngestion()
            raw_path, train_path, test_path = ingestion.initiate_data_ingestion()

            # 2. Transformation
            transformer = DataTransformation(raw_data_path=raw_path)
            df = transformer.initiate_data_transformation()
            X, y, encoders = transformer.get_classification_features(df)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 3. Training
            trainer = ModelTrainer(save_path="artifacts/models/best_model_task3.pkl")
            best_name, best_model, report = trainer.initiate_model_trainer(
                X_train, y_train, X_test, y_test
            )

            logging.info(f"=== TRAIN PIPELINE COMPLETE -- best model: {best_name} ===")
            print(f"Training complete. Best model: {best_name}")
            for name, res in report.items():
                print(f"  {name}: accuracy={res['accuracy']:.3f}  f1_macro={res['f1_macro']:.3f}")

            return best_name, best_model
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run()