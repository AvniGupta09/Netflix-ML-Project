import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.utils import save_object


class DataTransformation:
    def __init__(self, raw_data_path="artifacts/Dataset.csv"):
        self.raw_data_path = raw_data_path
        self.preprocessor_dir = "artifacts/models"

    # ---------- CLEANING ----------
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting data cleaning")
            df = df.copy()

            # Fill nulls (informed by EDA)
            for col in ["director", "country"]:
                df[col] = df[col].fillna("Unknown")
            df["rating"] = df["rating"].fillna(df["rating"].mode()[0])
            df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")

            # Duration split by type: Movies -> minutes, TV Shows -> seasons
            df["duration_num"] = df["duration"].str.extract(r"(\d+)").astype(float)

            # Genre as list (for TF-IDF / multi-hot later)
            df["genre_list"] = df["listed_in"].fillna("").apply(lambda x: [g.strip() for g in x.split(",")])

            # Group rare rating classes (Task 3 imbalance fix)
            rating_counts = df["rating"].value_counts()
            rare_ratings = rating_counts[rating_counts < 10].index
            df["rating_grouped"] = df["rating"].apply(lambda r: "Other" if r in rare_ratings else r)

            logging.info("Data cleaning complete")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- TASK 1: TEXT FEATURES FOR RECOMMENDATION ----------
    def get_tfidf_matrix(self, df: pd.DataFrame):
        try:
            # No 'description' or 'cast' columns in this dataset, so the text
            # signal comes from genre (listed_in), director, and country --
            # thinner than a description-based approach, but still captures
            # genre/style/origin similarity reasonably well.
            df["tags"] = (
                df["listed_in"].fillna("") + " " +
                df["director"].fillna("") + " " +
                df["country"].fillna("") + " " +
                df["type"].fillna("")
            ).str.lower()

            tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf_matrix = tfidf.fit_transform(df["tags"])

            save_object(f"{self.preprocessor_dir}/tfidf.pkl", tfidf)
            logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            return tfidf_matrix, tfidf
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- TASK 3: ENCODED FEATURES FOR CLASSIFICATION ----------
    def get_classification_features(self, df: pd.DataFrame):
        try:
            features = df[["type", "duration_num", "release_year", "country"]].copy()
            features["country"] = features["country"].apply(lambda x: x.split(",")[0].strip())

            encoders = {}
            for col in ["type", "country"]:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                encoders[col] = le

            features["duration_num"] = features["duration_num"].fillna(features["duration_num"].median())

            target_le = LabelEncoder()
            target = target_le.fit_transform(df["rating_grouped"].astype(str))
            encoders["rating_target"] = target_le

            save_object(f"{self.preprocessor_dir}/classification_encoders.pkl", encoders)
            return features, target, encoders
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- TASK 4: SCALED FEATURES FOR CLUSTERING ----------
    def get_clustering_features(self, df: pd.DataFrame):
        try:
            features = df[["duration_num", "release_year"]].copy()
            features["duration_num"] = features["duration_num"].fillna(features["duration_num"].median())

            # add top-genre one-hot flags (keep it simple: top 10 genres as binary columns)
            top_genres = df["genre_list"].explode().value_counts().head(10).index
            for genre in top_genres:
                features[f"genre_{genre.replace(' ', '_')}"] = df["genre_list"].apply(lambda gl: int(genre in gl))

            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)

            save_object(f"{self.preprocessor_dir}/cluster_scaler.pkl", scaler)
            return scaled, features
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- MAIN ENTRY ----------
    def initiate_data_transformation(self):
        try:
            df = pd.read_csv(self.raw_data_path)
            df = self.clean_data(df)
            return df
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    dt = DataTransformation()
    cleaned_df = dt.initiate_data_transformation()
    print(cleaned_df.head())