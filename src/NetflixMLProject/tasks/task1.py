import os
import sys

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.components.data_transformation import DataTransformation
from src.NetflixMLProject.utils import save_object, evaluate_models


class Recommender:
    def __init__(self):
        self.dt = DataTransformation()
        self.df = None
        self.similarity = None
 
    def build(self):
        try:
            self.df = self.dt.initiate_data_transformation()
            tfidf_matrix, _ = self.dt.get_tfidf_matrix(self.df)
 
            logging.info("Computing cosine similarity matrix")
            self.similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
 
            save_object("artifacts/models/similarity.pkl", self.similarity)
            save_object("artifacts/models/rec_titles.pkl", self.df["title"].reset_index(drop=True))
            logging.info("Recommendation model built and saved")
        except Exception as e:
            raise CustomException(e, sys)
 
    def get_recommendations(self, title: str, top_n: int = 5):
        try:
            titles = self.df["title"].reset_index(drop=True)
            if title not in titles.values:
                return f"Title '{title}' not found in dataset."
 
            idx = titles[titles == title].index[0]
            scores = list(enumerate(self.similarity[idx]))
            ranked = sorted(enumerate(self.similarity[idx]), key=lambda x: x[1], reverse=True)
            results, seen = [], {title}
            for i, _ in ranked:
                name = titles.iloc[i]
                if i == idx or name in seen:
                    continue
                seen.add(name)
                results.append(name)
                if len(results) == top_n:
                    break
            return results
        except Exception as e:
            raise CustomException(e, sys)
 
 
if __name__ == "__main__":
    rec = Recommender()
    rec.build()