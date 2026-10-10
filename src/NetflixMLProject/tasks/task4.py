import sys
import pandas as pd
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.components.data_transformation import DataTransformation
from src.NetflixMLProject.utils import save_object, evaluate_models
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

class ContentSegmenter:
    def __init__(self):
        self.dt = DataTransformation()
 
    def find_best_k(self, scaled_features, k_range=range(2, 13)):
        try:
            scores = {}
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(scaled_features)
                scores[k] = silhouette_score(scaled_features, labels)
                logging.info(f"k={k} silhouette={scores[k]:.3f}")
            best_k = max(scores, key=scores.get)
            return best_k, scores
        except Exception as e:
            raise CustomException(e, sys)
 
    def run(self):
        try:
            df = self.dt.initiate_data_transformation()
            scaled, feature_df = self.dt.get_clustering_features(df)
 
            best_k, scores = self.find_best_k(scaled)
            print(f"Best k by silhouette score: {best_k}")
            print(scores)
 
            km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            df["cluster"] = km.fit_predict(scaled)
 
            save_object("artifacts/models/kmeans_task4.pkl", km)
 
            # PCA for 2D visualization
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(scaled)
            print("PCA variance explained:", pca.explained_variance_ratio_.round(3),
                  "| total:", round(pca.explained_variance_ratio_.sum(), 3))
            logging.info(f"PCA variance explained (2D): {pca.explained_variance_ratio_.sum():.3f}")
            plt.figure(figsize=(8, 6))
            plt.scatter(reduced[:, 0], reduced[:, 1], c=df["cluster"], cmap="tab10", alpha=0.6)
            plt.title(f"Netflix Content Clusters (k={best_k}, PCA-reduced)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.savefig("artifacts/models/cluster_plot.png")
            plt.show()
 
            # Interpretation: dominant genre + country + avg year per cluster
            print("\n--- Cluster Profiles ---")
            for c in sorted(df["cluster"].unique()):
                sub = df[df["cluster"] == c]
                top_genre = sub["genre_list"].explode().value_counts().idxmax()
                top_country = sub["country"].apply(lambda x: x.split(",")[0]).value_counts().idxmax()
                avg_year = sub["release_year"].mean()
                print(f"Cluster {c}: n={len(sub)}, top_genre={top_genre}, "
                      f"top_country={top_country}, avg_release_year={avg_year:.0f}")
                
            df[["title", "cluster"]].to_csv("artifacts/cluster_results.csv", index=False)
            return df
           
        except Exception as e:
            raise CustomException(e, sys)
 
 
if __name__ == "__main__":
    seg = ContentSegmenter()
    seg.run()
 