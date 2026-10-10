import os
import sys
import pandas as pd
from flask import Flask, render_template, request

from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.utils import load_object
from src.NetflixMLProject.pipelines.prediction_pipeline import PredictPipeline

from src.NetflixMLProject.components.data_ingestion import DataIngestion
from src.NetflixMLProject.tasks.task1 import Recommender
from src.NetflixMLProject.tasks.task3 import RatingClassifier
from src.NetflixMLProject.tasks.task4 import ContentSegmenter
from src.NetflixMLProject.tasks.task6 import SuccessAnalyticsEngine

app = Flask(__name__)

ARTIFACTS = "artifacts"
MODELS = os.path.join(ARTIFACTS, "models")

# Charts written by task6 into static/task6/ (file name, heading, caption)
TASK6_CHART_DIR = os.path.join("static", "task6")
TASK6_CHARTS = [
    ("model_comparison.png", "Model comparison",
     "Accuracy vs macro F1 for each model. Macro F1 is the fairer metric because rating classes are imbalanced."),
    ("feature_importance.png", "What drives rating predictions",
     "Relative importance of each input feature in the best-performing model."),
    ("type_split.png", "Movies vs TV Shows",
     "Share of the catalogue by content type."),
    ("titles_per_year.png", "Titles added per year",
     "Catalogue additions over time. The latest year may be incomplete."),
    ("top_countries.png", "Top content-producing countries",
     "Based on the first country listed for each title."),
    ("genre_shift.png", "Genre mix: older vs recent titles",
     "Title counts for the most common genres, split at the median release year."),
]


# ======================================================================
# STARTUP: build whatever artifacts are missing, once, before serving
# ======================================================================
def ensure_artifacts():
    os.makedirs(MODELS, exist_ok=True)

    # ---- Ingestion: must run first, everything else reads from its output ----
    raw_data_path = os.path.join(ARTIFACTS, "Dataset.csv")
    if not os.path.exists(raw_data_path):
        try:
            logging.info("Raw data artifact missing -- running data ingestion")
            print("[startup] Running data ingestion...")
            ingestion = DataIngestion()
            ingestion.initiate_data_ingestion()
        except Exception as e:
            logging.info(f"Data ingestion failed: {e}")
            print(f"[startup] Data ingestion FAILED: {e}")
            print("[startup] Cannot continue without raw_data.csv -- stopping artifact build.")
            return
    else:
        print("[startup] Raw data artifact already present, skipping ingestion.")

    # ---- Task 1: Recommendation ----
    sim_path = os.path.join(MODELS, "similarity.pkl")
    titles_path = os.path.join(MODELS, "rec_titles.pkl")
    if not (os.path.exists(sim_path) and os.path.exists(titles_path)):
        try:
            logging.info("Task 1 artifacts missing -- building recommendation model")
            print("[startup] Building Task 1 (recommendation)...")
            rec = Recommender()
            rec.build()
        except Exception as e:
            logging.info(f"Task 1 build failed: {e}")
            print(f"[startup] Task 1 build FAILED: {e}")
    else:
        print("[startup] Task 1 artifacts already present, skipping.")

    # ---- Task 3: Rating Classification ----
    model_path = os.path.join(MODELS, "best_model_task3.pkl")
    encoders_path = os.path.join(MODELS, "classification_encoders.pkl")
    if not (os.path.exists(model_path) and os.path.exists(encoders_path)):
        try:
            logging.info("Task 3 artifacts missing -- training rating classifier")
            print("[startup] Building Task 3 (rating classification)...")
            clf = RatingClassifier()
            clf.run()
        except Exception as e:
            logging.info(f"Task 3 build failed: {e}")
            print(f"[startup] Task 3 build FAILED: {e}")
    else:
        print("[startup] Task 3 artifacts already present, skipping.")

    # ---- Task 4: Segmentation ----
    cluster_plot_path = os.path.join(MODELS, "cluster_plot.png")
    cluster_results_path = os.path.join(ARTIFACTS, "cluster_results.csv")
    if not (os.path.exists(cluster_plot_path) and os.path.exists(cluster_results_path)):
        try:
            logging.info("Task 4 artifacts missing -- running segmentation")
            print("[startup] Building Task 4 (segmentation)...")
            seg = ContentSegmenter()
            df_with_clusters = seg.run()
            # ensure the results table the app needs actually gets saved
            if "title" in df_with_clusters.columns and "cluster" in df_with_clusters.columns:
                df_with_clusters[["title", "cluster"]].to_csv(cluster_results_path, index=False)
        except Exception as e:
            logging.info(f"Task 4 build failed: {e}")
            print(f"[startup] Task 4 build FAILED: {e}")
    else:
        print("[startup] Task 4 artifacts already present, skipping.")

    # ---- Task 6: Success Analytics ----
    comparison_path = os.path.join(ARTIFACTS, "model_comparison_task6.csv")
    insights_path = os.path.join(ARTIFACTS, "insights_task6.txt")
    chart_path = os.path.join(TASK6_CHART_DIR, "model_comparison.png")
    if not (os.path.exists(comparison_path) and os.path.exists(insights_path)
            and os.path.exists(chart_path)):
        try:
            logging.info("Task 6 artifacts missing -- running success analytics")
            print("[startup] Building Task 6 (success analytics)...")
            engine = SuccessAnalyticsEngine()
            comparison_df, insights = engine.run()
            # ensure files the app needs actually get saved
            if not os.path.exists(comparison_path):
                comparison_df.to_csv(comparison_path)
            if not os.path.exists(insights_path):
                with open(insights_path, "w") as f:
                    f.write("\n".join(insights))
        except Exception as e:
            logging.info(f"Task 6 build failed: {e}")
            print(f"[startup] Task 6 build FAILED: {e}")
    else:
        print("[startup] Task 6 artifacts already present, skipping.")

    print("[startup] Artifact check complete.\n")


# ======================================================================
# TASK 1: RECOMMENDATIONS
# ======================================================================
@app.route("/", methods=["GET"])
@app.route("/recommendations", methods=["GET", "POST"])
def recommendations():
    sim_path = os.path.join(MODELS, "similarity.pkl")
    titles_path = os.path.join(MODELS, "rec_titles.pkl")
    model_ready = os.path.exists(sim_path) and os.path.exists(titles_path)

    titles_list, results, selected_title, error = [], None, None, None

    if model_ready:
        titles = load_object(titles_path)
        titles_list = sorted(titles.tolist())

        if request.method == "POST":
            selected_title = request.form.get("title")
            matches = titles[titles == selected_title]
            if matches.empty:
                error = f"'{selected_title}' not found in dataset."
            else:
                similarity = load_object(sim_path)
                idx = matches.index[0]
                ranked = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
                results, seen = [], {selected_title}
                for i, _ in ranked:
                    name = titles.iloc[i]
                    if i == idx or name in seen:      # skip the title itself and repeated names
                        continue
                    seen.add(name)
                    results.append(name)
                    if len(results) == 5:
                        break
                    

    return render_template(
        "recommendations.html",
        active="recommendations",
        model_ready=model_ready,
        titles_list=titles_list,
        results=results,
        selected_title=selected_title,
        error=error,
    )


# ======================================================================
# TASK 3: RATING PREDICTION
# ======================================================================
@app.route("/ratings", methods=["GET", "POST"])
def ratings():
    model_path = os.path.join(MODELS, "best_model_task3.pkl")
    encoders_path = os.path.join(MODELS, "classification_encoders.pkl")
    model_ready = os.path.exists(model_path) and os.path.exists(encoders_path)

    prediction, error, form_values = None, None, {}
    history = None

    metrics_path = os.path.join(ARTIFACTS, "metrics_log.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        df = df[df["task"] == "task3_rating_classification"]
        if not df.empty:
            history = df.to_dict(orient="records")

    if model_ready and request.method == "POST":
        form_values = {
            "type": request.form.get("type"),
            "duration_num": request.form.get("duration_num"),
            "release_year": request.form.get("release_year"),
            "country": request.form.get("country"),
        }
        try:
            pipeline = PredictPipeline(model_path=model_path, encoders_path=encoders_path)
            input_dict = {
                "type": form_values["type"],
                "duration_num": float(form_values["duration_num"]),
                "release_year": int(form_values["release_year"]),
                "country": form_values["country"],
            }
            prediction = pipeline.predict(input_dict)
        except Exception as e:
            error = f"Prediction failed: {e}"

    return render_template(
        "ratings.html",
        active="ratings",
        model_ready=model_ready,
        prediction=prediction,
        error=error,
        form_values=form_values,
        history=history,
    )


# ======================================================================
# TASK 4: CONTENT CLUSTERS
# ======================================================================
@app.route("/clusters", methods=["GET"])
def clusters():
    plot_path = os.path.join(MODELS, "cluster_plot.png")
    results_path = os.path.join(ARTIFACTS, "cluster_results.csv")

    plot_ready = os.path.exists(plot_path)
    results_ready = os.path.exists(results_path)
    cluster_rows = None

    if results_ready:
        df = pd.read_csv(results_path)
        cluster_rows = df.head(20).to_dict(orient="records")

    return render_template(
        "clusters.html",
        active="clusters",
        plot_ready=plot_ready,
        results_ready=results_ready,
        cluster_rows=cluster_rows,
    )


@app.route("/static-artifact/cluster_plot.png")
def cluster_plot_image():
    from flask import send_file
    return send_file(os.path.join(MODELS, "cluster_plot.png"))


# ======================================================================
# TASK 6: SUCCESS ANALYTICS
# ======================================================================
@app.route("/analytics", methods=["GET"])
def analytics():
    comparison_path = os.path.join(ARTIFACTS, "model_comparison_task6.csv")
    insights_path = os.path.join(ARTIFACTS, "insights_task6.txt")

    comparison_ready = os.path.exists(comparison_path)
    insights_ready = os.path.exists(insights_path)
    comparison_rows, insights = None, None

    if comparison_ready:
        df = pd.read_csv(comparison_path, index_col=0)
        comparison_rows = df.reset_index().rename(columns={"index": "model"}).to_dict(orient="records")

    if insights_ready:
        with open(insights_path, "r") as f:
            insights = [line.strip() for line in f if line.strip()]

    # only list charts that actually exist, so the page never shows broken images
    charts = [
        {"file": f, "title": title, "caption": caption}
        for f, title, caption in TASK6_CHARTS
        if os.path.exists(os.path.join(TASK6_CHART_DIR, f))
    ]

    return render_template(
        "analytics.html",
        active="analytics",
        comparison_ready=comparison_ready,
        insights_ready=insights_ready,
        comparison_rows=comparison_rows,
        insights=insights,
        charts=charts,
    )


if __name__ == "__main__":
    try:
        ensure_artifacts()
    except Exception as e:
        raise CustomException(e, sys)

    # use_reloader=False is intentional: the reloader would re-run this whole
    # file (including ensure_artifacts) a second time, which would re-check
    # or re-train everything twice on every startup.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)