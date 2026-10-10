import sys
import os
import pandas as pd
from datetime import datetime

from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.exception import CustomException


class ModelMonitor:
    """
    Records every training run's metrics to a persistent CSV so you can track
    improvement over time, compare tasks, and back up claims in your README
    with actual numbers instead of memory.

    Also does a lightweight data-drift check: compares null percentages and
    row counts between two ingestion runs, useful if you ever rerun the
    pipeline on a refreshed dataset.
    """

    def __init__(self, log_path="artifacts/metrics_log.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # ---------- METRIC LOGGING ----------
    def log_metrics(self, task_name: str, model_name: str, metrics: dict, notes: str = ""):
        """
        metrics example: {"accuracy": 0.81, "f1_macro": 0.74}
        Call this once per model per run -- appends a row, never overwrites.
        """
        try:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task": task_name,
                "model": model_name,
                "notes": notes,
                **metrics,
            }

            if os.path.exists(self.log_path):
                df = pd.read_csv(self.log_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])

            df.to_csv(self.log_path, index=False)
            logging.info(f"Logged metrics for {task_name}/{model_name}: {metrics}")
        except Exception as e:
            raise CustomException(e, sys)

    def log_report(self, task_name: str, report: dict, notes: str = ""):
        """
        Convenience wrapper: takes the full `report` dict returned by
        evaluate_models() (multiple models at once) and logs each one.
        """
        try:
            for model_name, res in report.items():
                self.log_metrics(
                    task_name,
                    model_name,
                    {"accuracy": res["accuracy"], "f1_macro": res["f1_macro"]},
                    notes=notes,
                )
        except Exception as e:
            raise CustomException(e, sys)

    def get_history(self, task_name: str = None) -> pd.DataFrame:
        """Returns the full metrics history, optionally filtered to one task."""
        try:
            if not os.path.exists(self.log_path):
                logging.warning("No metrics log found yet.")
                return pd.DataFrame()

            df = pd.read_csv(self.log_path)
            if task_name:
                df = df[df["task"] == task_name]
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def best_run(self, task_name: str, metric: str = "f1_macro"):
        """Returns the single best-scoring row for a given task and metric."""
        try:
            df = self.get_history(task_name)
            if df.empty:
                return None
            return df.loc[df[metric].idxmax()]
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- BASIC DATA DRIFT CHECK ----------
    def check_null_drift(self, current_df: pd.DataFrame, previous_null_pct: dict, threshold: float = 5.0):
        """
        Compares current null percentages per column against a previously
        recorded baseline (e.g. from your first EDA run). Flags any column
        whose null % changed by more than `threshold` percentage points.

        previous_null_pct example: {"country": 9.4, "director": 30.2}
        """
        try:
            current_null_pct = (current_df.isnull().mean() * 100).round(2).to_dict()
            drifted = {}

            for col, prev_pct in previous_null_pct.items():
                curr_pct = current_null_pct.get(col, 0.0)
                if abs(curr_pct - prev_pct) > threshold:
                    drifted[col] = {"previous": prev_pct, "current": curr_pct}

            if drifted:
                logging.warning(f"Data drift detected in columns: {drifted}")
            else:
                logging.info("No significant null-percentage drift detected.")

            return drifted
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    monitor = ModelMonitor()

    # example usage after running a task's evaluate_models() report
    example_report = {
        "RandomForest": {"accuracy": 0.81, "f1_macro": 0.74},
        "DecisionTree": {"accuracy": 0.77, "f1_macro": 0.68},
    }
    monitor.log_report("task3_rating_classification", example_report, notes="baseline run")

    print(monitor.get_history("task3_rating_classification"))
    print("Best run:", monitor.best_run("task3_rating_classification"))