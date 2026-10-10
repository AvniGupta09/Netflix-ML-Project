import os
import sys
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from src.NetflixMLProject.logger import logging
from src.NetflixMLProject.exception import CustomException
from src.NetflixMLProject.utils import evaluate_models, save_object
from src.NetflixMLProject.components.data_transformation import DataTransformation

# Charts are saved here. Flask serves the static/ folder automatically.
REPORT_DIR = os.path.join("static", "task6")

# Colours match the web UI theme (dark background, orange + teal accents)
BG, FG, GRID = "#1B1F23", "#EDEDED", "#2A2F35"
ACCENT, TEAL = "#E4572E", "#2BB3A3"


class SuccessAnalyticsEngine:
    """
    NOTE ON APPROACH:
    The raw dataset has no revenue/ratings 'success' column, so this task is
    framed as a Business Intelligence deliverable rather than forcing a fake
    prediction target. It:
      1. Builds a richer feature set than Task 3
      2. Trains & compares 3 models (reusing Task 3's rating target, since
         it's the closest proxy available, but with more features)
      3. Generates automated, human-readable insights from the raw data
         (growth trends, genre shifts, geography) -- this is the real
         "business intelligence" deliverable of this task.
    """

    def __init__(self):
        self.dt = DataTransformation()

    # ---------- MODEL COMPARISON (richer feature set than Task 3) ----------
    def run_model_comparison(self, df):
        try:
            X, y, encoders = self.dt.get_classification_features(df)

            # add extra engineered feature: number of genres per title
            X["genre_count"] = df["genre_list"].apply(len)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                "RandomForest": RandomForestClassifier(random_state=42, class_weight="balanced"),
                "GradientBoosting": GradientBoostingClassifier(random_state=42),
            }
            report = evaluate_models(X_train, y_train, X_test, y_test, models)

            comparison_df = pd.DataFrame({
                name: {"accuracy": res["accuracy"], "f1_macro": res["f1_macro"]}
                for name, res in report.items()
            }).T
            print("\n--- Model Comparison ---")
            print(comparison_df)

            best_name = comparison_df["f1_macro"].idxmax()
            # keep these for the feature-importance chart
            self.best_name = best_name
            self.best_model = report[best_name]["model"]
            self.feature_names = list(X.columns)
            save_object("artifacts/models/best_model_task6.pkl", report[best_name]["model"])
            comparison_df.to_csv("artifacts/model_comparison_task6.csv")

            return comparison_df
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- AUTOMATED BUSINESS INSIGHTS ----------
    def generate_insights(self, df):
        try:
            insights = []

            # 1. Content growth trend
            df["year_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.year
            yearly_counts = df["year_added"].value_counts().sort_index()
            if len(yearly_counts) >= 2:
                growth = yearly_counts.pct_change().iloc[-1] * 100
                insights.append(f"Titles added grew/shrank {growth:.1f}% in the most recent year of data.")

            # 2. Movie vs TV Show balance
            type_pct = df["type"].value_counts(normalize=True) * 100
            insights.append(f"Content split: {type_pct.round(1).to_dict()}")

            # 3. Top growing genre (compare first half vs second half of years present)
            genre_exploded = df[["genre_list", "release_year"]].explode("genre_list")
            median_year = df["release_year"].median()
            recent = genre_exploded[genre_exploded["release_year"] >= median_year]["genre_list"].value_counts()
            older = genre_exploded[genre_exploded["release_year"] < median_year]["genre_list"].value_counts()
            genre_growth = (recent - older).sort_values(ascending=False).head(3)
            insights.append(f"Fastest-growing genres (recent vs older half): {genre_growth.to_dict()}")

            # 4. Top content-producing country
            top_country = df["country"].apply(lambda x: str(x).split(",")[0]).value_counts().head(3)
            insights.append(f"Top 3 content-producing countries: {top_country.to_dict()}")

            for i in insights:
                logging.info(i)
                print("-", i)

            return insights
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- VISUAL REPORTS ----------
    def _new_fig(self, title, grid=True, figsize=(7, 4)):
        """Themed figure. Uses Figure directly (no pyplot), so it is safe inside Flask."""
        fig = Figure(figsize=figsize, facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=12, loc="left")
        ax.tick_params(colors=FG)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if grid:
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        return fig, ax

    def _save_fig(self, fig, filename):
        fig.tight_layout()
        path = os.path.join(REPORT_DIR, filename)
        fig.savefig(path, dpi=130, facecolor=BG)
        return path

    def generate_visual_reports(self, df, comparison_df):
        """Saves PNG charts to static/task6/. One failing chart never blocks the others."""
        os.makedirs(REPORT_DIR, exist_ok=True)
        saved = []

        def attempt(name, builder):
            try:
                saved.append(builder())
            except Exception as e:
                logging.info(f"Task 6 chart '{name}' failed: {e}")
                print(f"[task6] chart '{name}' failed: {e}")

        # 1. Model comparison: accuracy vs macro F1
        def model_comparison():
            fig, ax = self._new_fig("Model comparison: accuracy vs macro F1")
            x, w = np.arange(len(comparison_df)), 0.38
            ax.bar(x - w / 2, comparison_df["accuracy"], w, label="Accuracy", color=ACCENT)
            ax.bar(x + w / 2, comparison_df["f1_macro"], w, label="F1 (macro)", color=TEAL)
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df.index, color=FG)
            ax.set_ylim(0, 1)
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", color=FG, fontsize=8)
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG)
            return self._save_fig(fig, "model_comparison.png")

        # 2. Movie vs TV Show split
        def type_split():
            counts = df["type"].value_counts()
            fig, ax = self._new_fig("Content split: Movies vs TV Shows", grid=False, figsize=(5, 4))
            ax.pie(counts.values, labels=counts.index, colors=[ACCENT, TEAL][:len(counts)],
                   autopct="%1.1f%%", startangle=90, textprops={"color": FG},
                   wedgeprops={"edgecolor": BG})
            return self._save_fig(fig, "type_split.png")

        # 3. Titles added per year
        def titles_per_year():
            years = pd.to_datetime(df["date_added"], errors="coerce").dt.year.dropna().astype(int)
            yearly = years.value_counts().sort_index()
            fig, ax = self._new_fig("Titles added to the catalogue per year")
            ax.plot(yearly.index, yearly.values, color=ACCENT, marker="o")
            ax.annotate("latest year may be incomplete",
                        xy=(yearly.index[-1], yearly.values[-1]), xytext=(-12, 18),
                        textcoords="offset points", ha="right", color=FG, fontsize=8,
                        arrowprops=dict(arrowstyle="->", color=FG))
            return self._save_fig(fig, "titles_per_year.png")

        # 4. Top 10 countries (first listed country)
        def top_countries():
            first = df["country"].astype(str).str.split(",").str[0].str.strip()
            top = first[first != "Unknown"].value_counts().head(10).iloc[::-1]
            fig, ax = self._new_fig("Top 10 content-producing countries")
            ax.barh(top.index, top.values, color=TEAL)
            ax.bar_label(ax.containers[0], color=FG, fontsize=8, padding=3)
            return self._save_fig(fig, "top_countries.png")

        # 5. Genres: older vs recent titles
        def genre_shift():
            g = df[["genre_list", "release_year"]].explode("genre_list")
            g = g[g["genre_list"].notna() & (g["genre_list"] != "")]
            cutoff = int(df["release_year"].median())
            recent = g[g["release_year"] >= cutoff]["genre_list"].value_counts()
            older = g[g["release_year"] < cutoff]["genre_list"].value_counts()
            top = recent.add(older, fill_value=0).sort_values(ascending=False).head(8).index[::-1]
            y = np.arange(len(top))
            fig, ax = self._new_fig(f"Top genres: released before vs from {cutoff}", figsize=(7, 4.5))
            ax.barh(y - 0.2, older.reindex(top).fillna(0), 0.4, label=f"Before {cutoff}", color=TEAL)
            ax.barh(y + 0.2, recent.reindex(top).fillna(0), 0.4, label=f"{cutoff} onwards", color=ACCENT)
            ax.set_yticks(y)
            ax.set_yticklabels(top, color=FG)
            ax.set_xlim(0, max(recent.max(), older.max()) * 1.35)  # room for the legend
            ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower right")
            return self._save_fig(fig, "genre_shift.png")

        # 6. What the best model relies on
        def feature_importance():
            model = getattr(self, "best_model", None)
            if model is None or not hasattr(model, "feature_importances_"):
                raise ValueError("best model has no feature_importances_")
            imp = pd.Series(model.feature_importances_, index=self.feature_names).sort_values()
            fig, ax = self._new_fig(f"What drives rating predictions ({self.best_name})")
            ax.barh(imp.index, imp.values, color=TEAL)
            ax.bar_label(ax.containers[0], fmt="%.2f", color=FG, fontsize=8, padding=3)
            return self._save_fig(fig, "feature_importance.png")

        attempt("model_comparison", model_comparison)
        attempt("type_split", type_split)
        attempt("titles_per_year", titles_per_year)
        attempt("top_countries", top_countries)
        attempt("genre_shift", genre_shift)
        attempt("feature_importance", feature_importance)

        logging.info(f"Task 6 charts saved: {saved}")
        print(f"[task6] saved {len(saved)} chart(s) to {REPORT_DIR}/")
        return saved

    def run(self):
        try:
            df = self.dt.initiate_data_transformation()
            comparison_df = self.run_model_comparison(df)
            insights = self.generate_insights(df)
            self.generate_visual_reports(df, comparison_df)
            return comparison_df, insights
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    engine = SuccessAnalyticsEngine()
    engine.run()