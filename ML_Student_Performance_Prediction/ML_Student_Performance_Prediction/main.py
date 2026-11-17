"""Student Performance Prediction mini-project.

Run from this directory with: python main.py
The bundled student records are synthetic and generated reproducibly.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TARGET = "performance_level"
FEATURES = [
    "weekly_study_hours",
    "attendance_percent",
    "previous_exam_score",
    "assignment_average",
    "quiz_average",
    "sleep_hours",
    "class_participation",
    "parental_support_level",
    "tutoring",
    "internet_access",
    "extracurricular_activity",
    "travel_time_minutes",
]
CLASS_ORDER = ["Needs Support", "Developing", "High Performer"]


def generate_dataset(path: Path = DATA_PATH, n: int = 1500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a reproducible, clearly labelled synthetic learning dataset."""
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(13.0, 6.0, n), 0, 40)
    attendance = np.clip(rng.normal(84, 10, n), 45, 100)
    previous = np.clip(rng.normal(66, 14, n), 20, 100)
    assignment = np.clip(0.62 * previous + 0.38 * (50 + study * 2.1) + rng.normal(0, 10, n), 0, 100)
    quiz = np.clip(0.55 * previous + 0.45 * (48 + study * 2.2) + rng.normal(0, 12, n), 0, 100)
    sleep = np.clip(rng.normal(7.0, 1.15, n), 3.5, 10)
    participation = np.clip(np.rint(3.0 + study / 7.5 + rng.normal(0, 1.5, n)), 1, 10).astype(int)
    parental = np.clip(np.rint(rng.normal(3.2, 1.15, n)), 1, 5).astype(int)
    tutoring = rng.binomial(1, np.clip(0.18 + (60 - previous) / 180, 0.08, 0.48), n)
    internet = rng.binomial(1, 0.88, n)
    extracurricular = rng.binomial(1, 0.45, n)
    travel = np.clip(rng.gamma(2.0, 14.0, n), 0, 100)

    # Construct a plausible academic score using only the generated inputs.
    score = (
        0.25 * previous
        + 0.22 * assignment
        + 0.18 * quiz
        + 0.16 * attendance
        + 0.19 * np.minimum(study * 2.5, 100)
        + 0.75 * (participation - 5)
        + 0.75 * (parental - 3)
        + 1.8 * tutoring
        + 0.65 * (sleep - 7)
        + 0.6 * internet
        - 0.025 * travel
        + rng.normal(0, 4.5, n)
    )
    final_score = np.clip(score, 0, 100)
    # Three performance bands: below 60, 60-74.99, and 75-100.
    levels = np.where(final_score < 60, CLASS_ORDER[0], np.where(final_score < 75, CLASS_ORDER[1], CLASS_ORDER[2]))
    df = pd.DataFrame({
        "weekly_study_hours": np.round(study, 1),
        "attendance_percent": np.round(attendance, 1),
        "previous_exam_score": np.round(previous, 1),
        "assignment_average": np.round(assignment, 1),
        "quiz_average": np.round(quiz, 1),
        "sleep_hours": np.round(sleep, 1),
        "class_participation": participation,
        "parental_support_level": parental,
        "tutoring": tutoring,
        "internet_access": internet,
        "extracurricular_activity": extracurricular,
        "travel_time_minutes": np.round(travel, 1),
        "final_score": np.round(final_score, 1),
        TARGET: levels,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])


def train_and_evaluate(df: pd.DataFrame) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Random Forest": build_pipeline(RandomForestClassifier(
            n_estimators=350, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )),
        "Extra Trees": build_pipeline(ExtraTreesClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )),
        "Gradient Boosting": build_pipeline(GradientBoostingClassifier(
            n_estimators=160, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        )),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = []
    fitted = {}
    for name, pipeline in models.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=None)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        comparison.append({
            "Model": name,
            "CV Macro F1 Mean": float(cv_scores.mean()),
            "CV Macro F1 Std": float(cv_scores.std()),
            "Test Accuracy": float(accuracy_score(y_test, preds)),
            "Test Macro F1": float(f1_score(y_test, preds, average="macro")),
        })
        fitted[name] = pipeline

    comparison_df = pd.DataFrame(comparison).sort_values("CV Macro F1 Mean", ascending=False)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = str(comparison_df.iloc[0]["Model"])
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    metrics = {
        "project": "Student Performance Prediction",
        "dataset_type": "synthetic educational data generated by this project",
        "records": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "class_order": CLASS_ORDER,
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "selected_model": best_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
        "classification_report": classification_report(y_test, predictions, labels=CLASS_ORDER, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASS_ORDER).tolist(),
        "class_distribution": df[TARGET].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int).to_dict(),
        "model_comparison": comparison,
        "random_state": RANDOM_STATE,
        "test_size": 0.25,
        "cross_validation_folds": 5,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, predictions, labels=CLASS_ORDER, zero_division=0)
    )
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps({
        "selected_model": best_name,
        "features": FEATURES,
        "class_order": CLASS_ORDER,
        "target": TARGET,
    }, indent=2))

    # Persist an untouched test sample and its prediction for quick verification.
    sample = X_test.iloc[[0]].copy()
    sample.to_csv(OUTPUT_DIR / "sample_input.csv", index=False)
    (OUTPUT_DIR / "sample_prediction.txt").write_text(
        f"Predicted performance level: {best_model.predict(sample)[0]}\n"
    )

    # Visual outputs.
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Distribution of Student Performance Levels")
    ax.set_xlabel("Performance level")
    ax.set_ylabel("Number of students")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    cm = np.array(metrics["confusion_matrix"])
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    plot_df = comparison_df.sort_values("CV Macro F1 Mean")
    ax.barh(plot_df["Model"], plot_df["CV Macro F1 Mean"], xerr=plot_df["CV Macro F1 Std"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("5-fold cross-validation macro F1")
    ax.set_title("Model Comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Model-agnostic permutation importance on held-out data.
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10,
                                  random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1)
    importance = pd.DataFrame({"Feature": FEATURES, "Importance Mean": perm.importances_mean,
                               "Importance Std": perm.importances_std}).sort_values("Importance Mean", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    imp_plot = importance.sort_values("Importance Mean")
    ax.barh(imp_plot["Feature"], imp_plot["Importance Mean"], xerr=imp_plot["Importance Std"])
    ax.set_xlabel("Decrease in macro F1 after shuffling")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close(fig)
    return metrics


def main() -> None:
    df = generate_dataset()
    metrics = train_and_evaluate(df)
    print("Student Performance Prediction completed successfully")
    print(f"Records: {metrics['records']} | Features: {len(FEATURES)}")
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
