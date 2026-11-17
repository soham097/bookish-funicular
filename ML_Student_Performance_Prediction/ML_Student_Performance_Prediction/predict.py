"""Predict a student's performance category from a CSV input."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
FEATURES = [
    "weekly_study_hours", "attendance_percent", "previous_exam_score",
    "assignment_average", "quiz_average", "sleep_hours", "class_participation",
    "parental_support_level", "tutoring", "internet_access",
    "extracurricular_activity", "travel_time_minutes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict student performance level from a CSV file.")
    parser.add_argument("--input", default="outputs/sample_input.csv", help="CSV with one or more student rows")
    parser.add_argument("--output", default="outputs/predictions.csv", help="Where to save predictions")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model is missing. Run `python main.py` first.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    frame = pd.read_csv(input_path)
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Input CSV is missing required columns: " + ", ".join(missing))
    model = joblib.load(MODEL_PATH)
    frame["predicted_performance_level"] = model.predict(frame[FEATURES])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(frame[["predicted_performance_level"]].to_string(index=False))
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
