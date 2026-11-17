# Dataset notes

`student_performance.csv` contains 1,500 reproducibly generated synthetic student records. It is created by `main.py` using NumPy's default random generator with seed 42.

The dataset is not an official UCI dataset and does not represent real identifiable students. Its purpose is to demonstrate a complete tabular classification workflow in a self-contained educational project.

The `final_score` field is used only to derive `performance_level`; the model trains on the 12 predictor columns defined in `main.py`, not on `final_score` or `performance_level`.
