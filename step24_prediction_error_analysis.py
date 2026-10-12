
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "delhi_bus_demand.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load data and model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Prepare features and target
X = df.drop(columns=["passenger_count"])
y = df["passenger_count"]

# Reproduce the same test split as Step 23
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Predict demand
predictions = model.predict(X_test)

# Build error analysis report
report = X_test.copy()
report["actual_passengers"] = y_test.to_numpy()
report["predicted_passengers"] = np.round(predictions, 2)
report["prediction_error"] = (
    report["predicted_passengers"] - report["actual_passengers"]
)
report["absolute_error"] = report["prediction_error"].abs()

report["prediction_status"] = np.where(
    report["prediction_error"] > 0,
    "Overestimated",
    np.where(
        report["prediction_error"] < 0,
        "Underestimated",
        "Exact"
    )
)

# Save report
output_path = OUTPUT_DIR / "prediction_error_analysis.csv"
report.to_csv(output_path, index=False)

# Display summary
print("PREDICTION ERROR ANALYSIS")
print("-------------------------")
print(f"Total observations: {len(report)}")
print(f"Mean absolute error: {report['absolute_error'].mean():.3f}")

print("\nPrediction status:")
print(report["prediction_status"].value_counts())

print("\nLargest prediction errors:")
print(
    report.nlargest(10, "absolute_error")[
        [
            "route_id",
            "stop_name",
            "hour",
            "actual_passengers",
            "predicted_passengers",
            "prediction_error",
            "prediction_status"
        ]
    ].to_string(index=False)
)

print(f"\nReport saved to: {output_path}")
