
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "delhi_bus_demand.csv"
MODEL_FILE = PROJECT_DIR / "models" / "demand_model.pkl"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data and the original trained model
df = pd.read_csv(DATA_FILE)
saved_model = joblib.load(MODEL_FILE)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

features = [
    "day_of_week",
    "is_weekend",
    "route_id",
    "stop_id",
    "time_slot",
    "hour",
    "traffic_level",
    "weather"
]

target = "passenger_count"

X = df[features].copy()
y = df[target].copy()

# Match the preprocessing used in Step 20
categorical_columns = [
    "route_id",
    "stop_id",
    "time_slot",
    "traffic_level",
    "weather"
]

for column in categorical_columns:
    X[column] = X[column].astype(str)

# Use the earliest 80% of dates for training
# and the latest 20% for testing.
unique_dates = sorted(df["date"].unique())
split_index = int(len(unique_dates) * 0.80)

if split_index == 0 or split_index >= len(unique_dates):
    raise ValueError("Not enough distinct dates for temporal evaluation.")

split_date = unique_dates[split_index]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("===== TEMPORAL EVALUATION =====")
print(f"Training ends before: {split_date}")
print(f"Training observations: {len(X_train)}")
print(f"Testing observations: {len(X_test)}")
print(f"Testing starts: {df.loc[test_mask, 'date'].min().date()}")
print(f"Testing ends: {df.loc[test_mask, 'date'].max().date()}")

# Clone the saved model so the original model stays unchanged.
# Fit only on earlier dates.
temporal_model = clone(saved_model)
temporal_model.fit(X_train, y_train)

predictions = temporal_model.predict(X_test)

# Calculate model metrics
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Baseline uses only the training-period mean
baseline_predictions = [y_train.mean()] * len(y_test)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)
baseline_rmse = mean_squared_error(
    y_test, baseline_predictions
) ** 0.5
baseline_r2 = r2_score(
    y_test, baseline_predictions
)

print("\nTemporal Random Forest:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.4f}")

print("\nTraining-mean baseline:")
print(f"MAE  : {baseline_mae:.3f}")
print(f"RMSE : {baseline_rmse:.3f}")
print(f"R²   : {baseline_r2:.4f}")

# Save actual and predicted values
results = pd.DataFrame({
    "date": df.loc[test_mask, "date"].dt.strftime("%Y-%m-%d").to_numpy(),
    "actual_passengers": y_test.to_numpy(),
    "predicted_passengers": predictions
})

results["residual"] = (
    results["actual_passengers"]
    - results["predicted_passengers"]
)

results.to_csv(
    OUTPUT_DIR / "temporal_evaluation_results.csv",
    index=False
)

# Plot actual versus predicted demand
plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.5)

lower = min(y_test.min(), predictions.min())
upper = max(y_test.max(), predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--"
)

plt.xlabel("Actual Passenger Count")
plt.ylabel("Predicted Passenger Count")
plt.title("TransitAI: Future-Date Test")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temporal_actual_vs_predicted.png",
    dpi=150
)
plt.close()

# Plot residual distribution
plt.figure(figsize=(7, 5))
plt.hist(results["residual"], bins=30)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Observations")
plt.title("TransitAI: Future-Date Prediction Errors")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temporal_residuals.png",
    dpi=150
)
plt.close()

print("\nTemporal evaluation completed.")
print("Results saved in the outputs folder.")
