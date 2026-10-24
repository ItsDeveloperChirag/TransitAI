
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

# Load dataset and trained model
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# Use the same input features as the training script
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

# Match the preprocessing used by the original model.
# Route and stop IDs are categorical identifiers.
categorical_columns = [
    "route_id",
    "stop_id",
    "time_slot",
    "traffic_level",
    "weather"
]

for column in categorical_columns:
    X[column] = X[column].astype(str)

# Use the same random split as the original training script.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Predict test-set passenger counts
predictions = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

# Baseline: always predict the mean of the training targets
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

print("\n===== TRANSITAI MODEL EVALUATION =====")
print(f"Test observations: {len(y_test)}")

print("\nRandom Forest:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.4f}")

print("\nMean-prediction baseline:")
print(f"MAE  : {baseline_mae:.3f}")
print(f"RMSE : {baseline_rmse:.3f}")
print(f"R²   : {baseline_r2:.4f}")

# Save actual and predicted values
results = pd.DataFrame({
    "actual_passengers": y_test.to_numpy(),
    "predicted_passengers": predictions
})

results["residual"] = (
    results["actual_passengers"]
    - results["predicted_passengers"]
)

results.to_csv(
    OUTPUT_DIR / "model_evaluation_results.csv",
    index=False
)

# Plot 1: Actual vs predicted
plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.5)

min_value = min(y_test.min(), predictions.min())
max_value = max(y_test.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Passenger Count")
plt.ylabel("Predicted Passenger Count")
plt.title("TransitAI: Actual vs Predicted Demand")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=150
)
plt.close()

# Plot 2: Residual distribution
plt.figure(figsize=(7, 5))
plt.hist(results["residual"], bins=30)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Number of Observations")
plt.title("TransitAI: Prediction Error Distribution")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residual_distribution.png",
    dpi=150
)
plt.close()

# Plot 3: Actual vs predicted for first 100 test rows
plot_data = results.head(100)

plt.figure(figsize=(10, 5))
plt.plot(
    range(len(plot_data)),
    plot_data["actual_passengers"],
    label="Actual"
)
plt.plot(
    range(len(plot_data)),
    plot_data["predicted_passengers"],
    label="Predicted"
)

plt.xlabel("Test Observation")
plt.ylabel("Passenger Count")
plt.title("TransitAI: Actual vs Predicted Test Observations")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "prediction_comparison.png",
    dpi=150
)
plt.close()

print("\nEvaluation completed.")
print(f"Results saved in: {OUTPUT_DIR}")
print("Generated files:")
print("- model_evaluation_results.csv")
print("- actual_vs_predicted.png")
print("- residual_distribution.png")
print("- prediction_comparison.png")
