
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "delhi_bus_demand.csv"
MODEL_PATH = BASE_DIR / "models" / "demand_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset and trained model
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

# Prepare features and target
X = df.drop(columns=["passenger_count"])
y = df["passenger_count"]

# Use the same random split settings as the initial evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Generate predictions
predictions = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("ACTUAL VS PREDICTED DEMAND")
print("--------------------------")
print(f"Test observations: {len(y_test)}")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.4f}")

# Save actual and predicted values
comparison = pd.DataFrame({
    "actual_passengers": y_test.to_numpy(),
    "predicted_passengers": predictions
})

comparison.to_csv(
    OUTPUT_DIR / "actual_vs_predicted_data.csv",
    index=False
)

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)

# Reference line: perfect predictions
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Passenger Count")
plt.ylabel("Predicted Passenger Count")
plt.title("Actual vs Predicted Passenger Demand")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "step23_actual_vs_predicted.png",
    dpi=300
)
plt.show()

print("\nFiles saved in the outputs folder:")
print("- actual_vs_predicted_data.csv")
print("- step23_actual_vs_predicted.png")
