
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prediction_error_analysis.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load error report
df = pd.read_csv(DATA_PATH)

# 1. Distribution of prediction errors
plt.figure(figsize=(8, 5))
plt.hist(df["prediction_error"], bins=25, edgecolor="black")
plt.axvline(0, linestyle="--")
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Number of Observations")
plt.title("Distribution of Passenger Demand Prediction Errors")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step25_error_distribution.png", dpi=300)
plt.show()

# 2. Absolute error by hour
hourly_error = df.groupby("hour")["absolute_error"].mean()

plt.figure(figsize=(8, 5))
hourly_error.plot(kind="bar", edgecolor="black")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Absolute Error (Passengers)")
plt.title("Prediction Error by Hour")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step25_error_by_hour.png", dpi=300)
plt.show()

# 3. Absolute error by route
route_error = df.groupby("route_id")["absolute_error"].mean()

plt.figure(figsize=(8, 5))
route_error.plot(kind="bar", edgecolor="black")
plt.xlabel("Route")
plt.ylabel("Mean Absolute Error (Passengers)")
plt.title("Prediction Error by Route")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "step25_error_by_route.png", dpi=300)
plt.show()

print("Error visualizations created successfully:")
print("- step25_error_distribution.png")
print("- step25_error_by_hour.png")
print("- step25_error_by_route.png")
