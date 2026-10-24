import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prediction_error_analysis.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# Load prediction error report
df = pd.read_csv(DATA_PATH)

# Calculate mean absolute error for each stop
stop_error = (
    df.groupby(["route_id", "stop_name"])["absolute_error"]
    .mean()
    .reset_index()
    .sort_values("absolute_error", ascending=False)
)

stop_error.rename(
    columns={"absolute_error": "mean_absolute_error"},
    inplace=True
)

# Save stop-level analysis
output_path = OUTPUT_DIR / "stop_error_analysis.csv"
stop_error.to_csv(output_path, index=False)

# Display the 10 stops with the highest error
print("TOP 10 STOPS WITH HIGHEST PREDICTION ERROR")
print("------------------------------------------")
print(stop_error.head(10).to_string(index=False))

# Plot the top 10 stops
top_stops = stop_error.head(10).sort_values("mean_absolute_error")

plt.figure(figsize=(10, 6))
plt.barh(
    top_stops["stop_name"],
    top_stops["mean_absolute_error"],
    edgecolor="black"
)
plt.xlabel("Mean Absolute Error (Passengers)")
plt.ylabel("Bus Stop")
plt.title("Top 10 Stops by Prediction Error")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "step26_stop_error_analysis.png",
    dpi=300
)
plt.show()

print("\nFiles saved:")
print("- stop_error_analysis.csv")
print("- step26_stop_error_analysis.png")
