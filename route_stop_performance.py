import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prediction_error_analysis.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load prediction error data
df = pd.read_csv(DATA_PATH)

# Calculate route-level performance
route_summary = (
    df.groupby("route_id")
    .agg(
        total_predictions=("absolute_error", "count"),
        mean_absolute_error=("absolute_error", "mean"),
        average_actual_demand=("actual_passengers", "mean"),
        average_predicted_demand=("predicted_passengers", "mean")
    )
    .reset_index()
)

route_summary["mean_absolute_error"] = (
    route_summary["mean_absolute_error"].round(3)
)
route_summary["average_actual_demand"] = (
    route_summary["average_actual_demand"].round(2)
)
route_summary["average_predicted_demand"] = (
    route_summary["average_predicted_demand"].round(2)
)

route_summary = route_summary.sort_values(
    "mean_absolute_error", ascending=True
)

# Calculate stop-level performance
stop_summary = (
    df.groupby(["route_id", "stop_id", "stop_name"])
    .agg(
        total_predictions=("absolute_error", "count"),
        mean_absolute_error=("absolute_error", "mean"),
        average_actual_demand=("actual_passengers", "mean"),
        average_predicted_demand=("predicted_passengers", "mean")
    )
    .reset_index()
)

stop_summary["mean_absolute_error"] = (
    stop_summary["mean_absolute_error"].round(3)
)
stop_summary["average_actual_demand"] = (
    stop_summary["average_actual_demand"].round(2)
)
stop_summary["average_predicted_demand"] = (
    stop_summary["average_predicted_demand"].round(2)
)

stop_summary = stop_summary.sort_values(
    "mean_absolute_error", ascending=True
)

# Save both summaries
route_summary.to_csv(
    OUTPUT_DIR / "step27_route_performance.csv", index=False
)

stop_summary.to_csv(
    OUTPUT_DIR / "step27_stop_performance.csv", index=False
)

# Print route performance
print("ROUTE PERFORMANCE SUMMARY")
print("--------------------------")
print(route_summary.to_string(index=False))

# Print best and worst performing stops
print("\nTOP 5 STOPS WITH LOWEST PREDICTION ERROR")
print("----------------------------------------")
print(stop_summary.head(5).to_string(index=False))

print("\nTOP 5 STOPS WITH HIGHEST PREDICTION ERROR")
print("-----------------------------------------")
print(
    stop_summary.sort_values(
        "mean_absolute_error", ascending=False
    ).head(5).to_string(index=False)
)

# Plot route-level MAE
plt.figure(figsize=(8, 5))
plt.bar(
    route_summary["route_id"],
    route_summary["mean_absolute_error"],
    edgecolor="black"
)
plt.xlabel("Route ID")
plt.ylabel("Mean Absolute Error (Passengers)")
plt.title("Prediction Error Comparison Across Routes")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "step27_route_performance.png",
    dpi=300
)
plt.show()

print("\nFiles created successfully:")
print("- outputs/step27_route_performance.csv")
print("- outputs/step27_stop_performance.csv")
print("- outputs/step27_route_performance.png")
