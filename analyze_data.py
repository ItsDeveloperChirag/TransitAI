
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Locate the dataset
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "delhi_bus_demand.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load the dataset
df = pd.read_csv(DATA_FILE)

# 3. Validate the dataset
required_columns = [
    "route_id",
    "route_name",
    "stop_name",
    "hour",
    "passenger_count",
    "is_weekend",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df.empty:
    raise ValueError("The dataset is empty.")

df["passenger_count"] = pd.to_numeric(
    df["passenger_count"], errors="raise"
)

print("=" * 55)
print("TRANSITAI - STEP 2: DATA ANALYSIS")
print("=" * 55)

print(f"\nTotal records: {len(df)}")
print(f"Unique routes: {df['route_id'].nunique()}")
print(f"Unique stops: {df['stop_name'].nunique()}")
print(
    f"Average passenger count per observation: "
    f"{df['passenger_count'].mean():.2f}"
)

# 4. Route-wise passenger demand
route_demand = (
    df.groupby(["route_id", "route_name"])["passenger_count"]
    .mean()
    .sort_values(ascending=False)
)

print("\n1. Average passenger count by route:")
print(route_demand.round(2).to_string())

# 5. Hour-wise passenger demand
hour_demand = (
    df.groupby("hour")["passenger_count"]
    .mean()
    .sort_index()
)

peak_hour = hour_demand.idxmax()

print("\n2. Average passenger count by hour:")
print(hour_demand.round(2).to_string())
print(f"\nPeak demand hour: {peak_hour}:00")

# 6. Weekday versus weekend
df["day_group"] = df["is_weekend"].map({
    True: "Weekend",
    False: "Weekday",
    "True": "Weekend",
    "False": "Weekday",
})

day_demand = df.groupby("day_group")["passenger_count"].mean()

print("\n3. Average passenger count by day type:")
print(day_demand.round(2).to_string())

# 7. Busiest stops
stop_demand = (
    df.groupby("stop_name")["passenger_count"]
    .mean()
    .sort_values(ascending=False)
)

print("\n4. Top five busiest stops by average demand:")
print(stop_demand.head(5).round(2).to_string())

# 8. Graph 1: Route comparison
route_demand.sort_values().plot(
    kind="barh",
    title="Average Passenger Demand by Route",
    xlabel="Average passenger count per observation",
    ylabel="Route",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "route_demand.png", dpi=150)
plt.show()
plt.close()

# 9. Graph 2: Hourly demand
hour_demand.plot(
    kind="line",
    marker="o",
    title="Average Passenger Demand by Hour",
    xlabel="Hour of day",
    ylabel="Average passenger count",
    grid=True,
)
plt.xticks(range(7, 19))
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hourly_demand.png", dpi=150)
plt.show()
plt.close()

# 10. Graph 3: Weekday versus weekend
day_demand.reindex(["Weekday", "Weekend"]).plot(
    kind="bar",
    title="Weekday vs Weekend Demand",
    xlabel="Day type",
    ylabel="Average passenger count",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "day_type_demand.png", dpi=150)
plt.show()
plt.close()

# 11. Graph 4: Busiest stops
stop_demand.head(5).sort_values().plot(
    kind="barh",
    title="Top Five Stops by Average Demand",
    xlabel="Average passenger count",
    ylabel="Bus stop",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "busiest_stops.png", dpi=150)
plt.show()
plt.close()

print("\nGraphs saved in:", OUTPUT_DIR)
print("STEP 2 ANALYSIS COMPLETED")
