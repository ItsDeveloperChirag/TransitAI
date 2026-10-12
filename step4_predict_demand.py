
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "delhi_bus_demand.csv"
MODEL_FILE = PROJECT_DIR / "models" / "demand_model.pkl"

print("=" * 55)
print("TRANSITAI - STEP 4: PASSENGER DEMAND PREDICTION")
print("=" * 55)

# Check that the required files exist
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

# Load dataset and trained model
df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

# Build route and stop options from the dataset
routes = (
    df[["route_id", "route_name"]]
    .drop_duplicates()
    .sort_values("route_id")
)

print("\nAvailable routes:")
for _, row in routes.iterrows():
    print(f"{row['route_id']} - {row['route_name']}")

route_id = input("\nEnter route ID (e.g. D01): ").strip().upper()

route_row = routes[routes["route_id"] == route_id]
if route_row.empty:
    raise ValueError("Invalid route ID. Please use a listed route.")

route_name = route_row.iloc[0]["route_name"]

stops = (
    df.loc[df["route_id"] == route_id, ["stop_id", "stop_name"]]
    .drop_duplicates()
    .sort_values("stop_id")
)

print(f"\nStops on {route_name}:")
for _, row in stops.iterrows():
    print(f"{row['stop_id']} - {row['stop_name']}")

stop_id = input("\nEnter stop ID from the list: ").strip().upper()

if stop_id not in stops["stop_id"].astype(str).values:
    raise ValueError("Invalid stop ID for the selected route.")

hour = int(input("Enter hour (7 to 18): "))
if hour < 7 or hour > 18:
    raise ValueError("Hour must be between 7 and 18.")

is_weekend = input("Is it a weekend? (yes/no): ").strip().lower()
if is_weekend not in ("yes", "no"):
    raise ValueError("Enter yes or no.")

traffic = input("Traffic level (Low/Medium/High): ").strip().capitalize()
if traffic not in ("Low", "Medium", "High"):
    raise ValueError("Traffic must be Low, Medium, or High.")

weather = input("Weather (Clear/Cloudy/Rainy): ").strip().capitalize()
if weather not in ("Clear", "Cloudy", "Rainy"):
    raise ValueError("Weather must be Clear, Cloudy, or Rainy.")

# Use a time-slot format matching the dataset
time_slot = f"{hour:02d}:00"

# Prepare input in the exact format used during training
sample = pd.DataFrame([{
    "route_id": route_id,
    "stop_id": stop_id,
    "time_slot": time_slot,
    "traffic_level": traffic,
    "weather": weather,
    "hour": hour,
    "is_weekend": 1 if is_weekend == "yes" else 0,
}])

predicted_demand = max(0, model.predict(sample)[0])

print("\n" + "-" * 40)
print("TRANSITAI PREDICTION RESULT")
print("-" * 40)
print(f"Route: {route_id} - {route_name}")
print(f"Stop: {stops.loc[stops['stop_id'] == stop_id, 'stop_name'].iloc[0]}")
print(f"Time: {time_slot}")
print(f"Day type: {'Weekend' if is_weekend == 'yes' else 'Weekday'}")
print(f"Traffic: {traffic}")
print(f"Weather: {weather}")
print(f"\nPredicted passenger count: {predicted_demand:.0f}")
print("\nSTEP 4 PREDICTION COMPLETED")
