
import pandas as pd
import joblib
from datetime import date, timedelta

# Load the existing dataset and trained model
df = pd.read_csv("data/delhi_bus_demand.csv")
model = joblib.load("models/demand_model.pkl")

BUS_CAPACITY = 50
TARGET_OCCUPANCY = 0.80

# Select a route and stop
print("\nAvailable routes:")
routes = df[["route_id", "route_name"]].drop_duplicates()
print(routes.to_string(index=False))

route_id = input("\nEnter route ID (e.g., D01): ").strip().upper()

route_data = df[df["route_id"] == route_id]

if route_data.empty:
    print("Invalid route ID.")
    raise SystemExit

print("\nAvailable stops:")
stops = route_data[["stop_id", "stop_name"]].drop_duplicates()
print(stops.to_string(index=False))

stop_id = input("\nEnter stop ID: ").strip().upper()

stop_data = route_data[route_data["stop_id"] == stop_id]

if stop_data.empty:
    print("Invalid stop ID for this route.")
    raise SystemExit

# Use the selected stop's typical traffic and weather
traffic = stop_data["traffic_level"].mode()[0]
weather = stop_data["weather"].mode()[0]

# Forecast the next 7 days, starting tomorrow
forecast_rows = []
tomorrow = date.today() + timedelta(days=1)

for i in range(7):
    forecast_date = tomorrow + timedelta(days=i)
    weekday = forecast_date.weekday()
    is_weekend = weekday >= 5

    # Forecast the average hourly demand across operating hours
    hourly_predictions = []

    for hour in range(7, 19):
        sample = pd.DataFrame([{
            "day_of_week": weekday,
            "is_weekend": int(is_weekend),
            "route_id": route_id,
            "stop_id": stop_id,
            "time_slot": f"{hour:02d}:00",
            "hour": hour,
            "traffic_level": traffic,
            "weather": weather
        }])

        prediction = max(0, float(model.predict(sample)[0]))
        hourly_predictions.append(prediction)

    # Average passengers per hourly time slot
    avg_demand = round(
        sum(hourly_predictions) / len(hourly_predictions), 2
    )

    buses = max(
        1,
        int(
            (avg_demand + BUS_CAPACITY * TARGET_OCCUPANCY - 1)
            // (BUS_CAPACITY * TARGET_OCCUPANCY)
        )
    )

    forecast_rows.append({
        "date": forecast_date.isoformat(),
        "day": forecast_date.strftime("%A"),
        "average_passengers_per_hour": avg_demand,
        "recommended_buses_per_hour": buses
    })

forecast = pd.DataFrame(forecast_rows)

print("\n7-DAY DEMAND FORECAST")
print(forecast.to_string(index=False))

forecast.to_csv("outputs/7_day_forecast.csv", index=False)
print("\nForecast saved to outputs/7_day_forecast.csv")
