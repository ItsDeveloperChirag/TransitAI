
import csv
import random
from datetime import date, timedelta
from pathlib import Path

# Reproducible data: running this code again produces the same dataset.
random.seed(42)

# Project and output paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "delhi_bus_demand.csv"

# Fictional routes and stops for our Delhi-based prototype
routes = {
    "D01": {
        "route_name": "Dwarka - Connaught Place",
        "stops": [
            "Dwarka Sector 10",
            "Palam",
            "Dhaula Kuan",
            "AIIMS",
            "Karol Bagh",
            "Connaught Place",
        ],
        "base_demand": 28,
    },
    "D02": {
        "route_name": "Rohini - Kashmere Gate",
        "stops": [
            "Rohini Sector 18",
            "Pitampura",
            "Netaji Subhash Place",
            "Shalimar Bagh",
            "Azadpur",
            "Kashmere Gate",
        ],
        "base_demand": 24,
    },
    "D03": {
        "route_name": "Saket - Anand Vihar",
        "stops": [
            "Saket",
            "Hauz Khas",
            "INA",
            "Lajpat Nagar",
            "ITO",
            "Anand Vihar",
        ],
        "base_demand": 26,
    },
}

time_slots = [
    "07:00", "08:00", "09:00", "10:00",
    "11:00", "12:00", "13:00", "14:00",
    "15:00", "16:00", "17:00", "18:00",
]

bus_capacity = 50

fieldnames = [
    "date",
    "day_of_week",
    "is_weekend",
    "city",
    "route_id",
    "route_name",
    "stop_id",
    "stop_name",
    "time_slot",
    "hour",
    "passenger_count",
    "bus_capacity",
    "scheduled_buses",
    "total_bus_capacity",
    "avg_wait_time_min",
    "traffic_level",
    "weather",
]

# Synthetic observations cover 30 days.
start_date = date(2026, 9, 1)
number_of_days = 30
rows_written = 0

with CSV_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    for day_offset in range(number_of_days):
        current_date = start_date + timedelta(days=day_offset)

        day_name = current_date.strftime("%A")
        is_weekend = current_date.weekday() >= 5

        for route_id, route in routes.items():
            for stop_index, stop_name in enumerate(route["stops"]):
                stop_id = f"{route_id}_S{stop_index + 1:02d}"

                for time_slot in time_slots:
                    hour = int(time_slot.split(":")[0])

                    # Higher passenger demand during rush hours.
                    if hour in (8, 9):
                        time_factor = 1.8
                    elif hour in (17, 18):
                        time_factor = 1.7
                    elif hour in (7, 16):
                        time_factor = 1.3
                    elif hour in (12, 13):
                        time_factor = 1.0
                    else:
                        time_factor = 0.65

                    # Weekend demand is lower in this simulation.
                    day_factor = 0.65 if is_weekend else 1.0

                    # Stops near the middle/end of routes may be busier.
                    stop_factor = 0.8 + (stop_index * 0.08)

                    traffic_level = random.choices(
                        ["Low", "Medium", "High"],
                        weights=[25, 45, 30],
                    )[0]

                    weather = random.choices(
                        ["Clear", "Rainy", "Cloudy"],
                        weights=[65, 15, 20],
                    )[0]

                    traffic_factor = {
                        "Low": 0.95,
                        "Medium": 1.0,
                        "High": 1.10,
                    }[traffic_level]

                    weather_factor = 0.85 if weather == "Rainy" else 1.0

                    expected_demand = (
                        route["base_demand"]
                        * time_factor
                        * day_factor
                        * stop_factor
                        * traffic_factor
                        * weather_factor
                    )

                    passenger_count = max(
                        0,
                        round(expected_demand + random.gauss(0, 4))
                    )

                    # Simulated number of buses assigned in this hour.
                    scheduled_buses = (
                        3 if hour in (8, 9, 17, 18)
                        else 2 if hour in (7, 10, 16)
                        else 1
                    )

                    total_capacity = scheduled_buses * bus_capacity

                    # Estimate waiting time from demand and service capacity.
                    demand_ratio = passenger_count / max(total_capacity, 1)

                    avg_wait_time = round(
                        5
                        + demand_ratio * 10
                        + (4 if traffic_level == "High" else 0)
                        + random.uniform(0, 3),
                        1,
                    )

                    writer.writerow({
                        "date": current_date.isoformat(),
                        "day_of_week": day_name,
                        "is_weekend": is_weekend,
                        "city": "Delhi",
                        "route_id": route_id,
                        "route_name": route["route_name"],
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "time_slot": time_slot,
                        "hour": hour,
                        "passenger_count": passenger_count,
                        "bus_capacity": bus_capacity,
                        "scheduled_buses": scheduled_buses,
                        "total_bus_capacity": total_capacity,
                        "avg_wait_time_min": avg_wait_time,
                        "traffic_level": traffic_level,
                        "weather": weather,
                    })

                    rows_written += 1

print("TRANSITAI DATASET CREATED SUCCESSFULLY")
print("-" * 45)
print(f"City: Delhi, India (simulated)")
print(f"Routes: {len(routes)}")
print(f"Stops per route: 6")
print(f"Days: {number_of_days}")
print(f"Time slots per day: {len(time_slots)}")
print(f"Total records: {rows_written}")
print(f"CSV file: {CSV_FILE}")

# Verify the generated file using Python's CSV reader.
with CSV_FILE.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    sample_rows = [next(reader) for _ in range(5)]

print("\nFirst five records:")
for row in sample_rows:
    print(
        row["date"],
        row["route_id"],
        row["stop_name"],
        row["time_slot"],
        row["passenger_count"],
    )

print("\nVerification: CSV contains a header and five readable records.")
