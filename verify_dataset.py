
import csv
from pathlib import Path

# Locate the generated CSV
project_dir = Path(__file__).resolve().parent
csv_file = project_dir / "data" / "delhi_bus_demand.csv"

expected_columns = [
    "date", "day_of_week", "is_weekend", "city",
    "route_id", "route_name", "stop_id", "stop_name",
    "time_slot", "hour", "passenger_count",
    "bus_capacity", "scheduled_buses", "total_bus_capacity",
    "avg_wait_time_min", "traffic_level", "weather"
]

print("TRANSITAI DATASET VERIFICATION")
print("=" * 40)

# Check 1: Does the CSV exist?
if not csv_file.exists():
    print("FAIL: CSV file not found.")
    print("Expected location:", csv_file)
    raise SystemExit(1)

print("PASS: CSV file exists.")

# Check 2: Read and verify the file
with csv_file.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    actual_columns = reader.fieldnames
    rows = list(reader)

# Check 3: Verify column names
if actual_columns == expected_columns:
    print("PASS: All 17 expected columns are present.")
else:
    print("FAIL: Column mismatch.")
    print("Expected:", expected_columns)
    print("Found:", actual_columns)

# Check 4: Verify record count
print(f"Total data records: {len(rows)}")

if len(rows) == 6480:
    print("PASS: Record count is correct.")
else:
    print(f"FAIL: Expected 6480 records, found {len(rows)}.")

# Check 5: Verify the main fields contain data
required_fields = [
    "date", "route_id", "stop_name",
    "time_slot", "passenger_count"
]

if rows and all(
    row.get(field, "").strip()
    for row in rows
    for field in required_fields
):
    print("PASS: Required fields are populated.")
else:
    print("FAIL: Empty required fields found.")

# Check 6: Show a small sample
print("\nFirst three records:")
for row in rows[:3]:
    print(
        row["date"],
        row["route_id"],
        row["stop_name"],
        row["time_slot"],
        row["passenger_count"]
    )

# Overall result
if (
    actual_columns == expected_columns
    and len(rows) == 6480
    and rows
    and all(
        row.get(field, "").strip()
        for row in rows
        for field in required_fields
    )
):
    print("\nFINAL RESULT: ALL CHECKS PASSED")
else:
    print("\nFINAL RESULT: VERIFICATION INCOMPLETE")
