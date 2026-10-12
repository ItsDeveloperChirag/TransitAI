from pathlib import Path
from io import BytesIO
import datetime
import math

import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# --------------------------------------------------
# Project paths and application setup
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "delhi_bus_demand.csv"
MODEL_FILE = PROJECT_DIR / "models" / "demand_model.pkl"

app = Flask(__name__)

df = pd.read_csv(DATA_FILE)
model = joblib.load(MODEL_FILE)

routes = (
    df[["route_id", "route_name"]]
    .drop_duplicates()
    .sort_values("route_id")
)

BUS_CAPACITY = 50
TARGET_OCCUPANCY = 0.80
CAPACITY_PER_BUS = BUS_CAPACITY * TARGET_OCCUPANCY


# --------------------------------------------------
# Home page: single passenger-demand prediction
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    recommended_buses = None
    error = None
    selected = {}

    route_id = routes.iloc[0]["route_id"]

    if request.method == "POST":
        try:
            route_id = request.form["route_id"]
            stop_id = request.form["stop_id"]
            hour = int(request.form["hour"])
            day_type = request.form["day_type"]
            traffic = request.form["traffic"]
            weather = request.form["weather"]

            valid_routes = routes["route_id"].tolist()
            if route_id not in valid_routes:
                raise ValueError("Please select a valid route.")

            valid_stops = df.loc[
                df["route_id"] == route_id, "stop_id"
            ].unique().tolist()

            if stop_id not in valid_stops:
                raise ValueError("Please select a valid stop for this route.")

            if not 7 <= hour <= 18:
                raise ValueError("Hour must be between 7 and 18.")

            if day_type not in ["Weekday", "Weekend"]:
                raise ValueError("Invalid day type.")

            if traffic not in ["Low", "Medium", "High"]:
                raise ValueError("Invalid traffic level.")

            if weather not in ["Clear", "Cloudy", "Rainy"]:
                raise ValueError("Invalid weather condition.")

            sample = pd.DataFrame([{
                "route_id": route_id,
                "stop_id": stop_id,
                "time_slot": f"{hour:02d}:00",
                "traffic_level": traffic,
                "weather": weather,
                "hour": hour,
                "is_weekend": 1 if day_type == "Weekend" else 0,
            }])

            prediction = round(max(0, float(model.predict(sample)[0])))

            recommended_buses = max(
                1,
                math.ceil(prediction / CAPACITY_PER_BUS)
            )

            selected = {
                "route_id": route_id,
                "stop_id": stop_id,
                "stop_name": df.loc[
                    (df["route_id"] == route_id)
                    & (df["stop_id"] == stop_id),
                    "stop_name"
                ].iloc[0],
                "route_name": routes.loc[
                    routes["route_id"] == route_id, "route_name"
                ].iloc[0],
                "hour": f"{hour:02d}:00",
                "day_type": day_type,
                "traffic": traffic,
                "weather": weather,
            }

        except (KeyError, ValueError) as exc:
            error = str(exc)

    available_stops = (
        df.loc[df["route_id"] == route_id, ["stop_id", "stop_name"]]
        .drop_duplicates()
        .sort_values("stop_id")
    )

    return render_template(
        "index.html",
        routes=routes.to_dict("records"),
        stops=available_stops.to_dict("records"),
        selected=selected,
        prediction=prediction,
        recommended_buses=recommended_buses,
        error=error,
        default_route=route_id,
    )


# --------------------------------------------------
# API: return stops for a selected route
# --------------------------------------------------
@app.route("/stops")
def get_stops():
    route_id = request.args.get("route_id", "")

    stops = (
        df.loc[df["route_id"] == route_id, ["stop_id", "stop_name"]]
        .drop_duplicates()
        .sort_values("stop_id")
    )

    return jsonify(stops.to_dict("records"))


# --------------------------------------------------
# Analytics dashboard
# --------------------------------------------------
@app.route("/dashboard")
def dashboard():
    # Average demand by route
    route_stats = (
        df.groupby("route_name")["passenger_count"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )

    # Identify the busiest bus stop
    all_stop_stats = (
        df.groupby("stop_name")["passenger_count"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )

    busiest_stop = all_stop_stats.index[0]
    busiest_stop_demand = float(all_stop_stats.iloc[0])

    priority_route = route_stats.index[0]
    priority_demand = float(route_stats.iloc[0])

    # Average demand by hour
    hourly_stats = (
        df.groupby("hour")["passenger_count"]
        .mean()
        .round(2)
        .sort_index()
    )

    # Average demand by day type
    day_labels = ["Weekday", "Weekend"]
    day_values = []

    for is_weekend in [False, True]:
        average = df.loc[
            df["is_weekend"].astype(str).str.lower()
            == str(is_weekend).lower(),
            "passenger_count"
        ].mean()
        day_values.append(round(float(average), 2))

    # Five busiest stops for the dashboard chart
    top_stops = (
        df.groupby("stop_name")["passenger_count"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .sort_values()
        .round(2)
    )

    peak_hour = int(hourly_stats.idxmax())
    peak_demand = float(hourly_stats.max())

    # Passenger demand distribution
    low_demand = int((df["passenger_count"] < 20).sum())

    moderate_demand = int(
        (
            (df["passenger_count"] >= 20)
            & (df["passenger_count"] <= 35)
        ).sum()
    )

    high_demand = int((df["passenger_count"] > 35).sum())
    total_observations = len(df)

    demand_summary = [
        {
            "level": "Low",
            "count": low_demand,
            "percentage": round(low_demand / total_observations * 100, 2),
        },
        {
            "level": "Moderate",
            "count": moderate_demand,
            "percentage": round(
                moderate_demand / total_observations * 100, 2
            ),
        },
        {
            "level": "High",
            "count": high_demand,
            "percentage": round(high_demand / total_observations * 100, 2),
        },
    ]
    # Actual vs predicted demand for dashboard
    X = df.drop(columns=["passenger_count"])
    y = df["passenger_count"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    actual_vs_predicted = model.predict(X_test)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    dashboard_mae = mean_absolute_error(y_test, actual_vs_predicted)
    dashboard_rmse = np.sqrt(
        mean_squared_error(y_test, actual_vs_predicted)
    )
    dashboard_r2 = r2_score(y_test, actual_vs_predicted)
    # Select 30 observations for a readable chart
    comparison_df = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": actual_vs_predicted,
    }).reset_index(drop=True)

    comparison_df = comparison_df.head(30)

    comparison_labels = list(
        range(1, len(comparison_df) + 1)
    )

    actual_values = comparison_df["actual"].tolist()
    # Route-wise average passenger demand
    route_demand = (
        df.groupby("route_name")["passenger_count"]
        .mean()
        .round(2)
    )

    route_labels = route_demand.index.tolist()
    route_values = route_demand.values.tolist()
    predicted_values = comparison_df["predicted"].round(2).tolist()
    return render_template(
        "dashboard.html",
        total_records=len(df),
        total_routes=df["route_id"].nunique(),
        total_stops=df["stop_name"].nunique(),
        average_demand=round(float(df["passenger_count"].mean()), 2),

        route_labels=route_stats.index.tolist(),
        route_values=route_stats.tolist(),
        priority_route=priority_route,
        priority_demand=round(priority_demand, 2),

        busiest_stop=busiest_stop,
        busiest_stop_demand=round(busiest_stop_demand, 2),

        demand_distribution_labels=["Low", "Moderate", "High"],
        demand_distribution_values=[
            low_demand,
            moderate_demand,
            high_demand,
        ],
        demand_summary=demand_summary,

        hour_labels=[f"{int(h):02d}:00" for h in hourly_stats.index],
        hour_values=hourly_stats.tolist(),
        peak_hour=f"{peak_hour:02d}:00",
        peak_demand=round(peak_demand, 2),

        day_labels=day_labels,
        day_values=day_values,

        stop_labels=top_stops.index.tolist(),
        stop_values=top_stops.tolist(),

        comparison_labels=comparison_labels,
        actual_values=actual_values,
        predicted_values=predicted_values,

        dashboard_mae=round(float(dashboard_mae), 3),
        dashboard_rmse=round(float(dashboard_rmse), 3),
        dashboard_r2=round(float(dashboard_r2), 4),
    )


# --------------------------------------------------
# Seven-day forecast page
# Uses peak hourly prediction for bus recommendations
# --------------------------------------------------
@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    forecast_data = []
    selected_route = request.values.get("route_id", "").strip()
    selected_stop = ""
    error = None

    route_options = (
        df[["route_id", "route_name"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    available_stops = []

    if request.method == "POST":
        selected_route = request.form.get("route_id", "").strip()
        selected_stop = request.form.get("stop_id", "").strip()

        route_data = df[df["route_id"] == selected_route]
        stop_data = route_data[route_data["stop_id"] == selected_stop]

        if route_data.empty or stop_data.empty:
            error = "Please select a valid route and stop."
        else:
            traffic = stop_data["traffic_level"].mode()[0]
            weather = stop_data["weather"].mode()[0]
            tomorrow = datetime.date.today() + datetime.timedelta(days=1)

            for i in range(7):
                forecast_date = tomorrow + datetime.timedelta(days=i)
                weekday = forecast_date.weekday()
                is_weekend = weekday >= 5

                hourly_predictions = []
                forecast_hours = list(range(7, 19))

                for hour in forecast_hours:
                    sample = pd.DataFrame([{
                        "day_of_week": weekday,
                        "is_weekend": int(is_weekend),
                        "route_id": selected_route,
                        "stop_id": selected_stop,
                        "time_slot": f"{hour:02d}:00",
                        "hour": hour,
                        "traffic_level": traffic,
                        "weather": weather,
                    }])

                    prediction = max(
                        0, float(model.predict(sample)[0])
                    )
                    hourly_predictions.append(prediction)

                avg_demand = round(
                    sum(hourly_predictions) / len(hourly_predictions), 2
                )
                peak_demand = round(max(hourly_predictions), 2)
                peak_hour = forecast_hours[
                    hourly_predictions.index(max(hourly_predictions))
                ]

                recommended_buses = max(
                    1,
                    math.ceil(peak_demand / CAPACITY_PER_BUS)
                )

                if peak_demand > CAPACITY_PER_BUS:
                    alert = (
                        "HIGH DEMAND - Additional bus capacity may be needed"
                    )
                else:
                    alert = "NORMAL - One bus may be sufficient"

                if peak_demand < 20:
                    demand_level = "Low"
                elif peak_demand <= 35:
                    demand_level = "Moderate"
                else:
                    demand_level = "High"

                forecast_data.append({
                    "date": forecast_date.isoformat(),
                    "day": forecast_date.strftime("%A"),
                    "average_passengers_per_hour": avg_demand,
                    "peak_passengers_per_hour": peak_demand,
                    "peak_hour": f"{peak_hour:02d}:00",
                    "recommended_buses_per_hour": recommended_buses,
                    "alert": alert,
                    "demand_level": demand_level,
                })

    if selected_route:
        available_stops = (
            df[df["route_id"] == selected_route][["stop_id", "stop_name"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )

    return render_template(
        "forecast.html",
        forecast_data=forecast_data,
        routes=route_options,
        available_stops=available_stops,
        selected_route=selected_route,
        selected_stop=selected_stop,
        error=error,
    )


# --------------------------------------------------
# Download the same seven-day forecast as a CSV
# Includes peak hour and peak-based bus recommendations
# --------------------------------------------------
@app.route("/download-forecast", methods=["POST"])
def download_forecast():
    selected_route = request.form.get("route_id", "").strip()
    selected_stop = request.form.get("stop_id", "").strip()

    route_data = df[df["route_id"] == selected_route]
    stop_data = route_data[route_data["stop_id"] == selected_stop]

    if route_data.empty or stop_data.empty:
        return "Invalid route or stop.", 400

    traffic = stop_data["traffic_level"].mode()[0]
    weather = stop_data["weather"].mode()[0]
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    rows = []

    for i in range(7):
        forecast_date = tomorrow + datetime.timedelta(days=i)
        weekday = forecast_date.weekday()
        is_weekend = weekday >= 5

        forecast_hours = list(range(7, 19))
        hourly_predictions = []

        for hour in forecast_hours:
            sample = pd.DataFrame([{
                "day_of_week": weekday,
                "is_weekend": int(is_weekend),
                "route_id": selected_route,
                "stop_id": selected_stop,
                "time_slot": f"{hour:02d}:00",
                "hour": hour,
                "traffic_level": traffic,
                "weather": weather,
            }])

            hourly_predictions.append(
                max(0, float(model.predict(sample)[0]))
            )

        avg_demand = round(
            sum(hourly_predictions) / len(hourly_predictions), 2
        )
        peak_demand = round(max(hourly_predictions), 2)
        peak_hour = forecast_hours[
            hourly_predictions.index(max(hourly_predictions))
        ]

        recommended_buses = max(
            1,
            math.ceil(peak_demand / CAPACITY_PER_BUS)
        )

        if peak_demand > CAPACITY_PER_BUS:
            alert = "HIGH DEMAND - Additional bus capacity may be needed"
        else:
            alert = "NORMAL - One bus may be sufficient"

        if peak_demand < 20:
            demand_level = "Low"
        elif peak_demand <= 35:
            demand_level = "Moderate"
        else:
            demand_level = "High"

        rows.append({
            "date": forecast_date.isoformat(),
            "day": forecast_date.strftime("%A"),
            "average_passengers_per_hour": avg_demand,
            "peak_passengers_per_hour": peak_demand,
            "peak_hour": f"{peak_hour:02d}:00",
            "demand_level": demand_level,
            "recommended_buses_per_hour": recommended_buses,
            "alert": alert,
        })

    output = pd.DataFrame(rows)
    csv_buffer = BytesIO(output.to_csv(index=False).encode("utf-8"))
    csv_buffer.seek(0)

    return send_file(
        csv_buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"forecast_{selected_route}_{selected_stop}.csv",
    )

# --------------------------------------------------
# Interactive route map
# --------------------------------------------------

# --------------------------------------------------
# Real-data GTFS route map
# --------------------------------------------------
@app.route("/map")
def route_map():
    gtfs_dir = PROJECT_DIR / "data" / "gtfs"

    stops_df = pd.read_csv(gtfs_dir / "stops.txt")
    routes_df = pd.read_csv(gtfs_dir / "routes.txt")
    trips_df = pd.read_csv(gtfs_dir / "trips.txt")
    stop_times_df = pd.read_csv(gtfs_dir / "stop_times.txt")

    # Clean column names
    for frame in [stops_df, routes_df, trips_df, stop_times_df]:
        frame.columns = frame.columns.str.strip()

    # Ensure stop coordinates are numeric
    stops_df["stop_lat"] = pd.to_numeric(
        stops_df["stop_lat"], errors="coerce"
    )
    stops_df["stop_lon"] = pd.to_numeric(
        stops_df["stop_lon"], errors="coerce"
    )
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])

    # Pick one trip per route to avoid drawing every repeated journey
    trip_routes = trips_df[["route_id", "trip_id"]].drop_duplicates(
        subset=["route_id"], keep="first"
    )

    route_stops = (
        trip_routes
        .merge(
            stop_times_df[["trip_id", "stop_id", "stop_sequence"]],
            on="trip_id",
            how="left"
        )
        .merge(
            stops_df[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
            on="stop_id",
            how="inner"
        )
        .merge(
            routes_df[["route_id", "route_short_name", "route_long_name"]],
            on="route_id",
            how="left"
        )
        .sort_values(["route_id", "stop_sequence"])
    )

    # Keep the map responsive by displaying up to 30 routes
    route_ids = route_stops["route_id"].drop_duplicates().head(30)
    route_stops = route_stops[
        route_stops["route_id"].isin(route_ids)
    ]

    map_routes = []

    for route_id, group in route_stops.groupby("route_id", sort=False):
        first = group.iloc[0]

        map_routes.append({
            "route_id": str(route_id),
            "route_name": (
                str(first.get("route_short_name", ""))
                + " - "
                + str(first.get("route_long_name", ""))
            ).strip(" -"),
            "stops": [
                {
                    "stop_id": str(row.stop_id),
                    "stop_name": str(row.stop_name),
                    "lat": float(row.stop_lat),
                    "lon": float(row.stop_lon),
                    "sequence": int(row.stop_sequence),
                }
                for row in group.itertuples()
            ],
        })

    return render_template("map.html", map_routes=map_routes)

if __name__ == "__main__":
    app.run(debug=True)
