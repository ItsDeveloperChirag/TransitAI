
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "delhi_bus_demand.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "demand_model.pkl"

# Load dataset
df = pd.read_csv(DATA_FILE)

print("=" * 55)
print("TRANSITAI - STEP 3: MACHINE LEARNING")
print("=" * 55)

# Features used for prediction
categorical_features = [
    "route_id",
    "stop_id",
    "time_slot",
    "traffic_level",
    "weather",
]

numerical_features = [
    "hour",
    "is_weekend",
]

target = "passenger_count"

# Validate required columns
required = categorical_features + numerical_features + [target]
missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[required].isnull().any().any():
    raise ValueError("Missing values found in model columns.")

# Prepare features and target
X = df[categorical_features + numerical_features].copy()
y = df[target].copy()

# Ensure weekend flag is numeric
X["is_weekend"] = X["is_weekend"].astype(int)

# Encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", "passthrough", numerical_features),
    ]
)

# Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        ),
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print(f"\nTotal records: {len(df)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

# Train model
print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("-" * 35)
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R2   : {r2:.4f}")

# Display example predictions
comparison = pd.DataFrame({
    "Actual passengers": y_test.to_numpy()[:10],
    "Predicted passengers": predictions[:10].round(2),
})

print("\nSAMPLE PREDICTIONS")
print(comparison.to_string(index=False))

# Save trained model
joblib.dump(model, MODEL_FILE)

print(f"\nModel saved to: {MODEL_FILE}")
print("\nSTEP 3 TRAINING COMPLETED")
