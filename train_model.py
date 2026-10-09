import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load dataset
data = pd.read_csv("Household energy unit data.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", data.shape)


# 2. Define input features and target
features = [
    "num_rooms",
    "num_people",
    "housearea",
    "is_ac",
    "is_tv",
    "is_flat",
    "num_children",
    "is_urban"
]

target = "units"


# 3. Clean the dataset
data = data.copy()

# Treat -1 as missing for these two features
data["num_rooms"] = data["num_rooms"].replace(-1, float("nan"))
data["num_people"] = data["num_people"].replace(-1, float("nan"))

# Remove rows with missing or negative target values
data = data.dropna(subset=[target])
data = data[data[target] >= 0]

X = data[features]
y = data[target]


# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5. Define models
models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# 6. Train and evaluate each model
results = {}
trained_models = {}

for name, regressor in models.items():

    # Impute missing feature values using training data only
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", regressor)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = model

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.3f}")


# 7. Select the model with the lowest MAE
best_name = min(results, key=lambda name: results[name]["MAE"])
best_model = trained_models[best_name]

print("\n" + "=" * 40)
print("Best model:", best_name)
print("Lowest MAE:", f"{results[best_name]['MAE']:.2f}")


# 8. Save the selected model
joblib.dump(best_model, "model.pkl")

print("\nBest model saved successfully as model.pkl")


# 9. Display feature importance for tree-based models
regressor = best_model.named_steps["regressor"]

if hasattr(regressor, "feature_importances_"):
    importance = pd.Series(
        regressor.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    print("\nFeature importance:")
    print(importance)


# 10. Test predictions with different household inputs
test_data = pd.DataFrame([
    [2, 4, 500, 0, 1, 1, 1, 1],
    [6, 4, 1800, 1, 1, 0, 1, 0],
    [3, 2, 700, 0, 0, 1, 0, 1],
    [8, 8, 3000, 1, 1, 0, 4, 0],
    [1, 1, 300, 0, 0, 1, 0, 1]
], columns=features)

print("\nPredictions for different households:")

for i, units in enumerate(best_model.predict(test_data), start=1):
    print(f"Test {i}: {units:.2f} units")