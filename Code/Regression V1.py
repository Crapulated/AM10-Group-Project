import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
df = pd.read_csv("../data/movie_metadata.csv")

print("Columns:", df.columns)

# Assuming the column to predict is called 'ratings'
# If your column is named differently (e.g., imdb_score), modify this line:
target_column = "imdb_score"

# Drop rows with missing target
df = df.dropna(subset=[target_column])

# Select numeric features only (basic approach)
X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_column])
y = df[target_column]

# Handle remaining NaNs
X = X.fillna(X.mean())

# ----------------------------------------------------
# 2. TRAIN–TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 3. DEFINE MODELS
# ----------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42
    ),
}

# ----------------------------------------------------
# 4. TRAIN + EVALUATE
# ----------------------------------------------------
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Linear + Ridge use scaled data
    if name in ["Linear Regression", "Ridge Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = (mse, r2)
    print(f"{name} → MSE: {mse:.3f} | R²: {r2:.3f}")

# ----------------------------------------------------
# 5. SHOW RESULTS
# ----------------------------------------------------
print("\n===== Model Performance Summary =====")
for name, (mse, r2) in results.items():
    print(f"{name:20} | MSE: {mse:.3f} | R²: {r2:.3f}")
