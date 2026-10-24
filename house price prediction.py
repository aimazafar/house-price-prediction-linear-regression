# Project 1: House Price Prediction (Linear Regression)
# Run: pip install pandas numpy scikit-learn joblib
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

CSV = "house_data.csv"

if not os.path.exists(CSV):
    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 4000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 4, n),
        "age_years": rng.integers(0, 50, n),
    })
    df["price"] = (df.area_sqft*150 + df.bedrooms*10000 + df.bathrooms*8000
                   - df.age_years*900 + rng.normal(0, 20000, n)).round(0)
    df.loc[rng.choice(n, 15, replace=False), "age_years"] = np.nan  # missing values (cleaning demo)
    df.to_csv(CSV, index=False)

# ---- Step 1: Load + clean + explore ----
df = pd.read_csv(CSV)
print("Shape:", df.shape)
print(df.head(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
df = df.fillna(df.median(numeric_only=True)).drop_duplicates()
print(df.describe().round(2), "\n")
print("Correlation with price:\n", df.corr()["price"].round(3), "\n")

# ---- Step 2: Features, split, train ----
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# ---- Step 3: Evaluate ----
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"RMSE: {rmse:,.2f}")
print(f"R2 score: {r2_score(y_test, pred):.4f}\n")
print("Coefficients (price change per 1 unit increase):")
for name, c in zip(X.columns, model.coef_):
    print(f"  {name:12s}: {c:,.2f}")
print(f"  Intercept   : {model.intercept_:,.2f}\n")

# ---- Step 4: Save model + example predictions ----
joblib.dump(model, "house_model.pkl")
loaded = joblib.load("house_model.pkl")
samples = pd.DataFrame({"area_sqft": [1200, 2500], "bedrooms": [2, 4],
                        "bathrooms": [1, 3], "age_years": [10, 3]})
for i, p in enumerate(loaded.predict(samples)):
    print(f"Example {i+1}: {samples.iloc[i].to_dict()} -> Predicted price: {p:,.0f}")
