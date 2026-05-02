"""
train.py  –  Trains all regression models from the notebook and saves the best one.
Run once before launching the app:  python train.py
"""
import pandas as pd, numpy as np, pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, mean_absolute_percentage_error)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("house_price_dataset.csv")
print(f"Dataset: {df.shape}")

X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────────────────────────
models = {
    "Linear Regression":  LinearRegression(),
    "Decision Tree":      DecisionTreeRegressor(random_state=42),
    "Random Forest":      RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":  GradientBoostingRegressor(random_state=42),
    "KNN":                KNeighborsRegressor(),
    "SVR":                SVR(),
}

results = []
trained = {}
for name, m in models.items():
    m.fit(X_train_s, y_train)
    yp    = m.predict(X_test_s)
    n, p  = X_test_s.shape
    r2    = r2_score(y_test, yp)
    mae   = mean_absolute_error(y_test, yp)
    rmse  = np.sqrt(mean_squared_error(y_test, yp))
    mape  = mean_absolute_percentage_error(y_test, yp) * 100
    adj_r2 = 1 - (1 - r2)*(n - 1)/(n - p - 1)
    results.append([name, round(mae, 2), round(rmse, 2), round(mape, 4), round(r2, 4), round(adj_r2, 4)])
    trained[name] = m

results_df = pd.DataFrame(results, columns=["Model","MAE","RMSE","MAPE%","R2","Adj_R2"])
print("\nModel Comparison:")
print(results_df.sort_values("R2", ascending=False).to_string(index=False))

best_name = results_df.loc[results_df["R2"].idxmax(), "Model"]
print(f"\n✅ Best Model: {best_name}")

# ── Save best model ────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(trained[best_name], f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

results_df.to_csv("model_results.csv", index=False)
print("Saved model.pkl, scaler.pkl, model_results.csv")
