# 🏠 House Price Prediction — ML Project
**By Deepika**

A complete machine learning project that predicts house prices using multiple regression algorithms, deployed as an interactive web app with Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| Square_Footage | Total area of the house |
| Num_Bedrooms | Number of bedrooms |
| Num_Bathrooms | Number of bathrooms |
| Year_Built | Construction year |
| Lot_Size | Lot size in acres |
| Garage_Size | Number of cars the garage fits |
| Neighborhood_Quality | Quality rating 1–10 |
| **House_Price** | **Target variable** |

---

## 🤖 Models Trained

| Model | R² |
|---|---|
| Linear Regression | 0.9984 |
| Gradient Boosting | 0.9965 |
| **Random Forest** ✅ | **0.9939** |
| Decision Tree | 0.9847 |
| KNN | 0.8916 |
| SVR | ~0.00 |

---

## 🖥️ App Pages

| Page | What it shows |
|---|---|
| 🔮 Predict Price | Input property details → get instant price estimate |
| 📊 EDA & Visualisation | Univariate, bivariate, and correlation analysis |
| 🤖 Model Comparison | Side-by-side comparison of all 6 models |
| 📈 Best Model Performance | Actual vs Predicted, Residuals, Feature Importance |
| 🔁 Cross Validation | 5-fold CV results for Linear Regression & Random Forest |

---

## 🚀 Run Locally

```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

python train.py            # generates model.pkl and scaler.pkl

streamlit run app.py
```

App opens at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (include `model.pkl` and `scaler.pkl`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → branch `main` → file `app.py`
4. Click **Deploy** ✅

---

## 📁 Project Structure

```
house-price-prediction/
├── app.py                   # Streamlit app (5 pages)
├── train.py                 # Training script
├── model.pkl                # Saved best model
├── scaler.pkl               # Fitted scaler
├── house_price_dataset.csv  # Dataset (1000 rows)
├── model_results.csv        # Model comparison results
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠 Tech Stack

`Python` · `Scikit-learn` · `XGBoost` · `Streamlit` · `Pandas` · `NumPy` · `Plotly`
