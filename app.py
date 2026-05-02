import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (r2_score, mean_absolute_error,
                              mean_squared_error, mean_absolute_percentage_error)

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("house_price_dataset.csv")

@st.cache_resource
def train_all_models():
    df = load_data()
    X = df.drop("House_Price", axis=1)
    y = df["House_Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
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
        yp     = m.predict(X_test_s)
        n, p   = X_test_s.shape
        r2     = r2_score(y_test, yp)
        mae    = mean_absolute_error(y_test, yp)
        rmse   = np.sqrt(mean_squared_error(y_test, yp))
        mape   = mean_absolute_percentage_error(y_test, yp) * 100
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        results.append({"Model": name, "MAE": round(mae,2), "RMSE": round(rmse,2),
                         "MAPE%": round(mape,4), "R²": round(r2,4), "Adj R²": round(adj_r2,4)})
        trained[name] = (m, yp, y_test)
    return pd.DataFrame(results), trained, scaler, trained["Random Forest"][0], X, y

df = load_data()

st.sidebar.title("🏠 House Price Predictor")
st.sidebar.markdown("**By Deepika**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🔮 Predict Price","📊 EDA & Visualisation",
    "🤖 Model Comparison","📈 Best Model Performance","🔁 Cross Validation"])

with st.spinner("⏳ Training models, please wait..."):
    results_df, trained, scaler, best_model, X, y = train_all_models()

if page == "🔮 Predict Price":
    st.title("🏠 House Price Prediction")
    col1, col2 = st.columns(2)
    with col1:
        square_footage = st.number_input("Square Footage", 500, 10000, 2000, 100)
        num_bedrooms   = st.slider("Number of Bedrooms", 1, 10, 3)
        num_bathrooms  = st.slider("Number of Bathrooms", 1, 8, 2)
        year_built     = st.number_input("Year Built", 1900, 2024, 2000, 1)
    with col2:
        lot_size       = st.number_input("Lot Size (acres)", 0.1, 10.0, 0.5, 0.1)
        garage_size    = st.slider("Garage Size (cars)", 0, 5, 2)
        neighborhood_q = st.slider("Neighborhood Quality (1-10)", 1, 10, 7)
    st.markdown("---")
    if st.button("Predict Price", type="primary", use_container_width=True):
        features = np.array([[square_footage, num_bedrooms, num_bathrooms,
                               year_built, lot_size, garage_size, neighborhood_q]])
        pred = best_model.predict(scaler.transform(features))[0]
        st.success(f"### Estimated Price: ${pred:,.0f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Low Estimate",  f"${pred*0.95:,.0f}")
        c2.metric("Predicted",     f"${pred:,.0f}")
        c3.metric("High Estimate", f"${pred*1.05:,.0f}")
        feat_names = df.drop("House_Price", axis=1).columns.tolist()
        fig = px.bar(x=best_model.feature_importances_, y=feat_names, orientation="h",
                     title="Feature Importance", color=best_model.feature_importances_,
                     color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 EDA & Visualisation":
    st.title("📊 Exploratory Data Analysis")
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", f"{df.shape[1]-1}")
    c3.metric("Avg Price", f"${df['House_Price'].mean():,.0f}")
    st.markdown("---")
    st.subheader("Univariate Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df, x="House_Price", nbins=40, title="House Price Distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="Square_Footage", nbins=40, title="Square Footage Distribution"), use_container_width=True)
        st.plotly_chart(px.box(df, x="House_Price", title="House Price Outliers"), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x="Lot_Size", nbins=30, title="Lot Size Distribution"), use_container_width=True)
        st.plotly_chart(px.bar(df["Num_Bedrooms"].value_counts().sort_index(), title="Bedroom Count"), use_container_width=True)
        st.plotly_chart(px.bar(df["Neighborhood_Quality"].value_counts().sort_index(), title="Neighbourhood Quality"), use_container_width=True)
    st.markdown("---")
    st.subheader("Bivariate Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.scatter(df, x="Square_Footage", y="House_Price", color="Neighborhood_Quality",
            title="Square Footage vs House Price", color_continuous_scale="Viridis"), use_container_width=True)
        st.plotly_chart(px.box(df, x="Num_Bedrooms", y="House_Price", title="Bedrooms vs House Price"), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(df, x="Num_Bathrooms", y="House_Price", title="Bathrooms vs House Price"), use_container_width=True)
        st.plotly_chart(px.box(df, x="Neighborhood_Quality", y="House_Price", title="Neighbourhood Quality vs House Price"), use_container_width=True)
    st.markdown("---")
    st.subheader("Correlation Heatmap")
    st.plotly_chart(px.imshow(df.corr(numeric_only=True), text_auto=".2f",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix"), use_container_width=True)
    st.dataframe(df.head(20), use_container_width=True)

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    best_idx = results_df["R²"].idxmax()
    st.dataframe(results_df.style.highlight_max(subset=["R²","Adj R²"], color="#d4edda")
                                  .highlight_min(subset=["MAE","RMSE","MAPE%"], color="#d4edda"), use_container_width=True)
    st.success(f"Best Model: {results_df.loc[best_idx,'Model']} — R² = {results_df.loc[best_idx,'R²']}")
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(results_df.sort_values("R²"), x="Model", y="R²",
            title="R² Score", color="R²", color_continuous_scale="Blues"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(results_df.sort_values("RMSE"), x="Model", y="RMSE",
            title="RMSE (lower=better)", color="RMSE", color_continuous_scale="Reds_r"), use_container_width=True)
    col3,col4 = st.columns(2)
    with col3:
        st.plotly_chart(px.bar(results_df.sort_values("MAE"), x="Model", y="MAE",
            title="MAE", color="MAE", color_continuous_scale="Oranges_r"), use_container_width=True)
    with col4:
        st.plotly_chart(px.bar(results_df, x="Model", y="MAPE%",
            title="MAPE%", color="MAPE%", color_continuous_scale="Purples_r"), use_container_width=True)

elif page == "📈 Best Model Performance":
    st.title("📈 Best Model: Random Forest")
    _, y_pred_rf, y_test_rf = trained["Random Forest"]
    r2     = r2_score(y_test_rf, y_pred_rf)
    mae    = mean_absolute_error(y_test_rf, y_pred_rf)
    rmse   = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    mape   = mean_absolute_percentage_error(y_test_rf, y_pred_rf)*100
    adj_r2 = 1-(1-r2)*(len(y_test_rf)-1)/(len(y_test_rf)-7-1)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("R²", f"{r2:.4f}"); c2.metric("Adj R²", f"{adj_r2:.4f}")
    c3.metric("MAE", f"${mae:,.0f}"); c4.metric("RMSE", f"${rmse:,.0f}"); c5.metric("MAPE%", f"{mape:.2f}%")
    col1,col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(y_test_rf), y=list(y_pred_rf), mode="markers",
                                  marker=dict(color="steelblue", opacity=0.5), name="Predictions"))
        mn,mx = min(y_test_rf), max(y_test_rf)
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
                                  line=dict(color="red", dash="dash"), name="Perfect Fit"))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        residuals = np.array(y_test_rf) - np.array(y_pred_rf)
        fig2 = px.histogram(residuals, nbins=40, title="Residuals Distribution",
                             color_discrete_sequence=["#EF553B"])
        fig2.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig2, use_container_width=True)
    feat_names = df.drop("House_Price", axis=1).columns.tolist()
    st.plotly_chart(px.bar(x=best_model.feature_importances_, y=feat_names, orientation="h",
        title="Feature Importances", color=best_model.feature_importances_,
        color_continuous_scale="Blues"), use_container_width=True)

elif page == "🔁 Cross Validation":
    st.title("🔁 Cross Validation (5-Fold)")
    X_cv = df.drop("House_Price", axis=1)
    y_cv = df["House_Price"]
    sc = StandardScaler()
    X_cv_s = sc.fit_transform(X_cv)
    with st.spinner("Running cross-validation..."):
        lr_scores  = cross_val_score(LinearRegression(), X_cv_s, y_cv, cv=5, scoring="r2")
        rf_scores  = cross_val_score(RandomForestRegressor(random_state=42), X_cv_s, y_cv, cv=5, scoring="r2")
        mae_scores = -cross_val_score(RandomForestRegressor(random_state=42), X_cv_s, y_cv,
                                       cv=5, scoring="neg_mean_absolute_error")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Linear Regression R²")
        fold_lr = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(5)], "R²": lr_scores.round(4)})
        st.plotly_chart(px.bar(fold_lr, x="Fold", y="R²", color="R²",
            color_continuous_scale="Blues", title=f"Avg R² = {lr_scores.mean():.4f}"), use_container_width=True)
        st.metric("Average R²", f"{lr_scores.mean():.4f}")
    with col2:
        st.subheader("Random Forest R²")
        fold_rf = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(5)], "R²": rf_scores.round(4)})
        st.plotly_chart(px.bar(fold_rf, x="Fold", y="R²", color="R²",
            color_continuous_scale="Greens", title=f"Avg R² = {rf_scores.mean():.4f}"), use_container_width=True)
        st.metric("Average R²", f"{rf_scores.mean():.4f}")
    st.subheader("Random Forest MAE per fold")
    mae_df = pd.DataFrame({"Fold":[f"Fold {i+1}" for i in range(5)], "MAE": mae_scores.round(2)})
    st.plotly_chart(px.bar(mae_df, x="Fold", y="MAE", color="MAE",
        color_continuous_scale="Reds_r", title=f"Avg MAE = ${mae_scores.mean():,.0f}"), use_container_width=True)
    st.metric("Average MAE", f"${mae_scores.mean():,.0f}")
