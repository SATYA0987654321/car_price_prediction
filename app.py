import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Used Car Price Prediction System",
    page_icon="🚗",
    layout="wide"
)

# Helper function to format Indian currency (Lakhs / Crores)
def format_inr(val):
    if val >= 10000000:
        return f"₹{val / 10000000:.2f} Crore (₹{val:,.0f})"
    elif val >= 100000:
        return f"₹{val / 100000:.2f} Lakh (₹{val:,.0f})"
    else:
        return f"₹{val:,.0f}"

# Load dataset and model
@st.cache_data
def load_dataset():
    df = pd.read_csv("car_data.csv")
    df = df.dropna().drop_duplicates()
    return df

@st.cache_resource
def load_model():
    if not os.path.exists("car_price_model.pkl"):
        import train_model
        train_model.train()
    with open("car_price_model.pkl", "rb") as f:
        data = pickle.load(f)
    return data

df = load_dataset()
model_info = load_model()
best_model_name = model_info["model_name"]
metrics = model_info["metrics"]

# Header
st.title("🚗 Used Car Price Prediction System")
st.markdown("Predict the market valuation of used cars in the Indian automotive market using Machine Learning.")

# Sidebar navigation
menu = st.sidebar.radio(
    "Navigation",
    ["Price Predictor", "Model Comparison", "Feature Importance", "EDA & Dataset"]
)

# -------------------------------------------------------------
# 1. Price Predictor
# -------------------------------------------------------------
if menu == "Price Predictor":
    st.header("Indian Car Price Predictor")
    st.write(f"Select your car specifications below. Predictions are powered by **{best_model_name}**.")

    # Distinct Brands & Model Mapping
    brand_models = df.groupby("Brand")["Model"].unique().to_dict()
    brands = sorted(list(brand_models.keys()))

    col1, col2 = st.columns(2)

    with col1:
        # Default to Maruti Suzuki if present
        default_brand_idx = brands.index("Maruti Suzuki") if "Maruti Suzuki" in brands else 0
        brand = st.selectbox("Car Brand", brands, index=default_brand_idx)
        
        models = sorted(list(brand_models[brand]))
        model = st.selectbox("Model", models)
        
        year = st.slider("Year of Manufacture", min_value=2012, max_value=2024, value=2019)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=50000, step=5000)

    with col2:
        fuel_options = sorted(list(df[df["Brand"] == brand]["Fuel_Type"].unique()))
        fuel = st.selectbox("Fuel Type", fuel_options)
        
        trans_options = sorted(list(df[df["Brand"] == brand]["Transmission"].unique()))
        transmission = st.selectbox("Transmission", trans_options)
        
        # Engine displacement recommendation based on selected model
        sample_engine = float(df[df["Model"] == model]["Engine_Size"].median())
        engine = st.slider(
            "Engine Size (in Liters)",
            min_value=0.0 if fuel == "Electric" else 0.8,
            max_value=3.5,
            value=0.0 if fuel == "Electric" else sample_engine,
            step=0.1
        )
        
        model_choices = list(metrics.keys())
        selected_algo = st.selectbox(
            "Regression Algorithm",
            model_choices,
            index=model_choices.index(best_model_name)
        )

    st.write("")
    if st.button("Calculate Predicted Price", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Brand": brand,
            "Model": model,
            "Year": year,
            "Kilometers_Driven": float(km_driven),
            "Fuel_Type": fuel,
            "Transmission": transmission,
            "Engine_Size": float(engine)
        }])

        model_to_use = model_info["all_models"][selected_algo]
        prediction = float(model_to_use.predict(input_data)[0])
        prediction = max(90000, round(prediction, -2))
        
        mae_val = metrics[selected_algo]["MAE"]
        lower = max(70000, round(prediction - mae_val, -2))
        upper = round(prediction + mae_val, -2)

        st.success(f"""
        ### 🏷️ Estimated Market Value: **{format_inr(prediction)}**
        - **Expected Range (±MAE):** {format_inr(lower)} — {format_inr(upper)}
        - **Predicted with:** {selected_algo} Regressor
        """)

# -------------------------------------------------------------
# 2. Model Comparison
# -------------------------------------------------------------
elif menu == "Model Comparison":
    st.header("📊 Model Evaluation & Benchmarks")
    st.write("Models were evaluated on an 80/20 train-test split using **MAE, RMSE, and R²**.")

    metrics_df = pd.DataFrame(metrics).T
    
    # Format table for display
    formatted_df = metrics_df.copy()
    formatted_df["MAE (INR)"] = formatted_df["MAE"].map(lambda x: format_inr(x))
    formatted_df["RMSE (INR)"] = formatted_df["RMSE"].map(lambda x: format_inr(x))
    formatted_df["R² Score"] = formatted_df["R2"].map(lambda x: f"{x:.4f} ({x*100:.1f}%)")
    formatted_df = formatted_df.drop(columns=["MAE", "RMSE", "R2"])
    
    st.dataframe(formatted_df, use_container_width=True)

    st.info(f"🏆 **Winner:** **{best_model_name}** achieved the highest accuracy with an R² of `{metrics[best_model_name]['R2']:.4f}`.")

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=metrics_df.index, y=metrics_df["R2"], palette="Blues_r", ax=ax)
        ax.set_ylabel("R² (Higher is better)")
        ax.set_ylim(0.85, 1.0)
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h - 0.015),
                        ha='center', va='center', color='white', fontweight='bold')
        st.pyplot(fig)

    with chart_c2:
        st.subheader("Error Comparison: MAE vs RMSE")
        err_df = metrics_df[["MAE", "RMSE"]].reset_index().melt(id_vars="index", var_name="Metric", value_name="Value")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=err_df, x="index", y="Value", hue="Metric", palette="rocket", ax=ax)
        ax.set_ylabel("Error in INR (₹)")
        ax.set_xlabel("")
        st.pyplot(fig)

# -------------------------------------------------------------
# 3. Feature Importance
# -------------------------------------------------------------
elif menu == "Feature Importance":
    st.header("🔍 Important Factors Affecting Car Price")
    st.write("Understand which parameters have the greatest influence on Indian used car valuations.")

    # Calculate feature importances from XGBoost
    regressor = model_info["all_models"]["XGBoost"].named_steps["regressor"]
    preprocessor = model_info["all_models"]["XGBoost"].named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    # Aggregate by original feature category
    categories = ["Brand", "Model", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission", "Engine_Size"]
    grouped_importance = {}
    for col in categories:
        mask = [f.startswith(f"num__{col}") or f.startswith(f"cat__{col}_") for f in feature_names]
        grouped_importance[col] = float(np.sum(importances[mask]))

    total_imp = sum(grouped_importance.values())
    imp_df = pd.DataFrame([
        {"Factor": k.replace("_", " "), "Importance (%)": round((v / total_imp) * 100, 2)}
        for k, v in grouped_importance.items()
    ]).sort_values("Importance (%)", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(imp_df["Factor"], imp_df["Importance (%)"], color="#2563eb")
    ax.set_xlabel("Relative Importance (%)")
    ax.set_xlim(0, max(imp_df["Importance (%)"]) + 10)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va='center', fontweight='bold')
    st.pyplot(fig)

    st.markdown("""
    **Key Market Takeaways in India:**
    - **Brand & Model:** Brands like Toyota and Maruti Suzuki command strong resale multipliers, while luxury imports depreciate faster.
    - **Year (Age):** Annual vehicle age accounts for significant depreciation.
    - **Kilometers Driven:** High mileage reduces value due to road condition wear and service costs.
    - **Fuel Type & Transmission:** CNG and Hybrid cars hold resale premiums due to high fuel efficiency.
    """)

# -------------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
elif menu == "EDA & Dataset":
    st.header("📈 Exploratory Data Analysis (EDA)")
    st.write(f"The dataset includes **{len(df):,}** listings from top Indian car brands (Maruti Suzuki, Tata, Mahindra, Hyundai, Toyota, etc.).")

    st.subheader("Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution (INR)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Price"] / 100000, kde=True, color="#0284c7", bins=30, ax=ax)
        ax.set_xlabel("Price (in ₹ Lakhs)")
        ax.set_ylabel("Number of Cars")
        st.pyplot(fig)

    with col2:
        st.subheader("Price vs. Kilometers Driven")
        fig, ax = plt.subplots(figsize=(6, 4))
        sample_df = df.sample(min(700, len(df)), random_state=42)
        sns.scatterplot(
            data=sample_df,
            x="Kilometers_Driven",
            y=sample_df["Price"] / 100000,
            hue="Fuel_Type",
            alpha=0.7,
            ax=ax
        )
        ax.set_xlabel("Kilometers Driven")
        ax.set_ylabel("Price (in ₹ Lakhs)")
        st.pyplot(fig)

    st.subheader("Average Price by Car Brand in India")
    avg_price = (df.groupby("Brand")["Price"].mean() / 100000).sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=avg_price, x="Brand", y="Price", palette="Blues_r", ax=ax)
    plt.xticks(rotation=45)
    ax.set_ylabel("Average Price (in ₹ Lakhs)")
    st.pyplot(fig)
