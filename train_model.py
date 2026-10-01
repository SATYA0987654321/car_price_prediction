import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    print("Loading car dataset...")
    df = pd.read_csv("car_data.csv")

    # Data Cleaning
    df = df.dropna()
    df = df.drop_duplicates()

    # Features and target
    categorical_cols = ["Brand", "Model", "Fuel_Type", "Transmission"]
    numerical_cols = ["Year", "Kilometers_Driven", "Engine_Size"]
    
    X = df[categorical_cols + numerical_cols]
    y = df["Price"]

    # Train test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # Models to train
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=60, max_depth=15, min_samples_split=4, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    results = {}
    trained_pipelines = {}

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
        trained_pipelines[name] = pipeline

        print(f"{name:18} -> MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R2: {r2:.4f}")

    # Compare models
    results_df = pd.DataFrame(results).T
    print("\nModel Comparison Table:")
    print(results_df)

    # Select best model by R2
    best_model_name = max(results, key=lambda k: results[k]["R2"])
    best_pipeline = trained_pipelines[best_model_name]
    print(f"\nBest Model: {best_model_name} with R2: {results[best_model_name]['R2']}")

    # Save best model and evaluation metrics to pickle
    model_data = {
        "model": best_pipeline,
        "model_name": best_model_name,
        "metrics": results,
        "all_models": trained_pipelines
    }

    with open("car_price_model.pkl", "wb") as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Best model and metrics saved to car_price_model.pkl successfully!")

if __name__ == "__main__":
    train()
