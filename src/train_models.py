import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import from local modules
from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    DATASET_PATH,
    MODELS_DIR,
    LINEAR_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
    RANDOM_STATE,
    TEST_SIZE
)
from utils import preprocess_data, get_preprocessor, save_model, evaluate_model

def load_and_prepare_data():
    """Load data and prepare for modeling"""
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run generate_tunisia_data.py first.")
    
    # Load data
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Basic data exploration
    print("\nData Overview:")
    print(df.head())
    print("\nData Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Remove address column (not used for modeling)
    if 'address' in df.columns:
        df = df.drop('address', axis=1)
    
    # Handle missing values if any
    df = df.dropna()
    print(f"\nShape after handling missing values: {df.shape}")
    
    # Extract features and target
    X, y = preprocess_data(df)
    
    # Convert boolean columns to integers (for modeling)
    for col in ['has_elevator', 'has_garden', 'has_parking']:
        if col in X.columns:
            X[col] = X[col].astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, df

def train_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple regression models"""
    # Create a preprocessor
    preprocessor = get_preprocessor()
    
    # Create models dictionary
    models = {
        "Linear Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100, 
                max_depth=20,
                min_samples_split=5,
                random_state=RANDOM_STATE
            ))
        ]),
        
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=RANDOM_STATE
            ))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        
        print(f"{name} performance:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Save model
        if name == "Linear Regression":
            save_model(model, "linear_regression.pkl")
        elif name == "Random Forest":
            save_model(model, "random_forest.pkl")
        elif name == "XGBoost":
            save_model(model, "xgboost.pkl")
    
    return models, results

def visualize_results(results, df):
    """Visualize model performance and data insights"""
    # Create the plots directory if it doesn't exist
    plots_dir = Path("plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Model Performance Comparison
    metrics = list(results[list(results.keys())[0]].keys())
    model_names = list(results.keys())
    
    for metric in metrics:
        values = [results[model][metric] for model in model_names]
        
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, values)
        plt.title(f'Model Comparison: {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}_comparison.png")
        plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png")
    plt.close()

    # 3. Distribution of prices by region
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='region', y='price', data=df)
    plt.title('House Price Distribution by Region')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(plots_dir / "price_by_region.png")
    plt.close()
    
    # 4. Distribution of prices by property type
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='property_type', y='price', data=df)
    plt.title('House Price Distribution by Property Type')
    plt.tight_layout()
    plt.savefig(plots_dir / "price_by_property_type.png")
    plt.close()
    
    # 5. Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET], kde=True)
    plt.title(f'Distribution of {TARGET}')
    plt.tight_layout()
    plt.savefig(plots_dir / "target_distribution.png")
    plt.close()
    
    # 6. Scatter plot of price vs area_sqm
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='area_sqm', y='price', hue='property_type', data=df, alpha=0.6)
    plt.title('Price vs Area')
    plt.tight_layout()
    plt.savefig(plots_dir / "price_vs_area.png")
    plt.close()
    
    print(f"\nVisualization plots saved in {plots_dir}")

def main():
    """Main function to orchestrate model training"""
    print("Starting model training process...")
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, df = load_and_prepare_data()
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    visualize_results(results, df)
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
