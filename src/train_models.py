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
    
    # Create models dictionary with optimized parameters for realistic predictions
    models = {
        "Linear Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression(
                # Linear regression is relatively simple but can still be effective
                # when features have good correlation with target
                fit_intercept=True,  # Include intercept term
                positive=False,      # Don't constrain coefficients to be positive
                n_jobs=-1            # Use all available cores
            ))
        ]),
        
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,          # More trees for better stability
                max_depth=15,              # Prevent overfitting while capturing patterns
                min_samples_split=8,       # Require more samples to split nodes
                min_samples_leaf=4,        # Ensure leaf nodes represent multiple samples
                max_features='sqrt',       # Standard feature selection approach
                bootstrap=True,            # Use bootstrapping for tree diversity
                oob_score=True,            # Use out-of-bag samples to estimate accuracy
                n_jobs=-1,                 # Use all available cores
                random_state=RANDOM_STATE,
                verbose=0
            ))
        ]),
        
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=300,           # More boosting rounds
                learning_rate=0.03,         # Lower learning rate for better generalization
                max_depth=6,                # Moderate tree depth to prevent overfitting
                min_child_weight=3,         # Minimum sum of instance weight in children
                subsample=0.8,              # Use 80% of data per tree to prevent overfitting
                colsample_bytree=0.8,       # Use 80% of features per tree
                gamma=1,                    # Minimum loss reduction for split
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=1.0,             # L2 regularization
                random_state=RANDOM_STATE,
                n_jobs=-1,                  # Use all available cores
                verbosity=0
            ))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    feature_importances = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        
        print(f"{name} performance:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Calculate and store feature importances if available
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = (NUMERICAL_FEATURES + 
                            [f"{col}_{val}" for col in CATEGORICAL_FEATURES 
                             for val in X_train[col].unique()])
            
            # Store feature importances
            importances = model.named_steps['regressor'].feature_importances_
            if len(importances) == len(feature_names):
                feature_importances[name] = dict(zip(feature_names, importances))
            
            # Print top 5 important features
            indices = np.argsort(importances)[::-1]
            if len(indices) > 0:
                print(f"\nTop 5 important features for {name}:")
                for i in range(min(5, len(indices))):
                    if i < len(feature_names):
                        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Save model
        if name == "Linear Regression":
            save_model(model, "linear_regression.pkl")
        elif name == "Random Forest":
            save_model(model, "random_forest.pkl")
        elif name == "XGBoost":
            save_model(model, "xgboost.pkl")
    
    return models, results, feature_importances

def visualize_results(results, df, feature_importances=None):
    """Visualize model performance and data insights"""
    # Create the plots directory if it doesn't exist
    plots_dir = Path("plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Model Performance Comparison
    metrics = list(results[list(results.keys())[0]].keys())
    model_names = list(results.keys())
    
    for metric in metrics:
        values = [results[model][metric] for model in model_names]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(model_names, values, color=['#4A7CCF', '#1A4A94', '#F8B400'])
        plt.title(f'Model Comparison: {metric}', fontsize=16)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (max(values)-min(values))*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}_comparison.png")
        plt.close()
    
    # 2. Correlation heatmap
    plt.figure(figsize=(14, 12))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, annot_kws={"size": 10})
    plt.title('Feature Correlation Heatmap', fontsize=18, pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png")
    plt.close()

    # 3. Distribution of prices by region
    plt.figure(figsize=(16, 9))
    region_order = df.groupby('region')['price'].median().sort_values(ascending=False).index
    sns.boxplot(x='region', y='price', data=df, order=region_order)
    plt.title('House Price Distribution by Region', fontsize=18, pad=20)
    plt.xlabel('Region', fontsize=14)
    plt.ylabel('Price (TND)', fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "price_by_region.png")
    plt.close()
    
    # 4. Distribution of prices by property type
    plt.figure(figsize=(14, 8))
    type_order = df.groupby('property_type')['price'].median().sort_values(ascending=False).index
    sns.boxplot(x='property_type', y='price', data=df, order=type_order, palette='Blues')
    plt.title('House Price Distribution by Property Type', fontsize=18, pad=20)
    plt.xlabel('Property Type', fontsize=14)
    plt.ylabel('Price (TND)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "price_by_property_type.png")
    plt.close()
    
    # 5. Distribution of target variable with improved appearance
    plt.figure(figsize=(12, 7))
    sns.histplot(df[TARGET], kde=True, bins=50, color='#4A7CCF')
    plt.title('Distribution of Property Prices in Tunisia', fontsize=18, pad=20)
    plt.xlabel('Price (TND)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "target_distribution.png")
    plt.close()
    
    # 6. Scatter plot of price vs area_sqm with improved appearance
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='area_sqm', y='price', hue='property_type', data=df, 
                   alpha=0.7, palette='viridis', s=70)
    plt.title('Price vs Area Square Meters', fontsize=18, pad=20)
    plt.xlabel('Area (m²)', fontsize=14)
    plt.ylabel('Price (TND)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title='Property Type', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(plots_dir / "price_vs_area.png")
    plt.close()
    
    # 7. Feature importance visualization if available
    if feature_importances:
        for model_name, importances in feature_importances.items():
            # Get top 10 features
            top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(list(top_features.keys()), list(top_features.values()), color='#4A7CCF')
            plt.title(f'Top 10 Important Features - {model_name}', fontsize=18, pad=20)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.gca().invert_yaxis()  # Highest importance at the top
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + max(top_features.values())*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', va='center', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{model_name.replace(' ', '_').lower()}_feature_importance.png")
            plt.close()
    
    print(f"\nVisualization plots saved in {plots_dir}")

def main():
    """Main function to execute the model training process"""
    print("Starting model training process...")
    
    # Ensure models directory exists
    models_dirs = [
        MODELS_DIR,  # From config
        os.path.join('/mount/src/ml-project', 'models'),  # Deployment environment
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')  # Local absolute path
    ]
    
    # Create models directory if it doesn't exist
    for dir_path in models_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Successfully created/verified models directory at: {dir_path}")
        except Exception as e:
            print(f"Warning: Could not create models directory at {dir_path}: {e}")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, df = load_and_prepare_data()
    
    # Train models
    models, results, feature_importances = train_models(X_train, X_test, y_train, y_test)
    
    # Visualize results
    visualize_results(results, df, feature_importances)
    
    # Deploy models in case of deployment environment
    deployment_models_dir = os.path.join('/mount/src/ml-project', 'models')
    if MODELS_DIR != deployment_models_dir and os.path.exists('/mount/src'):
        try:
            os.makedirs(deployment_models_dir, exist_ok=True)
            # Copy all trained models to deployment path
            for model_name, model in models.items():
                filename = f"{model_name.lower().replace(' ', '_')}.pkl"
                deployment_path = os.path.join(deployment_models_dir, filename)
                joblib.dump(model, deployment_path)
                print(f"Saved model to deployment path: {deployment_path}")
        except Exception as e:
            print(f"Note: Could not save models to deployment path {deployment_models_dir}: {e}")
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
