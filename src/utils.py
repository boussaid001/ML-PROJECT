import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from pathlib import Path

from config import (
    NUMERICAL_FEATURES, 
    CATEGORICAL_FEATURES, 
    TARGET,
    DATASET_PATH,
    MODELS_DIR,
    REGIONS,
    PROPERTY_TYPES,
    TUNISIA_CENTER_LAT,
    TUNISIA_CENTER_LON
)

def load_data():
    """Load the housing dataset"""
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Handle missing values
    df = df.copy()
    df = df.dropna()
    
    # Return features and target
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    
    return X, y

def get_preprocessor():
    """Create preprocessing pipeline for numerical and categorical features"""
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    
    return preprocessor

def save_model(model, filename):
    """Save model to disk"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODELS_DIR / filename)
    
def load_model(filename):
    """Load model from disk"""
    try:
        return joblib.load(MODELS_DIR / filename)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }

def plot_feature_importance(model, feature_names):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return fig
    return None

def plot_interactive_map(df):
    """Create an interactive map of housing prices in Tunisia"""
    # Create a copy with price in thousands for better display
    df_map = df.copy()
    df_map['price_thousands'] = df_map['price'] / 1000
    
    fig = px.scatter_geo(
        df_map,
        lat="latitude",
        lon="longitude",
        color="price",
        size="area_sqm",
        color_continuous_scale=px.colors.sequential.Reds,
        size_max=15,
        scope="africa",
        title="Housing Prices in Tunisia",
        hover_data=["price", "property_type", "region", "area_sqm", "bedrooms"],
        center={"lat": TUNISIA_CENTER_LAT, "lon": TUNISIA_CENTER_LON},
        projection="mercator",
        fitbounds="locations"
    )
    
    fig.update_geos(
        resolution=50,
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="LightGray",
        showocean=True,
        oceancolor="LightBlue",
        showcountries=True,
        countrycolor="Black",
        countrywidth=0.5,
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=600,
        coloraxis_colorbar=dict(
            title="Price (TND)",
            tickformat=",.0f"
        )
    )
    return fig

def plot_region_prices(df):
    """Plot average house prices by region"""
    region_avg = df.groupby('region')['price'].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        region_avg,
        x='region',
        y='price',
        color='price',
        color_continuous_scale=px.colors.sequential.Reds,
        title="Average House Price by Region",
        labels={'price': 'Average Price (TND)', 'region': 'Region'}
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        xaxis_tickangle=-45,
        yaxis_tickformat=',.0f'
    )
    
    return fig

def plot_property_type_prices(df):
    """Plot average house prices by property type"""
    type_avg = df.groupby('property_type')['price'].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(
        type_avg,
        x='property_type',
        y='price',
        color='price',
        color_continuous_scale=px.colors.sequential.Reds,
        title="Average House Price by Property Type",
        labels={'price': 'Average Price (TND)', 'property_type': 'Property Type'}
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        yaxis_tickformat=',.0f'
    )
    
    return fig

def plot_price_vs_area(df):
    """Plot price vs area with regression line"""
    fig = px.scatter(
        df,
        x='area_sqm',
        y='price',
        color='property_type',
        opacity=0.7,
        title="Price vs Area",
        labels={'area_sqm': 'Area (m²)', 'price': 'Price (TND)'},
        trendline="ols"
    )
    
    fig.update_layout(
        yaxis_tickformat=',.0f'
    )
    
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap of numerical features"""
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="Reds", 
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    return fig

def format_price(price, currency="TND"):
    """Format price with thousands separator and currency"""
    return f"{price:,.0f} {currency}"

def apply_custom_theme():
    st.markdown("""
    <style>
    /* Hide emotion cache elements */
    .st-emotion-cache-i4rl61,
    .st-emotion-cache-j7qwjs,
    .st-emotion-cache-1v0mbdj {
        display: none !important;
    }
    
    /* Main content styling */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
        background-color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        padding: 1rem;
        border-right: 1px solid #E41E25;
    }
    
    .css-1d391kg .sidebar-content {
        padding: 1rem;
    }
    
    .css-1d391kg .sidebar-content .block-container {
        padding-top: 0;
    }
    
    /* Navigation menu styling */
    .css-1d391kg .nav-link {
        color: #E41E25 !important;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        border: 1px solid transparent;
        font-weight: 500;
    }
    
    .css-1d391kg .nav-link:hover {
        background-color: rgba(228, 30, 37, 0.1);
        color: #E41E25 !important;
        border-color: #E41E25;
    }
    
    .css-1d391kg .nav-link-selected {
        background-color: #E41E25 !important;
        color: white !important;
        border-color: #E41E25;
    }
    
    /* Form elements styling */
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #E41E25;
    }
    
    .stSelectbox > div, .stNumberInput > div, .stTextInput > div {
        background-color: inherit;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #E41E25;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(228, 30, 37, 0.2);
    }
    
    .stButton > button:hover {
        background-color: #c4191f;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(228, 30, 37, 0.3);
    }
    
    /* Card styling */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(228, 30, 37, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #E41E25;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(228, 30, 37, 0.15);
    }
    
    .card h3 {
        color: #E41E25;
        margin-top: 0;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Text styling */
    p, label, span, div {
        color: #E41E25;
    }
    
    /* Metric styling */
    .stMetric {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(228, 30, 37, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #E41E25;
    }
    
    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(228, 30, 37, 0.1);
        background-color: #ffffff !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(228, 30, 37, 0.1);
        background-color: #ffffff;
        border: 1px solid #E41E25;
    }
    
    .dataframe thead th {
        background-color: #E41E25;
        color: white;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: rgba(228, 30, 37, 0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(228, 30, 37, 0.1);
        border: 1px solid #E41E25;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {
        display: none;
    }
    
    /* Fix sidebar width */
    .css-1d391kg {
        width: 250px !important;
    }
    
    /* Fix main content margin */
    .main .block-container {
        margin-left: 250px !important;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #E41E25;
        font-weight: 600;
    }
    
    /* Links styling */
    a {
        color: #E41E25;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    a:hover {
        color: #c4191f;
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)
