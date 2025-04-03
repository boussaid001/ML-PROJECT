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

# Import configuration
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
    
    # Add longitude and latitude if they don't exist in the dataframe
    if 'longitude' not in df.columns:
        df['longitude'] = 0.0  # Default value
    
    if 'latitude' not in df.columns:
        df['latitude'] = 0.0  # Default value
    
    # Return features and target
    if TARGET in df.columns:
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET]
        return X, y
    else:
        # For prediction, we only need features
        X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        return X

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
        
        # Get theme-aware colors
        theme = 'light'
        if 'theme' in st.session_state:
            theme = st.session_state.theme
            
        # Define colors based on theme
        cmap = "Blues" if theme == 'light' else "Blues_r"
        fig_bg = '#FFFFFF' if theme == 'light' else '#1E2A3E'
        text_color = '#333333' if theme == 'light' else '#E9EEF6'
        
        # Create figure with theme-aware styling and increased size
        fig, ax = plt.subplots(figsize=(14, 10))  # Increased from (10, 6)
        # Set figure properties after creation for better compatibility
        fig.patch.set_facecolor(fig_bg)
        ax.set_facecolor(fig_bg)
        
        # Plot bars
        bars = ax.bar(range(len(indices)), importances[indices], align='center', color='#4A7CCF')
        
        # Set labels and ticks
        ax.set_title('Feature Importance', color=text_color, fontsize=18, pad=20)
        ax.set_xlabel('Features', color=text_color, fontsize=14, labelpad=15)
        ax.set_ylabel('Importance', color=text_color, fontsize=14, labelpad=15)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, color=text_color, ha='right')
        ax.tick_params(axis='y', colors=text_color, labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        
        # Add value labels on top of bars
        for i, v in enumerate(importances[indices]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', color=text_color, fontsize=10)
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(text_color)
            
        # Add grid for better readability and set tight layout
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig
    return None

def plot_interactive_map(df):
    """Create an interactive map of housing prices in Tunisia"""
    # Create a copy with price in thousands for better display
    df_map = df.copy()
    df_map['price_thousands'] = df_map['price'] / 1000
    
    # Get theme-aware colors
    theme = 'light'
    if 'theme' in st.session_state:
        theme = st.session_state.theme
        
    color_scale = px.colors.sequential.Blues if theme == 'light' else px.colors.sequential.Blues_r
    bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
    text_color = '#333333' if theme == 'light' else '#E9EEF6'
    grid_color = '#EEEEEE' if theme == 'light' else '#2A3A4A'
    
    fig = px.scatter_geo(
        df_map,
        lat="latitude",
        lon="longitude",
        color="price",
        size="area_sqm",
        color_continuous_scale=color_scale,
        size_max=20,  # Increased from 15 for better visibility
        scope="africa",
        title="Housing Prices in Tunisia",
        hover_data=["price", "property_type", "region", "area_sqm", "bedrooms"],
        center={"lat": TUNISIA_CENTER_LAT, "lon": TUNISIA_CENTER_LON},
        projection="mercator",
        fitbounds="locations"
    )
    
    # Theme-aware map colors
    land_color = "LightGray" if theme == 'light' else "#2A3A4A"
    ocean_color = "LightBlue" if theme == 'light' else "#142438"
    coast_color = "Black" if theme == 'light' else "White"
    country_color = "Black" if theme == 'light' else "White"
    
    fig.update_geos(
        resolution=50,
        showcoastlines=True,
        coastlinecolor=coast_color,
        showland=True,
        landcolor=land_color,
        showocean=True,
        oceancolor=ocean_color,
        showcountries=True,
        countrycolor=country_color,
        countrywidth=0.5,
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},  # Increased top margin for title
        height=700,  # Increased from 600
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=14),  # Increased font size
        title=dict(font=dict(size=20)),  # Larger title font
        coloraxis_colorbar=dict(
            title="Price (TND)",
            tickformat=",.0f",
            len=0.8,  # Adjusted colorbar length
            thickness=20  # Increased colorbar thickness
        )
    )
    return fig

def plot_region_prices(df):
    """Plot average house prices by region"""
    region_avg = df.groupby('region')['price'].mean().sort_values(ascending=False).reset_index()
    
    # Get theme-aware colors
    theme = 'light'
    if 'theme' in st.session_state:
        theme = st.session_state.theme
        
    color_scale = px.colors.sequential.Blues if theme == 'light' else px.colors.sequential.Blues_r
    bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
    text_color = '#333333' if theme == 'light' else '#E9EEF6'
    grid_color = '#EEEEEE' if theme == 'light' else '#2A3A4A'
    
    fig = px.bar(
        region_avg,
        x='region',
        y='price',
        color='price',
        color_continuous_scale=color_scale,
        title="Average House Price by Region",
        labels={'price': 'Average Price (TND)', 'region': 'Region'},
        height=600  # Set explicit height
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        xaxis_tickangle=-45,
        yaxis_tickformat=',.0f',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=14),  # Increased font size
        title=dict(font=dict(size=20)),  # Larger title font
        margin=dict(l=50, r=50, t=80, b=100)  # Added margins for better spacing
    )
    
    # Add value labels on top of bars
    fig.update_traces(
        texttemplate='%{y:,.0f}',
        textposition='outside',
        textfont=dict(color=text_color, size=12)
    )
    
    # Update the grid color
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
    
    return fig

def plot_property_type_prices(df):
    """Plot average house prices by property type"""
    type_avg = df.groupby('property_type')['price'].mean().sort_values(ascending=False).reset_index()
    
    # Get theme-aware colors
    theme = 'light'
    if 'theme' in st.session_state:
        theme = st.session_state.theme
        
    color_scale = px.colors.sequential.Blues if theme == 'light' else px.colors.sequential.Blues_r
    bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
    text_color = '#333333' if theme == 'light' else '#E9EEF6'
    grid_color = '#EEEEEE' if theme == 'light' else '#2A3A4A'
    
    fig = px.bar(
        type_avg,
        x='property_type',
        y='price',
        color='price',
        color_continuous_scale=color_scale,
        title="Average House Price by Property Type",
        labels={'price': 'Average Price (TND)', 'property_type': 'Property Type'},
        height=550  # Set explicit height
    )
    
    fig.update_layout(
        xaxis={'categoryorder':'total descending'},
        yaxis_tickformat=',.0f',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=14),  # Increased font size
        title=dict(font=dict(size=20)),  # Larger title font
        margin=dict(l=50, r=50, t=80, b=50)  # Added margins
    )
    
    # Add value labels on top of bars
    fig.update_traces(
        texttemplate='%{y:,.0f}',
        textposition='outside',
        textfont=dict(color=text_color, size=12)
    )
    
    # Update the grid color
    fig.update_xaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, title_font=dict(size=14))
    
    return fig

def plot_price_vs_area(df):
    """Plot price vs area with regression line"""
    # Get theme-aware colors
    theme = 'light'
    if 'theme' in st.session_state:
        theme = st.session_state.theme
        
    bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
    text_color = '#333333' if theme == 'light' else '#E9EEF6'
    grid_color = '#EEEEEE' if theme == 'light' else '#2A3A4A'
    
    fig = px.scatter(
        df,
        x='area_sqm',
        y='price',
        color='property_type',
        opacity=0.7,
        size='price',  # Add size to enhance the visualization
        size_max=15,
        title="Price vs Area",
        labels={'area_sqm': 'Area (m²)', 'price': 'Price (TND)'},
        trendline="ols",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=600  # Set explicit height
    )
    
    fig.update_layout(
        yaxis_tickformat=',.0f',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=14),  # Increased font size
        title=dict(font=dict(size=20)),  # Larger title font
        margin=dict(l=50, r=50, t=80, b=50),  # Added margins
        legend=dict(
            font=dict(size=12),
            itemsizing='constant',
            bgcolor=bg_color,
            bordercolor=grid_color,
            borderwidth=1
        )
    )
    
    # Update the grid color
    fig.update_xaxes(
        showgrid=True, 
        gridcolor=grid_color, 
        title_font=dict(size=14),
        zeroline=True,
        zerolinecolor=grid_color,
        zerolinewidth=1.5
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor=grid_color, 
        title_font=dict(size=14),
        zeroline=True,
        zerolinecolor=grid_color,
        zerolinewidth=1.5
    )
    
    # Update legend
    fig.update_layout(
        legend=dict(
            bgcolor=bg_color,
            font=dict(color=text_color, size=12),
            title=dict(text="Property Type", font=dict(size=14)),
            borderwidth=1,
            bordercolor=grid_color
        )
    )
    
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features."""
    # Select numerical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    X = df[numerical_features]
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    
    # Create figure with larger size
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        square=True,
        cbar_kws={'shrink': .8}
    )
    
    # Customize the plot
    plt.title('Feature Correlation Heatmap', pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()

def format_price(price, currency="TND"):
    """Format price with thousands separator and currency"""
    return f"{price:,.0f} {currency}"

def apply_custom_theme():
    """Apply custom theme based on session state (light or dark mode)"""
    # Check for theme in session state
    theme = 'light'
    if 'theme' in st.session_state:
        theme = st.session_state.theme
    
    # Light theme colors
    light_theme = {
        'primary': '#2E5EAA',
        'primary_light': '#4A7CCF',
        'primary_dark': '#1A4A94',
        'secondary': '#F8B400',
        'accent': '#FC5C65',
        'text_dark': '#333333',
        'text_medium': '#555555',
        'text_light': '#777777',
        'bg_light': '#FFFFFF',
        'bg_medium': '#F5F7FA',
        'bg_dark': '#E9EEF6',
        'shadow': 'rgba(0, 0, 0, 0.05)',
    }
    
    # Dark theme colors
    dark_theme = {
        'primary': '#4A7CCF',
        'primary_light': '#5D8FE2',
        'primary_dark': '#2E5EAA',
        'secondary': '#F8B400',
        'accent': '#FC5C65',
        'text_dark': '#E9EEF6',
        'text_medium': '#C5D0E2',
        'text_light': '#A1AFCA',
        'bg_light': '#1E2A3E',
        'bg_medium': '#172030',
        'bg_dark': '#121A27',
        'shadow': 'rgba(0, 0, 0, 0.15)',
    }
    
    # Choose theme based on session state
    colors = light_theme if theme == 'light' else dark_theme
    
    st.markdown(f"""
    <style>
    /* Global variables */
    :root {{
        --primary: {colors['primary']};
        --primary-light: {colors['primary_light']};
        --primary-dark: {colors['primary_dark']};
        --secondary: {colors['secondary']};
        --accent: {colors['accent']};
        --text-dark: {colors['text_dark']};
        --text-medium: {colors['text_medium']};
        --text-light: {colors['text_light']};
        --bg-light: {colors['bg_light']};
        --bg-medium: {colors['bg_medium']};
        --bg-dark: {colors['bg_dark']};
        --shadow: {colors['shadow']};
        --border-radius: 12px;
        --spacing-xs: 0.25rem;
        --spacing-sm: 0.5rem;
        --spacing-md: 1rem;
        --spacing-lg: 1.5rem;
        --spacing-xl: 2rem;
    }}
    
    /* Reset and base styles */
    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-dark);
        background-color: var(--bg-medium);
        line-height: 1.6;
    }}
    
    /* Dark theme overrides for Streamlit components */
    .stApp {{
        background-color: var(--bg-medium);
    }}
    
    /* Main layout */
    .main .block-container {{
        max-width: 1280px;
        background-color: var(--bg-light);
        border-radius: var(--border-radius);
        box-shadow: 0 4px 20px var(--shadow);
        padding: var(--spacing-xl);
        margin: var(--spacing-lg) auto;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-light);
        border-right: 1px solid var(--bg-dark);
        box-shadow: 2px 0 10px var(--shadow);
    }}
    
    section[data-testid="stSidebar"] > div {{
        padding: var(--spacing-lg);
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--primary-dark);
        font-weight: 700;
        margin-bottom: var(--spacing-md);
        letter-spacing: -0.025em;
    }}
    
    h1 {{
        font-size: 2.25rem;
        margin-bottom: var(--spacing-lg);
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: var(--spacing-sm);
        display: inline-block;
    }}
    
    h2 {{
        font-size: 1.75rem;
        color: var(--primary);
    }}
    
    h3 {{
        font-size: 1.5rem;
        color: var(--primary);
    }}
    
    /* Navigation menu */
    .stButton, [data-testid="stSidebarNavItems"] {{
        margin-top: var(--spacing-md);
    }}
    
    /* SELECT BOX FIXES - GLOBAL */
    /* Better styling for selectbox containers */
    div[data-baseweb="select"] {{
        margin-bottom: 1rem;
    }}
    
    div[data-baseweb="select"] > div {{
        background-color: var(--bg-light) !important;
        border-radius: 10px !important;
        border: 1px solid var(--primary) !important;
        transition: all 0.2s ease !important;
        padding: 12px 16px !important;
        height: auto !important;
        min-height: 50px !important;
    }}
    
    div[data-baseweb="select"] > div:hover {{
        border-color: var(--primary-light) !important;
        box-shadow: 0 3px 8px var(--shadow) !important;
    }}
    
    /* Core Fixes for All Text Elements in Dropdowns */
    /* Target every possible text element inside the select component */
    div[data-baseweb="select"] *, 
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] label,
    [data-testid="stSelectbox"] *,
    [data-testid="stMultiSelect"] * {{
        color: var(--text-dark) !important;
        line-height: 1.4 !important;
    }}
    
    /* Force color on the most specific elements */
    div[data-baseweb="select"] div[data-testid="stMarkdown"] p,
    div[data-baseweb="select"] [aria-selected="true"],
    div[data-baseweb="select"] [role="option"],
    div[data-baseweb="select"] span[title],
    div[data-baseweb="select"] div[title],
    div[data-baseweb="select"] span.st-emotion-cache-ue6h4q,
    div[data-baseweb="select"] span.st-emotion-cache-90p5uc,
    div[data-baseweb="select"] span.st-emotion-cache-1gk2i3z {{
        color: var(--text-dark) !important;
        font-weight: 500 !important;
        line-height: 1.4 !important;
    }}
    
    /* Ensure selected value is always visible */
    div[data-baseweb="select"] [aria-selected="true"] div,
    div[data-baseweb="select"] [aria-selected="true"] span,
    div[data-baseweb="select"] [class*="valueContainer"] div,
    div[data-baseweb="select"] [class*="valueContainer"] span {{
        color: var(--primary) !important;
        font-weight: 600 !important;
        line-height: 1.4 !important;
    }}
    
    /* Target the region and property type select boxes specifically */
    #root > div:nth-of-type(1) > div > div > div > div > section > div > div > div > div > div[data-baseweb="select"] div,
    #root > div:nth-of-type(1) > div > div > div > div > section > div > div > div > div > div[data-baseweb="select"] span {{
        color: var(--text-dark) !important;
        line-height: 1.4 !important;
    }}
    
    /* Fix for Preferred Region dropdown in ProfilePage */
    [data-testid="stSelectbox"] div[data-baseweb="select"] {{
        pointer-events: auto !important;
        cursor: pointer !important;
    }}
    
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {{
        position: relative !important;
        z-index: 100 !important;
        cursor: pointer !important;
    }}
    
    /* Ensure all dropdowns are clickable */
    div[role="combobox"] {{
        cursor: pointer !important;
        pointer-events: auto !important;
    }}
    
    /* Ensure dropdown button is clickable */
    [data-baseweb="select"] [role="button"],
    [data-baseweb="select"] [role="button"] svg {{
        pointer-events: auto !important;
        cursor: pointer !important;
        opacity: 1 !important;
    }}
    
    /* Dropdown menu styling */
    ul[role="listbox"] {{
        background-color: var(--bg-light) !important;
        border-color: var(--bg-dark) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px var(--shadow) !important;
        overflow: hidden !important;
        border: 1px solid var(--primary-light) !important;
        padding: 8px 0 !important;
    }}
    
    ul[role="listbox"] li {{
        color: var(--text-dark) !important;
        padding: 16px 16px !important;
        font-size: 15px !important;
        transition: background-color 0.2s ease !important;
        margin: 4px 0 !important;
        line-height: 1.4 !important;
        min-height: 50px !important;
        display: flex !important;
        align-items: center !important;
    }}
    
    /* Ensure option text is visible */
    ul[role="listbox"] li *,
    ul[role="listbox"] [role="option"] *,
    ul[role="listbox"] [role="option"] span,
    ul[role="listbox"] [role="option"] div {{
        color: var(--text-dark) !important;
        line-height: 1.4 !important;
    }}
    
    /* Fix line height in selectbox main input area */
    div[data-baseweb="select"] [data-testid="stMarkdown"] p {{
        margin: 0 !important;
        line-height: 1.4 !important;
        display: block !important;
        overflow: visible !important;
        white-space: normal !important;
        height: auto !important;
        padding: 8px 0 !important;
    }}
    
    /* Add additional fixes for the dropdown display */
    div[data-baseweb="select"] [data-testid="stMarkdown"] {{
        height: auto !important;
        line-height: 1.4 !important;
        min-height: 24px !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    
    /* Fix any height restrictions that might cut off text */
    div[data-baseweb="select"] [class*="valueContainer"] {{
        height: auto !important;
        min-height: 24px !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        display: flex !important;
        align-items: center !important;
    }}
    
    /* Hover state for options */
    ul[role="listbox"] li:hover {{
        background-color: var(--bg-medium) !important;
    }}
    
    /* Selected option styles with higher priority */
    [data-baseweb="select"] [aria-selected="true"],
    ul[role="listbox"] [aria-selected="true"],
    div[data-baseweb="select"] [aria-selected="true"] * {{
        color: var(--primary) !important;
        font-weight: 600 !important;
        background-color: var(--bg-medium) !important;
    }}
    
    /* Ensure placeholder and selected value are visible */
    div[data-baseweb="select"] span[title],
    div[data-baseweb="select"] div[title],
    [data-testid="stSelectbox"] label span {{
        color: var(--text-dark) !important;
    }}
    
    /* Target dynamic Streamlit class names */
    .st-emotion-cache-1w26yst,
    .st-emotion-cache-10trblm,
    .st-emotion-cache-16idsys p,
    .st-emotion-cache-16idsys,
    .st-emotion-cache-ue6h4q,
    .st-emotion-cache-90p5uc,
    .st-emotion-cache-1gk2i3z {{
        color: var(--text-dark) !important;
    }}
    
    /* Extra theme-specific overrides for select boxes */
    .stApp[data-theme="dark"] div[data-baseweb="select"] > div {{
        background-color: var(--bg-light) !important;
        border-color: var(--primary) !important;
    }}
    
    .stApp[data-theme="light"] div[data-baseweb="select"] > div {{
        background-color: var(--bg-light) !important;
        border-color: var(--primary) !important;
    }}
    
    /* Card components */
    div.stMetric, div.stDataFrame, div.stTable {{
        background-color: var(--bg-light);
        border-radius: var(--border-radius);
        box-shadow: 0 4px 10px var(--shadow);
        padding: var(--spacing-md);
        margin-bottom: var(--spacing-lg);
        border: 1px solid var(--bg-dark);
        transition: transform 0.2s ease;
    }}
    
    div.stMetric:hover, div.stDataFrame:hover, div.stTable:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 15px var(--shadow);
    }}
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }}
    
    div[data-testid="stMetricLabel"] {{
        font-size: 1rem;
        color: var(--text-medium);
    }}
    
    /* Form elements */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {{
        border-radius: var(--border-radius);
        border: 1px solid var(--bg-dark);
        padding: var(--spacing-md);
        font-size: 1rem;
        transition: all 0.3s ease;
        background-color: var(--bg-light);
        color: var(--text-dark);
    }}
    
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {{
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(46, 94, 170, 0.2);
    }}
    
    /* Button styling */
    button[kind="primary"] {{
        background-color: var(--primary);
        color: {colors['bg_light'] if theme == 'dark' else 'white'};
        border: none;
        border-radius: var(--border-radius);
        padding: var(--spacing-md) var(--spacing-lg);
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    button[kind="primary"]:hover {{
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    button[kind="secondary"] {{
        background-color: var(--bg-medium);
        color: var(--primary);
        border: 1px solid var(--primary);
        border-radius: var(--border-radius);
        padding: var(--spacing-md) var(--spacing-lg);
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    button[kind="secondary"]:hover {{
        background-color: var(--bg-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow);
    }}
    </style>
    """, unsafe_allow_html=True)

def display_matplotlib_fig(fig, title=None):
    """Helper function to safely display a matplotlib figure in Streamlit with correct theming"""
    theme = 'light'
    if 'theme' in st.session_state:
        theme = st.session_state.theme
        
    bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
    
    # Add title if provided
    if title:
        st.markdown(f"<h3 style='font-size: 24px; margin-bottom: 20px;'>{title}</h3>", unsafe_allow_html=True)
        
    # Ensure figure has correct background color
    if hasattr(fig, 'patch'):
        fig.patch.set_facecolor(bg_color)
    
    # Add container CSS to ensure proper sizing
    st.markdown("""
    <style>
    .stPlotlyChart, .element-container div[data-testid="stImage"] {
        height: auto !important;
        min-height: 400px;
        margin: 1rem 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display the figure with explicit width and background color
    st.pyplot(fig, use_container_width=True, facecolor=bg_color)
    
    # Add some spacing
    st.write("")
