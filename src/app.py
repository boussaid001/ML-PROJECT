import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import os
import sys
from pathlib import Path
from utils import (
    load_data, preprocess_data, load_model, plot_interactive_map,
    plot_region_prices, plot_property_type_prices, plot_price_vs_area,
    plot_correlation_heatmap, apply_custom_theme, format_price
)
from database import init_database
from auth import register_user, login_user, verify_token

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import from local modules
from config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    PAGE_TITLE,
    PAGE_ICON,
    LINEAR_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
    REGIONS,
    PROPERTY_TYPES
)

# Initialize database
init_database()

# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom theme
apply_custom_theme()

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None

# Load data
@st.cache_data
def get_data():
    return load_data()

# Load models
@st.cache_resource
def get_models():
    models = {
        "Linear Regression": load_model("linear_regression.pkl"),
        "Random Forest": load_model("random_forest.pkl"),
        "XGBoost": load_model("xgboost.pkl")
    }
    return {name: model for name, model in models.items() if model is not None}

# Initialize data and models
df = get_data()
models = get_models()

# Page header with Tunisia flag colors
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <div style="margin-right: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ce/Flag_of_Tunisia.svg" width="60">
    </div>
    <div>
        <h1>🏠 Tunisia House Price Prediction</h1>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Predict real estate prices across all regions of Tunisia using advanced machine learning models.
Explore property trends, compare prices by location, and get accurate predictions for your property investments.
""")

# Sidebar navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/ce/Flag_of_Tunisia.svg", width=100)
    st.title("Navigation")
    
    if st.session_state.user:
        # Logged in menu
        selected = option_menu(
            menu_title=None,
            options=["Explore Market", "Regional Analysis", "Prediction", "Profile", "Logout"],
            icons=["graph-up", "geo-alt", "calculator", "person", "box-arrow-right"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                "icon": {"color": "#E41E25", "font-size": "14px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee", "color": "#E41E25"},
                "nav-link-selected": {"background-color": "#E41E25", "color": "#FFFFFF"},
            }
        )
        
        if selected == "Logout":
            st.session_state.user = None
            st.rerun()
    else:
        # Not logged in menu
        selected = option_menu(
            menu_title=None,
            options=["Login", "Register"],
            icons=["box-arrow-in-right", "person-plus"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                "icon": {"color": "#E41E25", "font-size": "14px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee", "color": "#E41E25"},
                "nav-link-selected": {"background-color": "#E41E25", "color": "#FFFFFF"},
            }
        )

# Check if models are already trained
if not models:
    st.sidebar.warning("Models not found. Please train models first.")
    if st.sidebar.button("Generate Data & Train Models"):
        with st.spinner("Generating data and training models... This may take a few minutes."):
            import subprocess
            
            # First generate the data
            st.info("Step 1: Generating Tunisia housing dataset...")
            gen_result = subprocess.run(
                ["python", str(Path(__file__).parent / "generate_tunisia_data.py")], 
                capture_output=True, 
                text=True
            )
            
            if gen_result.returncode == 0:
                st.success("Dataset generated successfully!")
                
                # Then train the models
                st.info("Step 2: Training prediction models...")
                train_result = subprocess.run(
                    ["python", str(Path(__file__).parent / "train_models.py")], 
                    capture_output=True, 
                    text=True
                )
                
                if train_result.returncode == 0:
                    st.success("Models trained successfully!")
                    st.experimental_rerun()
                else:
                    st.error(f"Error training models: {train_result.stderr}")
            else:
                st.error(f"Error generating dataset: {gen_result.stderr}")

# Login page
if selected == "Login":
    st.header("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, result = login_user(username, password)
            if success:
                st.session_state.user = result
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(result)

# Register page
elif selected == "Register":
    st.header("📝 Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                success, message = register_user(username, email, password)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

# Profile page
elif selected == "Profile":
    st.header("👤 Profile")
    st.write(f"Username: {st.session_state.user['username']}")
    st.write(f"User ID: {st.session_state.user['user_id']}")

# Rest of the app (only accessible when logged in)
elif st.session_state.user:
    # Check if data is loaded successfully
    if df is None:
        st.error("Error: Could not load dataset. Please check if the dataset file exists.")
        st.info("Run the following command to generate the dataset: `python src/generate_tunisia_data.py`")
        st.stop()

    # Explore Market page
    if selected == "Explore Market":
        st.header("📊 Tunisia Real Estate Market Explorer")
        
        # Data overview
        with st.expander("Dataset Preview"):
            st.dataframe(df.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Shape:", df.shape)
            with col2:
                st.write("Missing Values:", df.isnull().sum().sum())
        
        # Price distribution
        st.subheader("Property Price Distribution")
        
        fig = px.histogram(
            df, 
            x="price", 
            nbins=50,
            color_discrete_sequence=["#E41E25"],
            title="Distribution of Property Prices in Tunisia",
            labels={"price": "Price (TND)"}
        )
        fig.update_layout(xaxis_tickformat=',.0f')
        st.plotly_chart(fig, use_container_width=True)
        
        # Price statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Price", format_price(df['price'].mean()))
        with col2:
            st.metric("Median Price", format_price(df['price'].median()))
        with col3:
            st.metric("Minimum Price", format_price(df['price'].min()))
        with col4:
            st.metric("Maximum Price", format_price(df['price'].max()))
        
        # Feature relationships
        st.subheader("Property Features Analysis")
        
        # Price vs Area
        st.plotly_chart(plot_price_vs_area(df), use_container_width=True)
        
        # Other feature relationships
        feature_cols = st.columns(2)
        
        with feature_cols[0]:
            # Price vs bedrooms
            bedroom_data = df.groupby('bedrooms')['price'].mean().reset_index()
            fig = px.bar(
                bedroom_data,
                x='bedrooms',
                y='price',
                title="Average Price by Number of Bedrooms",
                labels={"price": "Average Price (TND)", "bedrooms": "Number of Bedrooms"},
                color='price',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':sorted(df['bedrooms'].unique())})
            st.plotly_chart(fig, use_container_width=True)
        
        with feature_cols[1]:
            # Price vs property age
            fig = px.scatter(
                df,
                x='property_age',
                y='price',
                color='property_type',
                opacity=0.7,
                title="Price vs Property Age",
                labels={"price": "Price (TND)", "property_age": "Property Age (years)"}
            )
            fig.update_layout(yaxis_tickformat=',.0f')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        corr_fig = plot_correlation_heatmap(df)
        st.pyplot(corr_fig)
        
        # Impact of boolean features
        st.subheader("Impact of Property Amenities")
        
        amenity_cols = st.columns(3)
        
        bool_features = ['has_elevator', 'has_garden', 'has_parking']
        for i, feature in enumerate(bool_features):
            with amenity_cols[i]:
                avg_with = df[df[feature] == True]['price'].mean()
                avg_without = df[df[feature] == False]['price'].mean()
                diff_pct = (avg_with / avg_without - 1) * 100
                
                feature_name = feature.replace('has_', '').capitalize()
                
                fig = px.bar(
                    x=['With ' + feature_name, 'Without ' + feature_name],
                    y=[avg_with, avg_without],
                    color=[avg_with, avg_without],
                    color_continuous_scale=px.colors.sequential.Reds,
                    title=f"Price With/Without {feature_name}",
                    labels={"x": "", "y": "Average Price (TND)"}
                )
                fig.update_layout(yaxis_tickformat=',.0f')
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"Properties with {feature_name.lower()} are **{diff_pct:.1f}%** more expensive on average.")

    # Regional Analysis page
    elif selected == "Regional Analysis":
        st.header("🗺️ Regional Analysis")
        
        # Interactive map
        st.subheader("Property Prices Across Tunisia")
        map_fig = plot_interactive_map(df)
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Regional price comparison
        st.subheader("Regional Price Comparison")
        region_fig = plot_region_prices(df)
        st.plotly_chart(region_fig, use_container_width=True)
        
        # Property type distribution by region
        st.subheader("Property Types by Region")
        type_fig = plot_property_type_prices(df)
        st.plotly_chart(type_fig, use_container_width=True)

    # Prediction page
    elif selected == "Prediction":
        st.header("🏠 House Price Prediction")
        
        # Model selection
        model_name = st.selectbox("Select Model", list(models.keys()))
        model = models[model_name]
        
        # Input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                region = st.selectbox("Region", df['region'].unique())
                property_type = st.selectbox("Property Type", df['property_type'].unique())
                property_age = st.number_input("Property Age (years)", min_value=0, max_value=100)
                area_sqm = st.number_input("Area (sqm)", min_value=20, max_value=1000)
                bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10)
            
            with col2:
                bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10)
                distance_to_center = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, step=0.1)
                floor = st.number_input("Floor Number", min_value=0, max_value=50)
                has_elevator = st.checkbox("Has Elevator")
                has_garden = st.checkbox("Has Garden")
                has_parking = st.checkbox("Has Parking")
            
            submit = st.form_submit_button("Predict Price")
            
            if submit:
                # Prepare input data
                input_data = pd.DataFrame({
                    'region': [region],
                    'property_type': [property_type],
                    'property_age': [property_age],
                    'area_sqm': [area_sqm],
                    'bedrooms': [bedrooms],
                    'bathrooms': [bathrooms],
                    'distance_to_center': [distance_to_center],
                    'floor': [floor],
                    'has_elevator': [has_elevator],
                    'has_garden': [has_garden],
                    'has_parking': [has_parking]
                })
                
                # Preprocess input data
                X = preprocess_data(input_data)
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Display result
                st.success(f"Predicted Price: {format_price(prediction)}")
                
                # Show similar properties
                st.subheader("Similar Properties")
                similar_properties = df[
                    (df['region'] == region) &
                    (df['property_type'] == property_type) &
                    (df['bedrooms'] == bedrooms)
                ].head(5)
                
                if not similar_properties.empty:
                    st.dataframe(similar_properties[['region', 'property_type', 'bedrooms', 'price']])
                else:
                    st.info("No similar properties found in the dataset.")

    # About page
    elif selected == "About":
        st.header("ℹ️ About")
        st.write("""
        This application uses machine learning to predict house prices in Tunisia.
        The model takes into account various factors such as location, property features,
        and market conditions to provide accurate price estimates.
        
        ### Features
        - Interactive data visualization
        - Multiple prediction models
        - Regional analysis
        - Property type comparison
        
        ### Data Sources
        - Synthetic dataset based on Tunisian housing market characteristics
        - Covers all major regions in Tunisia
        - Includes various property types and features
        
        ### Models Used
        - Linear Regression
        - Random Forest
        - XGBoost
        
        ### Performance Metrics
        - RMSE (Root Mean Square Error)
        - MAE (Mean Absolute Error)
        - R² Score
        """)
else:
    st.warning("Please login to access the application.")
