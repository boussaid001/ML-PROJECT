import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from .BasePage import BasePage
from utils import preprocess_data, format_price, plot_feature_importance, display_matplotlib_fig, apply_custom_theme
from database import save_user_prediction

class PredictionPage(BasePage):
    """Page for house price prediction."""
    
    def __init__(self, df, models):
        super().__init__(title="Prediction")
        self.df = df
        self.models = models
    
    def render(self):
        """Render the price prediction page."""
        st.header("🏠 House Price Prediction")
        
        # Apply theme-aware styling
        apply_custom_theme()
        
        # Get theme colors
        theme = 'light'
        if 'theme' in st.session_state:
            theme = st.session_state.theme
            
        bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
        text_color = '#333333' if theme == 'light' else '#E9EEF6'
        primary = '#2E5EAA' if theme == 'light' else '#4A7CCF'
        primary_light = '#4A7CCF' if theme == 'light' else '#5D8FE2'
        
        # Custom CSS for better visualization layout
        st.markdown("""
        <style>
        /* Better spacing for charts */
        .element-container {
            margin-bottom: 2rem !important;
        }
        
        /* Chart container styling */
        .stPlotlyChart, .element-container div[data-testid="stImage"] {
            height: auto !important;
            min-height: 450px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
        }
        
        /* Section headers styling */
        h3 {
            margin-top: 2rem !important;
            margin-bottom: 1.5rem !important;
            font-size: 1.8rem !important;
        }
        
        /* Better spacing between sections */
        .section-spacing {
            margin-top: 2.5rem;
            margin-bottom: 1rem;
        }
        
        /* Model selection styling */
        div[data-baseweb="select"] > div {
            font-size: 16px !important;
            padding: 8px !important;
        }
        
        /* Prediction result card styling */
        .prediction-card {
            background-color: var(--bg-light);
            border-radius: 12px;
            padding: 20px 30px;
            margin: 30px 0;
            border-left: 5px solid var(--primary);
            box-shadow: 0 4px 15px var(--shadow);
        }
        
        .prediction-title {
            color: var(--primary);
            margin-top: 0;
            font-size: 20px;
            font-weight: 600;
        }
        
        .prediction-value {
            font-size: 36px;
            font-weight: bold;
            margin: 15px 0;
        }
        
        .prediction-source {
            font-size: 14px;
            opacity: 0.8;
        }
        
        /* Custom input form styling */
        .prediction-form {
            background-color: var(--bg-light);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px var(--shadow);
            margin-bottom: 30px;
        }
        
        /* Prediction result styling */
        .prediction-result {
            background-color: var(--bg-light);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px var(--shadow);
            margin-top: 20px;
            border-left: 5px solid var(--primary);
            text-align: center;
        }
        
        .prediction-range {
            font-size: 18px;
            color: var(--text-medium);
            margin-bottom: 10px;
        }
        
        /* Feature importance styling */
        .feature-importance {
            background-color: var(--bg-light);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px var(--shadow);
            margin-top: 30px;
        }
        
        /* Input label styling */
        .input-label {
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-dark);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Model selection with better styling
        st.markdown("<h3 style='margin-top: 0.5rem;'>Select Model</h3>", unsafe_allow_html=True)
        model_name = st.selectbox(
            "Choose a prediction model",
            list(self.models.keys()),
            help="Select the machine learning model to use for prediction"
        )
        model = self.models[model_name]
        
        # Add explanatory text about the model
        model_descriptions = {
            "Linear Regression": "A simple model that assumes a linear relationship between variables. Fast and interpretable, but may miss complex patterns.",
            "Random Forest": "An ensemble of decision trees that provides better accuracy. Good for capturing non-linear relationships in the data.",
            "XGBoost": "An advanced gradient boosting model with high accuracy. Excellent for complex data relationships and generally provides the best predictions."
        }
        
        if model_name in model_descriptions:
            st.markdown(
                f"<div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; "
                f"border-left: 3px solid var(--primary); margin-bottom: 20px;'>"
                f"<p style='color: {text_color}; margin: 0;'><strong>{model_name}:</strong> {model_descriptions[model_name]}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Input form with improved layout
        st.markdown("<h3>Property Details</h3>", unsafe_allow_html=True)
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                region = st.selectbox("Region", self.df['region'].unique())
                property_type = st.selectbox("Property Type", self.df['property_type'].unique())
                property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5)
                area_sqm = st.number_input("Area (m²)", min_value=20, max_value=1000, value=100)
                bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2)
            
            with col2:
                bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)
                distance_to_center = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=50.0, step=0.1, value=5.0)
                floor = st.number_input("Floor Number", min_value=0, max_value=50, value=1)
                has_elevator = st.checkbox("Has Elevator")
                has_garden = st.checkbox("Has Garden")
                has_parking = st.checkbox("Has Parking")
            
            # More prominent submit button
            submit = st.form_submit_button(
                "Predict Price", 
                use_container_width=True,
                help="Click to predict the house price based on these features"
            )
            
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
                    'has_parking': [has_parking],
                    'latitude': [0.0],  # Placeholder
                    'longitude': [0.0]  # Placeholder
                })
                
                # Preprocess input data
                X = preprocess_data(input_data)
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Calculate prediction range (±10%)
                lower_bound = prediction * 0.9
                upper_bound = prediction * 1.1
                
                # Create a styled card for the prediction result - using CSS classes
                st.markdown(f"""
                <div class="prediction-card">
                    <h3 class="prediction-title">Predicted Price</h3>
                    <p class="prediction-value" style="color: {text_color};">
                        {format_price(prediction)}
                    </p>
                    <p class="prediction-source" style="color: {text_color};">
                        Based on the {model_name} model
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add visual spacing
                st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
                
                # Show feature importance for the selected model
                if hasattr(model, 'feature_importances_'):
                    st.markdown("<h3>Feature Importance Analysis</h3>", unsafe_allow_html=True)
                    
                    # Explanatory text about feature importance
                    st.markdown(
                        "<p style='margin-bottom: 20px;'>The chart below shows which features had the most influence on the prediction. "
                        "Taller bars indicate features that were more important in determining the predicted price.</p>",
                        unsafe_allow_html=True
                    )
                    
                    # Get feature names
                    feature_names = np.array([
                        'Region', 'Property Type', 'Age', 'Area', 'Bedrooms', 
                        'Bathrooms', 'Distance to Center', 'Floor', 'Elevator', 
                        'Garden', 'Parking', 'Latitude', 'Longitude'
                    ])
                    
                    # Get the feature importance plot
                    importance_fig = plot_feature_importance(model, feature_names)
                    
                    # Display the feature importance plot using our helper function
                    display_matplotlib_fig(importance_fig)
                    
                    # Add insights about the most important features
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    top_features = [feature_names[i] for i in indices[:3]]
                    
                    st.markdown(
                        f"<div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; "
                        f"border-left: 3px solid var(--primary); margin: 20px 0;'>"
                        f"<p style='color: {text_color}; margin: 0;'><strong>Key Insights:</strong> "
                        f"For this prediction, the most influential factors were <strong>{top_features[0]}</strong>, "
                        f"<strong>{top_features[1]}</strong>, and <strong>{top_features[2]}</strong>.</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                # Add visual spacing
                st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
                
                # Show similar properties
                st.markdown("<h3>Similar Properties</h3>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='margin-bottom: 20px;'>These are actual properties in the dataset with similar characteristics. "
                    "Compare them to see how your predicted price compares to real market data.</p>",
                    unsafe_allow_html=True
                )
                
                similar_properties = self.df[
                    (self.df['region'] == region) &
                    (self.df['property_type'] == property_type) &
                    (self.df['bedrooms'] == bedrooms)
                ].head(5)
                
                if not similar_properties.empty:
                    # Create styled property cards
                    for _, property in similar_properties.iterrows():
                        price_diff = abs(property['price'] - prediction) / prediction * 100
                        diff_text = f"{price_diff:.1f}% {'higher' if property['price'] > prediction else 'lower'}"
                        diff_color = "#dc3545" if property['price'] > prediction else "#28a745"
                        
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; margin-bottom: 15px; 
                             border-left: 3px solid var(--primary); display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 3;">
                                <p style="font-weight: bold; color: {text_color}; margin: 0; font-size: 18px;">
                                    {property['property_type']} in {property['region']}
                                </p>
                                <p style="color: {text_color}; opacity: 0.8; margin: 8px 0;">
                                    {property['bedrooms']} bedrooms • {property['bathrooms']} bathrooms • {property['area_sqm']} m²
                                </p>
                                <p style="color: {text_color}; opacity: 0.7; margin: 5px 0; font-size: 14px;">
                                    Age: {property['property_age']} years • Floor: {property['floor']} • 
                                    {property['distance_to_center']:.1f}km from center
                                </p>
                            </div>
                            <div style="flex: 1; text-align: right;">
                                <p style="font-weight: bold; color: var(--primary); margin: 0; font-size: 20px;">
                                    {format_price(property['price'])}
                                </p>
                                <p style="color: {diff_color}; margin: 5px 0; font-size: 14px; font-weight: 500;">
                                    {diff_text}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No similar properties found in the dataset. Try different property characteristics.")
                
                # Save prediction to database if user is logged in
                if st.session_state.get('logged_in', False) and st.session_state.get('user_id'):
                    save_user_prediction(
                        st.session_state.user_id,
                        region,
                        property_type,
                        float(area_sqm),
                        int(bedrooms),
                        int(bathrooms),
                        float(prediction)
                    )
                    st.success("Prediction saved to your profile!")
                
                # Additional analysis - price per square meter
                price_per_sqm = prediction / area_sqm
                avg_price_per_sqm = self.df[self.df['region'] == region]['price'].mean() / self.df[self.df['region'] == region]['area_sqm'].mean()
                
                # Create comparison
                st.subheader("Price per Square Meter Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Your Property", 
                        f"{price_per_sqm:,.0f} TND/m²", 
                        f"{((price_per_sqm - avg_price_per_sqm) / avg_price_per_sqm) * 100:.1f}% vs Regional Average"
                    )
                
                with col2:
                    st.metric(
                        f"Average in {region}", 
                        f"{avg_price_per_sqm:,.0f} TND/m²"
                    )
                
        # Instruction section
        with st.expander("How does this work?"):
            st.markdown("""
            Our prediction model is built using machine learning algorithms trained on real Tunisian property data. Here's how it works:
            
            1. **Enter property details**: Provide information about the property you're interested in.
            2. **AI prediction**: Our model analyzes the features and compares them with thousands of similar properties.
            3. **Get estimate**: Receive an estimated market price based on current trends and comparable properties.
            
            The prediction provides a reasonable estimate, but actual market values may vary based on additional factors like exact location, property condition, and market fluctuations.
            """)
            
        # Display some statistics to help users understand the market
        with st.expander("Market Insights"):
            # Calculate statistics
            avg_price = self.df['price'].mean()
            avg_price_apartment = self.df[self.df['property_type'] == 'Apartment']['price'].mean()
            avg_price_house = self.df[self.df['property_type'] == 'House']['price'].mean() if 'House' in self.df['property_type'].unique() else 0
            avg_price_villa = self.df[self.df['property_type'] == 'Villa']['price'].mean() if 'Villa' in self.df['property_type'].unique() else 0
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Average Property Price", f"{avg_price:,.0f} TND")
            col2.metric("Average Apartment", f"{avg_price_apartment:,.0f} TND")
            col3.metric("Average Villa", f"{avg_price_villa:,.0f} TND" if avg_price_villa > 0 else "N/A") 