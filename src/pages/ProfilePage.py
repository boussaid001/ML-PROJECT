import streamlit as st
import pandas as pd
import numpy as np
from .BasePage import BasePage
from utils import format_price
from database import get_user_searches, get_user_predictions, get_saved_properties, delete_saved_property

class ProfilePage(BasePage):
    """User profile page."""
    
    def __init__(self, df):
        super().__init__(title="Profile")
        self.df = df
    
    def render(self):
        """Render the user profile page."""
        if not st.session_state.logged_in:
            st.warning("Please log in to view your profile.")
            return
        
        try:
            # Get user data from database
            user_searches = get_user_searches(st.session_state.user_id)
            user_predictions = get_user_predictions(st.session_state.user_id)
            saved_properties = get_saved_properties(st.session_state.user_id)
            
            # Determine current theme
            theme = 'light'
            if 'theme' in st.session_state:
                theme = st.session_state.theme
                
            # Theme-specific colors
            bg_color = '#FFFFFF' if theme == 'light' else '#1E2A3E'
            bg_medium = '#F5F7FA' if theme == 'light' else '#172030'
            text_color = '#333333' if theme == 'light' else '#E9EEF6'
            text_medium = '#555555' if theme == 'light' else '#C5D0E2'
            primary = '#2E5EAA' if theme == 'light' else '#4A7CCF'
            primary_dark = '#1A4A94' if theme == 'light' else '#2E5EAA'
            border_color = '#E9EEF6' if theme == 'light' else '#121A27'
            shadow = 'rgba(0, 0, 0, 0.05)' if theme == 'light' else 'rgba(0, 0, 0, 0.15)'
            
            # Simple profile card with modern theme colors
            username_initial = st.session_state.username[0].upper() if st.session_state.username else "U"
            
            # HTML content with updated styling to match theme
            profile_html = f"""
            <style>
            .profile-card {{
                background-color: {bg_color};
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 15px {shadow};
                margin-bottom: 20px;
                border-left: 5px solid {primary};
            }}
            .profile-header {{
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }}
            .profile-avatar {{
                background-color: {primary};
                color: {bg_color};
                width: 80px;
                height: 80px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 36px;
                margin-right: 20px;
                box-shadow: 0 4px 8px rgba(46, 94, 170, 0.2);
            }}
            .profile-name {{
                font-size: 24px;
                font-weight: bold;
                color: {primary_dark};
            }}
            .profile-id {{
                font-size: 14px;
                color: {text_medium};
            }}
            .profile-stats {{
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
            }}
            .stat-box {{
                text-align: center;
                padding: 15px;
                background-color: {bg_medium};
                border-radius: 12px;
                width: 30%;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .stat-box:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px {shadow};
            }}
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: {primary};
            }}
            .stat-label {{
                font-size: 14px;
                color: {text_medium};
                margin-top: 5px;
            }}
            hr {{
                border: none;
                height: 1px;
                background-color: {border_color};
                margin: 15px 0;
            }}
            </style>
            
            <div class="profile-card">
                <div class="profile-header">
                    <div class="profile-avatar">{username_initial}</div>
                    <div>
                        <div class="profile-name">{st.session_state.username}</div>
                        <div class="profile-id">User ID: {st.session_state.user_id}</div>
                    </div>
                </div>
                <hr>
                <div class="profile-stats">
                    <div class="stat-box">
                        <div class="stat-number">{len(user_searches)}</div>
                        <div class="stat-label">Searches</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{len(user_predictions)}</div>
                        <div class="stat-label">Predictions</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{len(saved_properties)}</div>
                        <div class="stat-label">Saved Properties</div>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(profile_html, unsafe_allow_html=True)
            
            # User preferences section
            st.subheader("🎯 Your Preferences")
            
            col1, col2 = st.columns(2)
            with col1:
                preferred_region = st.selectbox("Preferred Region", options=sorted(self.df['region'].unique()))
                preferred_min_price = st.number_input("Minimum Budget (TND)", min_value=0, max_value=5000000, value=100000, step=50000)
            
            with col2:
                preferred_property_type = st.selectbox("Preferred Property Type", options=sorted(self.df['property_type'].unique()))
                preferred_max_price = st.number_input("Maximum Budget (TND)", min_value=0, max_value=5000000, value=500000, step=50000)
            
            if st.button("Save Preferences"):
                st.success("Preferences saved successfully!")
            
            # Recent activity section - using tabs to organize different types of data
            st.subheader("📊 Your Activity")
            
            tab1, tab2, tab3 = st.tabs(["Recent Searches", "Recent Predictions", "Saved Properties"])
            
            with tab1:
                if user_searches:
                    # Convert to DataFrame for display
                    searches_df = pd.DataFrame(user_searches)
                    # Format the data for display
                    searches_display = pd.DataFrame({
                        "Date": [s['search_date'] for s in user_searches],
                        "Search Term": [s['search_term'] for s in user_searches],
                        "Region": [s['region'] or "Any" for s in user_searches],
                        "Property Type": [s['property_type'] or "Any" for s in user_searches],
                        "Price Range": [f"{format_price(s['min_price'])} - {format_price(s['max_price'])}" if s['min_price'] and s['max_price'] else "Any" for s in user_searches]
                    })
                    st.dataframe(searches_display, use_container_width=True)
                else:
                    st.info("You haven't performed any searches yet.")
            
            with tab2:
                if user_predictions:
                    # Convert to DataFrame for display
                    predictions_display = pd.DataFrame({
                        "Date": [p['prediction_date'] for p in user_predictions],
                        "Region": [p['region'] for p in user_predictions],
                        "Property Type": [p['property_type'] for p in user_predictions],
                        "Area (m²)": [p['area_sqm'] for p in user_predictions],
                        "Bedrooms": [p['bedrooms'] or "N/A" for p in user_predictions],
                        "Predicted Price": [format_price(p['predicted_price']) for p in user_predictions]
                    })
                    st.dataframe(predictions_display, use_container_width=True)
                else:
                    st.info("You haven't made any price predictions yet.")
            
            with tab3:
                if saved_properties:
                    st.write("Your saved properties:")
                    
                    # Custom CSS for property cards
                    st.markdown(f"""
                    <style>
                    .property-card {{
                        background-color: {bg_color};
                        border-radius: 12px;
                        padding: 15px;
                        margin-bottom: 15px;
                        border-left: 3px solid {primary};
                        box-shadow: 0 2px 8px {shadow};
                        transition: transform 0.2s ease;
                        position: relative;
                    }}
                    .property-card:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px {shadow};
                    }}
                    .property-title {{
                        font-size: 18px;
                        font-weight: bold;
                        color: {primary_dark};
                        margin-bottom: 5px;
                    }}
                    .property-price {{
                        font-size: 16px;
                        font-weight: bold;
                        color: {primary};
                        margin-bottom: 5px;
                    }}
                    .property-details {{
                        font-size: 14px;
                        color: {text_medium};
                        margin-bottom: 5px;
                    }}
                    .property-address {{
                        font-size: 14px;
                        color: {text_medium};
                        font-style: italic;
                    }}
                    .property-date {{
                        font-size: 12px;
                        color: {text_medium};
                        position: absolute;
                        top: 15px;
                        right: 15px;
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display each saved property with a delete button
                    for i, property in enumerate(saved_properties):
                        col1, col2 = st.columns([10, 1])
                        
                        with col1:
                            property_html = f"""
                            <div class="property-card">
                                <div class="property-date">Saved on {property['date_saved'].strftime('%Y-%m-%d')}</div>
                                <div class="property-title">
                                    {"🏢" if property['property_type'] == 'Apartment' else "🏘️" if property['property_type'] == 'Villa' else "🏠" if property['property_type'] == 'House' else "🏡"} 
                                    {property['property_type']} in {property['region']}
                                </div>
                                <div class="property-price">Price: {format_price(property['price'])}</div>
                                <div class="property-details">{property['bedrooms'] or 'N/A'} bedrooms • {property['bathrooms'] or 'N/A'} bathrooms • {property['area_sqm']} m²</div>
                                <div class="property-address">Address: {property['address'] or 'Not available'}</div>
                            </div>
                            """
                            st.markdown(property_html, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("🗑️", key=f"delete_{property['id']}"):
                                if delete_saved_property(st.session_state.user_id, property['id']):
                                    st.rerun()
                else:
                    st.info("You haven't saved any properties yet.")
            
            # Recommendations based on activity
            st.subheader("🏡 Recommended Properties")
            
            # Filter properties based on user preferences
            if preferred_region and preferred_property_type:
                recommended_properties = self.df[
                    (self.df['region'] == preferred_region) &
                    (self.df['property_type'] == preferred_property_type) &
                    (self.df['price'] >= preferred_min_price) &
                    (self.df['price'] <= preferred_max_price)
                ].head(3)
                
                if len(recommended_properties) > 0:
                    for _, property in recommended_properties.iterrows():
                        col1, col2 = st.columns([10, 1])
                        
                        with col1:
                            property_html = f"""
                            <div class="property-card">
                                <div class="property-title">
                                    {"🏢" if property['property_type'] == 'Apartment' else "🏘️" if property['property_type'] == 'Villa' else "🏠" if property['property_type'] == 'House' else "🏡"} 
                                    {property['property_type']} in {property['region']}
                                </div>
                                <div class="property-price">Price: {format_price(property['price'])}</div>
                                <div class="property-details">{property['bedrooms']} bedrooms • {property['bathrooms']} bathrooms • {property['area_sqm']} m²</div>
                                <div class="property-address">Address: {property['address'] if 'address' in property else 'Not available'}</div>
                            </div>
                            """
                            st.markdown(property_html, unsafe_allow_html=True)
                        
                        # Add save property button
                        with col2:
                            if st.button("💾", key=f"save_{_}"):
                                property_data = {
                                    'property_id': int(_) if isinstance(_, int) else None,
                                    'region': property['region'],
                                    'property_type': property['property_type'],
                                    'area_sqm': float(property['area_sqm']),
                                    'bedrooms': int(property['bedrooms']) if not np.isnan(property['bedrooms']) else None,
                                    'bathrooms': int(property['bathrooms']) if not np.isnan(property['bathrooms']) else None,
                                    'price': float(property['price']),
                                    'address': property['address'] if 'address' in property else None,
                                    'description': property['description'] if 'description' in property else None
                                }
                                
                                from database import save_property
                                if save_property(st.session_state.user_id, property_data):
                                    st.success("Property saved!")
                                    st.rerun()
                else:
                    st.info("No properties match your preferences. Try adjusting your criteria.")
            else:
                st.info("Please select your preferred region and property type to see recommendations.")
        
        except Exception as e:
            st.error(f"Error loading profile: {str(e)}")
            st.info("Please try logging in again.")
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_id = ""
            st.rerun() 