import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

# Import from local modules
from config import (
    PAGE_TITLE,
    PAGE_ICON,
    LINEAR_REGRESSION_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH
)
from utils import (
    load_data, load_model, apply_custom_theme
)
from database import init_database

# Import page classes directly instead of importing the entire module
from pages.LoginPage import LoginPage
from pages.RegisterPage import RegisterPage
from pages.ProfilePage import ProfilePage
from pages.ExploreMarketPage import ExploreMarketPage
from pages.RegionalAnalysisPage import RegionalAnalysisPage
from pages.PredictionPage import PredictionPage
from pages.AboutPage import AboutPage
from pages.LogoutPage import LogoutPage
from pages.SidebarComponent import SidebarComponent

# Clean up sidebar by removing modules from st._main.pages
# This prevents class names from showing up in the sidebar
def clean_sidebar_modules():
    # Get the _main module from streamlit
    main = sys.modules.get("streamlit.web.bootstrap")
    if main and hasattr(main, "_pages"):
        # Filter out pages with our class names
        class_names = [
            "BasePage", "LoginPage", "RegisterPage", "ProfilePage", 
            "ExploreMarketPage", "RegionalAnalysisPage", "PredictionPage",
            "AboutPage", "LogoutPage", "SidebarComponent", "app"
        ]
        # Attempt to clear module pages from sidebar
        try:
            for name in class_names:
                for i, page in enumerate(main._pages):
                    if name in page["page_name"]:
                        main._pages.pop(i)
                        break
        except:
            pass  # Ignore errors if structure changed

# Run cleaning function to remove unwanted sidebar entries
clean_sidebar_modules()

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

# Initialize session state (keep this simpler)
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.user_id = ""

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

# Modern header with clean design
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 20px;">
    <div style="margin-right: 20px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/ce/Flag_of_Tunisia.svg" width="60">
    </div>
    <div>
        <h1 style="margin: 0; color: #1A4A94;">Tunisia Housing Analytics</h1>
        <p style="margin: 5px 0 0 0; color: #555555; font-size: 16px;">AI-powered insights for property markets</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Brief app description
st.markdown("""
<div style="background-color: #F5F7FA; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #2E5EAA;">
    <p style="margin: 0; color: #333333;">
    Explore Tunisia's property market with advanced analytics. Get accurate price predictions, 
    visualize regional trends, and make informed decisions about your property investments.
    </p>
</div>
""", unsafe_allow_html=True)

# Hide the default Streamlit menu and footer
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.css-18ni7ap.e8zbici2 {display: none !important;}
.stDeployButton {display: none !important;}
/* Hide component names in sidebar */
div[data-testid="stSidebarNav"] ul {display: none !important;}
/* Optional - if previous approach doesn't work */
section[data-testid="stSidebar"] div.element-container:first-child {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize and render the sidebar component
sidebar = SidebarComponent()
selected = sidebar.render()

# Check if models are already trained
if not models:
    # Styled warning message
    st.markdown("""
    <div style="background-color: #FFF3CD; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #F8B400;">
        <p style="margin: 0; color: #856404; font-weight: 600;">Models not found</p>
        <p style="margin: 5px 0 0 0; color: #856404;">Please train models using the button below.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Generate Data & Train Models", type="primary"):
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

# Initialize page instances
login_page = LoginPage()
register_page = RegisterPage()
profile_page = ProfilePage(df)
explore_market_page = ExploreMarketPage(df)
regional_analysis_page = RegionalAnalysisPage(df)
prediction_page = PredictionPage(df, models)
about_page = AboutPage()
logout_page = LogoutPage()

# Route to the appropriate page based on selection
if selected == "Login":
    login_page.render()
elif selected == "Register":
    register_page.render()
elif selected == "About":
    about_page.render()
elif st.session_state.logged_in:
    # Routes for logged-in users
    if selected == "Profile":
        profile_page.render()
    elif selected == "Explore Market":
        explore_market_page.render()
    elif selected == "Regional Analysis":
        regional_analysis_page.render()
    elif selected == "Prediction":
        prediction_page.render()
    elif selected == "Logout":
        logout_page.render()
else:
    if selected != "About":
        st.warning("Please login to access the application.")
