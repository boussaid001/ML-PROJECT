import streamlit as st
from .BasePage import BasePage

class AboutPage(BasePage):
    """About page with project information."""
    
    def __init__(self):
        super().__init__(title="About")
    
    def render(self):
        """Render the about page."""
        st.header("ℹ️ About Tunisia Housing Analytics")
        
        # Introduction section with modern styling
        st.markdown("""
        <div style="background-color: #F5F7FA; padding: 20px; border-radius: 12px; margin-bottom: 20px; border-left: 5px solid #2E5EAA;">
            <h3 style="color: #1A4A94; margin-top: 0;">Project Overview</h3>
            <p style="margin-bottom: 0; color: #333333;">
                Tunisia Housing Analytics uses advanced machine learning to predict house prices across Tunisia.
                Our models analyze location, property features, and market conditions to provide accurate price estimates
                and valuable insights for property buyers, sellers, and investors.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features section with cards
        st.subheader("Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="font-size: 32px;">📊</span>
                </div>
                <h4 style="color: #1A4A94; text-align: center;">Interactive Visualization</h4>
                <p style="color: #555555; text-align: center;">
                    Explore property data through interactive charts and filters to understand market trends and patterns.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="font-size: 32px;">🔍</span>
                </div>
                <h4 style="color: #1A4A94; text-align: center;">Regional Analysis</h4>
                <p style="color: #555555; text-align: center;">
                    Compare property prices and trends across different regions of Tunisia to identify opportunities.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-top: 20px;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="font-size: 32px;">🧠</span>
                </div>
                <h4 style="color: #1A4A94; text-align: center;">AI Prediction</h4>
                <p style="color: #555555; text-align: center;">
                    Multiple ML models provide accurate price predictions based on property features and location.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; height: 100%; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-top: 20px;">
                <div style="text-align: center; margin-bottom: 15px;">
                    <span style="font-size: 32px;">📱</span>
                </div>
                <h4 style="color: #1A4A94; text-align: center;">User Profiles</h4>
                <p style="color: #555555; text-align: center;">
                    Save your preferences and get personalized property recommendations based on your criteria.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Data sources section
        st.subheader("Data Sources")
        
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
            <ul style="color: #555555; margin-bottom: 0;">
                <li>Synthetic dataset based on Tunisian housing market characteristics</li>
                <li>Coverage of all major regions in Tunisia, including urban and rural areas</li>
                <li>Comprehensive property attributes: size, age, amenities, and location details</li>
                <li>Historical price trends incorporated for time-series analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Models section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Models Used")
            
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <ul style="color: #555555; margin-bottom: 0;">
                    <li><strong>Linear Regression</strong>: Baseline model for price prediction</li>
                    <li><strong>Random Forest</strong>: Ensemble learning for complex feature relationships</li>
                    <li><strong>XGBoost</strong>: Advanced gradient boosting for highest accuracy</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Performance Metrics")
            
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                <ul style="color: #555555; margin-bottom: 0;">
                    <li><strong>RMSE</strong>: Root Mean Square Error</li>
                    <li><strong>MAE</strong>: Mean Absolute Error</li>
                    <li><strong>R² Score</strong>: Coefficient of determination</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Contact section
        st.subheader("Contact Us")
        
        st.markdown("""
        <div style="background-color: #F5F7FA; padding: 20px; border-radius: 12px; margin-top: 20px; text-align: center;">
            <p style="color: #555555; margin-bottom: 10px;">
                Have questions or feedback about Tunisia Housing Analytics?
            </p>
            <p style="color: #1A4A94; font-weight: 500; margin-bottom: 0;">
                contact@tunisiahousing.ai
            </p>
        </div>
        """, unsafe_allow_html=True) 