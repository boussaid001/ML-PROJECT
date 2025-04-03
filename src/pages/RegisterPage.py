import streamlit as st
from .BasePage import BasePage
from auth import register_user

class RegisterPage(BasePage):
    """Registration page for new users."""
    
    def __init__(self):
        super().__init__(title="Register")
    
    def render(self):
        """Render the registration page."""
        # Center the form on the page
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <h1 style="text-align: center; margin-bottom: 20px; color: #1A4A94;">Create Account</h1>
            <p style="text-align: center; margin-bottom: 30px; color: #555555;">
                Sign up to start exploring Tunisia's housing market data
            </p>
            """, unsafe_allow_html=True)
            
            # Modern card-style form
            with st.form("register_form"):
                st.markdown("""
                <style>
                div[data-testid="stForm"] {
                    background-color: #FFFFFF;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Form icon
                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="background-color: #E9EEF6; width: 70px; height: 70px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;">
                        <span style="color: #2E5EAA; font-size: 30px;">📝</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                username = st.text_input("Username")
                email = st.text_input("Email")
                
                # Two columns for passwords
                col1, col2 = st.columns(2)
                
                with col1:
                    password = st.text_input("Password", type="password")
                
                with col2:
                    confirm_password = st.text_input("Confirm Password", type="password")
                
                # Terms and privacy agreement
                st.markdown("""
                <div style="margin: 15px 0;">
                    <div style="display: flex; align-items: flex-start;">
                        <div style="min-width: 24px; margin-right: 10px; padding-top: 3px;">
                            <input type="checkbox" style="transform: scale(1.2);">
                        </div>
                        <div style="font-size: 14px; color: #555555;">
                            I agree to the <a href="#" style="color: #2E5EAA; text-decoration: none;">Terms of Service</a> and 
                            <a href="#" style="color: #2E5EAA; text-decoration: none;">Privacy Policy</a>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Register button
                submit = st.form_submit_button("Create Account")
                
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
            
            # Login link at bottom
            st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <p style="color: #555555; font-size: 14px;">
                    Already have an account? <a href="?page=Login" style="color: #2E5EAA; text-decoration: none;">Sign in</a>
                </p>
            </div>
            """, unsafe_allow_html=True) 