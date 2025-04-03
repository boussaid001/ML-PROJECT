import streamlit as st
from .BasePage import BasePage
from auth import login_user

class LoginPage(BasePage):
    """Login page for the application."""
    
    def __init__(self):
        super().__init__(title="Login")
    
    def render(self):
        """Render the login page."""
        # Center the form on the page
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <h1 style="text-align: center; margin-bottom: 20px; color: #1A4A94;">Welcome Back</h1>
            <p style="text-align: center; margin-bottom: 30px; color: #555555;">
                Sign in to access your Tunisia Housing Analytics dashboard
            </p>
            """, unsafe_allow_html=True)
            
            # Modern card-style form
            with st.form("login_form"):
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
                        <span style="color: #2E5EAA; font-size: 30px;">🔐</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                # Remember me checkbox with nicer styling
                col1, col2 = st.columns(2)
                with col1:
                    remember = st.checkbox("Remember me")
                
                with col2:
                    st.markdown("""
                    <div style="text-align: right; padding-top: 5px;">
                        <a href="#" style="color: #2E5EAA; text-decoration: none; font-size: 14px;">Forgot password?</a>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sign in button with custom styling
                submit = st.form_submit_button("Sign In")
                
                if submit:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_id = str(result)  # Store user ID as string
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(result)
            
            # Register link at bottom
            st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <p style="color: #555555; font-size: 14px;">
                    Don't have an account? <a href="?page=Register" style="color: #2E5EAA; text-decoration: none;">Create account</a>
                </p>
            </div>
            """, unsafe_allow_html=True) 