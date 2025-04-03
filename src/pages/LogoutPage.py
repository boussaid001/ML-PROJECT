import streamlit as st
from .BasePage import BasePage

class LogoutPage(BasePage):
    """Logout page for the application."""
    
    def __init__(self):
        super().__init__(title="Logout")
    
    def render(self):
        """Handle logout and redirect."""
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_id = ""
        st.success("You have been logged out successfully!")
        st.info("Redirecting to login page...")
        st.rerun() 