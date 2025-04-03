import streamlit as st
from streamlit_option_menu import option_menu

class SidebarComponent:
    """Sidebar component for the application."""
    
    def __init__(self):
        """Initialize the sidebar component."""
        # Initialize theme in session state if not exists
        if 'theme' not in st.session_state:
            st.session_state.theme = 'light'
    
    def render(self):
        """Render the sidebar with appropriate navigation based on login status."""
        with st.sidebar:
            # Center all elements in the sidebar
            st.markdown("""
            <style>
            section[data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
            }
            section[data-testid="stSidebar"] .stMarkdown div.element-container {
                text-align: center;
            }
            
            /* Custom theme toggle switch */
            .theme-switch-wrapper {
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 15px 0;
            }
            
            .theme-switch {
                display: inline-block;
                height: 28px;
                position: relative;
                width: 54px;
            }
            
            .theme-switch input {
                display: none;
            }
            
            .slider {
                background-color: #ccc;
                bottom: 0;
                cursor: pointer;
                left: 0;
                position: absolute;
                right: 0;
                top: 0;
                transition: .4s;
                border-radius: 34px;
            }
            
            .slider:before {
                background-color: white;
                bottom: 4px;
                content: "";
                height: 20px;
                left: 4px;
                position: absolute;
                transition: .4s;
                width: 20px;
                border-radius: 50%;
            }
            
            input:checked + .slider {
                background-color: #2E5EAA;
            }
            
            input:checked + .slider:before {
                transform: translateX(26px);
            }
            
            .theme-icon {
                margin: 0 10px;
                font-size: 18px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Logo and app name with refined styling - centered
            st.markdown("""
            <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 25px; text-align: center;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/c/ce/Flag_of_Tunisia.svg" width="60" style="margin-bottom: 10px;">
                <div>
                    <div style="font-size: 22px; font-weight: 600; color: #1A4A94;">Tunisia Housing</div>
                    <div style="font-size: 16px; color: #555555;">Market Analytics</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a prominent theme toggle button with clear visuals
            current_theme = "Light" if st.session_state.theme == 'light' else "Dark"
            opposite_theme = "Dark" if st.session_state.theme == 'light' else "Light"
            icon = "☀️" if st.session_state.theme == 'light' else "🌙"
            next_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
            
            # Create a styled button
            button_style = f"""
            <style>
            div[data-testid="stButton"] button {{
                background-color: {"#F5F7FA" if st.session_state.theme == 'light' else "#172030"};
                color: {"#1A4A94" if st.session_state.theme == 'light' else "#4A7CCF"};
                border: 1px solid {"#E9EEF6" if st.session_state.theme == 'light' else "#121A27"};
                border-radius: 12px;
                padding: 10px;
                font-weight: bold;
                width: 100%;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            div[data-testid="stButton"] button:hover {{
                background-color: {"#E9EEF6" if st.session_state.theme == 'light' else "#121A27"};
                transform: translateY(-2px);
            }}
            </style>
            """
            st.markdown(button_style, unsafe_allow_html=True)
            
            if st.button(f"Switch to {opposite_theme} Mode {next_icon}", key="theme_toggle", use_container_width=True):
                st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
                st.rerun()
                
            # Add a small hint under the button
            st.markdown(f"<div style='text-align: center; font-size: 12px; color: {'#555555' if st.session_state.theme == 'light' else '#A1AFCA'}; margin-bottom: 15px;'>Currently in {current_theme} Mode {icon}</div>", unsafe_allow_html=True)
            
            # Feature highlights with theme-specific styling
            features_bg = "#F5F7FA" if st.session_state.theme == 'light' else "#172030"
            features_color = "#555555" if st.session_state.theme == 'light' else "#C5D0E2"
            features_title_color = "#1A4A94" if st.session_state.theme == 'light' else "#4A7CCF"
            
            st.markdown(f"""
            <div style="background-color: {features_bg}; border-radius: 10px; padding: 15px; margin: 20px 0; text-align: center;">
                <div style="font-weight: 600; color: {features_title_color}; margin-bottom: 10px; font-size: 16px;">Features</div>
                <div style="color: {features_color}; font-size: 14px; margin-bottom: 8px;">
                    📊 Market Exploration
                </div>
                <div style="color: {features_color}; font-size: 14px; margin-bottom: 8px;">
                    🔍 Regional Analysis
                </div>
                <div style="color: {features_color}; font-size: 14px; margin-bottom: 8px;">
                    🧠 AI-Powered Predictions
                </div>
                <div style="color: {features_color}; font-size: 14px;">
                    💼 User Profiles
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.logged_in:
                # User greeting with theme-specific styling
                greeting_bg = "#F5F7FA" if st.session_state.theme == 'light' else "#172030"
                greeting_color = "#555555" if st.session_state.theme == 'light' else "#C5D0E2"
                greeting_name_color = "#1A4A94" if st.session_state.theme == 'light' else "#4A7CCF"
                
                st.markdown(f"""
                <div style="margin-bottom: 20px; padding: 15px; background-color: {greeting_bg}; border-radius: 10px; text-align: center;">
                    <div style="font-size: 14px; color: {greeting_color};">Welcome back,</div>
                    <div style="font-size: 18px; font-weight: 600; color: {greeting_name_color};">{st.session_state.username}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Logged in menu with theme-aware styling
                menu_container_bg = "#F5F7FA" if st.session_state.theme == 'light' else "#172030"
                
                selected = option_menu(
                    menu_title=None,
                    options=["Explore Market", "Regional Analysis", "Prediction", "Profile", "About", "Logout"],
                    icons=["graph-up", "geo-alt", "calculator", "person", "info-circle", "box-arrow-right"],
                    menu_icon="cast",
                    default_index=0,
                    styles={
                        "container": {"padding": "0!important", "background-color": menu_container_bg, "border-radius": "10px"},
                        "icon": {"color": "#4A7CCF", "font-size": "16px", "margin-right": "5px"}, 
                        "nav-link": {"font-size": "14px", "text-align": "center", "margin":"4px auto", "padding": "10px 15px", "--hover-color": "#E9EEF6" if st.session_state.theme == 'light' else "#121A27", "border-radius": "8px", "width": "90%"},
                        "nav-link-selected": {"background-color": "#2E5EAA", "color": "#FFFFFF" if st.session_state.theme == 'light' else "#E9EEF6"},
                    }
                )
            else:
                # Not logged in menu with theme-aware styling
                menu_container_bg = "#F5F7FA" if st.session_state.theme == 'light' else "#172030"
                
                selected = option_menu(
                    menu_title=None,
                    options=["Login", "Register", "About"],
                    icons=["box-arrow-in-right", "person-plus", "info-circle"],
                    menu_icon="cast",
                    default_index=0,
                    styles={
                        "container": {"padding": "0!important", "background-color": menu_container_bg, "border-radius": "10px"},
                        "icon": {"color": "#4A7CCF", "font-size": "16px", "margin-right": "5px"}, 
                        "nav-link": {"font-size": "14px", "text-align": "center", "margin":"4px auto", "padding": "10px 15px", "--hover-color": "#E9EEF6" if st.session_state.theme == 'light' else "#121A27", "border-radius": "8px", "width": "90%"},
                        "nav-link-selected": {"background-color": "#2E5EAA", "color": "#FFFFFF" if st.session_state.theme == 'light' else "#E9EEF6"},
                    }
                )
            
            # Add a separator with theme-aware color
            separator_color = "#E9EEF6" if st.session_state.theme == 'light' else "#121A27"
            st.markdown(f"<hr style='margin: 20px 0; border: none; height: 1px; background-color: {separator_color};'>", unsafe_allow_html=True)
            
            # Team information with theme-aware colors
            team_title_color = "#1A4A94" if st.session_state.theme == 'light' else "#4A7CCF"
            team_text_color = "#555555" if st.session_state.theme == 'light' else "#C5D0E2"
            
            st.markdown(f"""
            <div style="margin: 15px 0; text-align: center;">
                <div style="font-weight: 500; color: {team_title_color}; margin-bottom: 5px; font-size: 14px;">
                    Developed by
                </div>
                <div style="color: {team_text_color}; font-size: 12px;">
                    Student Team
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # App version info with theme-aware color
            version_color = "#777777" if st.session_state.theme == 'light' else "#A1AFCA"
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; margin-bottom: 20px;">
                <div style="color: {version_color}; font-size: 12px; margin: 0;">
                    Tunisia Housing Analytics v1.0
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            return selected 