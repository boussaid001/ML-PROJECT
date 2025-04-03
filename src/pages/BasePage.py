import streamlit as st

class BasePage:
    """Base class for all pages in the application."""
    
    def __init__(self, title=""):
        """Initialize with page title."""
        self.title = title
    
    def render(self):
        """Render method to be implemented by each page class."""
        raise NotImplementedError("Each page class must implement the render method") 