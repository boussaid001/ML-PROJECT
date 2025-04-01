from abc import ABC, abstractmethod

class BasePage(ABC):
    """Abstract base class for all pages"""
    def __init__(self, df, models=None):
        self.df = df
        self.models = models

    @abstractmethod
    def render(self):
        """Render the page content"""
        pass 