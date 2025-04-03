import os
from pathlib import Path

# Project structure
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Dataset
DATASET_PATH = DATA_DIR / "tunisia_housing_10k.csv"

# Models
LINEAR_REGRESSION_PATH = MODELS_DIR / "linear_regression.pkl"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.pkl"
XGBOOST_PATH = MODELS_DIR / "xgboost.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Tunisia-specific settings
REGIONS = [
    "Tunis", "Ariana", "Ben Arous", "Manouba", "Nabeul", "Zaghouan", "Bizerte", 
    "Béja", "Jendouba", "Kef", "Siliana", "Sousse", "Monastir", "Mahdia", 
    "Sfax", "Kairouan", "Kasserine", "Sidi Bouzid", "Gabès", "Medenine", 
    "Tataouine", "Gafsa", "Tozeur", "Kebili"
]

PROPERTY_TYPES = ["Apartment", "House", "Villa", "Studio", "Duplex", "Penthouse"]

# Map settings
TUNISIA_CENTER_LAT = 34.0
TUNISIA_CENTER_LON = 9.0

# Features
NUMERICAL_FEATURES = [
    "longitude", "latitude", "property_age", "area_sqm", "bedrooms",
    "bathrooms", "distance_to_center", "floor"
]
CATEGORICAL_FEATURES = ["region", "property_type"]
TARGET = "price"

# Streamlit page settings
PAGE_TITLE = "Tunisia House Price Prediction"
PAGE_ICON = "🏠"
