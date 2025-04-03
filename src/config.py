import os
from pathlib import Path

# Determine base directory based on environment
# In deployment, the data directory might be at '/mount/src/ml-project/data'
if os.path.exists('/mount/src/ml-project'):
    BASE_DIR = Path('/mount/src/ml-project')
else:
    # Local development path
    BASE_DIR = Path(__file__).resolve().parent.parent

# Create data directory if it doesn't exist
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset file options in priority order
DATASET_OPTIONS = [
    DATA_DIR / "tunisia_housing_10k.csv",
    DATA_DIR / "tunisia_housing.csv",
    # Add more fallback options if needed
]

# Try to find an existing dataset file
DATASET_PATH = None
for dataset_option in DATASET_OPTIONS:
    if dataset_option.exists():
        DATASET_PATH = dataset_option
        break

# If no dataset found, default to the 10k version (which will need to be generated)
if DATASET_PATH is None:
    DATASET_PATH = DATA_DIR / "tunisia_housing_10k.csv"

# Project structure
MODELS_DIR = BASE_DIR / "models"

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
