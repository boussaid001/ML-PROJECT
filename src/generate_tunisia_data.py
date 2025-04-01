import pandas as pd
import numpy as np
from pathlib import Path
import os
import random
from faker import Faker

# Import config
import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    DATA_DIR, 
    DATASET_PATH, 
    REGIONS, 
    PROPERTY_TYPES,
    TUNISIA_CENTER_LAT,
    TUNISIA_CENTER_LON
)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()

def generate_tunisia_housing_data(n_samples=5000):
    """Generate synthetic data for Tunisia housing prices"""
    
    # Region coordinates (approximate)
    region_coords = {
        "Tunis": (36.8065, 10.1815),
        "Ariana": (36.8625, 10.1956),
        "Ben Arous": (36.7533, 10.2283),
        "Manouba": (36.8094, 10.0970),
        "Nabeul": (36.4513, 10.7375),
        "Zaghouan": (36.4028, 10.1433),
        "Bizerte": (37.2744, 9.8739),
        "Béja": (36.7256, 9.1817),
        "Jendouba": (36.5011, 8.7811),
        "Kef": (36.1826, 8.7149),
        "Siliana": (36.0844, 9.3754),
        "Kairouan": (35.6781, 10.0996),
        "Kasserine": (35.1722, 8.8304),
        "Sidi Bouzid": (35.0383, 9.4858),
        "Sousse": (35.8245, 10.6346),
        "Monastir": (35.7775, 10.8262),
        "Mahdia": (35.5046, 11.0622),
        "Sfax": (34.7406, 10.7603),
        "Gabès": (33.8814, 10.0982),
        "Medenine": (33.3399, 10.5015),
        "Tataouine": (32.9326, 10.4509),
        "Gafsa": (34.4254, 8.7840),
        "Tozeur": (33.9197, 8.1335),
        "Kebili": (33.7041, 8.9686)
    }
    
    # Base prices by region (in Tunisian Dinar)
    region_base_prices = {
        "Tunis": 350000,       # Higher prices in the capital
        "Ariana": 320000,
        "Ben Arous": 280000,
        "Manouba": 250000,
        "Nabeul": 270000,
        "Zaghouan": 200000,
        "Bizerte": 230000,
        "Béja": 180000,
        "Jendouba": 170000,
        "Kef": 165000,
        "Siliana": 160000,
        "Kairouan": 190000,
        "Kasserine": 170000,
        "Sidi Bouzid": 165000,
        "Sousse": 300000,      # Higher prices in coastal tourist areas
        "Monastir": 290000,
        "Mahdia": 260000,
        "Sfax": 280000,
        "Gabès": 220000,
        "Medenine": 240000,
        "Tataouine": 180000,
        "Gafsa": 190000,
        "Tozeur": 210000,
        "Kebili": 170000
    }
    
    # Property type price multipliers
    property_type_multipliers = {
        "Apartment": 1.0,
        "House": 1.3,
        "Villa": 2.2,
        "Studio": 0.7,
        "Duplex": 1.5,
        "Penthouse": 1.8
    }
    
    # Generate data
    data = []
    
    for _ in range(n_samples):
        # Select region and coordinates with small random variation
        region = str(np.random.choice(REGIONS))  # Convert to string to ensure compatibility
        base_lat, base_lon = region_coords[region]
        lat_variation = np.random.uniform(-0.05, 0.05)
        lon_variation = np.random.uniform(-0.05, 0.05)
        latitude = base_lat + lat_variation
        longitude = base_lon + lon_variation
        
        # Select property type
        property_type = str(np.random.choice(PROPERTY_TYPES, p=[0.4, 0.25, 0.1, 0.1, 0.1, 0.05]))
        
        # Generate other features
        property_age = np.random.randint(0, 50)
        area_sqm = np.random.randint(30, 500)
        bedrooms = np.random.randint(1, 7)
        bathrooms = np.random.randint(1, 4)
        distance_to_center = np.random.uniform(0.1, 10.0)
        floor = np.random.randint(0, 10) if property_type in ["Apartment", "Studio", "Duplex", "Penthouse"] else 0
        has_elevator = np.random.choice([True, False]) if floor > 0 else False
        has_garden = np.random.choice([True, False], p=[0.3, 0.7])
        has_parking = np.random.choice([True, False], p=[0.6, 0.4])
        
        # Calculate price based on region, property type, and features
        base_price = region_base_prices[region]
        type_multiplier = property_type_multipliers[property_type]
        
        # Price calculations with some randomness
        price = base_price * type_multiplier
        price *= (area_sqm / 100) * np.random.uniform(0.9, 1.1)  # Area effect
        price *= (1 - 0.005 * property_age)  # Age effect
        price *= (1 + 0.1 * bedrooms)  # Bedrooms effect
        price *= (1 + 0.05 * bathrooms)  # Bathrooms effect
        price *= (1 - 0.03 * distance_to_center)  # Distance effect
        
        if has_elevator:
            price *= 1.05
        if has_garden:
            price *= 1.1
        if has_parking:
            price *= 1.05
            
        # Add some random fluctuation
        price *= np.random.uniform(0.9, 1.1)
        
        # Round price to nearest thousand
        price = round(price / 1000) * 1000
        
        # Create data point
        data_point = {
            "longitude": longitude,
            "latitude": latitude,
            "region": region,
            "property_type": property_type,
            "property_age": property_age,
            "area_sqm": area_sqm,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "distance_to_center": round(distance_to_center, 2),
            "floor": floor,
            "has_elevator": has_elevator,
            "has_garden": has_garden,
            "has_parking": has_parking,
            "price": price
        }
        
        data.append(data_point)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some addresses using faker
    addresses = []
    for region in df['region']:
        addresses.append(f"{fake.street_address()}, {region}, Tunisia")
    
    df['address'] = addresses
    
    return df

def save_dataset():
    """Generate and save the Tunisia housing dataset"""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Generate data
    print("Generating Tunisia housing dataset...")
    df = generate_tunisia_housing_data(n_samples=5000)
    
    # Save to CSV
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset saved to {DATASET_PATH}")
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Print some statistics
    print("\nStatistics by region:")
    region_stats = df.groupby('region')['price'].agg(['count', 'mean', 'min', 'max'])
    print(region_stats)
    
    print("\nStatistics by property type:")
    type_stats = df.groupby('property_type')['price'].agg(['count', 'mean', 'min', 'max'])
    print(type_stats)

if __name__ == "__main__":
    save_dataset() 