import pandas as pd
import numpy as np
from pathlib import Path
import os
import random
from faker import Faker
from tqdm import tqdm
import time

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
fake = Faker('fr_FR')  # Using French locale for more appropriate names

# Define regions with their central coordinates
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

# Define approximate regional population (in thousands) for weighting
region_population = {
    "Tunis": 634,
    "Ariana": 350,
    "Ben Arous": 600,
    "Manouba": 400,
    "Nabeul": 200,
    "Zaghouan": 150,
    "Bizerte": 150,
    "Béja": 100,
    "Jendouba": 100,
    "Kef": 120,
    "Siliana": 80,
    "Kairouan": 150,
    "Kasserine": 120,
    "Sidi Bouzid": 160,
    "Sousse": 270,
    "Monastir": 200,
    "Mahdia": 150,
    "Sfax": 330,
    "Gabès": 200,
    "Medenine": 200,
    "Tataouine": 100,
    "Gafsa": 120,
    "Tozeur": 70,
    "Kebili": 80
}

# Define region base prices in Tunisian Dinar (reflecting market differences)
region_base_prices = {
    "Tunis": 300000,
    "Ariana": 250000,
    "Ben Arous": 230000,
    "Manouba": 190000,
    "Nabeul": 210000,
    "Zaghouan": 160000,
    "Bizerte": 170000,
    "Béja": 120000,
    "Jendouba": 110000,
    "Kef": 105000,
    "Siliana": 100000,
    "Kairouan": 130000,
    "Kasserine": 110000,
    "Sidi Bouzid": 100000,
    "Sousse": 250000,
    "Monastir": 240000,
    "Mahdia": 210000,
    "Sfax": 230000,
    "Gabès": 160000,
    "Medenine": 170000,
    "Tataouine": 120000,
    "Gafsa": 130000,
    "Tozeur": 140000,
    "Kebili": 110000
}

# Define property types and their multipliers
property_type_multipliers = {
    "Apartment": 1.0,
    "House": 1.4,
    "Villa": 2.5,
    "Studio": 0.6,
    "Duplex": 1.6,
    "Penthouse": 2.0
}

# Additional data for realistic address generation
street_prefixes = ["Rue", "Avenue", "Boulevard", "Impasse", "Place"]
street_names = [
    "Habib Bourguiba", "Mohamed V", "Farhat Hached", "Ibn Khaldoun", "14 Janvier",
    "Liberté", "République", "Carthage", "Hannibal", "Bardo", "Ali Belhouane",
    "Mokhtar Attia", "Charles de Gaulle", "Paris", "Abou El Kacem Chebbi",
    "Hédi Chaker", "Mongi Slim", "Tahar Sfar", "Kheireddine", "Omar Ibn Al Khattab",
    "Abou Hamed El Ghazali", "Tahar Haddad", "Ibn Sina", "El Ferdaous", "El Ghazela"
]

# Additional location-specific streets to make more realistic addresses
region_specific_streets = {
    "Tunis": ["La Marsa", "Bab Saadoun", "Bab Bhar", "El Menzah", "El Manar", "Lafayette"],
    "Ariana": ["El Ghazela", "Ennasr", "Borj Louzir", "Riadh El Andalous", "Jardins d'El Menzah"],
    "Sousse": ["Khezama", "Sahloul", "Hammam Sousse", "Bouhsina", "Kantaoui"],
    "Sfax": ["Sfax El Jadida", "Chihia", "Afran", "Sakiet Ezzit", "Gremda"],
    "Monastir": ["Skanes", "Khniss", "Centre Ville", "Cité Essalem", "Beni Hassen"]
}

def generate_batch(batch_size, region_distribution, urban_regions):
    """Generate a batch of properties with vectorized operations"""
    
    # Sample regions based on population distribution
    regions = np.random.choice(
        list(region_distribution.keys()), 
        size=batch_size, 
        p=list(region_distribution.values())
    )
    
    # Initialize data dictionary
    data = {
        "region": regions,
        "region_population_thousands": [region_population[r] for r in regions]
    }
    
    # Generate coordinates
    base_coords = np.array([region_coords[r] for r in regions])
    variation = np.random.uniform(-0.025, 0.025, size=(batch_size, 2))
    coords = base_coords + variation
    data["latitude"] = coords[:, 0]
    data["longitude"] = coords[:, 1]
    
    # Generate property types with different distributions for urban vs. rural
    is_urban = np.array([r in urban_regions for r in regions])
    property_types = np.empty(batch_size, dtype=object)
    
    # Urban property type distribution
    urban_mask = is_urban
    urban_count = np.sum(urban_mask)
    if urban_count > 0:
        urban_probs = [0.55, 0.12, 0.12, 0.15, 0.03, 0.03]
        property_types[urban_mask] = np.random.choice(
            PROPERTY_TYPES, 
            size=urban_count, 
            p=urban_probs
        )
    
    # Rural property type distribution
    rural_mask = ~is_urban
    rural_count = np.sum(rural_mask)
    if rural_count > 0:
        rural_probs = [0.35, 0.40, 0.10, 0.05, 0.07, 0.03]
        property_types[rural_mask] = np.random.choice(
            PROPERTY_TYPES, 
            size=rural_count, 
            p=rural_probs
        )
    
    data["property_type"] = property_types
    
    # Initialize property features
    property_age = np.zeros(batch_size, dtype=int)
    area_sqm = np.zeros(batch_size, dtype=int)
    bedrooms = np.zeros(batch_size, dtype=int)
    bathrooms = np.zeros(batch_size, dtype=int)
    floor = np.zeros(batch_size, dtype=int)
    
    # Vectorized assignment for each property type
    for prop_type in PROPERTY_TYPES:
        mask = property_types == prop_type
        count = np.sum(mask)
        
        if count == 0:
            continue
            
        if prop_type == "Villa":
            property_age[mask] = np.random.randint(0, 25, size=count)
            area_sqm[mask] = np.random.randint(160, 450, size=count)
            bedrooms[mask] = np.random.randint(4, 8, size=count)
            bathrooms[mask] = np.random.randint(2, 5, size=count)
            # Villas are ground level
            floor[mask] = 0
            
        elif prop_type == "Apartment":
            property_age[mask] = np.random.randint(0, 35, size=count)
            area_sqm[mask] = np.random.randint(70, 180, size=count)
            bedrooms[mask] = np.random.randint(1, 5, size=count)
            bathrooms[mask] = np.random.randint(1, 3, size=count)
            floor[mask] = np.random.randint(0, 10, size=count)
            
        elif prop_type == "Studio":
            property_age[mask] = np.random.randint(0, 30, size=count)
            area_sqm[mask] = np.random.randint(30, 70, size=count)
            bedrooms[mask] = 0  # Studios don't have separate bedrooms
            bathrooms[mask] = 1
            floor[mask] = np.random.randint(0, 10, size=count)
            
        elif prop_type == "House":
            property_age[mask] = np.random.randint(3, 55, size=count)
            area_sqm[mask] = np.random.randint(110, 300, size=count)
            bedrooms[mask] = np.random.randint(2, 6, size=count)
            bathrooms[mask] = np.random.randint(1, 4, size=count)
            # Houses are ground level
            floor[mask] = 0
            
        elif prop_type == "Duplex":
            property_age[mask] = np.random.randint(0, 30, size=count)
            area_sqm[mask] = np.random.randint(110, 220, size=count)
            bedrooms[mask] = np.random.randint(2, 5, size=count)
            bathrooms[mask] = np.random.randint(2, 4, size=count)
            floor[mask] = np.random.randint(0, 8, size=count)
            
        elif prop_type == "Penthouse":
            property_age[mask] = np.random.randint(0, 20, size=count)
            area_sqm[mask] = np.random.randint(130, 300, size=count)
            bedrooms[mask] = np.random.randint(2, 5, size=count)
            bathrooms[mask] = np.random.randint(2, 4, size=count)
            floor[mask] = np.random.randint(5, 15, size=count)
    
    data["property_age"] = property_age
    data["area_sqm"] = area_sqm
    data["bedrooms"] = bedrooms
    data["bathrooms"] = bathrooms
    data["floor"] = floor
    
    # Distance to center depends on urban vs rural
    distance_to_center = np.zeros(batch_size)
    distance_to_center[is_urban] = np.random.uniform(0.5, 6.0, size=np.sum(is_urban))
    distance_to_center[~is_urban] = np.random.uniform(0.5, 15.0, size=np.sum(~is_urban))
    data["distance_to_center"] = np.round(distance_to_center, 1)
    
    # Generate amenities
    high_floor_mask = floor > 2
    data["has_elevator"] = np.zeros(batch_size, dtype=bool)
    data["has_elevator"][high_floor_mask] = np.random.choice(
        [True, False], 
        size=np.sum(high_floor_mask),
        p=[0.7, 0.3]
    )
    
    # Garden is common for houses and villas
    data["has_garden"] = np.zeros(batch_size, dtype=bool)
    house_villa_mask = (property_types == "Villa") | (property_types == "House")
    data["has_garden"][house_villa_mask] = True
    
    other_prop_mask = ~house_villa_mask
    other_prop_count = np.sum(other_prop_mask)
    if other_prop_count > 0:
        data["has_garden"][other_prop_mask] = np.random.choice(
            [True, False], 
            size=other_prop_count, 
            p=[0.2, 0.8]
        )
    
    # Parking availability
    data["has_parking"] = np.zeros(batch_size, dtype=bool)
    luxury_mask = (property_types == "Villa") | (property_types == "Penthouse")
    data["has_parking"][luxury_mask] = True
    
    other_prop_mask = ~luxury_mask
    other_prop_count = np.sum(other_prop_mask)
    if other_prop_count > 0:
        data["has_parking"][other_prop_mask] = np.random.choice(
            [True, False], 
            size=other_prop_count, 
            p=[0.5, 0.5]
        )
    
    # Calculate prices
    base_prices = np.array([region_base_prices[r] for r in regions])
    type_multipliers = np.array([property_type_multipliers[t] for t in property_types])
    
    # Start with base price * type multiplier
    prices = base_prices * type_multipliers
    
    # Apply area factor (non-linear relationship)
    area_factors = (area_sqm / 100) ** 1.1
    prices *= area_factors
    
    # Apply age depreciation
    age_factors = np.maximum(0.7, 1 - 0.008 * property_age)
    prices *= age_factors
    
    # Adjust for rooms
    prices *= (1 + 0.08 * bedrooms)
    prices *= (1 + 0.06 * bathrooms)
    
    # Adjust for location
    center_factors = np.maximum(0.75, 1 - 0.03 * distance_to_center)
    prices *= center_factors
    
    # Apply amenity premiums
    prices[data["has_elevator"]] *= 1.04
    prices[data["has_garden"]] *= 1.08
    prices[data["has_parking"]] *= 1.05
    
    # Apply random market fluctuation
    market_fluctuation = np.random.uniform(0.92, 1.08, size=batch_size)
    prices *= market_fluctuation
    
    # Round to nearest thousand
    data["price"] = (np.round(prices / 1000) * 1000).astype(int)
    
    return data

def generate_addresses(df):
    """Generate realistic Tunisian addresses for each property"""
    addresses = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating addresses"):
        region = row['region']
        
        # Use region-specific streets when available
        if region in region_specific_streets and random.random() < 0.6:
            street_name = random.choice(region_specific_streets[region])
        else:
            street_name = random.choice(street_names)
        
        street_num = random.randint(1, 200)
        street_prefix = random.choice(street_prefixes)
        
        # Apartments and similar properties may have apartment numbers
        if row['property_type'] in ["Apartment", "Studio", "Duplex", "Penthouse"] and random.random() < 0.8:
            apt_num = random.randint(1, 40)
            address = f"{street_num} {street_prefix} {street_name}, Apt. {apt_num}, {region}, Tunisia"
        else:
            address = f"{street_num} {street_prefix} {street_name}, {region}, Tunisia"
        
        addresses.append(address)
    
    return addresses

def generate_tunisia_housing_data(total_properties=10000, batch_size=5000):
    """Generate a large synthetic dataset for Tunisia housing with optimized batch processing"""
    print(f"Generating {total_properties} properties...")
    start_time = time.time()
    
    # Calculate the total population
    total_pop = sum(region_population.values())
    
    # Create a probability distribution based on population
    region_distribution = {region: pop/total_pop for region, pop in region_population.items()}
    
    # Define urban regions
    urban_regions = ["Tunis", "Ariana", "Ben Arous", "Manouba", "Sousse", "Monastir", "Sfax"]
    
    # Process in batches for memory efficiency
    all_data = []
    num_batches = total_properties // batch_size
    remainder = total_properties % batch_size
    
    for i in tqdm(range(num_batches), desc="Generating property batches"):
        batch_data = generate_batch(batch_size, region_distribution, urban_regions)
        all_data.append(pd.DataFrame(batch_data))
    
    # Handle remainder
    if remainder > 0:
        batch_data = generate_batch(remainder, region_distribution, urban_regions)
        all_data.append(pd.DataFrame(batch_data))
    
    # Combine all batches
    df = pd.concat(all_data, ignore_index=True)
    
    # Add addresses (this is done separately because it's harder to vectorize)
    df['address'] = generate_addresses(df)
    
    # Add some neighborhood names for more realism
    neighborhoods = generate_neighborhood_names(df)
    df['neighborhood'] = neighborhoods
    
    # Add construction quality as a feature
    df['construction_quality'] = generate_construction_quality(df)
    
    # Add energy efficiency rating
    df['energy_rating'] = generate_energy_rating(df)
    
    # Add renovation status
    df['last_renovation'] = generate_renovation_status(df)
    
    # Add transaction date (for time series analysis)
    df['transaction_date'] = generate_transaction_dates(df)
    
    # Add price per square meter as a derived feature
    df['price_per_sqm'] = (df['price'] / df['area_sqm']).round().astype(int)
    
    end_time = time.time()
    print(f"Dataset generated in {end_time - start_time:.2f} seconds")
    
    return df

def generate_neighborhood_names(df):
    """Generate realistic neighborhood names for each property"""
    neighborhoods = []
    
    # Common neighborhood name patterns
    prefixes = ["Cité", "Quartier", "Résidence", "Hay", "El"]
    neighborhood_types = ["Jardins", "Parc", "Lac", "Oliveraie", "Palmeraie", "Centre", "Médina"]
    
    # Region-specific neighborhoods
    region_neighborhoods = {
        "Tunis": ["La Marsa", "Carthage", "Le Bardo", "Menzah", "Manar", "Lac 1", "Lac 2", "Centre Ville", "Médina"],
        "Ariana": ["Ennasr", "Ghazela", "Menzah 5", "Menzah 6", "Riadh Andalous", "Borj Louzir"],
        "Sousse": ["Khezama", "Sahloul", "Kantaoui", "Bouhsina", "Médina"],
        "Sfax": ["Nouvelle Ville", "Médina", "Chihia", "Gremda", "Afran"],
        "Monastir": ["Skanes", "Centre Ville", "Zone Touristique", "Cité Essalem"]
    }
    
    for i, row in df.iterrows():
        region = row['region']
        
        # Use region-specific neighborhoods when available
        if region in region_neighborhoods and random.random() < 0.7:
            neighborhood = random.choice(region_neighborhoods[region])
        else:
            # Generate a random neighborhood name
            if random.random() < 0.6:
                prefix = random.choice(prefixes)
                type_name = random.choice(neighborhood_types)
                neighborhood = f"{prefix} {type_name}"
            else:
                # Some properties don't have specific neighborhoods
                neighborhood = None
        
        neighborhoods.append(neighborhood)
    
    return neighborhoods

def generate_construction_quality(df):
    """Generate construction quality ratings based on property type and age"""
    quality_options = ["Basic", "Standard", "Good", "Premium", "Luxury"]
    quality = []
    
    for _, row in df.iterrows():
        property_type = row['property_type']
        age = row['property_age']
        
        # Base probabilities depend on property type
        if property_type in ["Villa", "Penthouse"]:
            probs = [0.05, 0.15, 0.25, 0.35, 0.2]
        elif property_type in ["Duplex", "House"]:
            probs = [0.1, 0.25, 0.4, 0.2, 0.05]
        elif property_type == "Apartment":
            probs = [0.15, 0.35, 0.35, 0.1, 0.05]
        else:  # Studio
            probs = [0.25, 0.4, 0.25, 0.08, 0.02]
        
        # Adjust for age - older properties tend to have lower quality
        if age > 30:
            # Shift probabilities toward lower quality
            probs = [0.3, 0.4, 0.2, 0.08, 0.02]
        elif age > 15:
            # Slight shift toward lower quality
            probs = [0.2, 0.35, 0.3, 0.1, 0.05]
        
        quality.append(np.random.choice(quality_options, p=probs))
    
    return quality

def generate_energy_rating(df):
    """Generate energy efficiency ratings based on property age and quality"""
    ratings = []
    rating_options = ["A", "B", "C", "D", "E", "F", "G"]
    
    for _, row in df.iterrows():
        age = row['property_age']
        quality = row['construction_quality']
        
        # Base probabilities based on age
        if age < 5:
            probs = [0.2, 0.35, 0.25, 0.1, 0.05, 0.03, 0.02]  # Newer buildings tend to have better ratings
        elif age < 15:
            probs = [0.05, 0.15, 0.3, 0.25, 0.15, 0.07, 0.03]
        elif age < 30:
            probs = [0.01, 0.04, 0.15, 0.3, 0.3, 0.15, 0.05]
        else:
            probs = [0.0, 0.02, 0.08, 0.15, 0.25, 0.3, 0.2]  # Older buildings tend to have worse ratings
        
        # Adjust for construction quality
        if quality in ["Premium", "Luxury"]:
            # Shift probabilities toward better ratings
            probs = [p * 1.5 for p in probs[:3]] + [p * 0.7 for p in probs[3:]]
            # Normalize probabilities
            probs = [p / sum(probs) for p in probs]
        elif quality == "Basic":
            # Shift probabilities toward worse ratings
            probs = [p * 0.5 for p in probs[:3]] + [p * 1.3 for p in probs[3:]]
            # Normalize probabilities
            probs = [p / sum(probs) for p in probs]
        
        ratings.append(np.random.choice(rating_options, p=probs))
    
    return ratings

def generate_renovation_status(df):
    """Generate last renovation year based on property age"""
    renovation_years = []
    current_year = 2024
    
    for _, row in df.iterrows():
        age = row['property_age']
        
        if age < 5:
            # Very new properties haven't been renovated
            renovation_years.append(None)
        elif age < 15:
            # Some newer properties might have minor renovations
            if random.random() < 0.3:
                renovation_years.append(current_year - random.randint(1, 3))
            else:
                renovation_years.append(None)
        else:
            # Older properties are more likely to have been renovated
            if random.random() < 0.7:
                # Renovation usually within the last 15 years
                renovation_years.append(current_year - random.randint(1, 15))
            else:
                renovation_years.append(None)
    
    return renovation_years

def generate_transaction_dates(df):
    """Generate realistic transaction dates within the last two years"""
    dates = []
    start_date = pd.Timestamp('2022-01-01')
    end_date = pd.Timestamp('2024-03-31')
    date_range = (end_date - start_date).days
    
    for _ in range(len(df)):
        random_days = random.randint(0, date_range)
        date = start_date + pd.Timedelta(days=random_days)
        dates.append(date)
    
    return dates

def save_dataset():
    """Generate and save the Tunisia housing dataset"""
    # Check multiple possible data directory locations
    # First try the deployment environment path
    possible_data_dirs = [
        DATA_DIR,  # From config
        os.path.join('/mount/src/ml-project', 'data'),  # Deployment environment
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')  # Local absolute path
    ]
    
    # Create data directory if it doesn't exist
    data_dir = None
    for dir_path in possible_data_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Successfully created/verified data directory at: {dir_path}")
            data_dir = dir_path
            break
        except Exception as e:
            print(f"Warning: Could not create data directory at {dir_path}: {e}")
    
    if data_dir is None:
        print("Error: Could not create any data directory, using current directory")
        data_dir = "."
    
    # Define custom dataset path for the dataset
    dataset_path = os.path.join(data_dir, "tunisia_housing_10k.csv")
    
    print("Generating Tunisia housing dataset with 10,000 properties...")
    df = generate_tunisia_housing_data(10000, batch_size=5000)
    
    # Adjust the price distribution to be more realistic
    # Add some price outliers for more realistic data
    outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    df.loc[outlier_indices, 'price'] = df.loc[outlier_indices, 'price'] * np.random.uniform(1.3, 1.8, size=len(outlier_indices))
    
    # Add noise to prices to improve model training
    noise = np.random.normal(0, 0.05, size=len(df))
    df['price'] = df['price'] * (1 + noise)
    df['price'] = (np.round(df['price'] / 1000) * 1000).astype(int)
    
    # Update price_per_sqm to reflect the changes
    df['price_per_sqm'] = (df['price'] / df['area_sqm']).round().astype(int)
    
    # Ensure consistent relationships between features for better model accuracy
    # Adjust bathroom count for very large properties
    large_property_mask = df['area_sqm'] > 300
    df.loc[large_property_mask & (df['bathrooms'] < 3), 'bathrooms'] += 1
    
    # Adjust bedroom count for very small apartments
    small_apt_mask = (df['area_sqm'] < 50) & (df['property_type'] == 'Apartment')
    df.loc[small_apt_mask & (df['bedrooms'] > 1), 'bedrooms'] = 1
    
    print(f"Saving dataset to {dataset_path}...")
    # Create parent directories if they don't exist (additional safety)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    df.to_csv(dataset_path, index=False)
    
    print(f"Dataset saved successfully!")
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Print some statistics
    print("\nStatistics by region:")
    region_stats = df.groupby('region')['price'].agg(['count', 'mean', 'min', 'max'])
    print(region_stats)
    
    print("\nStatistics by property type:")
    type_stats = df.groupby('property_type')['price'].agg(['count', 'mean', 'min', 'max'])
    print(type_stats)
    
    # Dataset quality check
    print("\nData quality check:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Calculate and print correlation between key features and price
    print("\nFeature correlation with price:")
    numeric_cols = ['area_sqm', 'bedrooms', 'bathrooms', 'property_age', 'distance_to_center', 'price']
    correlations = df[numeric_cols].corr()['price'].sort_values(ascending=False)
    print(correlations)
    
    # Save a copy to deployment path if in that environment
    deployment_path = os.path.join('/mount/src/ml-project/data', "tunisia_housing_10k.csv")
    if DATA_DIR != '/mount/src/ml-project/data' and os.path.exists('/mount/src'):
        try:
            os.makedirs(os.path.dirname(deployment_path), exist_ok=True)
            df.to_csv(deployment_path, index=False)
            print(f"Also saved dataset to deployment path: {deployment_path}")
        except Exception as e:
            print(f"Note: Could not save to deployment path {deployment_path}: {e}")
    
    # Update the original dataset path to point to this new file
    print("\nUpdating config to use the new dataset...")
    try:
        # First try the config in the current directory
        config_paths = [
            os.path.join("src", "config.py"),  # Local relative path
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),  # Current directory
            os.path.join('/mount/src/ml-project/src', "config.py")  # Deployment path
        ]
        
        updated = False
        for config_path in config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # Update the dataset path - check different potential patterns
                patterns = [
                    'DATASET_PATH = DATA_DIR / "tunisia_housing.csv"',
                    'DATASET_PATH = DATA_DIR / "tunisia_housing_100k.csv"'
                ]
                
                for pattern in patterns:
                    if pattern in config_content:
                        config_content = config_content.replace(
                            pattern,
                            'DATASET_PATH = DATA_DIR / "tunisia_housing_10k.csv"'
                        )
                        updated = True
                
                # If we found and updated a pattern
                if updated:
                    with open(config_path, 'w') as f:
                        f.write(config_content)
                    
                    print(f"Config updated successfully at {config_path}!")
                    break
                else:
                    print(f"Could not find DATASET_PATH patterns in {config_path}. Please update manually.")
        
        if not updated:
            print("Could not update any config file. Please update DATASET_PATH manually.")
            
    except Exception as e:
        print(f"Error updating config: {e}")

if __name__ == "__main__":
    save_dataset() 