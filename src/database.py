import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def get_database_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL Database: {e}")
        return None

def init_database():
    connection = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    
    if connection.is_connected():
        cursor = connection.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {os.getenv('DB_NAME')}")
        
        # Use the database
        cursor.execute(f"USE {os.getenv('DB_NAME')}")
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_searches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_searches (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                search_term VARCHAR(255) NOT NULL,
                region VARCHAR(100),
                property_type VARCHAR(100),
                min_price INT,
                max_price INT,
                bedrooms INT,
                search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create user_predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                region VARCHAR(100) NOT NULL,
                property_type VARCHAR(100) NOT NULL,
                area_sqm FLOAT NOT NULL,
                bedrooms INT,
                bathrooms INT,
                predicted_price FLOAT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        # Create saved_properties table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS saved_properties (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                property_id INT,
                region VARCHAR(100) NOT NULL,
                property_type VARCHAR(100) NOT NULL,
                area_sqm FLOAT NOT NULL,
                bedrooms INT,
                bathrooms INT,
                price FLOAT NOT NULL,
                address VARCHAR(255),
                description TEXT,
                date_saved TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        
        connection.commit()
        print("Database initialized successfully")
    
    if connection.is_connected():
        cursor.close()
        connection.close()

# Helper functions for user searches

def save_user_search(user_id, search_term, region=None, property_type=None, min_price=None, max_price=None, bedrooms=None):
    """Save a user search to the database"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        
        query = """
            INSERT INTO user_searches 
            (user_id, search_term, region, property_type, min_price, max_price, bedrooms)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (user_id, search_term, region, property_type, min_price, max_price, bedrooms))
        connection.commit()
        
        search_id = cursor.lastrowid
        
        cursor.close()
        connection.close()
        
        return search_id
    except Error as e:
        print(f"Error saving user search: {e}")
        return None

def get_user_searches(user_id, limit=10):
    """Get user's recent searches"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT * FROM user_searches
            WHERE user_id = %s
            ORDER BY search_date DESC
            LIMIT %s
        """
        
        cursor.execute(query, (user_id, limit))
        searches = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return searches
    except Error as e:
        print(f"Error retrieving user searches: {e}")
        return []

# Helper functions for user predictions

def save_user_prediction(user_id, region, property_type, area_sqm, bedrooms, bathrooms, predicted_price):
    """Save a user prediction to the database"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        
        query = """
            INSERT INTO user_predictions 
            (user_id, region, property_type, area_sqm, bedrooms, bathrooms, predicted_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (user_id, region, property_type, area_sqm, bedrooms, bathrooms, predicted_price))
        connection.commit()
        
        prediction_id = cursor.lastrowid
        
        cursor.close()
        connection.close()
        
        return prediction_id
    except Error as e:
        print(f"Error saving user prediction: {e}")
        return None

def get_user_predictions(user_id, limit=10):
    """Get user's recent predictions"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT * FROM user_predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
            LIMIT %s
        """
        
        cursor.execute(query, (user_id, limit))
        predictions = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return predictions
    except Error as e:
        print(f"Error retrieving user predictions: {e}")
        return []

# Helper functions for saved properties

def save_property(user_id, property_data):
    """Save a property to the user's saved properties"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        
        query = """
            INSERT INTO saved_properties 
            (user_id, property_id, region, property_type, area_sqm, bedrooms, bathrooms, price, address, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            user_id, 
            property_data.get('property_id'),
            property_data.get('region'),
            property_data.get('property_type'),
            property_data.get('area_sqm'),
            property_data.get('bedrooms'),
            property_data.get('bathrooms'),
            property_data.get('price'),
            property_data.get('address'),
            property_data.get('description')
        ))
        connection.commit()
        
        saved_id = cursor.lastrowid
        
        cursor.close()
        connection.close()
        
        return saved_id
    except Error as e:
        print(f"Error saving property: {e}")
        return None

def get_saved_properties(user_id):
    """Get all properties saved by the user"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
            SELECT * FROM saved_properties
            WHERE user_id = %s
            ORDER BY date_saved DESC
        """
        
        cursor.execute(query, (user_id,))
        properties = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return properties
    except Error as e:
        print(f"Error retrieving saved properties: {e}")
        return []

def delete_saved_property(user_id, property_id):
    """Delete a property from the user's saved properties"""
    try:
        connection = get_database_connection()
        cursor = connection.cursor()
        
        query = """
            DELETE FROM saved_properties
            WHERE user_id = %s AND id = %s
        """
        
        cursor.execute(query, (user_id, property_id))
        connection.commit()
        
        deleted = cursor.rowcount > 0
        
        cursor.close()
        connection.close()
        
        return deleted
    except Error as e:
        print(f"Error deleting saved property: {e}")
        return False 