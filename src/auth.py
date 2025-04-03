import bcrypt
from database import get_database_connection
import os
from dotenv import load_dotenv

load_dotenv()

def register_user(username, email, password):
    connection = get_database_connection()
    if connection is None:
        return False, "Database connection failed"
    
    try:
        cursor = connection.cursor()
        
        # Check if username or email already exists
        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
        if cursor.fetchone():
            return False, "Username or email already exists"
        
        # Hash password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )
        connection.commit()
        return True, "User registered successfully"
    
    except Exception as e:
        return False, f"Registration failed: {str(e)}"
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def login_user(username, password):
    connection = get_database_connection()
    if connection is None:
        return False, "Database connection failed"
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Get user by username
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if not user:
            return False, "User not found"
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return False, "Invalid password"
        
        # Return user ID directly
        return True, user['id']
    
    except Exception as e:
        return False, f"Login failed: {str(e)}"
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Simple function to verify if user is authenticated (for API routes if needed)
def is_authenticated(username, user_id):
    connection = get_database_connection()
    if connection is None:
        return False
    
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND id = %s", (username, user_id))
        user = cursor.fetchone()
        return user is not None
    
    except Exception:
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close() 