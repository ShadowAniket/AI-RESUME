import sqlite3
import os
import sys

def get_database_connection():
    """Create and return a database connection"""
    # Check if the database file exists
    if not os.path.exists('resume_data.db'):
        print("Error: Database file 'resume_data.db' not found.")
        print("Make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    conn = sqlite3.connect('resume_data.db')
    return conn

def check_admin_exists(email):
    """Check if an admin with the given email exists"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT * FROM admin WHERE email = ?', (email,))
        result = cursor.fetchone()
        return bool(result)
    except Exception as e:
        print(f"Error checking admin: {str(e)}")
        return False
    finally:
        conn.close()

def add_default_admin():
    """Add the default admin user if it doesn't exist"""
    default_email = "admin@example.com"
    default_password = "admin123"
    
    # Check if default admin already exists
    if check_admin_exists(default_email):
        print(f"Default admin ({default_email}) already exists.")
        return True
    
    # Add default admin
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('INSERT INTO admin (email, password) VALUES (?, ?)', 
                      (default_email, default_password))
        conn.commit()
        print(f"Default admin ({default_email}) created successfully.")
        return True
    except Exception as e:
        print(f"Error adding default admin: {str(e)}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    print("\n===== Initialize Default Admin =====\n")
    add_default_admin()
    print("\nDefault admin credentials:")
    print("Email: admin@example.com")
    print("Password: admin123")
    print("\nYou can change these credentials using the change_admin_credentials.py script.")