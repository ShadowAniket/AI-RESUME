import sqlite3
import sys
import os

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

def update_admin_credentials(old_email, new_email, new_password):
    """Update admin credentials"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('UPDATE admin SET email = ?, password = ? WHERE email = ?', 
                      (new_email, new_password, old_email))
        conn.commit()
        if cursor.rowcount > 0:
            return True
        return False
    except Exception as e:
        print(f"Error updating admin credentials: {str(e)}")
        return False
    finally:
        conn.close()

def add_new_admin(email, password):
    """Add a new admin"""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('INSERT INTO admin (email, password) VALUES (?, ?)', (email, password))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding admin: {str(e)}")
        return False
    finally:
        conn.close()

def main():
    print("\n===== Change Admin Credentials =====\n")
    
    # Check if default admin exists
    default_admin_email = "admin@example.com"
    default_exists = check_admin_exists(default_admin_email)
    
    if default_exists:
        print(f"Default admin account ({default_admin_email}) found.")
    else:
        print(f"Default admin account ({default_admin_email}) not found.")
    
    # Get new credentials
    new_email = input("Enter new admin email: ")
    new_password = input("Enter new admin password: ")
    
    # Validate input
    if not new_email or not new_password:
        print("Error: Email and password cannot be empty.")
        return
    
    # Update or add admin
    if default_exists:
        success = update_admin_credentials(default_admin_email, new_email, new_password)
        if success:
            print(f"\nSuccess! Admin credentials updated from '{default_admin_email}' to '{new_email}'.")
        else:
            print("\nFailed to update admin credentials.")
    else:
        # Check if the new email already exists
        if check_admin_exists(new_email):
            print(f"\nAn admin with email '{new_email}' already exists.")
            update_choice = input("Do you want to update this admin's password? (y/n): ")
            if update_choice.lower() == 'y':
                success = update_admin_credentials(new_email, new_email, new_password)
                if success:
                    print(f"\nSuccess! Password updated for admin '{new_email}'.")
                else:
                    print("\nFailed to update admin password.")
        else:
            success = add_new_admin(new_email, new_password)
            if success:
                print(f"\nSuccess! New admin '{new_email}' created.")
            else:
                print("\nFailed to create new admin.")

if __name__ == "__main__":
    main()