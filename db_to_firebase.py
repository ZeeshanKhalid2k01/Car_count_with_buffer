import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import sqlite3
import logging
import time
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Maximum expected number of digits in row ID (adjust if needed)
MAX_ID_DIGITS = 10

def initialize_firebase():
    cred = credentials.Certificate('creds.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://vehicle-detection-2730c-default-rtdb.firebaseio.com/'
    })
    logging.info("Firebase initialized successfully")

def connect_to_sqlite(db_file):
    conn = sqlite3.connect(db_file)
    return conn

def get_new_rows(conn, last_row_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM vehicle_log WHERE id > ? ORDER BY id", (last_row_id, ))
    rows = cursor.fetchall()
    return rows

def send_to_firebase(rows, system_code):
    ref = db.reference('/vehicle_log')  # Adjust the path as needed
    updates = {}
    for row in rows:
        data = {
            'id': row[0],
            'timestamp': row[1],
            'up': row[2],
            'down': row[3],
            'type': row[4],
            'image': row[5],
            'gate': row[6]
        }
        # Pad the row ID with zeros
        padded_id = str(row[0]).zfill(MAX_ID_DIGITS)
        key = f"{padded_id}_{system_code}"
        updates[key] = data

    if updates:
        ref.update(updates)
        logging.info(f"Saved {len(updates)} records to Firebase")

def get_last_row_id():
    try:
        with open('last_row_id.txt', 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0  # If the file doesn't exist, start from 0

def save_last_row_id(row_id):
    with open('last_row_id.txt', 'w') as f:
        f.write(str(row_id))

def main(system_code, buffer_time):
    try:
        initialize_firebase()
        conn = connect_to_sqlite('vehicle_tracking_with_gate.db')  # Replace with your SQLite database path

        while True:
            logging.info("Checking for new records")
            last_row_id = get_last_row_id()
            new_rows = get_new_rows(conn, last_row_id)

            if new_rows:
                send_to_firebase(new_rows, system_code)
                last_row_id = new_rows[-1][0]  # Get the ID of the last row sent
                save_last_row_id(last_row_id)
            else:
                logging.info("No new records found")

            time.sleep(buffer_time)  # Wait for the specified buffer time (in seconds)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Sync vehicle log data to Firebase with custom system code and buffer.")
    
    # Add arguments for system code and buffer time
    parser.add_argument('--code', type=str, required=True, help='System code (e.g., H1, H2, etc.)')
    parser.add_argument('--buffer', type=int, required=True, help='Buffer time in seconds (e.g., 30 or 15 seconds)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Pass the arguments to the main function
    main(system_code=args.code, buffer_time=args.buffer)
