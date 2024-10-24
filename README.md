# Car Count with Buffer

**YOLO-based Traffic Analytics System**, integrated with Firebase. This system utilizes a buffer mechanism to append new records to Firebase periodically.

## Key Components

This system consists of two key Python files:

1. **`app3.py`**  
   Handles traffic counting and gate-related operations.
   
2. **`db_to_firebase.py`**  
   Syncs vehicle data from the local SQLite database to Firebase.

Both programs need to run simultaneously for the system to function properly.

## Usage

- **`app3.py`**  
  Use the `argparse` argument `--gate` to specify the gate number. For example:  
  python app3.py --gate 01


- **`db_to_firebase.py`**  
Use `argparse` arguments to specify the buffer time and system code. For example:
python db_to_firebase.py --buffer 30 --code B1


- `--buffer` specifies the buffer time in seconds (e.g., 30 seconds or 15 seconds).
- `--code` specifies a unique system code (e.g., B1, B2, or any identifier).

## Required Files

Make sure the following files are present in the working directory:

- `coco.txt` – Contains class labels for YOLO.
- `tracker.py` – Implements tracking logic.
- `creds` – Firebase credentials file (`creds.json`).

## Auto-Created Files

- **SQLite Database** – Stores vehicle logs.
- **Log File** – Records the system's activity.
  


