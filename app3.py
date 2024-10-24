import cv2
import pandas as pd
import numpy as np
import base64
from ultralytics import YOLO
import cvzone
from vidgear.gears import CamGear
from tracker import *
import time
import sqlite3
from datetime import datetime
import argparse

# Parse command-line arguments for the gate number
parser = argparse.ArgumentParser(description="Vehicle Tracking with Gate Number")
parser.add_argument('--gate', type=int, required=True, help="Gate number to be logged")
args = parser.parse_args()

# Load YOLO model
model = YOLO('yolov8m.pt')

# Stream YouTube video
stream = CamGear(source='https://www.youtube.com/watch?v=_TusTf0iZQU', stream_mode=True, logging=True).start()

# Define areas for polygon detection
area1 = [(578, 112), (247, 166), (254, 193), (634,130)]
area2 = [(631, 129), (255, 195), (269, 222), (662,136)]

downcar = {}
downcarcounter = []
upcar = {}
upcarcounter = []

# Initialize database connection
conn = sqlite3.connect('vehicle_tracking_with_gate.db')
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS vehicle_log (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    up INTEGER,
    down INTEGER,
    type TEXT,
    image TEXT,  -- Column for base64 encoded image
    gate INTEGER  -- Column for gate number
)
''')

# Define a mouse event callback function to show cursor coordinates
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print("Cursor Coordinates: ", x, y)

# Create a named window for video and set the mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Initialize Tracker
tracker = Tracker()

# For FPS calculation
start_time = time.time()

# Read class labels from a file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

padding=20

# Main loop to read frames from the video stream
count = 0
while True:
    frame = stream.read()
    if frame is None:
        break

    count += 1
    if count % 2 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection using YOLO
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # Filter out car, truck, bus, cycle, and bike objects
    car_list = []
    vehicle_types = {}  # Dictionary to store vehicle types
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        vehicle_type = class_list[int(d)]
        if vehicle_type in ['car', 'truck', 'bus', 'cycle', 'bike']:
            car_list.append([int(x1), int(y1), int(x2), int(y2)])
            vehicle_types[(int(x1), int(y1), int(x2), int(y2))] = vehicle_type  # Store type
            print("Vehicle detected:", vehicle_type)

    # Update tracker with car bounding boxes
    bbox_idx = tracker.update(car_list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id1 = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        # Detect the car in the polygon of area1
        result = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
        print(result)

        # Draw bounding box and ID if car is within area1
        if result > 0:
            downcar[id1] = (cx, cy)
        if id1 in downcar:
            result1 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
            if result1 > 0:
                # cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 1)
                vehicle_type = vehicle_types[(x3, y3, x4, y4)]
                # cvzone.putTextRect(frame, f'ID: {id1}, Type: {vehicle_type}', (x3, y3), 1, 1)

                # Only increment the counter once
                if downcarcounter.count(id1) == 0:
                    downcarcounter.append(id1)
                    print(f'ID: {id1}, Type: {vehicle_type}')
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


                    # Save cropped image as base64 string
                    cropped_image = frame[max(0, y3-padding):min(frame.shape[0], y4+padding), max(0, x3-padding):min(frame.shape[1], x4+padding)]

                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    cropped_image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Insert into database with base64 encoded image and gate number
                    cursor.execute('''
                    INSERT INTO vehicle_log (timestamp, up, down, type, image, gate)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, 0, 1, vehicle_type, cropped_image_base64, args.gate))
                    conn.commit()

        # Detect the car in the polygon of area2
        result2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
        print(result2)

        # Draw bounding box and ID if car is within area2
        if result2 > 0:
            upcar[id1] = (cx, cy)
        if id1 in upcar:
            result3 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
            if result3 > 0:
                # cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 1)
                vehicle_type = vehicle_types[(x3, y3, x4, y4)]
                # cvzone.putTextRect(frame, f'ID: {id1}, Type: {vehicle_type}', (x3, y3), 1, 1)

                # Only increment the counter once
                if upcarcounter.count(id1) == 0:
                    upcarcounter.append(id1)
                    print(f'ID: {id1}, Type: {vehicle_type}')
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    # Save cropped image as base64 string
                    cropped_image = frame[max(0, y3-padding):min(frame.shape[0], y4+padding), max(0, x3-padding):min(frame.shape[1], x4+padding)]

                    _, buffer = cv2.imencode('.jpg', cropped_image)
                    cropped_image_base64 = base64.b64encode(buffer).decode('utf-8')

                    # Insert into database with base64 encoded image and gate number
                    cursor.execute('''
                    INSERT INTO vehicle_log (timestamp, up, down, type, image, gate)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (timestamp, 1, 0, vehicle_type, cropped_image_base64, args.gate))
                    conn.commit()

    # Display local counter for cars moving down and up
    cvzone.putTextRect(frame, f'Local Down: {len(downcarcounter)}', (50, 50), scale=2, thickness=2, colorT=(0, 0, 0), colorR=(255, 255, 255, 150))
    cvzone.putTextRect(frame, f'Local Up: {len(upcarcounter)}', (50, 100), scale=2, thickness=2, colorT=(0, 0, 0), colorR=(255, 255, 255, 150))

    # Calculate FPS
    fps = 1.0 / (time.time() - start_time)
    start_time = time.time()

    # Display FPS on the video
    frame_width = 1020
    cvzone.putTextRect(frame, f'FPS: {int(fps)}', (frame_width - 200, 50), scale=2, thickness=2, colorR=(0, 0, 0), colorT=(0, 255, 0))

    # Display the frame
    cv2.imshow("RGB", frame)

    # Check for exit key press (Esc key)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

# Release video capture and close OpenCV windows
stream.stop()
cv2.destroyAllWindows()
conn.close()
