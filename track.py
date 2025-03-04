import cv2
from ultralytics import YOLO
import numpy as np
import time
import csv

# Load YOLOv11 model
model = YOLO("pingpong_11n.pt")  # Replace with your trained model path

# Open webcam
cap = cv2.VideoCapture(0)

# Resize frames for better FPS
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Kalman Filter Setup
kalman = cv2.KalmanFilter(4, 2)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                    [0, 1, 0, 1], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]], dtype=np.float32)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                     [0, 1, 0, 0]], dtype=np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 0.1

# Tracking variables
trajectory = []  # Stores (timestamp, x, y) positions
tracking_enabled = False  # Start with tracking OFF
tracking_active = False  # Controls whether to save data
last_position = None  # Store last detected position

# File to save trajectory data
trajectory_file = "trajectory_data.csv"

# Function to save trajectory data to CSV
def save_trajectory():
    if trajectory:
        with open(trajectory_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "X", "Y"])  # Header
            writer.writerows(trajectory)  # Save all tracked points
        print(f"Trajectory saved to {trajectory_file}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Display updated instructions in two lines
    cv2.putText(frame, "Press 'S' to Start | 'Y' to Stop & Save", 
                (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'R' to Reset | 'Q' to Quit", 
                (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Kalman predicts next position before detection
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0]), int(prediction[1])

    # Run YOLO on every frame (no skipping)
    results = model(frame, conf=0.5)

    detected = False
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < 0.5:
            continue  # Skip low-confidence detections

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        detected = True
        timestamp = time.time()  # Get current time

        # Save position if tracking is active
        if tracking_active:
            trajectory.append((timestamp, center_x, center_y))

        # Update Kalman Filter with real detection
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(measurement)

        last_position = (center_x, center_y)  # Save last valid detection

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Ping Pong Ball ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If no detection, use Kalman predicted position
    if not detected and last_position:
        center_x, center_y = pred_x, pred_y
        if tracking_active:
            trajectory.append((time.time(), center_x, center_y))

    # Draw trajectory (resets when 'R' is pressed)
    for i in range(1, len(trajectory)):
        x1, y1 = int(trajectory[i - 1][1]), int(trajectory[i - 1][2])
        x2, y2 = int(trajectory[i][1]), int(trajectory[i][2])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red trail

    # Show the frame
    cv2.imshow("Ping Pong Ball Tracking", frame)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF

    # (S) Start tracking
    if key == ord("s"):
        print("Tracking started.")
        tracking_enabled = True
        tracking_active = True  # Start saving trajectory
        trajectory = []  # Reset trajectory when tracking starts

    # (Y) Stop tracking & save
    if key == ord("y"):
        print("Tracking stopped. Saving data...")
        tracking_active = False
        save_trajectory()  # Save when tracking stops

    # (R) Reset trajectory (clear screen)
    if key == ord("r"):
        print("Trajectory reset.")
        trajectory = []  # Clear trajectory from screen

    # (Q) Quit the program
    if key == ord("q"):
        print("Exiting program... Saving trajectory (if tracking).")
        save_trajectory()  # Save trajectory if tracking was on
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
