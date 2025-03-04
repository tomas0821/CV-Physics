import cv2
from ultralytics import YOLO
import numpy as np
import time
import csv
from collections import deque

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
bounce_data = []  # Stores (bounce_time, peak_height, velocity_before, velocity_after)
tracking_active = False
last_position = None

# Bounce detection variables
rolling_window = deque(maxlen=7)  # Store the last 7 y-positions
bounces = []  # Stores (x, y, bounce_number)
bounce_count = 0  # Number of detected bounces
last_bounce_time = 0  # Last detected bounce (to avoid duplicates)

# File to save bounce data
bounce_file = "bounce_data.csv"

# Function to save bounce data to CSV
def save_bounce_data():
    if bounce_data:
        with open(bounce_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bounce Time", "Peak Height", "Velocity Before", "Velocity After"])
            writer.writerows(bounce_data)
        print(f"Bounce data saved to {bounce_file}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Display updated instructions
    cv2.putText(frame, "Press 'S' to Start Tracking", (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Y' to Stop & Save | 'R' to Reset | 'Q' to Quit", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Kalman predicts next position before detection
    prediction = kalman.predict()
    pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])

    # Run YOLO on every frame
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
        timestamp = time.time()

        # Save position if tracking is active
        if tracking_active:
            trajectory.append((timestamp, center_x, center_y))

        # Update Kalman Filter with real detection
        measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(measurement)

        last_position = (center_x, center_y)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Ping Pong Ball ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Only detect bounces if tracking is active
        if tracking_active:
            rolling_window.append(center_y)  # Add the current y-position to rolling window

            # Ensure we have enough points in the rolling window
            if len(rolling_window) == rolling_window.maxlen:
                min_y = min(rolling_window)
                min_index = list(rolling_window).index(min_y)

                # Detect a bounce: When the middle value of the rolling window is the lowest
                if min_index == len(rolling_window) // 2:
                    time_since_last_bounce = timestamp - last_bounce_time

                    if time_since_last_bounce > 0.1:  # Ignore bounces too close together
                        bounce_time = timestamp
                        peak_height = min_y  # Use min_y from rolling window
                        velocity_before = (trajectory[-2][2] - trajectory[-1][2]) / (trajectory[-1][0] - trajectory[-2][0]) if len(trajectory) > 1 else 0

                        # Store bounce data
                        bounce_data.append((bounce_time, peak_height, velocity_before, velocity_before))
                        bounce_count += 1
                        bounces.append((center_x, center_y, bounce_count))

                        last_bounce_time = timestamp  # Update last bounce time

                        print(f"Bounce {bounce_count} detected! Time: {bounce_time:.2f}s, Peak: {peak_height}px, V_before: {velocity_before:.2f}")

        # Update previous position
        previous_y = center_y

    # If no detection, use Kalman predicted position
    if not detected and last_position and tracking_active:
        center_x, center_y = pred_x, pred_y
        trajectory.append((time.time(), center_x, center_y))

    # Draw trajectory
    for i in range(1, len(trajectory)):
        x1, y1 = int(trajectory[i - 1][1]), int(trajectory[i - 1][2])
        x2, y2 = int(trajectory[i][1]), int(trajectory[i][2])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red trail

    # Draw detected bounces
    for (bx, by, bnum) in bounces:
        cv2.circle(frame, (bx, by), 6, (0, 0, 255), -1)  # Red dot for bounce
        cv2.putText(frame, f"B{bnum}", (bx + 10, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Ping Pong Ball Tracking", frame)

    # Keyboard control
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("Tracking started.")
        tracking_active = True
        trajectory = []  # Reset trajectory
        bounces = []  # Reset bounces
        bounce_count = 0

    if key == ord("y"):
        print("Tracking stopped. Saving bounce data...")
        tracking_active = False
        save_bounce_data()

    if key == ord("r"):
        print("Resetting trajectory and bounce data.")
        trajectory = []
        bounce_data = []
        bounces = []
        bounce_count = 0

    if key == ord("q"):
        print("Exiting program... Saving bounce data.")
        save_bounce_data()
        break

cap.release()
cv2.destroyAllWindows()
