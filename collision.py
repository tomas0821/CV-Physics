import cv2
from ultralytics import YOLO
import numpy as np
import time

# Load YOLOv11 model
model = YOLO("pingpong_11s.pt")  # Replace with your trained model path

# Open webcam
cap = cv2.VideoCapture(0)

# Resize frames for better FPS
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Tracking variables
trajectory_1 = []  # Ball 1 trajectory (Red)
trajectory_2 = []  # Ball 2 trajectory (Blue)
tracking_active = False

# Previous positions of Ball 1 and Ball 2
prev_ball_1 = None  # (x, y)
prev_ball_2 = None  # (x, y)

# Collision variables
COLLISION_THRESHOLD = 25  # Pixels (adjust as needed)
last_collision_time = 0  # To prevent duplicate collisions
collision_points = []  # Store detected collision points

# Function to match detected balls to their previous identities
def match_balls(detections, prev_ball_1, prev_ball_2):
    assigned_1, assigned_2 = None, None
    
    if len(detections) < 2:
        return None, None  # Need at least 2 balls to track collisions

    # Sort detections by x-coordinate (left to right)
    sorted_detections = sorted(detections, key=lambda b: b[0])

    # If there were no previous positions, assign first two detections
    if prev_ball_1 is None or prev_ball_2 is None:
        assigned_1 = sorted_detections[0]
        assigned_2 = sorted_detections[1]
        return assigned_1, assigned_2

    # Compute distances to previous positions
    distances = []
    for det in detections:
        x, y, _ = det
        d1 = np.linalg.norm([x - prev_ball_1[0], y - prev_ball_1[1]]) if prev_ball_1 else float("inf")
        d2 = np.linalg.norm([x - prev_ball_2[0], y - prev_ball_2[1]]) if prev_ball_2 else float("inf")
        distances.append((d1, d2, det))

    # Ensure the closest detection is assigned to each previous ball
    distances.sort()
    for d1, d2, det in distances:
        if assigned_1 is None and d1 < d2:
            assigned_1 = det
        elif assigned_2 is None:
            assigned_2 = det

    return assigned_1, assigned_2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Display instructions
    cv2.putText(frame, "Press 'S' to Start Tracking", (20, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Y' to Stop & Save | 'R' to Reset | 'Q' to Quit", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Run YOLO on every frame
    results = model(frame, conf=0.5)

    detected_balls = []  # Store (x, y, width) for detected balls

    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < 0.5:
            continue  # Skip low-confidence detections

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        width = x2 - x1  # Width of the bounding box

        detected_balls.append((center_x, center_y, width))

    # Match detections to previous positions
    ball_1, ball_2 = match_balls(detected_balls, prev_ball_1, prev_ball_2)

    timestamp = time.time()

    # Process Ball 1
    if ball_1:
        x1, y1, _ = ball_1
        prev_ball_1 = (x1, y1)  # Update previous position
        if tracking_active:
            trajectory_1.append((x1, y1))  # Append position

        cv2.circle(frame, (x1, y1), 10, (0, 0, 255), -1)  # Red for Ball 1
        cv2.putText(frame, "Ball 1", (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Process Ball 2
    if ball_2:
        x2, y2, _ = ball_2
        prev_ball_2 = (x2, y2)  # Update previous position
        if tracking_active:
            trajectory_2.append((x2, y2))  # Append position

        cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)  # Blue for Ball 2
        cv2.putText(frame, "Ball 2", (x2 + 10, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Detect Collision
    if ball_1 and ball_2:
        distance = np.linalg.norm([x1 - x2, y1 - y2])  # Compute Euclidean distance

        if distance < COLLISION_THRESHOLD and (timestamp - last_collision_time > 0.2):
            print(f"Collision detected at ({(x1 + x2) // 2}, {(y1 + y2) // 2}) at time {timestamp:.2f}s")
            last_collision_time = timestamp  # Update last collision time
            collision_points.append(((x1 + x2) // 2, (y1 + y2) // 2))  # Store collision point

    # Draw trajectory for Ball 1
    if len(trajectory_1) > 1:
        for i in range(1, len(trajectory_1)):
            cv2.line(frame, trajectory_1[i - 1], trajectory_1[i], (0, 0, 255), 2)

    # Draw trajectory for Ball 2
    if len(trajectory_2) > 1:
        for i in range(1, len(trajectory_2)):
            cv2.line(frame, trajectory_2[i - 1], trajectory_2[i], (255, 0, 0), 2)

    # Draw collision points (Yellow)
    for (cx, cy) in collision_points:
        cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)  # Yellow for collision
        cv2.putText(frame, "Collision!", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Ping Pong Ball Collision Tracking", frame)

    # Exit with "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
