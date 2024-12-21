import cv2
import numpy as np
import os

# frames directories
if not os.path.exists("frames_hw10"):
    os.makedirs("frames_hw10")

# Load video
video = cv2.VideoCapture("img/vid3.mp4")

# Read the first frame
ret, frame = video.read()
if not ret:
    print("Error: Cannot read video file.")
    exit()

# Resizing the frame for faster processing
frame = cv2.resize(frame, (640, 360))

# Selecting the bounding box for the object 
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Initialize KCF tracker
tracker_kcf = cv2.TrackerKCF_create()  
tracker_kcf.init(frame, bbox)

# Initialize CSRT tracker
tracker_csrt = cv2.TrackerCSRT_create()
tracker_csrt.init(frame, bbox)

frame_count = 0

while True:
    ret, frame = video.read()
    if not ret or frame_count >= 50:
        break

    frame = cv2.resize(frame, (640, 360))  # Resize to 640x360

    # Update KCF tracker
    ret_kcf, bbox_kcf = tracker_kcf.update(frame)
    # Update CSRT tracker
    ret_csrt, bbox_csrt = tracker_csrt.update(frame)

    # Draw the bounding box for KCF tracker (green)
    if ret_kcf:
        x, y, w, h = [int(v) for v in bbox_kcf]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the bounding box for CSRT tracker (blue)
    if ret_csrt:
        x, y, w, h = [int(v) for v in bbox_csrt]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result (with reduced window size)
    cv2.imshow("Tracking", frame)

    # Save the frame for KCF tracker
    frame_filename_kcf = f"frames_hw10/frame_{frame_count:03d}.jpg" 
    cv2.imwrite(frame_filename_kcf, frame)

    frame_count += 1

    # Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
