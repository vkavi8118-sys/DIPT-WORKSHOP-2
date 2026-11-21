# DIPT-WORKSHOP-2
# YOLOv8 Object Detection Using Laptop Camera
### NAME: KAVINAYA V
### REGISTER NO: 212225230133
## Aim
To access the laptop camera, capture an image, and detect objects using YOLOv8.

## Requirements
Anaconda
Jupyter Notebook
Laptop Camera
## Steps
Open Jupyter Notebook using Anaconda.
Access your laptop camera using OpenCV.
Wait for 5 seconds and capture an image.
Display the captured image.
Apply YOLOv8 object detection.
Display the detected image with bounding boxes and labels.
## Algorithm
```
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```
Student Task
Capture your own image using the laptop camera.
Detect the objects present in the image.
Display the final output.
Perform the experiment with 3 different scenes.
Submission
Jupyter Notebook (.ipynb)
Original captured images
YOLOv8 output images
Screenshot of the final result
GitHub Reference
https://github.com/ultralytics/ultralytics

Platform: Anaconda + Jupyter Notebook only.

## program
```from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load YOLO model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break

        # Perform object detection
        results = model(frame, conf=0.60, verbose=False)

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display in Jupyter
        clear_output(wait=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(annotated_frame)
        plt.title("Real-Time Object Detection")
        plt.axis("off")
        plt.show()

# Release webcam
cap.release()
cv2.destroyAllWindows()
```
## Output
<img width="897" height="702" alt="Screenshot 2026-11-21 102201" src="https://github.com/user-attachments/assets/3ba0b17f-a7e0-426c-b62c-18cb0596cf96" />
