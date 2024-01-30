from ultralytics import YOLO
from PIL import Image
import cv2

print("Loading Models")
pose_model = YOLO("yolov8n-pose.pt")
obj_model = YOLO("yolov8n.pt")
print("Models Loaded")

print("Starting Camera")
cap = cv2.VideoCapture(0)
print("Camera Started")

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Run inference
    results = pose_model.predict(source=frame, show=True)

    # Display the resulting frame
    # Press q to quit
    if cv2.waitKey(1) == ord("q"):
        break 