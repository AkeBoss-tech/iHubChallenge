from ultralytics import YOLO
from tags import get_tags, draw_tags
from faces import detect
import cv2

print("Loading Models")
pose_model = YOLO("yolov8n-pose.pt")
obj_model = YOLO("yolov8n.pt")
seg_model = YOLO("yolov8n-seg.pt")
print("Models Loaded")

print("Starting Camera")
cap = cv2.VideoCapture(1)
print("Camera Started")

# Set mode
# tags, pose, object, seg, face
current_mode = "pose"

while True:
    # Read frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if current_mode == "pose":
        # Run inference
        results = pose_model.predict(source=frame, show=True)
    elif current_mode == "object":
        # Run inference
        results = obj_model.predict(source=frame, show=True)
    elif current_mode == "tags":
        # Run inference
        try:
            tags = get_tags(frame)
        except Exception as e:
            print(e)
            current_mode = "object"
            continue
        frame = draw_tags(frame, tags)
        # Display the resulting frame
        cv2.imshow("AprilTag Detection", frame)
    elif current_mode == "face":
        cv2.imshow("Face Detection", detect(frame))
    elif current_mode == "seg":
        results = seg_model.predict(source=frame, show=True)
    else:
        print(f"Invalid Mode: {current_mode}")
        break

    # Display the resulting frame
    # Press q to quit
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        print("Quitting")
        break 
    
    if key == ord("1"):
        print("Changing Mode")
        current_mode = "pose"
        print(f"Mode: {current_mode}")
        cv2.destroyAllWindows()
    elif key == ord("2"):
        print("Changing Mode")
        current_mode = "object"
        print(f"Mode: {current_mode}")
        cv2.destroyAllWindows()
    elif key == ord("3"):
        print("Changing Mode")
        current_mode = "tags"
        print(f"Mode: {current_mode}")
        cv2.destroyAllWindows()
    elif key == ord("4"):
        print("Changing Mode")
        current_mode = "face"
        print(f"Mode: {current_mode}")
        cv2.destroyAllWindows()
    elif key == ord("5"):
        print("Changing Mode")
        current_mode = "seg"
        print(f"Mode: {current_mode}")
        cv2.destroyAllWindows()
    
    