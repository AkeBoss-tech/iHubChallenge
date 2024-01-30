from ultralytics import YOLO

pose_model = YOLO("yolov8m.pt")

pose_model.predict('https://www.youtube.com/watch?v=wbL2cnj8an8', show=True)
input()