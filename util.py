from ultralytics import YOLO
import cv2, numpy as np
from collections import defaultdict

# Store the track history
track_history = defaultdict(lambda: [])

def track_frame(frame, model):
    global track_history

    results = model.track(frame, persist=True)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    return annotated_frame

def clear_track_history():
    global track_history
    track_history = defaultdict(lambda: [])