from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO('yolov8m.pt')
 
#pip install pafy
#sudo pip install --upgrade youtube_dl
import cv2, pafy, numpy as np
from collections import defaultdict


url   = "https://www.youtube.com/watch?v=iiJG3GcI9NY"
video = pafy.new(url)
best  = video.getbest()
#documentation: https://pypi.org/project/pafy/
print(video)
print(video.streams)

capture = cv2.VideoCapture(best.url)

# Store the track history
track_history = defaultdict(lambda: [])

while True:
    # Read a frame from the video
    success, frame = capture.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
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

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

capture.release()
cv2.destroyAllWindows()