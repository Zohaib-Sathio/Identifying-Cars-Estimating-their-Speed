import cv2
from collections import defaultdict
from ultralytics import YOLO
import numpy as np

model = YOLO("Car Detection and Speed Estimation\\best_UAV_car_detection.pt")

video_path = "Car Detection and Speed Estimation\\y2mate.com - Crossroads Traffic Aerial View Static Drone Free Stock Video No Copyright_v720P (online-video-cutter.com).mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])
# speed_history = defaultdict(lambda: [])
average_speed_history = defaultdict(lambda: (0, 0))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

output_video_path = "processed_output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))


meters_per_pixel = 0.06

smoothing_window_size = frame_rate

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, show_boxes=False)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        annotated_frame = results[0].plot(conf=False, labels=False)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point

            if len(track) > 30: 
                track.pop(0)
            
            if len(track) > 1:
                # Get the current and previous points
                prev_x, prev_y = track[-2] 
                curr_x, curr_y = track[-1]  

                # Calculate the distance in pixels between the points
                pixel_distance = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)

                if pixel_distance < 0.9:
                    pixel_distance = 0

                # Convert pixel distance to meters
                real_distance = pixel_distance * meters_per_pixel

                # Time interval between frames
                time_interval = 1 / frame_rate

                speed_mps = real_distance / time_interval

                # Convert speed to kilometers per hour (km/h)
                speed_kmph = speed_mps * 3.6

                previous_average, count = average_speed_history[track_id]

                # Incrementally update the average speed using CMA
                new_count = count + 1
                new_average = (previous_average * count + speed_kmph) / new_count

                average_speed_history[track_id] = (new_average, new_count)

                # speed_history[track_id].append(speed_kmph)

                # # Limit the history to the smoothing window size
                # if len(speed_history[track_id]) > smoothing_window_size:
                #     speed_history[track_id].pop(0)

                # # Calculate the average speed for smoothing
                # smoothed_speed_kmph = np.mean(speed_history[track_id])

                if speed_kmph == 0:
                    pass
                    # cv2.putText(annotated_frame, f'Not Moving!',
                    #         (int(x - w / 2), int(y + h / 2 - 5)),
                    #         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, f'Speed: {new_average:.2f} km/h',
                            (int(x - w / 2), int(y + h / 2 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.rectangle(annotated_frame, 
                          (int(x - w / 2), int(y - h / 2)), 
                          (int(x + w / 2), int(y + h / 2)), 
                          (0, 255, 0),  
                          1)  


        out.write(annotated_frame)

        cv2.imshow("YOLO11 Tracking - Cars Only", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break if the end of the video
        break

cap.release()
out.release()
cv2.destroyAllWindows()