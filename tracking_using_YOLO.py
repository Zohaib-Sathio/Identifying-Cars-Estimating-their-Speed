from ultralytics import YOLO

# Load an official or custom model
model = YOLO("Car Detection and Speed Estimation\\best_UAV_car_detection.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# Perform tracking with the model
results = model.track("Car Detection and Speed Estimation\\y2mate.com - Crossroads Traffic Aerial View Static Drone Free Stock Video No Copyright_v720P (online-video-cutter.com).mp4", show=True, save=True)  # Tracking with default tracker
# results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # with ByteTrack

for result in results:
    print(result)