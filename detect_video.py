from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model("test.mp4", save=True)

print("Video detection complete!")