from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.track(
    source="test.mp4",
    save=True,
    show=True,
    persist=True
)

print("Object tracking complete!")