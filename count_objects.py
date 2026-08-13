from ultralytics import YOLO

model = YOLO("yolo11s.pt")

seen_ids = set()

results = model.track(
    source="test.mp4",
    persist=True
)

for result in results:
    if result.boxes is None or result.boxes.id is None:
        continue

    track_ids = result.boxes.id.int().cpu().tolist()

    for track_id in track_ids:
        seen_ids.add(track_id)

print("Unique objects tracked:", len(seen_ids))