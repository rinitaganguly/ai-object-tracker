from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model("test.jpg")

for result in results:
    if result.boxes is None:
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2%}")