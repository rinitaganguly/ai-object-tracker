from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("test.jpg")

results[0].show()