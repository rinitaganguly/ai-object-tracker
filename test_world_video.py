from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-worldv2.pt")

model.set_classes([
    "person",
    "smartphone",
    "wireless computer mouse",
    "petroleum jelly container",
    "keyboard",
    "laptop"
])

results = model("test.mp4", save=True)

print("YOLO-World video detection complete!")