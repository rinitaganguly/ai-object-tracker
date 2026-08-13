from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-worldv2.pt")

model.set_classes([
    "person",
    "cell phone",
    "computer mouse",
    "Vaseline bottle",
    "keyboard",
    "laptop"
])

results = model("test.jpg")

results[0].show()