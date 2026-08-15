from ultralytics import YOLO


MODEL_PATH = "yolo11s.pt"


def load_model():
    return YOLO(MODEL_PATH)


def get_class_ids(model, selected_classes):
    if not selected_classes:
        return None

    return [
        class_id
        for class_id, class_name in model.names.items()
        if class_name in selected_classes
    ]