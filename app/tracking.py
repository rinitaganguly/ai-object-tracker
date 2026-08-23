from deep_sort_realtime.deepsort_tracker import DeepSort


def create_tracker():
    return DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0
    )


def update_tracker(tracker, result, model, frame):
    detections = []

    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()

    for box, confidence, class_id in zip(
        boxes,
        confidences,
        class_ids
    ):
        x1, y1, x2, y2 = box

        width = x2 - x1
        height = y2 - y1

        detections.append(
            (
                [x1, y1, width, height],
                confidence,
                model.names[class_id]
            )
        )

    return tracker.update_tracks(
        detections,
        frame=frame
    )


def update_object_tracks(object_tracks, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_name = track.get_det_class()

        if class_name is None:
            continue

        if class_name not in object_tracks:
            object_tracks[class_name] = set()

        object_tracks[class_name].add(track_id)


def get_tracking_summary(object_tracks):
    total_unique_objects = sum(
        len(track_ids)
        for track_ids in object_tracks.values()
    )

    total_object_types = len(object_tracks)

    return total_unique_objects, total_object_types


def get_average_confidence(results):
    confidences = []

    for result in results:
        if result.boxes is None:
            continue

        if result.boxes.conf is None:
            continue

        confidences.extend(
            result.boxes.conf.cpu().tolist()
        )

    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)