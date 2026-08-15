def update_object_tracks(object_tracks, result, model):
    if result.boxes is None:
        return

    if result.boxes.id is None:
        return

    track_ids = (
        result.boxes.id
        .int()
        .cpu()
        .tolist()
    )

    class_ids = (
        result.boxes.cls
        .int()
        .cpu()
        .tolist()
    )

    for track_id, class_id in zip(track_ids, class_ids):

        class_name = model.names[class_id]

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