from collections import defaultdict

import cv2


def draw_trajectories(results, track_history):
    if results.boxes is None:
        return results.plot(), track_history

    if results.boxes.id is None:
        return results.plot(), track_history

    boxes = results.boxes.xywh.cpu()
    track_ids = results.boxes.id.int().cpu().tolist()

    annotated_frame = results.plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, _, _ = box
        center = (int(x), int(y))

        track = track_history[track_id]
        track.append(center)

        if len(track) > 30:
            track.pop(0)

        points = track

        for i in range(1, len(points)):
            cv2.line(
                annotated_frame,
                points[i - 1],
                points[i],
                (255, 255, 255),
                2
            )

    return annotated_frame, track_history


def create_track_history():
    return defaultdict(list)