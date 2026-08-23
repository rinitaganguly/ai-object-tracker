import os
import tempfile
from collections import defaultdict
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from detector import load_model, get_class_ids
from tracking import (
    create_tracker,
    update_tracker,
    update_object_tracks,
    get_tracking_summary,
    get_average_confidence
)


st.set_page_config(
    page_title="AI Vision Tracker",
    layout="wide"
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()


def draw_tracks(frame, tracks, track_history):
    annotated_frame = frame.copy()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_name = track.get_det_class()

        if class_name is None:
            class_name = "object"

        box = track.to_ltrb()

        if box is None:
            continue

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        label = f"{class_name} ID: {track_id}"

        cv2.rectangle(
            annotated_frame,
            (x1, max(0, y1 - 28)),
            (x1 + 180, y1),
            (17, 17, 17),
            -1
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1 + 6, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        history = track_history[track_id]
        history.append(center)

        if len(history) > 30:
            history.pop(0)

        for index in range(1, len(history)):
            cv2.line(
                annotated_frame,
                history[index - 1],
                history[index],
                (255, 255, 255),
                2
            )

    return annotated_frame


def process_video(
    input_path,
    output_path,
    confidence,
    selected_classes
):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    tracker = create_tracker()

    track_history = defaultdict(list)
    object_tracks = {}
    all_results = []

    class_ids = get_class_ids(
        model,
        selected_classes
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        prediction_args = {
            "source": frame,
            "conf": confidence,
            "verbose": False
        }

        if class_ids:
            prediction_args["classes"] = class_ids

        results = model.predict(**prediction_args)
        result = results[0]

        all_results.append(result)

        tracks = update_tracker(
            tracker,
            result,
            model,
            frame
        )

        update_object_tracks(
            object_tracks,
            tracks
        )

        annotated_frame = draw_tracks(
            frame,
            tracks,
            track_history
        )

        writer.write(annotated_frame)

    cap.release()
    writer.release()

    return (
        all_results,
        object_tracks,
        output_path
    )


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.confidence = 0.50
        self.selected_classes = []
        self.track_history = defaultdict(list)
        self.tracker = create_tracker()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        prediction_args = {
            "source": image,
            "conf": self.confidence,
            "verbose": False
        }

        class_ids = get_class_ids(
            model,
            self.selected_classes
        )

        if class_ids:
            prediction_args["classes"] = class_ids

        results = model.predict(**prediction_args)
        result = results[0]

        tracks = update_tracker(
            self.tracker,
            result,
            model,
            image
        )

        annotated_frame = draw_tracks(
            image,
            tracks,
            self.track_history
        )

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


with st.sidebar:
    st.title("AI Vision Tracker")

    st.markdown("---")

    st.subheader("Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    st.markdown("---")

    st.subheader("Object Filters")

    available_classes = list(
        model.names.values()
    )

    selected_classes = st.multiselect(
        "Track specific objects",
        options=available_classes,
        default=[]
    )

    st.markdown("---")

    st.caption("Computer Vision • YOLO • Deep SORT")


st.title("AI Vision Tracker")

st.write(
    "Detect and track objects using YOLO and Deep SORT "
    "from live webcam or uploaded video."
)


st.header("Live Webcam Tracking")

st.write(
    "Start your camera to detect and track objects in real time."
)

webrtc_ctx = webrtc_streamer(
    key="deep-sort-live-tracker",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.confidence = confidence
    webrtc_ctx.video_processor.selected_classes = selected_classes


st.markdown("---")

st.header("Video Analysis")

uploaded_file = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov"]
)


if uploaded_file is not None:
    st.success("Video uploaded successfully.")

    st.video(uploaded_file)

    if st.button("Analyze Video"):
        input_suffix = Path(
            uploaded_file.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_suffix
        ) as input_file:
            input_file.write(
                uploaded_file.read()
            )
            input_path = input_file.name

        output_path = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ).name

        try:
            with st.spinner(
                "Detecting, tracking and generating trajectories..."
            ):
                (
                    results,
                    object_tracks,
                    output_video
                ) = process_video(
                    input_path,
                    output_path,
                    confidence,
                    selected_classes
                )

            st.success("Analysis complete!")

            (
                total_unique_objects,
                total_object_types
            ) = get_tracking_summary(
                object_tracks
            )

            average_confidence = get_average_confidence(
                results
            )

            st.subheader("Analysis Overview")

            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric(
                    "Frames Analyzed",
                    len(results)
                )

            with metric2:
                st.metric(
                    "Unique Objects",
                    total_unique_objects
                )

            with metric3:
                st.metric(
                    "Object Types",
                    total_object_types
                )

            with metric4:
                st.metric(
                    "Avg Confidence",
                    f"{average_confidence:.0%}"
                )

            if object_tracks:
                st.subheader("Detection Summary")

                object_counts = {
                    object_name.title(): len(track_ids)
                    for object_name, track_ids
                    in object_tracks.items()
                }

                st.bar_chart(object_counts)

                st.subheader("Unique Objects Tracked")

                for object_name, track_ids in object_tracks.items():
                    st.write(
                        f"**{object_name.title()}**: "
                        f"{len(track_ids)}"
                    )
            else:
                st.subheader("Detection Summary")
                st.info(
                    "No trackable objects detected."
                )

            st.subheader(
                "Detection, Tracking & Trajectory Result"
            )

            st.video(output_video)

            with open(
                output_video,
                "rb"
            ) as video_file:
                st.download_button(
                    label="Download Result Video",
                    data=video_file,
                    file_name="ai_vision_tracking_result.mp4",
                    mime="video/mp4"
                )

        except Exception as error:
            st.error(
                f"Video processing failed: {error}"
            )

        finally:
            if os.path.exists(input_path):
                os.remove(input_path)