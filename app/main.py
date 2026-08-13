import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import tempfile
import os
from pathlib import Path


st.set_page_config(
    page_title="AI Vision Tracker",
    page_icon="👁️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")


model = load_model()


# -----------------------------
# Webcam processor
# -----------------------------

class YOLOVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.confidence = 0.50
        self.selected_classes = []

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        track_args = {
            "source": image,
            "conf": self.confidence,
            "persist": True,
            "verbose": False
        }

        if self.selected_classes:

            class_ids = [
                class_id
                for class_id, class_name in model.names.items()
                if class_name in self.selected_classes
            ]

            track_args["classes"] = class_ids

        results = model.track(**track_args)

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )


# -----------------------------
# Sidebar
# -----------------------------

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

    available_classes = list(model.names.values())

    selected_classes = st.multiselect(
        "Track specific objects",
        options=available_classes,
        default=[]
    )

    st.markdown("---")

    st.caption("Computer Vision • YOLO")


# -----------------------------
# Main application
# -----------------------------

st.title("AI Vision Tracker")

st.write(
    "Detect and track objects using YOLO in uploaded videos "
    "or directly from your webcam."
)


# -----------------------------
# Live Webcam
# -----------------------------

st.header("Live Webcam Tracking")

st.write(
    "Start your camera to detect and track objects in real time."
)

webrtc_ctx = webrtc_streamer(
    key="yolo-live-tracker",
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


# -----------------------------
# Uploaded Video
# -----------------------------

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

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(uploaded_file.name).suffix
        ) as temp_file:

            temp_file.write(uploaded_file.read())
            video_path = temp_file.name


        with st.spinner("Detecting and tracking objects..."):

            tracking_args = {
                "source": video_path,
                "conf": confidence,
                "save": True,
                "persist": True
            }


            if selected_classes:

                class_ids = [
                    class_id
                    for class_id, class_name in model.names.items()
                    if class_name in selected_classes
                ]

                tracking_args["classes"] = class_ids


            results = model.track(**tracking_args)


        st.success("Analysis complete!")


        # -----------------------------
        # Unique object tracking
        # -----------------------------

        object_tracks = {}


        for result in results:

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue


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


            for track_id, class_id in zip(
                track_ids,
                class_ids
            ):

                class_name = model.names[class_id]


                if class_name not in object_tracks:
                    object_tracks[class_name] = set()


                object_tracks[class_name].add(track_id)


        total_unique_objects = sum(
            len(track_ids)
            for track_ids in object_tracks.values()
        )


        total_object_types = len(object_tracks)


        # -----------------------------
        # Analysis Overview
        # -----------------------------

        st.subheader("Analysis Overview")

        st.write(
            f"**Frames Analyzed:** {len(results)}"
        )

        st.write(
            f"**Unique Objects:** {total_unique_objects}"
        )

        st.write(
            f"**Object Types:** {total_object_types}"
        )


        # -----------------------------
        # Object Summary
        # -----------------------------

        st.subheader("Unique Objects Tracked")


        if object_tracks:

            for object_name, track_ids in object_tracks.items():

                st.write(
                    f"**{object_name.title()}**: "
                    f"{len(track_ids)}"
                )

        else:

            st.write(
                "No trackable objects detected."
            )


        # -----------------------------
        # Processed Video
        # -----------------------------

        output_dir = Path(
            results[0].save_dir
        )


        video_files = [
            file
            for file in output_dir.iterdir()
            if file.suffix.lower()
            in [".mp4", ".avi", ".mov"]
        ]


        if video_files:

            output_video = video_files[0]


            st.subheader(
                "Detection & Tracking Result"
            )


            st.video(
                str(output_video)
            )


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


        else:

            st.warning(
                "Tracking completed, but the "
                "processed video could not be found."
            )


        os.remove(video_path)