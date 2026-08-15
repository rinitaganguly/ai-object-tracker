import os
import tempfile
from pathlib import Path

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from detector import load_model, get_class_ids
from tracking import update_object_tracks, get_tracking_summary
from utils import find_output_video


st.set_page_config(
    page_title="AI Vision Tracker",
    page_icon="👁️",
    layout="wide"
)


@st.cache_resource
def get_model():
    return load_model()


model = get_model()


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

        class_ids = get_class_ids(
            model,
            self.selected_classes
        )

        if class_ids:
            track_args["classes"] = class_ids

        results = model.track(**track_args)
        annotated_frame = results[0].plot()

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

    available_classes = list(model.names.values())

    selected_classes = st.multiselect(
        "Track specific objects",
        options=available_classes,
        default=[]
    )

    st.markdown("---")

    st.caption("Computer Vision • YOLO")


st.title("AI Vision Tracker")

st.write(
    "Detect and track objects using YOLO "
    "from live webcam or uploaded video."
)


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

        try:
            with st.spinner("Detecting and tracking objects..."):
                tracking_args = {
                    "source": video_path,
                    "conf": confidence,
                    "save": True,
                    "persist": True
                }

                class_ids = get_class_ids(
                    model,
                    selected_classes
                )

                if class_ids:
                    tracking_args["classes"] = class_ids

                results = model.track(**tracking_args)

            st.success("Analysis complete!")

            object_tracks = {}

            for result in results:
                update_object_tracks(
                    object_tracks,
                    result,
                    model
                )

            (
                total_unique_objects,
                total_object_types
            ) = get_tracking_summary(object_tracks)

            st.subheader("Analysis Overview")

            metric1, metric2, metric3 = st.columns(3)

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
                st.info("No trackable objects detected.")

            output_video = find_output_video(
                results[0].save_dir
            )

            if output_video:
                st.subheader("Detection & Tracking Result")

                st.video(str(output_video))

                with open(output_video, "rb") as video_file:
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

        finally:
            if os.path.exists(video_path):
                os.remove(video_path)