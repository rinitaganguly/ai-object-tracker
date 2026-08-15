from pathlib import Path


def find_output_video(save_dir):
    output_dir = Path(save_dir)

    if not output_dir.exists():
        return None

    video_files = [
        file
        for file in output_dir.iterdir()
        if file.suffix.lower() in [".mp4", ".avi", ".mov"]
    ]

    if not video_files:
        return None

    return video_files[0]


def get_file_extension(filename):
    return Path(filename).suffix