import argparse, os
from utils import vid_utils

# Environment variables that can be configured

# Boolean to display results
DISPLAY_SCREEN = bool(os.getenv("DISPLAY_SCREEN", default=True))
# Threshold for face segmentation algoriithm
THRESHOLD = float(os.getenv("THRESHOLD", default=0.5))
# Source of video for face segmentation algorithm
VIDEO_SOURCE_LANE = os.getenv("VIDEO_SOURCE_LANE","sample/sample.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face segmentation engine")
    parser.add_argument("--video_source", type=str, default=VIDEO_SOURCE_LANE)
    parser.add_argument(
        "--threshold_value", type=int, default=THRESHOLD
    )
    args = parser.parse_args()

    # configurable lane_hitzone to change the area of interest
    vid_utils.process_video(
        vid_path=args.video_source,
        roi = ((500, 100), (760, 720)),
        lane_hitzone = [600,350,700,450],
        gantry_id=1,
        threshold_value=args.threshold_value,
    )