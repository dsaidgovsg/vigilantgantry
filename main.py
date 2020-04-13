import argparse, os
from utils import vid_utils


ENDPOINT_HOST = os.getenv("EXAMPLE_HOST", default="example_host:5000")
ENDPOINT = "live_results"
DISPLAY_SCREEN = os.getenv("DISPLAY_SCREEN", default=True)
THRESHOLD = float(os.getenv("THRESHOLD", default=0.5))
VIDEO_SOURCE_LANE = os.getenv("VIDEO_SOURCE_LANE","sample/sample.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="face segmentation engine")
    parser.add_argument("--video_source", type=str, default=VIDEO_SOURCE_LANE)
    parser.add_argument(
        "--endpoint", type=str, default="http://" + ENDPOINT_HOST + "/" + ENDPOINT
    )    
    parser.add_argument(
        "--threshold_value", type=int, default=THRESHOLD
    )
    args = parser.parse_args()
    vid_utils.process_video(
        vid_path=args.video_source,
        roi = ((500, 100), (760, 720)),
        lane_hitzone = [600,350,700,450],
        endpoint=args.endpoint,
        gantry_id=1,
        threshold_value=args.threshold_value,
    )