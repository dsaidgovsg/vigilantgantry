# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This video processor module ingestes video streams from either RTSP, webcam or video file and performs face segmentation.


"""
import logging

import cv2
from vidgear.gears import CamGear

from standard_logs.logger import setup_logging
from video_processor.video_streamer import VideoStreamer
from video_processor.utils import if_rtsp, if_webcam

from pipeline.pipeline import run_pipeline

setup_logging()
logger = logging.getLogger(name="video_processor:video_processor.py")
logger = logging.getLogger(__name__)


class VideoProcessor(VideoStreamer):
    def __init__(
        self,
        video_source,
        video_width,
        video_height,
        person_detect_roi_boundary,
        person_detect_intercept_boundary,
        gantry_id,
        face_seg_threshold_value,
        full_screen_display,
    ):
        super(VideoProcessor, self).__init__(video_source, video_width, video_height)
        self.person_detect_roi_boundary = person_detect_roi_boundary
        self.person_detect_intercept_boundary = person_detect_intercept_boundary
        self.gantry_id = gantry_id
        self.face_seg_threshold_value = face_seg_threshold_value
        self.full_screen_display = full_screen_display

    def process_video(self, person_detector, face_detector, face_segmentor):
        self.person_detector = person_detector
        self.face_detector = face_detector
        self.face_segmentor = face_segmentor

        if if_rtsp(self.video_source) or if_webcam(self.video_source):
            vid = VideoStreamer(self.video_source).start()
        else:
            vid = CamGear(source=self.video_source).start()

        while True:
            frame = vid.read()
            if frame is None:
                break
            frame = self.process_frame(frame)
            cv2.imshow("VigilantGantry Face Segmentation AI Engine", frame)
            if cv2.waitKey(1) == 27:
                break
        vid.stop()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        roi = self.person_detect_roi_boundary
        cv2.rectangle(
            frame, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]), (255, 255, 255)
        )
        roi_frame = frame[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]]
        roi_frame = run_pipeline(
            frame,
            self.person_detector,
            self.face_detector,
            self.face_segmentor,
            self.face_seg_threshold_value,
            self.person_detect_roi_boundary,
            self.person_detect_intercept_boundary,
            self.gantry_id,
        )
        frame[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]] = roi_frame

        return frame
