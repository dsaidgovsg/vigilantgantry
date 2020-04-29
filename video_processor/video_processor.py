# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This video processor module ingest video streams from either RTSP, webcam or video file and processes

Author: GovTech Singapore
"""
import logging

import cv2
from vidgear.gears import CamGear

from standard_logs.logger import setup_logging
from video_processor.video_streamer import VideoStreamer
from video_processor.utils import if_rtsp, if_webcam, if_face_trigger_inside_person_roi

from pipeline.pipeline import run_pipeline

setup_logging()
logger = logging.getLogger(name="video_processor:video_processor.py")
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    VideoProcess processes video

    """

    def __init__(
        self,
        video_source,
        video_width,
        video_height,
        person_detect_roi_boundary,
        face_segmentation_trigger_boundary,
        gantry_id,
        face_seg_threshold_value,
        full_screen_display,
    ):
        """

        Initialises VideoProcess class.

        :param video_source: Video source path from video file, RTSP or webcam
        :type video_source: str
        :param video_width: Video source width
        :type video_width:  int
        :param video_height: Video source height
        :type video_height: int
        :param person_detect_roi_boundary: ROI for person detection 
        :type person_detect_roi_boundary: tuple
        :param face_segmentation_trigger_boundary: Trigger boundary for face segmentation
        :type face_segmentation_trigger_boundary: tuple
        :param gantry_id: Gantry ID
        :type gantry_id: int
        :param face_seg_threshold_value: Threshold for proportion of exposed face over total face area
        :type face_seg_threshold_value:  Float
        :param full_screen_display: Display fullscreen
        :type full_screen_display: True
        """
        self.video_source = video_source
        self.person_detect_roi_boundary = person_detect_roi_boundary
        self.face_segmentation_trigger_boundary = face_segmentation_trigger_boundary
        self.gantry_id = gantry_id
        self.face_seg_threshold_value = face_seg_threshold_value
        self.full_screen_display = full_screen_display

        if_face_trigger_inside_person_roi(
            person_detect_roi_boundary, face_segmentation_trigger_boundary
        )

    def process_video(self, person_detector, face_detector, face_segmentor):
        """
        Processes video with person detection, face detection and face segmentation.

        :param person_detector: Person Detection
        :type person_detector: PersonDetector class
        :param face_detector: Face Detection
        :type face_detector: FaceDetector class
        :param face_segmentor: Face Segmentation 
        :type face_segmentor: FaceSegmentation class
        """
        self.person_detector = person_detector
        self.face_detector = face_detector
        self.face_segmentor = face_segmentor

        if if_rtsp(self.video_source):
            vid = VideoStreamer(self.video_source).start()

        if if_webcam(self.video_source):
            vid = CamGear(source=int(self.video_source)).start()

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
        return None

    def process_frame(self, frame):
        """
        Processes indivdual frame within a defined region of interest (ROI)

        :param frame: Video frame
        :type frame: np.array
        :return: Processed video frame
        :rtype: np.array
        """
        roi = self.person_detect_roi_boundary
        cv2.rectangle(
            frame, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]), (255, 255, 255)
        )
        frame_inside_roi = frame[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]]
        processed_frame_inside_roi = run_pipeline(
            frame_inside_roi,
            self.person_detector,
            self.face_detector,
            self.face_segmentor,
            self.face_seg_threshold_value,
            self.person_detect_roi_boundary,
            self.face_segmentation_trigger_boundary,
            self.gantry_id,
        )
        frame[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]] = processed_frame_inside_roi

        return frame
