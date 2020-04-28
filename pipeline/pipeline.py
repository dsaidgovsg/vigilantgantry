# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
The run_pipeline and run_heuristic connects the various AI algo together.


"""
import cv2
import colorsys
import numpy as np
import threading

from pipeline.utils import (
    xyxy2xywh,
    xywh2xyxy,
    get_centriod,
    if_point_intersect_with_rect,
    get_frame_inside_roi,
    push_output_result,
    get_intercept_zone,
    display_bbox,
)


def run_pipeline(
    frame,
    person_detector,
    face_detector,
    face_segmentor,
    face_seg_threshold_value,
    person_detect_roi_boundary,
    face_segmentation_trigger_boundary,
    gantry_id,
):
    """
    A pipeline to run person detection, face detection and face segmentation.


    :param frame: Video frame
    :type frame: np.array
    :param person_detector: Person detector object
    :type person_detector: PersonDetector class
    :param face_detector: Face detector object
    :type face_detector: FaceDetector class
    :param face_segmentor:  Face segmentation object
    :type face_segmentor: Face segmentation class
    :param face_seg_threshold_value: Percentage of coverage acceptable
    :type face_seg_threshold_value: float
    :param person_detect_roi_boundary: ROI to detect person
    :type person_detect_roi_boundary: ruple
    :param face_segmentation_trigger_boundary: ROI to detect face
    :type face_segmentation_trigger_boundary: tuple
    :param gantry_id: ID of gantry 
    :type gantry_id: int
    :return: Processed Video frame
    :rtype: np.array
    """
    intercept_zone = get_intercept_zone(
        person_detect_roi_boundary, face_segmentation_trigger_boundary
    )
    cv2.rectangle(
        frame,
        (intercept_zone[0], intercept_zone[1]),
        (intercept_zone[2], intercept_zone[3]),
        (255, 255, 255),
        1,
    )
    persons = person_detector.get_human_bbox(frame, "person")
    for person in persons:
        bbox = np.array(person).astype(int)
        if (bbox > 1).all():
            if if_point_intersect_with_rect(intercept_zone, get_centriod(bbox)):
                run_heuristic(
                    frame,
                    bbox,
                    face_seg_threshold_value,
                    face_detector,
                    face_segmentor,
                    gantry_id,
                )
                cv2.circle(frame, get_centriod(bbox), 10, (0, 0, 255), -1)
    return frame


def run_heuristic(
    frame, bbox, threshold_value, face_detector, face_segmentor, gantry_id
):
    """
    Performs business logic for face segmentation.


    :param frame:  Video frame
    :type frame: np.array
    :param bbox: Face bouding box
    :type bbox: tuple
    :param threshold_value: Percentage of coverage acceptable
    :type threshold_value: float
    :param face_detector: FaceDetector class
    :type face_detector: Face detecto object
    :param face_segmentor: FaceSegmentor class
    :type face_segmentor: Face segmentor object
    :param gantry_id: ID of gantry 
    :type gantry_id: int
    :return: output from face segmentation
    :rtype: dict
    """
    frame_inside_roi = get_frame_inside_roi(frame, xywh2xyxy(bbox))
    boxes = face_detector.get_face_bbox(frame_inside_roi)
    computed_seg_value, _ = face_segmentor.get_segmentation_value(
        frame_inside_roi, xyxy2xywh(boxes[0])
    )
    if computed_seg_value > threshold_value:
        frame = display_bbox(frame, bbox, "NOT COVERED")
        output = push_output_result(gantry_id, 1)
    else:
        frame = display_bbox(frame, bbox, "COVERED", "red")
        output = push_output_result(gantry_id, 0)
    return output
