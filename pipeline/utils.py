# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This file containts helper functions that supports the pipepline.

Author: GovTech Singapore
"""
import uuid
import time

import cv2


def xyxy2xywh(coordinate):
    """
    Convert bounding box coordinates (minX, minY, maxX, maxY to minX, minY, height, width)

    :param x: Coordinates of bounding box (minX, minY, maxX, maxY)
    :type coordinates: tuple
    :return: Converted bounding box coordinates (minX, minY, height, width)
    :rtype: tuple
    """
    return (
        coordinate[0],
        coordinate[1],
        coordinate[2] - coordinate[0],
        coordinate[3] - coordinate[1],
    )


def xywh2xyxy(coordinate):
    """
    Convert bounding box coordinates (minX, minY, maxX, maxY to minX, minY, height, width)

    :param x: Coordinates of bounding box (minX, minY, height, width)
    :type coordinates: tuple
    :return: Converted bounding box coordinates (minX, minY, maxX, maxY)
    :rtype: tuple
    """

    return (
        coordinate[0],
        coordinate[1],
        coordinate[0] + coordinate[2],
        coordinate[1] + coordinate[3],
    )


def get_centriod(coordinate):
    """
    Get centriod of box coordinates

    :param x: Coordinates of bounding box (minX, minY, height, width)
    :type coordinates: tuple
    :return: Centrol of bounding box (X, Y)
    :rtype: tuple 
    """
    x1, y1, x2, y2 = coordinate
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def if_point_intersect_with_rect(rect, pt):
    """
    Check if point is inside rectange (minX, minY, maxX, maxY to minX)

    :param rect: Bounding box coordinates(minX, minY, maxX, maxY)
    :type rect: tuple
    :param pt: Points (X, Y)
    :type pt: tuple
    :return: If point is inside bounding box 
    :rtype: bool
    """
    return pt[0] > rect[0] and pt[0] < rect[2] and pt[1] > rect[1] and pt[1] < rect[3]


def get_frame_inside_roi(frame_inside_roi, bbox):
    """
    Frame inside of ROI

    :param frame_inside_roi: Video Frame
    :type frame_inside_roi: np.array
    :param bbox: Define bounding box (minX, minY, maxX, maxY to minX)
    :type bbox: np.array
    :return: Smaller frame inside of ROI
    :rtype: np.array
    """
    f_min_x, f_min_y, f_max_x, f_max_y = bbox
    h, w = f_max_y - f_min_y, f_max_x - f_min_x
    frame_inside_roi = frame_inside_roi[:, :, ::-1]
    frame_inside_roi = frame_inside_roi[f_min_y : f_min_y + h, f_min_x : f_min_x + w]
    return frame_inside_roi


def push_output_result(gantry_id, verify):
    """
    Dictionary Schema for Output 

    :param gantry_id: Gantry ID
    :type gantry_id: int
    :param verify: 1 for verified and 0 for not verified
    :type verify: int
    :return: Dictionary to be converted to JSON
    :rtype: dict
    """
    uuid_id = str(uuid.uuid4())
    millis = int(round(time.time() * 1000))
    output_json = {
        "request_id": uuid_id,
        "gantry_id": gantry_id,
        "timestamp": millis,
        "pass": verify,
        "check_type": "face_segmentation",
    }
    return output_json


def get_intercept_zone(person_detect_roi_boundary, face_segmentation_trigger_boundary):
    """

    Intercept Zone for Face Segmentation

    :param person_detect_roi_boundary: Defined ROI zone for person detection
    :type person_detect_roi_boundary: tuple 
    :param face_segmentation_trigger_boundary: Defined triggered boundary for face segmentation
    :type face_segmentation_trigger_boundary: tuple
    :return: Trigger boundary for face segmentation
    :rtype: tuple
    """
    return (
        face_segmentation_trigger_boundary[0][0] - person_detect_roi_boundary[0][0],
        face_segmentation_trigger_boundary[0][1] - person_detect_roi_boundary[0][1],
        face_segmentation_trigger_boundary[1][0] - person_detect_roi_boundary[0][0],
        face_segmentation_trigger_boundary[1][1] - person_detect_roi_boundary[0][1],
    )


def display_bbox(frame, bbox, text, color="green"):
    """
    Visualises Bounding box

    :param frame: Input video frame
    :type frame: np.array
    :param bbox: Bounding box coordinated
    :type bbox: tuple
    :param text: Text to be placed on top of bounding box
    :type text: str
    :param color: Color of bounding box ('red' or 'green'), defaults to "green"
    :type color: str, optional
    :return: Processed video frame
    :rtype: np.array
    """
    color = (0, 255, 0)
    if color == "red":
        color = (0, 0, 255)
    x1, y1, w, h = bbox
    p1 = (int(x1), int(y1))
    p2 = (int(x1 + w), int(y1 + h))
    cv2.rectangle(frame, p1, p2, color)

    l1 = (bbox[0], bbox[1] - 10)
    l2 = (bbox[2] + bbox[0], bbox[1])
    t1 = (bbox[0], bbox[1] - 3)
    cv2.rectangle(frame, l1, l2, color, cv2.FILLED)

    if sum(color) / 3 < 97:
        text_colour = (255, 255, 255)
    else:
        text_colour = (0, 0, 0)

    cv2.putText(
        frame, text, t1, cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_colour, 1, cv2.LINE_AA
    )

    return frame
