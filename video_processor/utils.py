# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This files contains helper functions for the VideoProcesor module

Author: GovTech Singapore
"""
from collections import namedtuple, Iterable


def if_rtsp(video_source):
    """
    Check if video source is RTSP.

    :param video_source: video source 
    :type video_source: str
    :return: True if video source is rtsp
    :rtype: bool
    """
    return True if video_source[:4] == "rtsp" else False


def if_webcam(video_source):
    """
    Check if video source is from webcam.

    :param video_source: video source
    :type video_source: str
    :return: True if video source is from webcam id string
    :rtype: str
    """
    try:
        if int(video_source) < 0:
            raise Exception(f"Webcam dev id '{int(video_source)}' cannot be negative")
        else:
            return True
    except ValueError:
        return False


def flatten(tuple):
    """
    Flattern nested tuple

    :param t: Nested tuple
    :type t: tuple
    :yield: Flat tuple
    :rtype: tuple
    """
    for x in tuple:
        if not isinstance(x, Iterable):
            yield x
        else:
            yield from flatten(x)


def contains(person_rect_roi, face_trigger_boundary):
    """
    Check if face segmentation trigger boundary is inside a person detection ROI.

    :param person_rect_roi: Coordinates of person detection ROI
    :type person_rect_roi: RectangleCoordinate
    :param face_trigger_boundary: Coordinates of trigger boundary for face segmentation
    :type face_trigger_boundary: RectangleCoordinate
    :return: True if face segmentation trigger boundary is inside a person detection ROI
    :rtype: bool
    """
    return (
        person_rect_roi.xmin
        < face_trigger_boundary.xmin
        < face_trigger_boundary.xmax
        < person_rect_roi.xmax
        and person_rect_roi.ymin
        < face_trigger_boundary.ymin
        < face_trigger_boundary.ymax
        < person_rect_roi.ymax
    )


def if_face_trigger_inside_person_roi(
    person_detect_roi_boundary, face_segmentation_trigger_boundary
):
    """
    Raises an exception if face segmentation trigger boundary is not inside a person detection ROI.

    :param person_detect_roi_boundary: Coordinates of person detection ROI
    :type person_detect_roi_boundary: tuple (nested)
    :param face_segmentation_trigger_boundary: Coordinates of trigger boundary for face segmentation
    :type face_segmentation_trigger_boundary: tuple (nested)
    :raises Exception: Warning if face segmentation trigger boundary is not inside a person detection ROI.
    """
    RectangleCoordinate = namedtuple("rect", "xmin ymin xmax ymax")
    person_rect_roi = RectangleCoordinate(*tuple(flatten(person_detect_roi_boundary)))
    face_trigger_boundary = RectangleCoordinate(
        *tuple(flatten(face_segmentation_trigger_boundary))
    )
    if not contains(person_rect_roi, face_trigger_boundary):
        raise Exception(
            "Face segmentation trigger zone cooordinates must be within Person detection ROI coordinates"
        )
