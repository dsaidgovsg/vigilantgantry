# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This files contains helper functions for the VideoProcesor module


"""
from collections import namedtuple, Iterable


def if_rtsp(video_source):
    """
    if_rtsp: Check if video source is RTSP

    :param video_source: video source 
    :type video_source: str
    :return: True if video source is rtsp
    :rtype: bool
    """
    return True if video_source[:4] == "rtsp" else False


def if_webcam(video_source):
    """
    if_webcam: Check if video source is from webcam

    :param video_source: video source
    :type video_source: str
    :return: True if video source is from webcam id string
    :rtype: str
    """
    return video_source.isdigit()


from collections import namedtuple


def flatten(t):
    for x in t:
        if not isinstance(x, Iterable):
            yield x
        else:
            yield from flatten(x)


def contains(person_rect_roi, face_trigger_boundary):
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
    RectangleCoordinate = namedtuple("rect", "xmin ymin xmax ymax")
    person_rect_roi = RectangleCoordinate(*tuple(flatten(person_detect_roi_boundary)))
    face_trigger_boundary = RectangleCoordinate(
        *tuple(flatten(face_segmentation_trigger_boundary))
    )
    if not contains(person_rect_roi, face_trigger_boundary):
        raise Exception(
            "Face segmentation trigger zone cooordinates must be within Person detection ROI coordinates"
        )
