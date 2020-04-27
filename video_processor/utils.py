# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This files contains helper functions for the VideoProcesor module


"""


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
