# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This files contains helper functions for the VideoProcesor module


"""


def if_rtsp(video_source):
    return True if video_source[:4] == "rtsp" else False


def if_webcam(video_source):
    return type(video_source) is int
