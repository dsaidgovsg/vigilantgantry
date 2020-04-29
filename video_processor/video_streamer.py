# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This module is a wrapper around OpenCV which allows for asynchronous video capturing from RTSP address.


Credits: http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/
"""
import logging

import cv2
import threading


from standard_logs.logger import setup_logging


setup_logging()
logger = logging.getLogger(name="video_processor:video_streamer.py")
logger = logging.getLogger(__name__)


import logging

import cv2
import threading


from standard_logs.logger import setup_logging


setup_logging()
logger = logging.getLogger(name="video_processor:video_streamer.py")
logger = logging.getLogger(__name__)


class VideoStreamer:
    def __init__(self, video_source):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.thread = None

    def set(self, var1, var2):
        self.cap.set(var1, var2)
        return None

    def start(self):
        if self.started:
            logger.info("VideoStreamer has started")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
        return None

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return frame

    def stop(self):
        self.started = False
        self.thread.join()
        return None

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
        return None
