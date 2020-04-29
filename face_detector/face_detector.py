# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This FaceDetector module performs face detection and return face bounding box.

Author: GovTech Singapore
"""
import logging

import torch
import numpy as np

from face_detector.vision.ssd.config.fd_config import define_img_size
from face_detector.vision.ssd.mb_tiny_RFB_fd import (
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)
from standard_logs.logger import setup_logging

setup_logging()
logger = logging.getLogger(name="video_processor:video_processor.py")
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    FaceDetector class that is configurable and returns face bounding boxes. 
    """

    def __init__(self):
        """
        __init__: For configuration

        """
        input_img_size = 480
        define_img_size(input_img_size)
        label_path = "face_detector/models/voc-model-labels.txt"
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.num_classes = len(self.class_names)
        self.test_device = "cuda:0"
        self.candidate_size = 1000
        self.threshold = 0.5
        self.model_path = "face_detector/models/Mb_Tiny_RFB_FD_train_input_320.pth"
        self.net = create_Mb_Tiny_RFB_fd(
            len(self.class_names), is_test=True, device=self.test_device
        )
        self.net.load(self.model_path)
        self.predictor = create_Mb_Tiny_RFB_fd_predictor(
            self.net, candidate_size=self.candidate_size, device=self.test_device
        )

    def get_face_bbox(self, image):
        """
        get_face_bbox: Return a list containing bounding boxes.

        :param image: np.array for frame
        :type image: np.array
        :return: list of containing a list of bounding boxes
        :rtype: list containing np.array
        """
        boxes, labels, probs = self.predictor.predict(
            image, self.candidate_size / 2, self.threshold
        )
        boxes_lst = list()
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            box = np.array((box[0], box[1], box[2], box[3])).astype(int)
            boxes_lst.append(box)
        return boxes_lst
