# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This FaceDetector module performs face detection and return bounding box.
"""
import torch

from face_detector.vision.ssd.config.fd_config import define_img_size
from face_detector.vision.ssd.mb_tiny_RFB_fd import (
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)
import numpy as np


class FaceDetector:
    def __init__(self):
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
        boxes, labels, probs = self.predictor.predict(
            image, self.candidate_size / 2, self.threshold
        )
        boxes_lst = list()
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            box[0], box[1], box[2], box[3] = (box[0], box[1], box[2], box[3])
            box = np.array(box).astype(int)
            boxes_lst.append(box)
        return boxes_lst
