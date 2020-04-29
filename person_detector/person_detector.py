# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This PersonDetector class performs human detection and return a list of bounding boxes.

Author: GovTech Singapore
"""
import logging

import torch
import numpy as np
from person_detector.util import load_classes, write_results
from person_detector.darknet import Darknet
from person_detector.preprocess import prep_image
from torch.autograd import Variable

from standard_logs.logger import setup_logging


setup_logging()
logger = logging.getLogger(name="person_detecor:person_detector.py")
logger = logging.getLogger(__name__)


class PersonDetector:
    """
    PersonDetector class that is configurable and returns persons bounding boxes. 

    """

    def __init__(
        self,
        class_path,
        config_path,
        weights_path,
        batch_size,
        nms_threshold,
        scales,
        confidence,
        resolution,
        num_classes,
    ):
        self.classes = load_classes(class_path)
        self.model = Darknet(config_path)
        self.model.load_weights(weights_path)
        self.model.net_info["height"] = resolution
        self.batch_size = batch_size
        self.nms_threshold = nms_threshold
        self.scales = scales
        self.confidence = confidence
        self.resolution = resolution
        self.num_classes = num_classes

        if torch.cuda.is_available():
            self.cuda = True
            self.model.cuda()

    def model(self):
        """
        model Returns YoloV3 Model

        :return: YoloV3 model
        :rtype: PyTorch Tensors
        """
        self.model.eval()
        return model

    def get_human_bbox(self, frame, img_class="person"):
        """
        get_human_bbox [summary]

        [extended_summary]

        :param frame: Video Frame
        :type frame: np.array
        :param img_class: label from COCO dataset, defaults to "person"
        :type img_class: str, optional
        :return: List of bounding boxes
        :rtype: List of np.array
        """
        bboxs, probs, clses = self.inference(frame)
        human_candidates = []
        for b, c in zip(bboxs, clses):
            if str(self.classes[int(c)]) == img_class:
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                human_candidates.append([x1, y1, w, h])
        return human_candidates

    def inference(self, images):
        start = 0
        imlist = [images]
        inp_dim = int(self.model.net_info["height"])
        batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
        if self.cuda:
            im_dim_list = im_dim_list.cuda()
        for batch in im_batches:
            if self.cuda:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = self.model(Variable(batch), self.cuda)
            output = write_results(
                prediction,
                self.confidence,
                self.num_classes,
                nms=True,
                nms_conf=self.nms_threshold,
            )
            if self.cuda:
                torch.cuda.synchronize()
        try:
            output
        except NameError:
            print("No detections were made")
            exit()
        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
        scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (
            inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)
        ) / 2
        output[:, [2, 4]] -= (
            inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)
        ) / 2
        output[:, 1:5] /= scaling_factor
        human_candidates = []
        scores = []
        im_id_list = []
        for i in range(len(output)):
            item = output[i]
            im_id = item[-1]
            im_id_list.append(im_id)
            if int(im_id) in [0, 7]:
                bbox = item[1:5].cpu().numpy()
                bbox = [round(i, 2) for i in list(bbox)]
                score = item[5]
                human_candidates.append(bbox)
                scores.append(score)
        scores = np.expand_dims(np.array(scores), 0)
        human_candidates = np.array(human_candidates)
        return human_candidates, scores, im_id_list
