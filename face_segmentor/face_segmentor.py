# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This FaceSegmentor module performs face segmentation.

"""
import logging
import PIL

import cv2
import torch
import numpy as np
from torchvision import transforms

from standard_logs.logger import setup_logging
from pipeline.utils import *
from face_segmentor.nets.MobileNetV2_unet import MobileNetV2_unet

setup_logging()
logger = logging.getLogger(name="video_processor:video_processor.py")
logger = logging.getLogger(__name__)


class FaceSegmentor:
    """
     FaceSegmentor class that is configurable and returns amount of face skin detected. 

    """

    def __init__(self):
        self.pre_trained = "face_segmentor/checkpoints/model.pt"
        self.model = self.load_model()

    def load_model(self):
        """
        load_model: Load face segmentation model

        :return: face segmentation model
        :rtype: PyTorch Tensors
        """
        model = MobileNetV2_unet(None).to("cuda")
        state_dict = torch.load(self.pre_trained, map_location="cuda")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_segmentation_value(self, image, face_coord):
        """
        get_segmentation_value:  Returns the percentage of exposed skin to face ration

        :param image: Frame
        :type image: np.array
        :param face_coord: Coordinates of face
        :type face_coord: list of
        :return: tuple
        :rtype: float and np.array
        """
        f_min_x, f_min_y, f_max_x, f_max_y = xywh2xyxy(face_coord)
        h, w = f_max_y - f_min_y, f_max_x - f_min_x
        pil_img_c = image[f_min_y : f_min_y + h, f_min_x : f_min_x + w]
        pil_img = PIL.Image.fromarray(pil_img_c)

        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        torch_img_c = transform(pil_img)
        torch_img = torch_img_c.unsqueeze(0)
        torch_img_a = torch_img.to("cuda")
        logits = self.model(torch_img_a)
        mask = torch.max(logits, 1)[1]
        masked_img_c = mask.squeeze().data.cpu().numpy()
        masked_img = np.uint8(masked_img_c)
        masked_img_b = np.where(masked_img == 2, 1, masked_img)
        masked_img_b = cv2.resize(
            masked_img_b, (pil_img_c.shape[1], pil_img_c.shape[0])
        )

        image = self.display_segmentation(image, masked_img_b, face_coord)

        return (
            (np.sum(masked_img_b) / (masked_img_b.shape[0] * masked_img_b.shape[1])),
            image,
        )

    def display_segmentation(self, image, masked_img_b, face_coord):
        """
        display_segmentation visualize face segmentation

        [extended_summary]

        :param image: frame
        :type image: np.array
        :param masked_img_b: image
        :type masked_img_b: np.array
        :param face_coord: Coordinate of person face
        :type face_coord: List of np.array
        :return: Frame
        :rtype: np.array
        """
        f_min_x, f_min_y, f_max_x, f_max_y = xywhTOx1y1x2y2_bbox(face_coord)
        masked_img_b = np.where(masked_img_b == 1, 255, masked_img_b)
        masked_img_b = np.where(masked_img_b == 2, 255, masked_img_b)
        masked_img = cv2.cvtColor(masked_img_b, cv2.COLOR_GRAY2BGR)
        indices = np.where(masked_img == 255)
        masked_img[indices[0], indices[1], :] = [0, 255, 0]
        image[
            f_min_y : f_min_y + masked_img.shape[0],
            f_min_x : f_min_x + masked_img.shape[1],
        ] = masked_img
        return image


def xywhTOx1y1x2y2_bbox(x):
    return [x[0], x[1], x[0] + x[2], x[1] + x[3]]
