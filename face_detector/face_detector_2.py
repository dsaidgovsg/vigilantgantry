# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
This FaceDetector module performs face detection and return bounding box.
"""
import torch

# from face_seg.nets.MobileNetV2_unet import MobileNetV2_unet
from face_detector.vision.ssd.config.fd_config import define_img_size
from face_detector.vision.ssd.mb_tiny_RFB_fd import (
    create_Mb_Tiny_RFB_fd,
    create_Mb_Tiny_RFB_fd_predictor,
)
from face_detector.vision.utils.misc import Timer
import numpy as np

pre_trained = "face_seg/checkpoints/model.pt"


# def load_model():
#     model = MobileNetV2_unet(None).to("cuda")
#     state_dict = torch.load(pre_trained, map_location="cuda")
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model


# model = load_model()
input_img_size = 480
define_img_size(input_img_size)

label_path = "face_det/models/voc-model-labels.txt"
# net_type = "mb_tiny_RFB_fd"
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = "cuda:0"
candidate_size = 1000
threshold = 0.5
model_path = "face_det/models/Mb_Tiny_RFB_FD_train_input_320.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
net.load(model_path)
predictor = create_Mb_Tiny_RFB_fd_predictor(
    net, candidate_size=candidate_size, device=test_device
)


class FaceDetector:
    def __init__(self):
        self.class_names = [name.strip() for name in open(label_path).readlines()]
        self.num_classes = len(class_names)
        self.test_device = "cuda:0"
        self.candidate_size = 1000
        self.threshold = 0.5
        self.model_path = "face_det/models/Mb_Tiny_RFB_FD_train_input_320.pth"
        self.net = create_Mb_Tiny_RFB_fd(
            len(class_names), is_test=True, device=test_device
        )
        self.net.load(model_path)
        self.predictor = create_Mb_Tiny_RFB_fd_predictor(
            net, candidate_size=candidate_size, device=test_device
        )

    def check(self):
        pass

    def model(self):
        model = MobileNetV2_unet(None).to("cuda")
        state_dict = torch.load(self.model_path, map_location="cuda")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def get_face_bbox(self, image):
        boxes, labels, probs = predictor.predict(
            image, self.candidate_size / 2, self.threshold
        )
        boxes_lst = list()
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            box[0], box[1], box[2], box[3] = (box[0], box[1], box[2], box[3])
            box = np.array(box).astype(int)
            boxes_lst.append(box)
        return boxes_lst

    def inference(self, images):
        pass
