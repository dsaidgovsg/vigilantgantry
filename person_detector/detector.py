# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~
Credits: https://github.com/ayooshkathuria/pytorch-yolo-v3
"""
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2

import os
import sys

yolo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, yolo_dir)

from util import *
import argparse
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import random

sys.path.pop(0)

num_classes = 80


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


class args:
    bs = 1
    nms_thresh = 0.4
    cfgfile = yolo_dir + "/cfg/yolov3.cfg"
    weightsfile = yolo_dir + "/yolov3.weights"
    reso = "416"
    scales = "1,2,3"
    confidence = 0.8


scales = args.scales
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()


def load_model():
    start = 0
    classes = load_classes(yolo_dir + "/data/coco.names")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()
    return model


def inference(images, model):
    start = 0
    classes = load_classes(yolo_dir + "/data/coco.names")

    imlist = [images]
    inp_dim = int(model.net_info["height"])
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        output = write_results(
            prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh
        )
        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        exit()
    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
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
